"""
Profile TitansGPT or Qwen-Titans training loop to identify performance bottlenecks.

This profiler closely matches the actual training loop for accurate measurements.

Usage:
    # Profile TitansGPT (nanoGPT-based)
    uv run python -m nanogpt_titans.profile_model
    uv run python -m nanogpt_titans.profile_model --compile --n_layer 12 --n_embd 768

    # Profile Qwen-Titans (HuggingFace-based)
    uv run python -m nanogpt_titans.profile_model --qwen
    uv run python -m nanogpt_titans.profile_model --qwen --model_name Qwen/Qwen2-0.5B
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import torch
from torch.profiler import ProfilerActivity, profile

from nanogpt_titans.model import TitansConfig, TitansGPT

# Optional 8-bit optimizer
_BITSANDBYTES_AVAILABLE = False
try:
    from bitsandbytes.optim import AdamW8bit

    _BITSANDBYTES_AVAILABLE = True
except ImportError:
    AdamW8bit = None  # type: ignore[misc, assignment]

# Optional Qwen/HuggingFace support
_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM

    from nanogpt_titans.qwen_titans import (
        TitansQwenConfig,
        freeze_base_model,
        get_gate_statistics,
        get_internal_losses,
        get_titans_layers,
        patch_qwen_with_titans,
    )

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def profile_training_iteration(
    model: TitansGPT,
    x: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    num_iters: int = 10,
    warmup_iters: int = 3,
    use_8bit: bool = False,
) -> dict[str, float]:
    """
    Profile a complete training iteration (forward + backward + optimizer step).

    This matches the actual training loop as closely as possible.
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]

    # Setup matching train.py
    device_type = "cuda" if device.type == "cuda" else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    # Optimizer selection
    if use_8bit:
        if not _BITSANDBYTES_AVAILABLE:
            raise ImportError("8-bit AdamW requires bitsandbytes: pip install bitsandbytes")
        print("Using 8-bit AdamW (bitsandbytes)")
        optimizer = AdamW8bit(model.parameters(), lr=1e-4)
    else:
        use_fused = device_type == "cuda"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)

    model.train()  # Training mode, not eval!

    # Warmup (important for torch.compile and CUDA)
    print(f"Warming up ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        memory_states = model.init_memory_states(batch_size, device)
        with ctx:
            _logits, loss, memory_states = model(x, targets=targets, memory_states=memory_states)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    # Timed runs
    print(f"Profiling ({num_iters} iterations)...")
    timings = {
        "forward": [],
        "backward": [],
        "optimizer": [],
        "total": [],
    }

    for _ in range(num_iters):
        optimizer.zero_grad(set_to_none=True)
        memory_states = model.init_memory_states(batch_size, device)

        # Forward
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with ctx:
            _logits, loss, memory_states = model(x, targets=targets, memory_states=memory_states)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings["forward"].append(t1 - t0)

        # Backward
        scaler.scale(loss).backward()

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        timings["backward"].append(t2 - t1)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        timings["optimizer"].append(t3 - t2)
        timings["total"].append(t3 - t0)

    # Compute averages
    results = {}
    for name, times in timings.items():
        results[f"{name}_ms"] = 1000 * sum(times) / len(times)

    results["tokens_per_sec"] = batch_size * seq_len / (results["total_ms"] / 1000)

    return results


def profile_with_torch_profiler(
    model: TitansGPT,
    x: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    num_iters: int = 3,
    export_trace: bool = False,
    use_8bit: bool = False,
) -> None:
    """
    Profile using torch.profiler for detailed CUDA kernel analysis.
    Profiles the complete training iteration (forward + backward).
    """
    batch_size = x.shape[0]

    device_type = "cuda" if device.type == "cuda" else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    # Optimizer selection
    if use_8bit:
        if not _BITSANDBYTES_AVAILABLE:
            raise ImportError("8-bit AdamW requires bitsandbytes: pip install bitsandbytes")
        print("Using 8-bit AdamW (bitsandbytes)")
        optimizer = AdamW8bit(model.parameters(), lr=1e-4)
    else:
        use_fused = device_type == "cuda"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)

    model.train()

    print("\n" + "=" * 70)
    print("TORCH PROFILER - TRAINING ITERATION ANALYSIS")
    print("=" * 70)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        memory_states = model.init_memory_states(batch_size, device)
        with ctx:
            _logits, loss, _ = model(x, targets=targets, memory_states=memory_states)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()

    # Profile complete training iteration
    print(f"Profiling {num_iters} training iterations (forward + backward + optim)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            memory_states = model.init_memory_states(batch_size, device)

            with ctx:
                _logits, loss, _ = model(x, targets=targets, memory_states=memory_states)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # Print results sorted by CUDA time
    print("\n" + "=" * 70)
    print("TOP CUDA OPERATIONS (by total CUDA time)")
    print("=" * 70)
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30,
        )
    )

    # Export Chrome trace if requested
    if export_trace:
        trace_path = "trace.json"
        prof.export_chrome_trace(trace_path)
        print(f"\nChrome trace exported to: {trace_path}")
        print("Open chrome://tracing and load the file to visualize.")

    # Summarize by category
    print("\n" + "=" * 70)
    print("KERNEL CATEGORY SUMMARY")
    print("=" * 70)

    categories = {
        "attention": 0.0,
        "matmul/gemm": 0.0,
        "memory_ops": 0.0,
        "elementwise": 0.0,
        "triton/fused": 0.0,
        "optimizer": 0.0,
        "other": 0.0,
    }
    other_ops: list[tuple[str, float]] = []  # Track what goes into "other"

    for e in prof.key_averages():
        # Get device time (works across PyTorch versions)
        cuda_time = getattr(e, "device_time_total", 0) or 0
        if cuda_time <= 0:
            cuda_time = getattr(e, "cuda_time_total", 0) or 0
        if cuda_time <= 0:
            continue

        key = e.key.lower()
        # Categorize by operation type
        if "attention" in key or "softmax" in key or "flash" in key or "sdpa" in key or "flex" in key:
            categories["attention"] += cuda_time
        elif "gemm" in key or "matmul" in key or "::mm" in key or "addmm" in key or "bmm" in key:
            categories["matmul/gemm"] += cuda_time
        elif "adamw" in key or "adam" in key or "optimizer" in key:
            categories["optimizer"] += cuda_time
        elif "triton" in key or "cross_entropy" in key:
            categories["triton/fused"] += cuda_time
        elif "copy" in key or "memcpy" in key or "memset" in key or "to_copy" in key:
            categories["memory_ops"] += cuda_time
        elif "elementwise" in key or "add_" in key or "mul_" in key or "foreach" in key or "vectorized" in key:
            categories["elementwise"] += cuda_time
        else:
            categories["other"] += cuda_time
            other_ops.append((e.key, cuda_time))

    total = sum(categories.values())
    print(f"\n{'Category':<20} {'Time (ms)':<12} {'%':<8}")
    print("-" * 40)
    for cat, time_us in sorted(categories.items(), key=lambda x: -x[1]):
        time_ms = time_us / 1000
        pct = 100 * time_us / total if total > 0 else 0
        print(f"{cat:<20} {time_ms:<12.2f} {pct:<8.1f}")
    print("-" * 40)
    print(f"{'Total':<20} {total / 1000:<12.2f}")

    # Show top operations in "other" category
    if other_ops:
        print("\nTop operations in 'other' category:")
        other_ops.sort(key=lambda x: -x[1])
        for op_name, op_time in other_ops[:10]:  # Top 10
            print(f"  {op_time / 1000:>8.2f} ms  {op_name[:60]}")


def diagnose_gradient_flow(
    model_name: str = "Qwen/Qwen2-0.5B",
    memory_layers: list[int] | None = None,
    segment_len: int = 512,
    batch_size: int = 2,
    seq_len: int = 1024,
    num_steps: int = 50,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    dtype_str: str = "bfloat16",
) -> dict:
    """
    Diagnose why mem_scale and gate aren't evolving.

    Tracks:
    - Gradient norms for all TITANS parameters
    - Parameter values before/after training
    - Learning rate applied to each param group
    - Whether gradients are being zeroed unexpectedly
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("Requires: pip install transformers")

    from nanogpt_titans.qwen_titans import TitansStateManager

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[dtype_str]
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("GRADIENT FLOW DIAGNOSTICS")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Learning rate: {learning_rate}")
    print(f"Steps: {num_steps}")
    print()

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    qwen_config = model.config
    n_layer = qwen_config.num_hidden_layers
    if memory_layers is None:
        memory_layers = [n_layer // 2]

    # Patch with TITANS
    titans_config = TitansQwenConfig.from_qwen_config(
        qwen_config,
        segment_len=segment_len,
        memory_layers=memory_layers,
        use_cms=True,
    )

    model = patch_qwen_with_titans(model, titans_config, variant="hope")
    freeze_base_model(model)
    titans_layers = get_titans_layers(model)

    model.to(device)

    # Store initial values
    initial_values = {}
    for layer in titans_layers:
        layer_key = f"layer_{layer.layer_idx}"
        for name, param in layer.named_parameters():
            if param.requires_grad:
                key = f"{layer_key}.{name}"
                initial_values[key] = param.data.clone()

    print(f"\nTracking {len(initial_values)} trainable parameters")

    # Key parameters to watch

    print("\n" + "-" * 70)
    print("INITIAL VALUES:")
    print("-" * 70)
    for layer in titans_layers:
        layer_key = f"layer_{layer.layer_idx}"
        print(f"\n{layer_key}:")
        if hasattr(layer, "mem_scale"):
            val = torch.sigmoid(layer.mem_scale).item()
            print(f"  mem_scale (sigmoid): {val:.6f}")
            print(f"  mem_scale (raw):     {layer.mem_scale.item():.6f}")
        if hasattr(layer, "gate") and hasattr(layer.gate, "gate_mlp"):
            bias = layer.gate.gate_mlp[2].bias.item()
            print(f"  gate bias (raw):     {bias:.6f}")
            print(f"  gate bias (sigmoid): {torch.sigmoid(torch.tensor(bias)).item():.6f}")

    # Setup optimizer - check param groups
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    print(f"\nOptimizer param groups: {len(optimizer.param_groups)}")
    print(f"Total trainable params in optimizer: {sum(p.numel() for g in optimizer.param_groups for p in g['params'])}")

    # State manager
    state_manager = TitansStateManager(model)

    # Create dummy data
    input_ids = torch.randint(0, qwen_config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    # Training loop with detailed gradient tracking
    print("\n" + "=" * 70)
    print("TRAINING WITH GRADIENT TRACKING")
    print("=" * 70)

    model.train()

    device_type = "cuda" if device.type == "cuda" else "cpu"
    ctx = torch.amp.autocast(device_type=device_type, dtype=dtype) if device_type == "cuda" else nullcontext()

    grad_history = []

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        state_manager.reset()
        state_manager.init_states(batch_size, device)

        total_loss = 0.0
        num_segments = 0

        # Process segments
        for start in range(0, seq_len, segment_len):
            end = min(start + segment_len, seq_len)
            seg_input = input_ids[:, start:end]
            seg_labels = labels[:, start:end]

            state_manager.sync_to_layers()

            with ctx:
                outputs = model(seg_input, labels=seg_labels, use_cache=False)
                loss = outputs.loss

            loss.backward()
            total_loss += loss.item()
            num_segments += 1

            state_manager.sync_from_layers()

        # Collect gradient info BEFORE optimizer step
        step_grads = {"step": step, "loss": total_loss / num_segments, "grads": {}, "values": {}}

        for layer in titans_layers:
            layer_key = f"layer_{layer.layer_idx}"

            # mem_scale
            if hasattr(layer, "mem_scale"):
                param = layer.mem_scale
                if param.grad is not None:
                    step_grads["grads"][f"{layer_key}.mem_scale"] = param.grad.item()
                else:
                    step_grads["grads"][f"{layer_key}.mem_scale"] = None
                step_grads["values"][f"{layer_key}.mem_scale"] = param.item()

            # gate bias
            if hasattr(layer, "gate") and hasattr(layer.gate, "gate_mlp"):
                param = layer.gate.gate_mlp[2].bias
                if param.grad is not None:
                    step_grads["grads"][f"{layer_key}.gate_bias"] = param.grad.item()
                else:
                    step_grads["grads"][f"{layer_key}.gate_bias"] = None
                step_grads["values"][f"{layer_key}.gate_bias"] = param.item()

            # gate weight norm
            if hasattr(layer, "gate") and hasattr(layer.gate, "gate_mlp"):
                param = layer.gate.gate_mlp[2].weight
                if param.grad is not None:
                    step_grads["grads"][f"{layer_key}.gate_weight_norm"] = param.grad.norm().item()
                else:
                    step_grads["grads"][f"{layer_key}.gate_weight_norm"] = None

            # mem_proj weight norm
            if hasattr(layer, "mem_proj"):
                for name, param in layer.mem_proj.named_parameters():
                    if param.grad is not None:
                        step_grads["grads"][f"{layer_key}.mem_proj.{name}"] = param.grad.norm().item()

        grad_history.append(step_grads)

        # Clip and step
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        # Print progress
        if step % 10 == 0 or step == num_steps - 1:
            print(f"\nStep {step}:")
            print(f"  Loss: {step_grads['loss']:.4f}")
            for key, grad in step_grads["grads"].items():
                if "mem_scale" in key or "gate_bias" in key:
                    val = step_grads["values"].get(key.replace("_norm", ""), "N/A")
                    grad_str = f"{grad:.2e}" if grad is not None else "None!"
                    print(f"  {key}: grad={grad_str}, value={val:.6f}")

    # Final analysis
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    # Compare initial vs final
    print("\nParameter Changes:")
    for layer in titans_layers:
        layer_key = f"layer_{layer.layer_idx}"
        print(f"\n{layer_key}:")

        if hasattr(layer, "mem_scale"):
            init_val = initial_values.get(f"{layer_key}.mem_scale", torch.tensor(0.0)).item()
            final_val = layer.mem_scale.item()
            delta = final_val - init_val
            print(f"  mem_scale: {init_val:.6f} → {final_val:.6f} (Δ={delta:+.6f})")
            print(
                f"  mem_scale (sigmoid): {torch.sigmoid(torch.tensor(init_val)).item():.4f} → {torch.sigmoid(layer.mem_scale).item():.4f}"
            )

        if hasattr(layer, "gate") and hasattr(layer.gate, "gate_mlp"):
            init_key = f"{layer_key}.gate.gate_mlp.2.bias"
            if init_key in initial_values:
                init_val = initial_values[init_key].item()
                final_val = layer.gate.gate_mlp[2].bias.item()
                delta = final_val - init_val
                print(f"  gate bias: {init_val:.6f} → {final_val:.6f} (Δ={delta:+.6f})")

    # Gradient flow analysis
    print("\n" + "-" * 70)
    print("GRADIENT FLOW ANALYSIS:")
    print("-" * 70)

    # Check if gradients were ever non-zero
    for key in ["mem_scale", "gate_bias"]:
        grads = [h["grads"].get(f"layer_{memory_layers[0]}.{key}") for h in grad_history]
        non_none = [g for g in grads if g is not None]
        non_zero = [g for g in non_none if abs(g) > 1e-10]

        print(f"\n{key}:")
        print(f"  Total steps: {len(grads)}")
        print(f"  Non-None gradients: {len(non_none)}")
        print(f"  Non-zero gradients: {len(non_zero)}")
        if non_zero:
            print(f"  Gradient range: [{min(non_zero):.2e}, {max(non_zero):.2e}]")
            print(f"  Gradient mean: {sum(non_zero) / len(non_zero):.2e}")
        else:
            print("  ⚠️  NO GRADIENTS FLOWING TO THIS PARAMETER!")

    # Diagnose issues
    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    print("=" * 70)

    issues = []

    # Check mem_scale gradients
    mem_scale_grads = [h["grads"].get(f"layer_{memory_layers[0]}.mem_scale") for h in grad_history]
    if all(g is None for g in mem_scale_grads):
        issues.append("mem_scale: No gradients (param not in computation graph)")
    elif all(g is None or abs(g) < 1e-10 for g in mem_scale_grads):
        issues.append("mem_scale: Gradients too small (vanishing gradient)")

    # Check gate gradients
    gate_grads = [h["grads"].get(f"layer_{memory_layers[0]}.gate_bias") for h in grad_history]
    if all(g is None for g in gate_grads):
        issues.append("gate: No gradients (param not in computation graph)")
    elif all(g is None or abs(g) < 1e-10 for g in gate_grads):
        issues.append("gate: Gradients too small (vanishing gradient)")

    # Check if values changed
    for layer in titans_layers:
        layer_key = f"layer_{layer.layer_idx}"
        if hasattr(layer, "mem_scale"):
            init = initial_values.get(f"{layer_key}.mem_scale", torch.tensor(0.0)).item()
            final = layer.mem_scale.item()
            if abs(final - init) < 1e-5:
                issues.append(f"mem_scale: Value unchanged after {num_steps} steps")

    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")

        print("\n💡 POSSIBLE FIXES:")
        if any("No gradients" in i for i in issues):
            print("  1. Check if mem_scale is used in forward pass")
            print("  2. Verify param.requires_grad = True")
            print("  3. Check freeze_base_model() unfreezes these params")
        if any("too small" in i for i in issues):
            print("  1. Increase learning rate (try 5e-4 or 1e-3)")
            print("  2. Check if loss is backpropagating through memory path")
            print("  3. Reduce gradient clipping threshold")
        if any("unchanged" in i for i in issues):
            print("  1. Learning rate may be too low")
            print("  2. Gradients may be clipped to zero")
    else:
        print("\n✅ Gradients are flowing correctly!")

    return {
        "grad_history": grad_history,
        "initial_values": {k: v.item() if v.numel() == 1 else v.tolist() for k, v in initial_values.items()},
        "issues": issues,
    }


def profile_qwen_training(
    model_name: str = "Qwen/Qwen2-0.5B",
    memory_layers: list[int] | None = None,
    segment_len: int = 256,
    batch_size: int = 2,
    seq_len: int = 512,
    num_iters: int = 10,
    warmup_iters: int = 3,
    device: str = "cuda",
    dtype_str: str = "bfloat16",
    detailed: bool = False,
    use_segments: bool = True,
) -> dict[str, float]:
    """
    Profile Qwen-Titans training iteration.

    Args:
        use_segments: If True, profile segment-by-segment processing (matches actual training).
                     If False, profile single forward pass (faster but less accurate).
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError("Qwen profiling requires: pip install transformers accelerate")

    from nanogpt_titans.qwen_titans import TitansStateManager

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[dtype_str]
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("QWEN-TITANS TRAINING PROFILER")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype_str}")
    print(f"Segment-by-segment: {use_segments}")
    print()

    # Load model - use SDPA for faster attention
    print(f"Loading {model_name}...")
    attn_impl = "sdpa" if device.type == "cuda" else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_impl,
    )
    print(f"Using attention implementation: {attn_impl}")

    # Get model config
    qwen_config = model.config
    n_layer = qwen_config.num_hidden_layers
    if memory_layers is None:
        memory_layers = [n_layer // 2]  # Middle layer

    print(f"Model layers: {n_layer}")
    print(f"Memory layers: {memory_layers}")
    print(f"Segment length: {segment_len}")
    print(f"Sequence length: {seq_len} ({seq_len // segment_len} segments)")

    # Create Titans config and patch
    titans_config = TitansQwenConfig.from_qwen_config(
        qwen_config,
        segment_len=segment_len,
        memory_layers=memory_layers,
    )

    print("\nPatching model with HOPE-Titans memory...")
    model = patch_qwen_with_titans(model, titans_config)
    freeze_base_model(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.1f}M")
    print(f"Trainable params: {trainable_params / 1e6:.1f}M ({100 * trainable_params / total_params:.2f}%)")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

    # Setup state manager for segment processing
    state_manager = TitansStateManager(model)

    # Setup autocast
    device_type = "cuda" if device.type == "cuda" else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    model.train()

    # Create dummy input
    input_ids = torch.randint(0, qwen_config.vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    print(f"\nInput shape: {input_ids.shape}")
    print(f"Tokens per iteration: {batch_size * seq_len:,}")
    print()

    # Warmup with segment processing
    print(f"Warming up ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        optimizer.zero_grad(set_to_none=True)
        state_manager.reset()
        state_manager.init_states(batch_size, device)

        if use_segments:
            for start in range(0, seq_len, segment_len):
                end = min(start + segment_len, seq_len)
                seg_input = input_ids[:, start:end]
                seg_labels = labels[:, start:end]

                state_manager.sync_to_layers()
                with ctx:
                    outputs = model(seg_input, labels=seg_labels, use_cache=False)
                    loss = outputs.loss
                    for int_loss in get_internal_losses(model):
                        if int_loss is not None:
                            loss = loss + 0.01 * int_loss
                scaler.scale(loss).backward()
                state_manager.sync_from_layers()
        else:
            with ctx:
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                for int_loss in get_internal_losses(model):
                    if int_loss is not None:
                        loss = loss + 0.01 * int_loss
            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs with detailed segment breakdown
    print(f"Profiling ({num_iters} iterations)...")

    if use_segments:
        # Detailed segment-by-segment profiling
        timings = {
            "state_init": [],
            "sync_to": [],
            "forward": [],
            "backward": [],
            "sync_from": [],
            "optimizer": [],
            "total": [],
        }
        segment_times: list[list[dict]] = []  # Per-iteration segment breakdown

        num_segments = seq_len // segment_len

        for _i in range(num_iters):
            optimizer.zero_grad(set_to_none=True)
            iter_segment_times = []

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            # State init
            t0 = time.perf_counter()
            state_manager.reset()
            state_manager.init_states(batch_size, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings["state_init"].append(time.perf_counter() - t0)

            # Process segments
            total_sync_to = 0.0
            total_forward = 0.0
            total_backward = 0.0
            total_sync_from = 0.0

            for seg_idx, start in enumerate(range(0, seq_len, segment_len)):
                end = min(start + segment_len, seq_len)
                seg_input = input_ids[:, start:end]
                seg_labels = labels[:, start:end]

                # Sync to layers
                t0 = time.perf_counter()
                state_manager.sync_to_layers()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_sync_to = time.perf_counter() - t0
                total_sync_to += t_sync_to

                # Forward
                t0 = time.perf_counter()
                with ctx:
                    outputs = model(seg_input, labels=seg_labels, use_cache=False)
                    loss = outputs.loss
                    for int_loss in get_internal_losses(model):
                        if int_loss is not None:
                            loss = loss + 0.01 * int_loss
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_fwd = time.perf_counter() - t0
                total_forward += t_fwd

                # Backward
                t0 = time.perf_counter()
                scaler.scale(loss).backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_bwd = time.perf_counter() - t0
                total_backward += t_bwd

                # Sync from layers
                t0 = time.perf_counter()
                state_manager.sync_from_layers()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_sync_from = time.perf_counter() - t0
                total_sync_from += t_sync_from

                iter_segment_times.append(
                    {
                        "seg": seg_idx,
                        "sync_to": t_sync_to * 1000,
                        "forward": t_fwd * 1000,
                        "backward": t_bwd * 1000,
                        "sync_from": t_sync_from * 1000,
                    }
                )

            timings["sync_to"].append(total_sync_to)
            timings["forward"].append(total_forward)
            timings["backward"].append(total_backward)
            timings["sync_from"].append(total_sync_from)

            # Optimizer
            t0 = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings["optimizer"].append(time.perf_counter() - t0)

            timings["total"].append(time.perf_counter() - t_start)
            segment_times.append(iter_segment_times)

        # Compute results
        results = {}
        for name, times in timings.items():
            results[f"{name}_ms"] = 1000 * sum(times) / len(times)
        results["tokens_per_sec"] = batch_size * seq_len / (results["total_ms"] / 1000)
        results["num_segments"] = num_segments

        # Print results
        print("\n" + "=" * 70)
        print("TRAINING ITERATION TIMING (segment-by-segment)")
        print("=" * 70)
        print(f"{'Phase':<20} {'Time (ms)':<12} {'%':<8}")
        print("-" * 40)
        total_ms = results["total_ms"]
        for phase in ["state_init", "sync_to", "forward", "backward", "sync_from", "optimizer"]:
            ms = results[f"{phase}_ms"]
            pct = 100 * ms / total_ms
            print(f"{phase:<20} {ms:<12.2f} {pct:<8.1f}")
        print("-" * 40)
        print(f"{'Total':<20} {total_ms:<12.2f} {'100.0':<8}")
        print()
        print(f"Throughput: {results['tokens_per_sec']:,.0f} tokens/sec")
        print(f"Segments per sequence: {num_segments}")
        print(f"Avg time per segment: {(results['forward_ms'] + results['backward_ms']) / num_segments:.2f} ms")

        # Per-segment breakdown (last iteration)
        if segment_times:
            print("\n" + "=" * 70)
            print("PER-SEGMENT TIMING (last iteration, ms)")
            print("=" * 70)
            print(f"{'Seg':<5} {'sync_to':<10} {'forward':<10} {'backward':<10} {'sync_from':<10}")
            print("-" * 45)
            last_iter = segment_times[-1]
            for st in last_iter[:5]:  # First 5
                print(
                    f"{st['seg']:<5} {st['sync_to']:<10.2f} {st['forward']:<10.2f} {st['backward']:<10.2f} {st['sync_from']:<10.2f}"
                )
            if len(last_iter) > 5:
                print("...")
                st = last_iter[-1]
                print(
                    f"{st['seg']:<5} {st['sync_to']:<10.2f} {st['forward']:<10.2f} {st['backward']:<10.2f} {st['sync_from']:<10.2f}"
                )

            # Bottleneck analysis
            avg_fwd = sum(s["forward"] for s in last_iter) / len(last_iter)
            avg_bwd = sum(s["backward"] for s in last_iter) / len(last_iter)
            avg_sync = sum(s["sync_to"] + s["sync_from"] for s in last_iter) / len(last_iter)

            print("\n" + "=" * 70)
            print("BOTTLENECK ANALYSIS")
            print("=" * 70)
            print(f"Avg forward/segment:  {avg_fwd:.2f} ms")
            print(f"Avg backward/segment: {avg_bwd:.2f} ms")
            print(f"Avg sync/segment:     {avg_sync:.2f} ms")

            if avg_fwd > avg_bwd and avg_fwd > avg_sync:
                print("\n→ FORWARD is the bottleneck")
                print("  Consider: larger segment_len, fewer memory layers, flash attention")
            elif avg_bwd > avg_fwd and avg_bwd > avg_sync:
                print("\n→ BACKWARD is the bottleneck")
                print("  Consider: gradient checkpointing, reduce memory layers")
            else:
                print("\n→ SYNC overhead is significant")
                print("  Consider: batch state updates, reduce memory layers")
    else:
        # Original non-segment profiling
        timings = {"forward": [], "backward": [], "optimizer": [], "total": []}

        for _i in range(num_iters):
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Forward
            with ctx:
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                for int_loss in get_internal_losses(model):
                    if int_loss is not None:
                        loss = loss + 0.01 * int_loss

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings["forward"].append(t1 - t0)

            # Backward
            scaler.scale(loss).backward()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            timings["backward"].append(t2 - t1)

            # Optimizer
            scaler.step(optimizer)
            scaler.update()

            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            timings["optimizer"].append(t3 - t2)
            timings["total"].append(t3 - t0)

        # Compute results
        results = {}
        for name, times in timings.items():
            results[f"{name}_ms"] = 1000 * sum(times) / len(times)
        results["tokens_per_sec"] = batch_size * seq_len / (results["total_ms"] / 1000)

        # Print results
        print("\n" + "=" * 70)
        print("TRAINING ITERATION TIMING")
        print("=" * 70)
        print(f"{'Phase':<20} {'Time (ms)':<12}")
        print("-" * 32)
        print(f"{'Forward':<20} {results['forward_ms']:<12.2f}")
        print(f"{'Backward':<20} {results['backward_ms']:<12.2f}")
        print(f"{'Optimizer':<20} {results['optimizer_ms']:<12.2f}")
        print("-" * 32)
        print(f"{'Total':<20} {results['total_ms']:<12.2f}")
        print()
        print(f"Throughput: {results['tokens_per_sec']:,.0f} tokens/sec")

    # Gate statistics
    print("\n" + "=" * 70)
    print("GATE STATISTICS")
    print("=" * 70)
    gate_stats = get_gate_statistics(model)
    if isinstance(gate_stats, dict):
        for layer_idx, stats in gate_stats.items():
            if isinstance(stats, dict):
                print(f"  Layer {layer_idx}: gate mean = {stats.get('mean_gate', 'N/A'):.4f}")
            else:
                print(f"  Layer {layer_idx}: {stats}")
    else:
        print(f"  Gate stats: {gate_stats}")

    # Memory info
    if device.type == "cuda":
        mem_allocated = torch.cuda.max_memory_allocated() / 1e9
        mem_reserved = torch.cuda.max_memory_reserved() / 1e9
        print(f"\nPeak GPU memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile TitansGPT or Qwen-Titans training iteration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    parser.add_argument("--qwen", action="store_true", help="Profile Qwen-Titans instead of TitansGPT")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Diagnose gradient flow for mem_scale/gate (use with --qwen)",
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B", help="Qwen model name (for --qwen)")
    parser.add_argument(
        "--memory_layers",
        type=str,
        default=None,
        help="Memory layer indices, comma-separated (for --qwen)",
    )

    # Profiling options
    parser.add_argument("--detailed", action="store_true", help="Use torch.profiler for kernel analysis")
    parser.add_argument("--export-trace", action="store_true", help="Export Chrome trace file")
    parser.add_argument("--num-iters", type=int, default=10, help="Number of iterations to profile")

    # Model config (for TitansGPT)
    parser.add_argument("--n_layer", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--block_size", type=int, default=1024, help="Context length")
    parser.add_argument("--segment_len", type=int, default=128, help="Segment length for memory")
    parser.add_argument("--num_persist_mem", type=int, default=4, help="Persistent memory tokens")
    parser.add_argument("--num_longterm_mem", type=int, default=16, help="Long-term memory tokens")

    # Training config
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length (for --qwen)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument(
        "--8bit",
        dest="use_8bit",
        action="store_true",
        help="Use 8-bit AdamW (requires bitsandbytes)",
    )
    parser.add_argument(
        "--no-segments",
        dest="use_segments",
        action="store_false",
        help="Profile single forward pass instead of segment-by-segment (faster but less accurate)",
    )

    args = parser.parse_args()

    # Parse memory_layers if provided
    memory_layers = None
    if args.memory_layers:
        memory_layers = [int(x.strip()) for x in args.memory_layers.split(",")]

    # Gradient flow diagnostics
    if args.diagnose:
        diagnose_gradient_flow(
            model_name=args.model_name,
            memory_layers=memory_layers,
            segment_len=args.segment_len,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_steps=args.num_iters * 5,  # More steps for diagnosis
            learning_rate=1e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype_str=args.dtype,
        )
        return

    # Qwen-Titans profiling
    if args.qwen:
        profile_qwen_training(
            model_name=args.model_name,
            memory_layers=memory_layers,
            segment_len=args.segment_len,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_iters=args.num_iters,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype_str=args.dtype,
            detailed=args.detailed,
            use_segments=args.use_segments,
        )
        return

    # TitansGPT profiling (original behavior)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    print("=" * 70)
    print("TITANS TRAINING PROFILER")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Compile: {args.compile}")
    print(f"8-bit optimizer: {args.use_8bit}")
    print()

    # Model config
    config = TitansConfig(
        block_size=args.block_size,
        vocab_size=50304,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        segment_len=args.segment_len,
        num_persist_mem=args.num_persist_mem,
        num_longterm_mem=args.num_longterm_mem,
    )

    print("Model config:")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_head: {config.n_head}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  block_size: {config.block_size}")
    print(f"  segment_len: {config.segment_len}")
    print()

    model = TitansGPT(config).to(device)
    num_params = model.get_num_params()
    print(f"Parameters: {num_params / 1e6:.2f}M")

    if args.compile:
        print("Compiling model (first run will be slow)...")
        model = torch.compile(model)

    # Test input matching training
    x = torch.randint(0, config.vocab_size, (args.batch_size, args.block_size), device=device)
    targets = x.clone()

    print(f"Input shape: {x.shape}")
    print(f"Tokens per iteration: {args.batch_size * args.block_size:,}")
    print()

    if args.detailed:
        profile_with_torch_profiler(
            model,
            x,
            targets,
            device,
            dtype,
            num_iters=args.num_iters,
            export_trace=args.export_trace,
            use_8bit=args.use_8bit,
        )
    else:
        results = profile_training_iteration(
            model,
            x,
            targets,
            device,
            dtype,
            num_iters=args.num_iters,
            use_8bit=args.use_8bit,
        )

        print("\n" + "=" * 70)
        print("TRAINING ITERATION TIMING")
        print("=" * 70)
        print(f"{'Phase':<20} {'Time (ms)':<12}")
        print("-" * 32)
        print(f"{'Forward':<20} {results['forward_ms']:<12.2f}")
        print(f"{'Backward':<20} {results['backward_ms']:<12.2f}")
        print(f"{'Optimizer':<20} {results['optimizer_ms']:<12.2f}")
        print("-" * 32)
        print(f"{'Total':<20} {results['total_ms']:<12.2f}")
        print()
        print(f"Throughput: {results['tokens_per_sec']:,.0f} tokens/sec")
        print()

        # Memory info
        if device.type == "cuda":
            mem_allocated = torch.cuda.max_memory_allocated() / 1e9
            mem_reserved = torch.cuda.max_memory_reserved() / 1e9
            print(f"Peak GPU memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")


if __name__ == "__main__":
    main()

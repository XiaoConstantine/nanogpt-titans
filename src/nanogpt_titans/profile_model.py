"""
Profile TitansGPT training loop to identify performance bottlenecks.

This profiler closely matches the actual training loop for accurate measurements.

Usage:
    uv run python -m nanogpt_titans.profile_model
    uv run python -m nanogpt_titans.profile_model --compile
    uv run python -m nanogpt_titans.profile_model --compile --n_layer 12 --n_embd 768
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import torch
from torch.profiler import ProfilerActivity, profile

from nanogpt_titans.model import TitansConfig, TitansGPT


def profile_training_iteration(
    model: TitansGPT,
    x: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    num_iters: int = 10,
    warmup_iters: int = 3,
) -> dict[str, float]:
    """
    Profile a complete training iteration (forward + backward + optimizer step).

    This matches the actual training loop as closely as possible.
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]

    # Setup matching train.py
    device_type = "cuda" if device.type == "cuda" else "cpu"
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=dtype)
    )
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    # Simple optimizer for profiling (fused=True uses single CUDA kernel)
    use_fused = device_type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)

    model.train()  # Training mode, not eval!

    # Warmup (important for torch.compile and CUDA)
    print(f"Warming up ({warmup_iters} iterations)...")
    for _ in range(warmup_iters):
        memory_states = model.init_memory_states(batch_size, device)
        with ctx:
            _logits, loss, memory_states = model(
                x, targets=targets, memory_states=memory_states
            )
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
            _logits, loss, memory_states = model(
                x, targets=targets, memory_states=memory_states
            )

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
) -> None:
    """
    Profile using torch.profiler for detailed CUDA kernel analysis.
    Profiles the complete training iteration (forward + backward).
    """
    batch_size = x.shape[0]

    device_type = "cuda" if device.type == "cuda" else "cpu"
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=dtype)
    )
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
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
        "triton_custom": 0.0,
        "other": 0.0,
    }

    for e in prof.key_averages():
        # Use self_cuda_time_total (actual kernel time, not including child calls)
        cuda_time = e.self_cuda_time_total
        if cuda_time <= 0:
            continue

        key = e.key.lower()
        if "attention" in key or "softmax" in key or "flash" in key or "sdpa" in key:
            categories["attention"] += cuda_time
        elif "gemm" in key or "matmul" in key or "aten::mm" in key or "addmm" in key:
            categories["matmul/gemm"] += cuda_time
        elif "triton" in key or "_cross_entropy" in key or "_fused" in key:
            categories["triton_custom"] += cuda_time
        elif "copy" in key or "cat" in key or "memcpy" in key or "memset" in key:
            categories["memory_ops"] += cuda_time
        elif "elementwise" in key or "aten::add" in key or "aten::mul" in key or "foreach" in key:
            categories["elementwise"] += cuda_time
        else:
            categories["other"] += cuda_time

    total = sum(categories.values())
    print(f"\n{'Category':<20} {'Time (ms)':<12} {'%':<8}")
    print("-" * 40)
    for cat, time_us in sorted(categories.items(), key=lambda x: -x[1]):
        time_ms = time_us / 1000
        pct = 100 * time_us / total if total > 0 else 0
        print(f"{cat:<20} {time_ms:<12.2f} {pct:<8.1f}")
    print("-" * 40)
    print(f"{'Total':<20} {total / 1000:<12.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile TitansGPT training iteration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Profiling options
    parser.add_argument("--detailed", action="store_true", help="Use torch.profiler for kernel analysis")
    parser.add_argument("--export-trace", action="store_true", help="Export Chrome trace file")
    parser.add_argument("--num-iters", type=int, default=10, help="Number of iterations to profile")

    # Model config (match your training setup)
    parser.add_argument("--n_layer", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--block_size", type=int, default=1024, help="Context length")
    parser.add_argument("--segment_len", type=int, default=128, help="Segment length for memory")
    parser.add_argument("--num_persist_mem", type=int, default=4, help="Persistent memory tokens")
    parser.add_argument("--num_longterm_mem", type=int, default=16, help="Long-term memory tokens")

    # Training config
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    print("=" * 70)
    print("TITANS TRAINING PROFILER")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Compile: {args.compile}")
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
            model, x, targets, device, dtype,
            num_iters=args.num_iters,
            export_trace=args.export_trace,
        )
    else:
        results = profile_training_iteration(
            model, x, targets, device, dtype,
            num_iters=args.num_iters,
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

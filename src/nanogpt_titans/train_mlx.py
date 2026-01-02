#!/usr/bin/env python3
"""
MLX-optimized training for TITANS memory on Apple Silicon.

This implementation matches the PyTorch HOPE architecture exactly:
1. ContinuumMemorySystem (CMS) - Multi-frequency memory (3 levels)
2. PositionDependentGate - Per-token gating
3. NeuralMemory - Memory MLP with key/value/query projections
4. HOPE Integration: output = hidden + gate * scale * LayerNorm(project(memory))

Usage:
    uv run python -m nanogpt_titans.train_mlx --model_name Qwen/Qwen2-0.5B --max_steps 500
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Import from MLX module
from nanogpt_titans.mlx import (
    CombinedModel,
    MLXTitansConfig,
    MLXTitansLayer,
    create_loss_fn,
    filter_titans_grads,
    get_lr,
)

# For loading HuggingFace models
try:
    from mlx_lm import load as mlx_load
except ImportError as e:
    raise ImportError("mlx-lm required: uv add mlx-lm") from e


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer using mlx-lm."""
    print(f"Loading {model_name} with MLX...")
    model, tokenizer = mlx_load(model_name)
    return model, tokenizer


def get_model_dim(model) -> int:
    """Get hidden dimension from model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        sample_layer = model.model.layers[0]
        if hasattr(sample_layer, "hidden_size"):
            return sample_layer.hidden_size
        if hasattr(sample_layer.self_attn, "hidden_size"):
            return sample_layer.self_attn.hidden_size
        return sample_layer.self_attn.q_proj.weight.shape[0]
    return 896  # Default for Qwen2-0.5B


def create_titans_layer(model, config: MLXTitansConfig) -> MLXTitansLayer:
    """Create a single TITANS layer matching model dimensions."""
    dim = get_model_dim(model)

    return MLXTitansLayer(
        dim=dim,
        use_cms=config.use_cms,
        num_cms_levels=config.num_cms_levels,
        cms_update_frequencies=config.cms_update_frequencies,
        memory_depth=config.memory_depth,
        memory_expansion=config.memory_expansion,
        adaptive_memory=config.adaptive_memory,
        memory_lr_max=config.memory_lr_max,
        gate_init_bias=config.gate_init_bias,
    )


def create_titans_layers(model, config: MLXTitansConfig) -> dict[int, MLXTitansLayer]:
    """Create TITANS layers for all specified layer indices."""
    dim = get_model_dim(model)
    num_layers = len(model.model.layers)

    titans_layers = {}
    for layer_idx in config.memory_layers:
        # Clamp to valid range
        idx = min(layer_idx, num_layers - 1)
        titans_layers[idx] = MLXTitansLayer(
            dim=dim,
            use_cms=config.use_cms,
            num_cms_levels=config.num_cms_levels,
            cms_update_frequencies=config.cms_update_frequencies,
            memory_depth=config.memory_depth,
            memory_expansion=config.memory_expansion,
            adaptive_memory=config.adaptive_memory,
            memory_lr_max=config.memory_lr_max,
            gate_init_bias=config.gate_init_bias,
            grad_clip=config.memory_grad_clip,
            surprise_threshold=config.surprise_threshold,
            use_cascade=config.use_cascade,
        )

    return titans_layers


def scale_grads_recursive(grad_tree: Any, factor: float) -> Any:
    """Recursively scale gradients by a constant factor (for averaging)."""
    if isinstance(grad_tree, dict):
        return {k: scale_grads_recursive(v, factor) for k, v in grad_tree.items()}
    if isinstance(grad_tree, mx.array):
        return grad_tree * factor
    return grad_tree


def accumulate_grads(accum_grads: dict | None, new_grads: dict) -> dict:
    """Add new gradients to accumulated gradients."""
    if accum_grads is None:
        return new_grads

    def add_grads(a, b):
        if isinstance(a, dict):
            return {k: add_grads(a[k], b[k]) for k in a}
        if isinstance(a, mx.array):
            return a + b
        return a

    return add_grads(accum_grads, new_grads)


def create_masked_grads(
    grads: dict[str, Any], keep_gate_scale: bool, path: str = "", freeze_gate: bool = False
) -> dict[str, Any]:
    """Create grads with zeros for non-target params."""
    if isinstance(grads, dict):
        return {
            k: create_masked_grads(v, keep_gate_scale, f"{path}.{k}" if path else k, freeze_gate)
            for k, v in grads.items()
        }
    if isinstance(grads, mx.array):
        is_gate = "gate" in path
        is_scale_adaptive = (
            "mem_scale" in path
            or "mem_ln" in path
            or "level_weights" in path
            or "to_lr" in path
            or "to_momentum" in path
            or "to_decay" in path
        )
        is_gate_scale = is_gate or is_scale_adaptive

        if freeze_gate and is_gate:
            return mx.zeros_like(grads)

        if keep_gate_scale:
            return grads if is_gate_scale else mx.zeros_like(grads)
        return mx.zeros_like(grads) if is_gate_scale else grads
    return grads


def get_layers_to_unfreeze(memory_layers: list, num_backbone_layers: int, radius: int) -> set:
    """Return backbone layer indices to unfreeze around TITANS layer positions."""
    if radius <= 0:
        return set()

    layers_to_unfreeze = set()
    for mem_idx in memory_layers:
        # Unfreeze layers before and after the TITANS insertion point
        for offset in range(-radius, radius + 1):
            layer_idx = mem_idx + offset
            if 0 <= layer_idx < num_backbone_layers:
                layers_to_unfreeze.add(layer_idx)

    return layers_to_unfreeze


def train(config: MLXTitansConfig):
    """Main MLX training loop with full HOPE architecture."""
    print("=" * 60)
    print("MLX TITANS Training (Full HOPE Architecture)")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Memory layers: {config.memory_layers}")
    print(f"CMS enabled: {config.use_cms}")
    if config.use_cms:
        print(f"CMS levels: {config.num_cms_levels}")
        print(f"CMS frequencies: {config.cms_update_frequencies}")
        print(f"CMS cascade mode: {config.use_cascade}")
    print(f"Max steps: {config.max_steps}")
    print(f"Internal loss: {config.use_internal_loss} (weight={config.internal_loss_weight})")
    if config.surprise_threshold > 0:
        print(f"Surprise threshold: {config.surprise_threshold}")
    if config.memory_grad_clip > 0:
        print(f"Memory grad clip: {config.memory_grad_clip}")
    if config.unfreeze_backbone_layers > 0:
        print(f"Backbone unfreezing: {config.unfreeze_backbone_layers} layers near TITANS")
    print()

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(config.model_name)

    # Create TITANS layers (independent memory per layer)
    titans_layers = create_titans_layers(model, config)
    num_titans_layers = len(titans_layers)
    print(f"Created {num_titans_layers} TITANS layer(s) at indices: {sorted(titans_layers.keys())}")

    # Count parameters
    titans_params = sum(p.size for layer in titans_layers.values() for _, p in tree_flatten(layer.parameters()))
    print(f"TITANS trainable params: {titans_params:,} ({titans_params // num_titans_layers:,} per layer)")

    # Separate param groups for different LRs
    def count_param_groups_from_module(module):
        """Count gate/scale vs memory params for any module."""
        gate_scale_count = 0
        memory_count = 0

        for name, param in tree_flatten(module.parameters()):
            name_str = ".".join(str(k) for k in name) if isinstance(name, tuple) else str(name)
            if (
                "gate" in name_str
                or "mem_scale" in name_str
                or "mem_ln" in name_str
                or "level_weights" in name_str
                or "to_lr" in name_str
                or "to_momentum" in name_str
                or "to_decay" in name_str
            ):
                gate_scale_count += param.size
            else:
                memory_count += param.size

        return gate_scale_count, memory_count

    gate_scale_count, memory_count = 0, 0
    for layer in titans_layers.values():
        g, m = count_param_groups_from_module(layer)
        gate_scale_count += g
        memory_count += m
    print(f"Gate/scale/CMS params: {gate_scale_count:,}")
    print(f"Memory params: {memory_count:,}")

    # Get number of backbone layers for unfreezing logic
    num_backbone_layers = len(model.model.layers)

    # Determine which backbone layers to unfreeze (if any)
    unfrozen_backbone_indices = get_layers_to_unfreeze(
        config.memory_layers, num_backbone_layers, config.unfreeze_backbone_layers
    )
    if unfrozen_backbone_indices:
        print(f"Unfreezing backbone layers: {sorted(unfrozen_backbone_indices)}")
        unfrozen_backbone_params = sum(
            p.size for idx in unfrozen_backbone_indices for _, p in tree_flatten(model.model.layers[idx].parameters())
        )
        print(f"Unfrozen backbone params: {unfrozen_backbone_params:,}")

    # Create optimizers for different param groups
    gate_lr = config.learning_rate * config.gate_lr_scale
    backbone_lr = config.learning_rate * 0.1  # Lower LR for backbone (fine-tuning)

    optimizer_memory = optim.AdamW(learning_rate=config.learning_rate, weight_decay=0.0)
    optimizer_gate = optim.AdamW(learning_rate=gate_lr, weight_decay=0.0)
    optimizer_backbone = (
        optim.AdamW(learning_rate=backbone_lr, weight_decay=0.01) if unfrozen_backbone_indices else None
    )

    print("\nOptimizer learning rates:")
    print(f"  Memory params: {config.learning_rate:.2e}")
    print(f"  Gate/scale params: {gate_lr:.2e} ({config.gate_lr_scale}x)")
    if unfrozen_backbone_indices:
        print(f"  Backbone params: {backbone_lr:.2e} (0.1x base LR)")

    # Load training data
    print("\nLoading training data...")
    try:
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        texts = [ex["text"] for ex in dataset["train"] if len(ex["text"]) > 100][:500]
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        texts = ["This is a test sentence for training. " * 50] * 100

    print(f"Training examples: {len(texts)}")

    # Tokenize
    print("Tokenizing...")
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) >= 100:
            all_tokens.extend(tokens[: config.max_length])

    all_tokens = mx.array(all_tokens)
    print(f"Total tokens: {len(all_tokens):,}")

    # Training loop
    print("\nStarting training...")
    log_history = []

    # Determine memory layer indices
    num_layers = len(model.model.layers)
    memory_layer_indices = sorted(titans_layers.keys())
    print(f"Applying TITANS at layers {memory_layer_indices} of {num_layers} total")

    # Create combined model wrapper
    combined_model = CombinedModel(
        model,
        titans_layers=titans_layers,
        use_internal_loss=config.use_internal_loss,
        internal_loss_weight=config.internal_loss_weight,
    )
    if config.use_internal_loss:
        print(f"Internal loss enabled with weight={config.internal_loss_weight}")

    # Create loss function with gate regularization
    loss_fn = create_loss_fn(
        combined_model, gate_min_value=config.gate_min_value, gate_reg_weight=config.gate_reg_weight
    )
    loss_and_grad_fn = nn.value_and_grad(combined_model, loss_fn)

    # Note: mx.compile doesn't work well with nn.value_and_grad wrapper
    # The optimization gains from async_eval and batched loss accumulation are sufficient

    # Helper function to evaluate gradient tree (prevents graph accumulation)
    def _eval_grad_tree(tree):
        """Evaluate all arrays in a gradient tree structure."""
        arrays = []

        def collect(t):
            if isinstance(t, dict):
                for v in t.values():
                    collect(v)
            elif isinstance(t, mx.array):
                arrays.append(t)

        collect(tree)
        if arrays:
            mx.eval(*arrays)

    if config.gate_min_value > 0:
        print(f"Gate regularization enabled: min_value={config.gate_min_value}, weight={config.gate_reg_weight}")

    t0 = time.time()
    step_times = []  # Track step times for averaging

    for step in range(config.max_steps):
        step_start = time.perf_counter()

        # Reset memory state at start of each step
        combined_model.reset_memory_state()

        # Gradient accumulation loop
        accumulated_grads = None
        accumulated_backbone_grads = None
        total_loss = mx.array(0.0)  # Keep as mx.array to avoid .item() in loop

        for micro_step in range(config.gradient_accumulation_steps):
            batch_idx = step * config.gradient_accumulation_steps + micro_step
            start_idx = (batch_idx * config.segment_len) % (len(all_tokens) - config.segment_len - 1)
            batch = all_tokens[start_idx : start_idx + config.segment_len + 1]

            input_ids = batch[:-1].reshape(1, -1)
            target_ids = batch[1:].reshape(1, -1)

            loss, full_grads = loss_and_grad_fn(combined_model, input_ids, target_ids)
            titans_grads = filter_titans_grads(full_grads)

            # Extract backbone gradients if unfreezing is enabled
            backbone_grads = None
            if unfrozen_backbone_indices and "base_model" in full_grads:
                base_grads = full_grads.get("base_model", {})
                model_grads = base_grads.get("model", {})
                layers_grads = model_grads.get("layers", {})
                if layers_grads:
                    backbone_grads = {
                        str(idx): layers_grads.get(str(idx), layers_grads.get(idx))
                        for idx in unfrozen_backbone_indices
                        if str(idx) in layers_grads or idx in layers_grads
                    }

            # OPTIMIZATION: Use async_eval to pipeline graph construction with computation
            # This returns control immediately, allowing next iteration to start building graph
            # while previous computation runs on GPU
            if config.eager_eval:
                mx.async_eval(loss)
                _eval_grad_tree(titans_grads)
                if backbone_grads:
                    _eval_grad_tree(backbone_grads)

            scale_factor = 1.0 / config.gradient_accumulation_steps
            titans_grads_scaled = scale_grads_recursive(titans_grads, scale_factor)

            accumulated_grads = accumulate_grads(accumulated_grads, titans_grads_scaled)

            # Accumulate backbone grads separately
            if backbone_grads:
                backbone_grads_scaled = scale_grads_recursive(backbone_grads, scale_factor)
                accumulated_backbone_grads = accumulate_grads(accumulated_backbone_grads, backbone_grads_scaled)

            # OPTIMIZATION: Collect losses without triggering eval
            # Avoid .item() in loop - batch the conversion later
            total_loss = total_loss + loss  # Keep as mx.array

        # OPTIMIZATION: Convert loss to float only once after accumulation
        avg_loss = float((total_loss / config.gradient_accumulation_steps).item())

        # Get current learning rate
        lr = get_lr(step, config)
        lr_gate = lr * config.gate_lr_scale

        # Extract TITANS gradients
        if "titans_layers" in accumulated_grads:
            titans_grads = accumulated_grads["titans_layers"]
        elif "titans_layer" in accumulated_grads:
            titans_grads = accumulated_grads["titans_layer"]
        else:
            titans_grads = accumulated_grads

        # Gate warmup: freeze gate params
        gate_warmup_active = step < config.gate_warmup_steps

        # Apply optimizers
        # OPTIMIZATION: Batch updates - set LR once, collect grads, then update
        params_to_eval = []

        # Set learning rates once (avoid repeated assignments)
        optimizer_memory.learning_rate = lr
        optimizer_gate.learning_rate = lr_gate

        t_opt_start = time.perf_counter()

        # Update each TITANS layer separately
        all_memory_grads = {}
        all_gate_grads = {}

        for layer_idx, _layer in titans_layers.items():
            # Get gradients for this layer
            if isinstance(titans_grads, dict) and str(layer_idx) in titans_grads:
                layer_grads = titans_grads[str(layer_idx)]
            elif isinstance(titans_grads, dict) and layer_idx in titans_grads:
                layer_grads = titans_grads[layer_idx]
            else:
                layer_grads = titans_grads  # Single layer case

            # Pre-compute masked grads
            all_memory_grads[layer_idx] = create_masked_grads(layer_grads, keep_gate_scale=False)
            all_gate_grads[layer_idx] = create_masked_grads(
                layer_grads, keep_gate_scale=True, freeze_gate=gate_warmup_active
            )

        # Apply all updates
        for layer_idx, layer in titans_layers.items():
            optimizer_memory.update(layer, all_memory_grads[layer_idx])
            optimizer_gate.update(layer, all_gate_grads[layer_idx])
            params_to_eval.extend(tree_flatten(layer.parameters()))

        # Update backbone layers if unfreezing is enabled
        if unfrozen_backbone_indices and accumulated_backbone_grads and optimizer_backbone:
            lr_backbone = lr * 0.1  # Lower LR for backbone
            optimizer_backbone.learning_rate = lr_backbone

            for idx in unfrozen_backbone_indices:
                layer_key = str(idx)
                if layer_key in accumulated_backbone_grads:
                    layer_grads = accumulated_backbone_grads[layer_key]
                    if layer_grads is not None:
                        backbone_layer = model.model.layers[idx]
                        optimizer_backbone.update(backbone_layer, layer_grads)
                        params_to_eval.extend(tree_flatten(backbone_layer.parameters()))

        _t_opt = time.perf_counter() - t_opt_start  # Keep for debugging

        # OPTIMIZATION: Release intermediate references before eval to reduce peak memory
        # This allows MLX to free gradient arrays before computing parameter updates
        del all_memory_grads, all_gate_grads
        del accumulated_grads
        if accumulated_backbone_grads is not None:
            del accumulated_backbone_grads

        # Eval all updated parameters
        t_eval_start = time.perf_counter()
        mx.eval(*[p for _, p in params_to_eval])
        _t_eval = time.perf_counter() - t_eval_start  # Keep for debugging

        # Track step time
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)

        # Logging
        if step % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Debug gradients at key steps
            if step == 0 or step == 50 or step == 90:
                print(f"\n  DEBUG step {step} gradients:")
                for path, g in tree_flatten(titans_grads):
                    name = ".".join(str(x) for x in path) if isinstance(path, tuple) else str(path)
                    if "mem_scale" in name or "linear2.bias" in name or "to_lr" in name:
                        grad_val = float(mx.mean(mx.abs(g)).item()) if g.size > 1 else float(g.item())
                        print(f"    {name}: grad_mean_abs={grad_val:.6f}")
                print(f"    LR memory={lr:.6f}, LR gate={lr_gate:.6f}")
                print(f"    Gate warmup active: {gate_warmup_active}")

            # Collect stats from all layers
            layer_stats = combined_model.get_layer_stats()
            first_layer_idx = combined_model.memory_layer_indices[0]

            # Use first layer for primary stats (backward compatible)
            mem_scale = layer_stats[first_layer_idx]["mem_scale"]
            gate_bias = layer_stats[first_layer_idx]["gate_bias"]
            gate_mean = 1 / (1 + math.exp(-gate_bias))

            # Get CMS weights if enabled
            cms_weights = {}
            if config.use_cms and titans_layers and first_layer_idx in titans_layers:
                # Get weights from first TITANS layer
                first_layer = titans_layers[first_layer_idx]
                if first_layer._use_cms:
                    weights = mx.softmax(first_layer.memory.level_weights)
                    for i in range(config.num_cms_levels):
                        cms_weights[f"cms_weight_{i}"] = float(weights[i].item())

            # Get internal loss if enabled
            internal_loss_val = 0.0
            if config.use_internal_loss:
                il = combined_model.get_internal_loss()
                if il is not None:
                    internal_loss_val = float(il.item())

            # Build log entry with all layer stats
            log_entry = {
                "step": step,
                "loss": avg_loss,
                "mem_scale": mem_scale,
                "gate_mean": gate_mean,
                "lr": lr,
                "time_ms": dt * 1000,
                "internal_loss": internal_loss_val,
                **cms_weights,
            }

            # Add per-layer stats
            for idx, stats in layer_stats.items():
                log_entry[f"layer_{idx}_mem_scale"] = stats["mem_scale"]
                log_entry[f"layer_{idx}_gate"] = 1 / (1 + math.exp(-stats["gate_bias"]))

            log_history.append(log_entry)

            # Format output string
            cms_str = ""
            if cms_weights:
                cms_str = ", cms=[" + ", ".join(f"{v:.3f}" for v in cms_weights.values()) + "]"

            il_str = ""
            if config.use_internal_loss:
                il_str = f", int_loss={internal_loss_val:.4f}"

            # Show per-layer gates if multiple layers
            num_memory_layers = len(titans_layers) if titans_layers else 0
            if num_memory_layers > 1:
                layer_gates = [f"L{idx}:{1 / (1 + math.exp(-s['gate_bias'])):.3f}" for idx, s in layer_stats.items()]
                gates_str = f", gates=[{', '.join(layer_gates)}]"
            else:
                gates_str = f", gate={gate_mean:.3f}"

            # Timing info - note: fwd_bwd/opt show graph build time, eval shows actual compute
            # With lazy evaluation, most computation happens during eval
            avg_step_time = sum(step_times[-10:]) / len(step_times[-10:]) if step_times else 0
            steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
            # Show actual step time (more accurate than sum of components due to lazy eval)
            timing_str = f" [{avg_step_time * 1000:.0f}ms, {steps_per_sec:.1f} step/s]"

            print(
                f"step {step}: loss={avg_loss:.4f}, mem_scale={mem_scale:.3f}{gates_str}{cms_str}{il_str}, lr={lr:.2e}{timing_str}"
            )

    # Save results
    log_path = output_dir / "mlx_training_log.json"
    with log_path.open("w") as f:
        json.dump(log_history, f, indent=2)
    print(f"\nSaved training log to {log_path}")

    # Save TITANS weights
    weights_path = output_dir / "titans_weights.safetensors"
    titans_weights = {}
    # Save each TITANS layer
    for layer_idx, layer in titans_layers.items():
        for name, param in tree_flatten(layer.parameters()):
            # name is already a dotted string like 'gate.linear1.weight'
            key = f"layer_{layer_idx}.{name}"
            titans_weights[key] = param
    mx.save_safetensors(str(weights_path), titans_weights)
    print(f"Saved TITANS weights to {weights_path}")

    # Summary
    if log_history:
        first = log_history[0]
        last = log_history[-1]
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Loss: {first['loss']:.4f} → {last['loss']:.4f}")
        print(f"mem_scale: {first['mem_scale']:.3f} → {last['mem_scale']:.3f}")

        # Show per-layer gate progression
        print("Gate values per layer:")
        layer_indices = sorted(titans_layers.keys())
        for layer_idx in layer_indices:
            first_gate = first.get(f"layer_{layer_idx}_gate", first.get("gate_mean", 0))
            last_gate = last.get(f"layer_{layer_idx}_gate", last.get("gate_mean", 0))
            print(f"  Layer {layer_idx}: {first_gate:.3f} → {last_gate:.3f}")

        if config.use_cms:
            print("CMS weights:")
            for i in range(config.num_cms_levels):
                key = f"cms_weight_{i}"
                print(
                    f"  Level {i} (freq={config.cms_update_frequencies[i]}): {first.get(key, 0):.3f} → {last.get(key, 0):.3f}"
                )


def parse_memory_layers(s: str) -> list:
    """Parse comma-separated layer indices."""
    if not s:
        return [12]
    return [int(x.strip()) for x in s.split(",")]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLX TITANS Training (Full HOPE)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument(
        "--memory_layers",
        type=str,
        default="12",
        help="Comma-separated layer indices for TITANS (e.g., '6,12,18')",
    )
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gate_lr_scale", type=float, default=50.0)
    parser.add_argument("--segment_len", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="out-mlx-titans")
    parser.add_argument("--use_cms", action="store_true", default=True)
    parser.add_argument("--no_cms", action="store_false", dest="use_cms")
    parser.add_argument("--gate_warmup_steps", type=int, default=0)
    parser.add_argument("--no_adaptive", action="store_true")
    parser.add_argument(
        "--use_internal_loss",
        action="store_true",
        default=True,
        help="Enable internal loss for memory to learn (default: True)",
    )
    parser.add_argument(
        "--no_internal_loss",
        action="store_false",
        dest="use_internal_loss",
        help="Disable internal loss",
    )
    parser.add_argument(
        "--internal_loss_weight",
        type=float,
        default=0.1,
        help="Weight for internal loss (default: 0.1)",
    )
    parser.add_argument(
        "--unfreeze_backbone_layers",
        type=int,
        default=0,
        help="Number of backbone layers to unfreeze near TITANS layers (0=frozen)",
    )
    parser.add_argument(
        "--gate_min_value",
        type=float,
        default=0.15,
        help="Minimum gate value for regularization (default: 0.15, like PyTorch)",
    )
    parser.add_argument(
        "--gate_reg_weight",
        type=float,
        default=1.0,
        help="Weight for gate regularization loss (default: 1.0)",
    )
    parser.add_argument(
        "--gate_init_bias",
        type=float,
        default=-2.0,
        help="Initial gate bias (default: -2.0 → sigmoid ≈ 0.12, like PyTorch)",
    )
    parser.add_argument(
        "--no_eager_eval",
        action="store_true",
        help="Disable eager evaluation (slower but uses less memory per micro-step)",
    )
    parser.add_argument(
        "--use_cascade",
        action="store_true",
        default=False,
        help="Use cascade mode for CMS (each level transforms previous level's output)",
    )
    parser.add_argument(
        "--surprise_threshold",
        type=float,
        default=0.0,
        help="Skip memory updates when grad norm below threshold (0=disabled)",
    )
    parser.add_argument(
        "--memory_grad_clip",
        type=float,
        default=1.0,
        help="Per-level gradient clipping for CMS",
    )

    args = parser.parse_args()

    # Parse memory layers
    memory_layers = parse_memory_layers(args.memory_layers)

    config = MLXTitansConfig(
        model_name=args.model_name,
        memory_layers=memory_layers,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        gate_lr_scale=args.gate_lr_scale,
        segment_len=args.segment_len,
        output_dir=args.output_dir,
        use_cms=args.use_cms,
        use_cascade=args.use_cascade,
        gate_warmup_steps=args.gate_warmup_steps,
        adaptive_memory=not args.no_adaptive,
        use_internal_loss=args.use_internal_loss,
        internal_loss_weight=args.internal_loss_weight,
        gate_min_value=args.gate_min_value,
        gate_reg_weight=args.gate_reg_weight,
        gate_init_bias=args.gate_init_bias,
        eager_eval=not args.no_eager_eval,
        unfreeze_backbone_layers=args.unfreeze_backbone_layers,
        surprise_threshold=args.surprise_threshold,
        memory_grad_clip=args.memory_grad_clip,
    )

    train(config)


if __name__ == "__main__":
    main()

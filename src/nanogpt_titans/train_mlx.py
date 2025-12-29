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
from typing import Dict, Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Import from MLX module
from nanogpt_titans.mlx import (
    MLXTitansConfig,
    MLXTitansLayer,
    CombinedModel,
    create_loss_fn,
    filter_titans_grads,
    get_lr,
)

# For loading HuggingFace models
try:
    from mlx_lm import load as mlx_load
except ImportError:
    raise ImportError("mlx-lm required: uv add mlx-lm")


# =============================================================================
# Model Loading
# =============================================================================


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer using mlx-lm."""
    print(f"Loading {model_name} with MLX...")
    model, tokenizer = mlx_load(model_name)
    return model, tokenizer


def create_titans_layer(model, config: MLXTitansConfig) -> MLXTitansLayer:
    """Create a TITANS layer matching model dimensions."""
    # Get hidden size from model
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        sample_layer = model.model.layers[0]
        if hasattr(sample_layer, 'hidden_size'):
            dim = sample_layer.hidden_size
        elif hasattr(sample_layer.self_attn, 'hidden_size'):
            dim = sample_layer.self_attn.hidden_size
        else:
            dim = sample_layer.self_attn.q_proj.weight.shape[0]
    else:
        dim = 896  # Default for Qwen2-0.5B

    print(f"Creating TITANS layer with dim={dim}, CMS={config.use_cms}")
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


# =============================================================================
# Gradient Utilities (inline for training-specific behavior)
# =============================================================================


def scale_grads_recursive(grad_tree: Any, factor: float) -> Any:
    """Recursively scale gradients by a constant factor (for averaging)."""
    if isinstance(grad_tree, dict):
        return {k: scale_grads_recursive(v, factor) for k, v in grad_tree.items()}
    elif isinstance(grad_tree, mx.array):
        return grad_tree * factor
    else:
        return grad_tree


def accumulate_grads(accum_grads: Dict | None, new_grads: Dict) -> Dict:
    """Add new gradients to accumulated gradients."""
    if accum_grads is None:
        return new_grads

    def add_grads(a, b):
        if isinstance(a, dict):
            return {k: add_grads(a[k], b[k]) for k in a}
        elif isinstance(a, mx.array):
            return a + b
        else:
            return a

    return add_grads(accum_grads, new_grads)


def create_masked_grads(
    grads: Dict[str, Any],
    keep_gate_scale: bool,
    path: str = "",
    freeze_gate: bool = False
) -> Dict[str, Any]:
    """Create grads with zeros for non-target params."""
    if isinstance(grads, dict):
        return {
            k: create_masked_grads(v, keep_gate_scale, f"{path}.{k}" if path else k, freeze_gate)
            for k, v in grads.items()
        }
    elif isinstance(grads, mx.array):
        is_gate = 'gate' in path
        is_scale_adaptive = ('mem_scale' in path or 'mem_ln' in path or
                            'level_weights' in path or 'to_lr' in path or
                            'to_momentum' in path or 'to_decay' in path)
        is_gate_scale = is_gate or is_scale_adaptive

        if freeze_gate and is_gate:
            return mx.zeros_like(grads)

        if keep_gate_scale:
            return grads if is_gate_scale else mx.zeros_like(grads)
        else:
            return mx.zeros_like(grads) if is_gate_scale else grads
    else:
        return grads


# =============================================================================
# Training Loop
# =============================================================================


def train(config: MLXTitansConfig):
    """Main MLX training loop with full HOPE architecture."""
    print("=" * 60)
    print("MLX TITANS Training (Full HOPE Architecture)")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Memory layer: {config.memory_layer}")
    print(f"CMS enabled: {config.use_cms}")
    if config.use_cms:
        print(f"CMS levels: {config.num_cms_levels}")
        print(f"CMS frequencies: {config.cms_update_frequencies}")
    print(f"Max steps: {config.max_steps}")
    print()

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(config.model_name)

    # Create TITANS layer
    titans_layer = create_titans_layer(model, config)

    # Count parameters
    titans_params = sum(p.size for _, p in tree_flatten(titans_layer.parameters()))
    print(f"TITANS trainable params: {titans_params:,}")

    # Separate param groups for different LRs
    def count_param_groups(layer):
        gate_scale_count = 0
        memory_count = 0

        for name, param in tree_flatten(layer.parameters()):
            name_str = ".".join(str(k) for k in name) if isinstance(name, tuple) else str(name)
            if ('gate' in name_str or 'mem_scale' in name_str or 'mem_ln' in name_str or
                'level_weights' in name_str or 'to_lr' in name_str or
                'to_momentum' in name_str or 'to_decay' in name_str):
                gate_scale_count += param.size
            else:
                memory_count += param.size

        return gate_scale_count, memory_count

    gate_scale_count, memory_count = count_param_groups(titans_layer)
    print(f"Gate/scale/CMS params: {gate_scale_count:,}")
    print(f"Memory params: {memory_count:,}")

    # Create TWO optimizers for different LRs
    gate_lr = config.learning_rate * config.gate_lr_scale
    optimizer_memory = optim.AdamW(learning_rate=config.learning_rate, weight_decay=0.0)
    optimizer_gate = optim.AdamW(learning_rate=gate_lr, weight_decay=0.0)

    print(f"\nOptimizer learning rates:")
    print(f"  Memory params: {config.learning_rate:.2e}")
    print(f"  Gate/scale params: {gate_lr:.2e} ({config.gate_lr_scale}x)")

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
            all_tokens.extend(tokens[:config.max_length])

    all_tokens = mx.array(all_tokens)
    print(f"Total tokens: {len(all_tokens):,}")

    # Training loop
    print(f"\nStarting training...")
    log_history = []

    # Determine memory layer index
    num_layers = len(model.model.layers)
    memory_layer_idx = min(config.memory_layer, num_layers - 1)
    print(f"Applying TITANS at layer {memory_layer_idx} of {num_layers}")

    # Create combined model wrapper
    combined_model = CombinedModel(model, titans_layer, memory_layer_idx)

    # Create loss function
    loss_fn = create_loss_fn(combined_model)
    loss_and_grad_fn = nn.value_and_grad(combined_model, loss_fn)

    t0 = time.time()

    for step in range(config.max_steps):
        # Reset memory state at start of each step
        combined_model.reset_memory_state()

        # Gradient accumulation loop
        accumulated_grads = None
        total_loss = 0.0

        for micro_step in range(config.gradient_accumulation_steps):
            batch_idx = step * config.gradient_accumulation_steps + micro_step
            start_idx = (batch_idx * config.segment_len) % (len(all_tokens) - config.segment_len - 1)
            batch = all_tokens[start_idx:start_idx + config.segment_len + 1]

            input_ids = batch[:-1].reshape(1, -1)
            target_ids = batch[1:].reshape(1, -1)

            loss, full_grads = loss_and_grad_fn(combined_model, input_ids, target_ids)
            grads = filter_titans_grads(full_grads)

            scale_factor = 1.0 / config.gradient_accumulation_steps
            grads_scaled = scale_grads_recursive(grads, scale_factor)

            accumulated_grads = accumulate_grads(accumulated_grads, grads_scaled)
            total_loss += float(loss.item())

        avg_loss = total_loss / config.gradient_accumulation_steps

        # Get current learning rate
        lr = get_lr(step, config)
        lr_gate = lr * config.gate_lr_scale

        # Extract TITANS layer gradients
        if 'titans_layer' in accumulated_grads:
            titans_grads = accumulated_grads['titans_layer']
        else:
            titans_grads = accumulated_grads

        # Gate warmup: freeze gate params
        gate_warmup_active = step < config.gate_warmup_steps

        # Apply memory optimizer
        memory_only_grads = create_masked_grads(titans_grads, keep_gate_scale=False)
        optimizer_memory.learning_rate = lr
        optimizer_memory.update(titans_layer, memory_only_grads)

        # Apply gate optimizer
        gate_only_grads = create_masked_grads(titans_grads, keep_gate_scale=True, freeze_gate=gate_warmup_active)
        optimizer_gate.learning_rate = lr_gate
        optimizer_gate.update(titans_layer, gate_only_grads)

        mx.eval(titans_layer.parameters())

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
                    if 'mem_scale' in name or 'linear2.bias' in name or 'to_lr' in name:
                        grad_val = float(mx.mean(mx.abs(g)).item()) if g.size > 1 else float(g.item())
                        print(f"    {name}: grad_mean_abs={grad_val:.6f}")
                print(f"    LR memory={lr:.6f}, LR gate={lr_gate:.6f}")
                print(f"    Gate warmup active: {gate_warmup_active}")

            mem_scale = float(mx.sigmoid(titans_layer.mem_scale).item())
            gate_bias = float(titans_layer.gate.linear2.bias.item())
            gate_mean = 1 / (1 + math.exp(-gate_bias))

            # Get CMS weights if enabled
            cms_weights = {}
            if config.use_cms and titans_layer._use_cms:
                weights = mx.softmax(titans_layer.memory.level_weights)
                for i in range(config.num_cms_levels):
                    cms_weights[f"cms_weight_{i}"] = float(weights[i].item())

            log_entry = {
                "step": step,
                "loss": avg_loss,
                "mem_scale": mem_scale,
                "gate_mean": gate_mean,
                "lr": lr,
                "time_ms": dt * 1000,
                **cms_weights,
            }
            log_history.append(log_entry)

            cms_str = ""
            if cms_weights:
                cms_str = ", cms=[" + ", ".join(f"{v:.3f}" for v in cms_weights.values()) + "]"

            print(f"step {step}: loss={avg_loss:.4f}, mem_scale={mem_scale:.3f}, gate={gate_mean:.3f}{cms_str}, lr={lr:.2e}")

    # Save results
    log_path = output_dir / "mlx_training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"\nSaved training log to {log_path}")

    # Summary
    if log_history:
        first = log_history[0]
        last = log_history[-1]
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Loss: {first['loss']:.4f} → {last['loss']:.4f}")
        print(f"mem_scale: {first['mem_scale']:.3f} → {last['mem_scale']:.3f}")
        print(f"gate_mean: {first['gate_mean']:.3f} → {last['gate_mean']:.3f}")
        if config.use_cms:
            print("CMS weights:")
            for i in range(config.num_cms_levels):
                key = f"cms_weight_{i}"
                print(f"  Level {i} (freq={config.cms_update_frequencies[i]}): {first.get(key, 0):.3f} → {last.get(key, 0):.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLX TITANS Training (Full HOPE)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--memory_layer", type=int, default=12)
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

    args = parser.parse_args()

    config = MLXTitansConfig(
        model_name=args.model_name,
        memory_layer=args.memory_layer,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        gate_lr_scale=args.gate_lr_scale,
        segment_len=args.segment_len,
        output_dir=args.output_dir,
        use_cms=args.use_cms,
        gate_warmup_steps=args.gate_warmup_steps,
        adaptive_memory=not args.no_adaptive,
    )

    train(config)


if __name__ == "__main__":
    main()

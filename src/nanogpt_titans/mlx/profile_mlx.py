#!/usr/bin/env python3
"""
MLX-specific profiler for TITANS training on Apple Silicon.

Identifies bottlenecks in:
- Memory retrieval and update operations
- Gradient computation (manual backward pass)
- Layer-by-layer forward pass
- Optimizer updates

Usage:
    uv run python -m nanogpt_titans.mlx.profile_mlx
    uv run python -m nanogpt_titans.mlx.profile_mlx --detailed
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from nanogpt_titans.mlx.config import MLXTitansConfig
from nanogpt_titans.mlx.decoder_layer import MLXTitansLayer
from nanogpt_titans.mlx.training import (
    CombinedModel,
    accumulate_grads,
    create_loss_fn,
    create_masked_grads,
    filter_titans_grads,
    scale_grads_recursive,
)


@dataclass
class TimingResult:
    """Stores timing results for a profiled operation."""

    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    samples: int


def _eval_result(result):
    """Recursively evaluate all mx.arrays in a result."""
    if isinstance(result, mx.array):
        mx.eval(result)
    elif isinstance(result, (list, tuple)):
        for item in result:
            _eval_result(item)
    elif hasattr(result, "__dict__"):
        for v in result.__dict__.values():
            _eval_result(v)
    elif isinstance(result, dict):
        for v in result.values():
            _eval_result(v)


def time_operation(fn, num_iters: int = 20, warmup: int = 10, eval_result: bool = True) -> TimingResult:
    """Time an operation with warmup.

    Args:
        fn: Function to time
        num_iters: Number of iterations
        warmup: Number of warmup iterations
        eval_result: If True, force evaluation of results (measure actual compute time)
    """
    name = fn.__name__ if hasattr(fn, "__name__") else "operation"

    # Warmup
    for _ in range(warmup):
        result = fn()
        if eval_result:
            _eval_result(result)
        mx.eval(mx.array(0))  # Final sync

    # Timed runs
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        result = fn()
        if eval_result:
            _eval_result(result)
        mx.eval(mx.array(0))  # Final sync
        times.append((time.perf_counter() - t0) * 1000)

    import statistics

    return TimingResult(
        name=name,
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0,
        min_ms=min(times),
        max_ms=max(times),
        samples=num_iters,
    )


def profile_memory_operations(dim: int = 896, batch_size: int = 2, seq_len: int = 512):
    """Profile memory retrieval and update operations."""
    print("\n" + "=" * 60)
    print("MEMORY OPERATIONS PROFILING")
    print("=" * 60)

    from nanogpt_titans.mlx.memory import MLXContinuumMemorySystem, MLXNeuralMemory

    # Create memory modules
    single_mem = MLXNeuralMemory(dim, depth=2, expansion=2, adaptive=True)
    cms = MLXContinuumMemorySystem(dim, num_levels=3, update_frequencies=(1, 4, 16))

    # Create input
    x = mx.random.normal((batch_size, seq_len, dim))
    mx.eval(x)

    # Initialize states
    single_state = single_mem.init_state(batch_size)
    cms_state = cms.init_state(batch_size)
    mx.eval(single_state.weights, cms_state.level_states[0].weights)

    results = {}

    # Profile single memory retrieval
    def single_retrieve():
        return single_mem(x, single_state)

    result = time_operation(single_retrieve)
    results["single_mem_retrieve"] = result
    print(f"  Single memory retrieval: {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile single memory update
    def single_update():
        return single_mem.update(x, single_state)

    result = time_operation(single_update)
    results["single_mem_update"] = result
    print(f"  Single memory update:    {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile CMS retrieval
    def cms_retrieve():
        return cms(x, cms_state)

    result = time_operation(cms_retrieve)
    results["cms_retrieve"] = result
    print(f"  CMS retrieval (3 levels):{result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile CMS update
    def cms_update():
        return cms.update(x, cms_state)

    result = time_operation(cms_update)
    results["cms_update"] = result
    print(f"  CMS update (3 levels):   {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile gradient computation
    def compute_grads():
        keys = single_mem.key_proj(x)
        values = single_mem.value_proj(x)
        return single_mem._compute_gradients(keys, values, single_state.weights)

    result = time_operation(compute_grads)
    results["gradient_computation"] = result
    print(f"  Gradient computation:    {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile internal loss
    def compute_internal_loss():
        return single_mem.compute_internal_loss(x, single_state)

    result = time_operation(compute_internal_loss)
    results["internal_loss"] = result
    print(f"  Internal loss:           {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    return results


def profile_titans_layer(dim: int = 896, batch_size: int = 2, seq_len: int = 512):
    """Profile TITANS layer operations."""
    print("\n" + "=" * 60)
    print("TITANS LAYER PROFILING")
    print("=" * 60)

    layer = MLXTitansLayer(
        dim=dim,
        use_cms=True,
        num_cms_levels=3,
        memory_depth=2,
        memory_expansion=2,
        adaptive_memory=True,
    )

    x = mx.random.normal((batch_size, seq_len, dim))
    state = layer.init_state(batch_size)
    mx.eval(x, state)

    results = {}

    # Profile full forward pass
    def full_forward():
        return layer(x, state)

    result = time_operation(full_forward)
    results["titans_layer_forward"] = result
    print(f"  Full forward pass:       {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile gate computation
    def gate_compute():
        return layer.gate(x)

    result = time_operation(gate_compute)
    results["gate_compute"] = result
    print(f"  Gate computation:        {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile mem projection
    def mem_projection():
        mem_retrieved = layer.memory(x, state)
        mem_pooled = mx.mean(mem_retrieved, axis=1, keepdims=True)
        return layer.mem_proj(mem_pooled)

    result = time_operation(mem_projection)
    results["mem_projection"] = result
    print(f"  Memory projection:       {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    return results


def profile_training_step(config: MLXTitansConfig):
    """Profile a complete training step."""
    print("\n" + "=" * 60)
    print("TRAINING STEP PROFILING")
    print("=" * 60)

    # Load model
    try:
        from mlx_lm import load as mlx_load
    except ImportError:
        print("mlx-lm not installed, skipping training step profiling")
        return {}

    print(f"Loading {config.model_name}...")
    model, _tokenizer = mlx_load(config.model_name)

    # Get model dimensions
    dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    num_layers = len(model.model.layers)

    print(f"Model dim: {dim}, layers: {num_layers}")

    # Create TITANS layers (support multiple)
    titans_layers = {}
    for layer_idx in config.memory_layers:
        idx = min(layer_idx, num_layers - 1)
        titans_layers[idx] = MLXTitansLayer(
            dim=dim,
            use_cms=config.use_cms,
            num_cms_levels=config.num_cms_levels,
            memory_depth=config.memory_depth,
            memory_expansion=config.memory_expansion,
            adaptive_memory=config.adaptive_memory,
        )

    print(f"TITANS layers: {list(titans_layers.keys())}")
    layer_idx = next(iter(titans_layers.keys()))  # Use first for single-layer tests

    # Create combined model
    combined_model = CombinedModel(
        model,
        titans_layers,
        use_internal_loss=config.use_internal_loss,
    )

    # Create loss function
    loss_fn = create_loss_fn(combined_model, config.gate_min_value, config.gate_reg_weight)
    loss_and_grad_fn = nn.value_and_grad(combined_model, loss_fn)

    # Create optimizers
    optimizer_memory = optim.AdamW(learning_rate=config.learning_rate)
    optimizer_gate = optim.AdamW(learning_rate=config.learning_rate * config.gate_lr_scale)

    # Create dummy data
    input_ids = mx.random.randint(0, 32000, (1, config.segment_len))
    target_ids = mx.random.randint(0, 32000, (1, config.segment_len))
    mx.eval(input_ids, target_ids)

    results = {}

    # Profile forward pass
    def forward_pass():
        combined_model.reset_memory_state()
        return combined_model(input_ids)

    result = time_operation(forward_pass, num_iters=5, warmup=2)
    results["forward_pass"] = result
    print(f"  Forward pass:            {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile forward + backward
    def forward_backward():
        combined_model.reset_memory_state()
        loss, grads = loss_and_grad_fn(combined_model, input_ids, target_ids)
        return loss, grads

    result = time_operation(forward_backward, num_iters=5, warmup=2)
    results["forward_backward"] = result
    print(f"  Forward + backward:      {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile gradient filtering
    _, full_grads = loss_and_grad_fn(combined_model, input_ids, target_ids)

    def grad_filter():
        return filter_titans_grads(full_grads)

    result = time_operation(grad_filter)
    results["grad_filtering"] = result
    print(f"  Gradient filtering:      {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile gradient masking
    filtered_grads = filter_titans_grads(full_grads)

    def grad_masking():
        return create_masked_grads(filtered_grads, keep_gate_scale=True)

    result = time_operation(grad_masking)
    results["grad_masking"] = result
    print(f"  Gradient masking:        {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile optimizer step
    memory_grads = create_masked_grads(filtered_grads, keep_gate_scale=False)
    gate_grads = create_masked_grads(filtered_grads, keep_gate_scale=True)

    # We need to extract the titans layer grads
    if "titans_layers" in memory_grads:
        layer_memory_grads = memory_grads["titans_layers"].get(
            str(layer_idx), memory_grads["titans_layers"].get(layer_idx, {})
        )
        layer_gate_grads = gate_grads["titans_layers"].get(
            str(layer_idx), gate_grads["titans_layers"].get(layer_idx, {})
        )
    else:
        layer_memory_grads = memory_grads
        layer_gate_grads = gate_grads

    titans_layer = titans_layers[layer_idx]  # Get first layer for single-layer tests

    def optimizer_step():
        optimizer_memory.update(titans_layer, layer_memory_grads)
        optimizer_gate.update(titans_layer, layer_gate_grads)

    result = time_operation(optimizer_step, num_iters=5)
    results["optimizer_step"] = result
    print(f"  Optimizer step (1 layer):{result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile optimizer step for ALL layers
    def optimizer_step_all():
        for _idx, layer in titans_layers.items():
            optimizer_memory.update(layer, layer_memory_grads)
            optimizer_gate.update(layer, layer_gate_grads)

    result = time_operation(optimizer_step_all, num_iters=5)
    results["optimizer_step_all"] = result
    print(f"  Optimizer step ({len(titans_layers)} layers):{result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    # Profile mx.eval
    def eval_params():
        mx.eval(titans_layer.parameters())

    result = time_operation(eval_params)
    results["mx_eval"] = result
    print(f"  mx.eval (parameters):    {result.mean_ms:.2f} ms (std={result.std_ms:.2f})")

    return results


def profile_gradient_accumulation(config: MLXTitansConfig, num_accum_steps: int = 4):
    """Profile gradient accumulation overhead."""
    print("\n" + "=" * 60)
    print("GRADIENT ACCUMULATION PROFILING")
    print("=" * 60)

    try:
        from mlx_lm import load as mlx_load
    except ImportError:
        print("mlx-lm not installed, skipping")
        return {}

    print(f"Loading {config.model_name}...")
    model, _ = mlx_load(config.model_name)

    dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    num_layers = len(model.model.layers)
    layer_idx = min(config.memory_layers[0], num_layers - 1)

    titans_layer = MLXTitansLayer(dim=dim, use_cms=config.use_cms)
    titans_layers = {layer_idx: titans_layer}
    combined_model = CombinedModel(model, titans_layers)

    loss_fn = create_loss_fn(combined_model)
    loss_and_grad_fn = nn.value_and_grad(combined_model, loss_fn)

    input_ids = mx.random.randint(0, 32000, (1, config.segment_len))
    target_ids = mx.random.randint(0, 32000, (1, config.segment_len))
    mx.eval(input_ids, target_ids)

    results = {}

    # Profile single forward+backward
    def single_fwd_bwd():
        combined_model.reset_memory_state()
        return loss_and_grad_fn(combined_model, input_ids, target_ids)

    result = time_operation(single_fwd_bwd, num_iters=5, warmup=2)
    results["single_step"] = result
    print(f"  Single forward+backward: {result.mean_ms:.2f} ms")

    # Profile accumulated steps
    def accumulated_steps():
        combined_model.reset_memory_state()
        accumulated_grads = None
        for _ in range(num_accum_steps):
            _loss, grads = loss_and_grad_fn(combined_model, input_ids, target_ids)
            filtered = filter_titans_grads(grads)
            scaled = scale_grads_recursive(filtered, 1.0 / num_accum_steps)
            accumulated_grads = accumulate_grads(accumulated_grads, scaled)
        return accumulated_grads

    result = time_operation(accumulated_steps, num_iters=3, warmup=1)
    results["accumulated_steps"] = result
    print(f"  {num_accum_steps}x accumulated steps:   {result.mean_ms:.2f} ms")

    # Overhead per step
    overhead = result.mean_ms - (results["single_step"].mean_ms * num_accum_steps)
    print(f"  Accumulation overhead:   {overhead:.2f} ms ({overhead / result.mean_ms * 100:.1f}%)")
    results["accumulation_overhead_ms"] = overhead

    return results


def profile_update_breakdown(dim: int = 896, batch_size: int = 2, seq_len: int = 512):
    """Profile individual operations within memory update to find exact bottleneck."""
    print("\n" + "=" * 60)
    print("DETAILED UPDATE BREAKDOWN")
    print("=" * 60)

    from nanogpt_titans.mlx.memory import MLXNeuralMemory

    # Create memory and inputs
    single_mem = MLXNeuralMemory(dim, depth=2, expansion=2, adaptive=True)
    x = mx.random.normal((batch_size, seq_len, dim))
    state = single_mem.init_state(batch_size)
    mx.eval(x, state.weights, state.last_momentum)

    results = {}

    # Profile key/value projection
    def profile_projections():
        keys = single_mem.key_proj(x)
        values = single_mem.value_proj(x)
        mx.eval(keys, values)
        return keys, values

    t0 = time.perf_counter()
    for _ in range(10):
        keys, values = profile_projections()
    results["projections"] = TimingResult("projections", (time.perf_counter() - t0) * 100, 0, 0, 0, 10)
    print(f"  Key/Value projections:     {results['projections'].mean_ms:.2f} ms avg")

    # Profile gradient computation only
    keys, values = profile_projections()

    def profile_grad_comp():
        grads, grad_norm = single_mem._compute_gradients(keys, values, state.weights)
        mx.eval(grads["w0"], grads["w1"], grad_norm)
        return grads, grad_norm

    t0 = time.perf_counter()
    for _ in range(10):
        grads, grad_norm = profile_grad_comp()
    results["gradient_comp"] = TimingResult("gradient_comp", (time.perf_counter() - t0) * 100, 0, 0, 0, 10)
    print(f"  Gradient computation:      {results['gradient_comp'].mean_ms:.2f} ms avg")

    # Profile .item() call (forces CPU-GPU sync)
    def profile_item_call():
        _ = float(grad_norm.item())

    t0 = time.perf_counter()
    for _ in range(10):
        profile_item_call()
    results["item_call"] = TimingResult("item_call", (time.perf_counter() - t0) * 100, 0, 0, 0, 10)
    print(f"  .item() sync call:         {results['item_call'].mean_ms:.2f} ms avg")

    # Profile adaptive parameter computation
    def profile_adaptive_params():
        adaptive_lr = mx.sigmoid(single_mem.to_lr(x)) * single_mem.lr_max
        adaptive_momentum = mx.sigmoid(single_mem.to_momentum(x))
        adaptive_decay = mx.sigmoid(single_mem.to_decay(x))
        lr_param = mx.mean(adaptive_lr, axis=1)[:, 0]
        mom_param = mx.mean(adaptive_momentum, axis=1)[:, 0]
        decay_param = mx.mean(adaptive_decay, axis=1)[:, 0]
        mx.eval(lr_param, mom_param, decay_param)
        return lr_param, mom_param, decay_param

    t0 = time.perf_counter()
    for _ in range(10):
        lr_param, mom_param, decay_param = profile_adaptive_params()
    results["adaptive_params"] = TimingResult("adaptive_params", (time.perf_counter() - t0) * 100, 0, 0, 0, 10)
    print(f"  Adaptive param compute:    {results['adaptive_params'].mean_ms:.2f} ms avg")

    # Profile .item() on adaptive params (3x calls)
    def profile_adaptive_item():
        _ = float(mx.mean(lr_param).item())
        _ = float(mx.mean(mom_param).item())
        _ = float(mx.mean(decay_param).item())

    t0 = time.perf_counter()
    for _ in range(10):
        profile_adaptive_item()
    results["adaptive_item"] = TimingResult("adaptive_item", (time.perf_counter() - t0) * 100, 0, 0, 0, 10)
    print(f"  Adaptive .item() (3x):     {results['adaptive_item'].mean_ms:.2f} ms avg")

    # Profile weight update loop (without .item())
    def profile_weight_update():
        new_weights = {}
        new_momentum = {}
        for name in state.weights:
            g = grads[name]
            m_prev = state.last_momentum[name]
            w_prev = state.weights[name]

            mom_expanded = mom_param.reshape((-1,) + (1,) * (g.ndim - 1))
            lr_expanded = lr_param.reshape((-1,) + (1,) * (g.ndim - 1))
            decay_expanded = decay_param.reshape((-1,) + (1,) * (g.ndim - 1))

            m = mom_expanded * m_prev + (1 - mom_expanded) * g
            decay_factor = 1 - decay_expanded
            w = decay_factor * w_prev - lr_expanded * m
            w = mx.clip(w, -10.0, 10.0)

            new_weights[name] = w
            new_momentum[name] = m
        mx.eval(new_weights["w0"], new_weights["w1"])
        return new_weights, new_momentum

    t0 = time.perf_counter()
    for _ in range(10):
        _new_weights, _new_momentum = profile_weight_update()
    results["weight_update"] = TimingResult("weight_update", (time.perf_counter() - t0) * 100, 0, 0, 0, 10)
    print(f"  Weight update loop:        {results['weight_update'].mean_ms:.2f} ms avg")

    # Profile full update (with all .item() calls)
    def full_update():
        return single_mem.update(x, state)

    state_copy = single_mem.init_state(batch_size)
    mx.eval(state_copy.weights)

    t0 = time.perf_counter()
    for _ in range(5):
        new_state, _metrics = full_update()
        mx.eval(new_state.weights)
    results["full_update"] = TimingResult("full_update", (time.perf_counter() - t0) * 200, 0, 0, 0, 5)
    print(f"  Full update (current):     {results['full_update'].mean_ms:.2f} ms avg")

    # Summary
    print("\n  --- Analysis ---")
    total_ops = (
        results["projections"].mean_ms
        + results["gradient_comp"].mean_ms
        + results["adaptive_params"].mean_ms
        + results["weight_update"].mean_ms
    )
    item_overhead = results["item_call"].mean_ms + results["adaptive_item"].mean_ms
    print(f"  Pure computation time:     {total_ops:.2f} ms")
    print(f"  .item() sync overhead:     {item_overhead:.2f} ms")
    print(f"  Full update actual:        {results['full_update'].mean_ms:.2f} ms")
    if results["full_update"].mean_ms > 0:
        overhead_pct = (item_overhead / results["full_update"].mean_ms) * 100
        print(f"  Sync overhead ratio:       {overhead_pct:.1f}%")

    return results


def identify_bottlenecks(all_results: dict[str, dict[str, Any]]):
    """Analyze results and identify bottlenecks."""
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)

    # Collect all timing results
    all_timings = []
    for category, results in all_results.items():
        for name, result in results.items():
            if isinstance(result, TimingResult):
                all_timings.append((f"{category}/{name}", result.mean_ms))

    # Sort by time
    all_timings.sort(key=lambda x: -x[1])

    print("\nRanked by time (slowest first):")
    print("-" * 50)
    for name, time_ms in all_timings[:10]:
        print(f"  {name:<35} {time_ms:>8.2f} ms")

    # Identify specific bottlenecks
    print("\nKey Insights:")
    print("-" * 50)

    # Check memory vs compute
    if "memory_ops" in all_results:
        mem_ops = all_results["memory_ops"]
        cms_time = mem_ops.get("cms_retrieve", TimingResult("", 0, 0, 0, 0, 0)).mean_ms
        cms_update = mem_ops.get("cms_update", TimingResult("", 0, 0, 0, 0, 0)).mean_ms
        grad_time = mem_ops.get("gradient_computation", TimingResult("", 0, 0, 0, 0, 0)).mean_ms

        if cms_time > 10:
            print(f"  - CMS retrieval is slow ({cms_time:.1f}ms) - consider reducing num_levels")
        if cms_update > 10:
            print(f"  - CMS update is slow ({cms_update:.1f}ms) - consider higher update_frequencies")
        if grad_time > 5:
            print(f"  - Gradient computation is slow ({grad_time:.1f}ms) - batching could help")

    # Check training step breakdown
    if "training_step" in all_results:
        train = all_results["training_step"]
        fwd_time = train.get("forward_pass", TimingResult("", 0, 0, 0, 0, 0)).mean_ms
        fwd_bwd = train.get("forward_backward", TimingResult("", 0, 0, 0, 0, 0)).mean_ms
        bwd_time = fwd_bwd - fwd_time if fwd_bwd > fwd_time else 0

        if bwd_time > fwd_time * 3:
            print(f"  - Backward pass ({bwd_time:.1f}ms) is >3x forward ({fwd_time:.1f}ms)")
            print("    Consider gradient checkpointing or simplifying loss computation")

    # Check gradient accumulation
    if "grad_accum" in all_results:
        accum = all_results["grad_accum"]
        overhead = accum.get("accumulation_overhead_ms", 0)
        if overhead > 50:
            print(f"  - Gradient accumulation overhead is high ({overhead:.1f}ms)")
            print("    Consider fusing accumulation operations")


def main():
    parser = argparse.ArgumentParser(description="Profile MLX TITANS training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--segment_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--detailed", action="store_true", help="Run all profiling tests")
    parser.add_argument("--memory-only", action="store_true", help="Only profile memory ops")
    parser.add_argument("--training-only", action="store_true", help="Only profile training step")
    parser.add_argument("--breakdown", action="store_true", help="Detailed update breakdown")
    args = parser.parse_args()

    print("=" * 60)
    print("MLX TITANS PROFILER")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Segment length: {args.segment_len}")
    print(f"Batch size: {args.batch_size}")

    # Test both single and multi-layer configs
    memory_layers = [8, 10, 12] if args.detailed else [12]
    print(f"Memory layers: {memory_layers}")

    config = MLXTitansConfig(
        model_name=args.model_name,
        segment_len=args.segment_len,
        batch_size=args.batch_size,
        memory_layers=memory_layers,
        use_cms=True,
        num_cms_levels=3,
    )

    all_results = {}

    if args.breakdown:
        all_results["update_breakdown"] = profile_update_breakdown()
    elif args.memory_only:
        all_results["memory_ops"] = profile_memory_operations()
        all_results["titans_layer"] = profile_titans_layer()
    elif args.training_only:
        all_results["training_step"] = profile_training_step(config)
    else:
        # Run all profiles
        all_results["memory_ops"] = profile_memory_operations()
        all_results["titans_layer"] = profile_titans_layer()
        all_results["training_step"] = profile_training_step(config)

        if args.detailed:
            all_results["grad_accum"] = profile_gradient_accumulation(config)

    identify_bottlenecks(all_results)

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

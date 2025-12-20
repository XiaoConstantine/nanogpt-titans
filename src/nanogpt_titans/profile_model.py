"""
Profile TitansGPT to identify performance bottlenecks.

Usage:
    uv run python -m nanogpt_titans.profile_model              # Simple timer profiling
    uv run python -m nanogpt_titans.profile_model --detailed   # torch.profiler for CUDA kernels
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from nanogpt_titans.model import TitansConfig, TitansGPT, parallel_momentum

if TYPE_CHECKING:
    from nanogpt_titans.model import MemoryState


class Timer:
    """Simple timer for profiling."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextmanager
    def track(self, name: str):
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if name not in self.times:
            self.times[name] = 0.0
            self.counts[name] = 0
        self.times[name] += elapsed
        self.counts[name] += 1

    def report(self) -> None:
        total = sum(self.times.values())
        print(f"\n{'=' * 60}")
        print(f"{'Component':<30} {'Time (ms)':<12} {'%':<8} {'Calls':<8}")
        print(f"{'=' * 60}")
        for name, t in sorted(self.times.items(), key=lambda x: -x[1]):
            pct = 100 * t / total if total > 0 else 0
            avg_ms = 1000 * t / self.counts[name]
            print(f"{name:<30} {avg_ms:<12.2f} {pct:<8.1f} {self.counts[name]:<8}")
        print(f"{'=' * 60}")
        print(f"{'TOTAL':<30} {1000 * total:<12.2f}")


def profile_forward_pass(
    model: TitansGPT,
    x: torch.Tensor,
    timer: Timer,
    num_iters: int = 10,
) -> None:
    """Profile a single forward pass with detailed breakdown."""
    device = x.device
    segment_len = model.config.segment_len
    b, t = x.shape
    num_segments = (t + segment_len - 1) // segment_len

    for _ in range(num_iters):
        states: list[MemoryState | None] = [
            model.transformer["h"][i].init_state(b, device)  # type: ignore[index]
            for i in range(len(model.transformer["h"]))  # type: ignore[arg-type]
        ]

        for seg_idx in range(num_segments):
            start = seg_idx * segment_len
            end = min(start + segment_len, t)
            seg_tokens = x[:, start:end]

            # Embedding
            with timer.track("1. token_embedding"):
                tok_emb = model.transformer["wte"](seg_tokens)

            with timer.track("2. position_embedding"):
                pos = torch.arange(start, end, dtype=torch.long, device=device)
                pos_emb = model.transformer["wpe"](pos)

            with timer.track("3. dropout"):
                h = model.transformer["drop"](tok_emb + pos_emb)

            # Process through blocks
            for i, block in enumerate(model.transformer["h"]):  # type: ignore[arg-type]
                has_mem = block.has_memory

                if has_mem:
                    # Memory retrieval
                    with timer.track("4. memory_retrieval"):
                        mem_context = block.memory(h, states[i])

                    with timer.track("5. persistent_mem_expand"):
                        persist = block.persist_mem.expand(b, -1, -1)

                    with timer.track("6. concat_context"):
                        prefix_len = block.num_longterm_mem + block.num_persist_mem
                        context = torch.cat([mem_context, persist, h], dim=1)

                    with timer.track("7. ln_1"):
                        ln_out = block.ln_1(context)

                    with timer.track("8. attention"):
                        attn_out = block.attn(ln_out, prefix_len=prefix_len)

                    with timer.track("9. residual_extract"):
                        h = h + attn_out[:, prefix_len:]

                    with timer.track("10. ln_2"):
                        ln2_out = block.ln_2(h)

                    with timer.track("11. mlp"):
                        h = h + block.mlp(ln2_out)

                    with timer.track("12. memory_update"):
                        states[i] = block.memory.update(h, states[i])
                else:
                    with timer.track("7. ln_1"):
                        ln_out = block.ln_1(h)

                    with timer.track("8. attention"):
                        attn_out = block.attn(ln_out)

                    with timer.track("9. residual"):
                        h = h + attn_out

                    with timer.track("10. ln_2"):
                        ln2_out = block.ln_2(h)

                    with timer.track("11. mlp"):
                        h = h + block.mlp(ln2_out)

            with timer.track("13. final_ln"):
                h = model.transformer["ln_f"](h)

            with timer.track("14. lm_head"):
                _ = model.lm_head(h)


def profile_memory_ops(
    model: TitansGPT,
    x: torch.Tensor,
    timer: Timer,
    num_iters: int = 10,
) -> None:
    """Profile memory operations in detail."""
    device = x.device
    b, _t = x.shape

    # Find the memory layer
    mem_block = None
    for block in model.transformer["h"]:  # type: ignore[union-attr]
        if block.has_memory:
            mem_block = block
            break

    if mem_block is None:
        print("No memory layer found!")
        return

    memory = mem_block.memory
    state = memory.init_state(b, device)

    # Simulate a segment
    seg_len = model.config.segment_len
    h = torch.randn(b, seg_len, model.config.n_embd, device=device)

    print(f"\n{'=' * 60}")
    print("MEMORY OPERATIONS BREAKDOWN")
    print(f"{'=' * 60}")

    for _ in range(num_iters):
        # Retrieval breakdown
        with timer.track("mem.query_proj"):
            if state.last_segment_output is not None:
                query_source = state.last_segment_output
            else:
                query_source = memory.init_query.expand(b, -1, -1)
            queries = memory.query_proj(query_source)

        with timer.track("mem.mlp_forward_loop"):
            retrieved_list = []
            for i in range(b):
                q_i = queries[i : i + 1]
                mem_out = memory._forward_with_weights(q_i, state.weights, i)
                retrieved_list.append(mem_out)

        with timer.track("mem.cat_retrieved"):
            retrieved = torch.cat(retrieved_list, dim=0)

        with timer.track("mem.adaptive_pool"):
            retrieved = retrieved.transpose(1, 2)
            retrieved = torch.nn.functional.adaptive_avg_pool1d(retrieved, memory.num_longterm_mem)
            retrieved = retrieved.transpose(1, 2)

        with timer.track("mem.out_proj"):
            _ = memory.out_proj(retrieved)

        # Update breakdown
        with timer.track("mem.key_value_proj"):
            keys = memory.key_proj(h.detach())
            values = memory.value_proj(h.detach())

        with timer.track("mem.vmap_grad"):

            def single_token_loss(
                params: dict[str, torch.Tensor], k: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                k = k.unsqueeze(0).unsqueeze(0)
                v = v.unsqueeze(0).unsqueeze(0)
                pred = torch.func.functional_call(memory.memory_mlp, params, (k,))
                return torch.nn.functional.mse_loss(pred, v)

            grad_fn = torch.func.grad(single_token_loss)
            params_in_dims: dict[str, Any] = dict.fromkeys(state.weights, 0)
            batched_grad_fn = torch.func.vmap(
                torch.func.vmap(grad_fn, in_dims=(None, 0, 0)),
                in_dims=(params_in_dims, 0, 0),
            )

            with torch.enable_grad():
                params_for_grad = {
                    name: w.clone().requires_grad_(True) for name, w in state.weights.items()
                }
                all_grads = batched_grad_fn(params_for_grad, keys, values)

        with timer.track("mem.momentum_update"):
            for name in state.weights:
                surprises = -all_grads[name]
                _ = parallel_momentum(
                    surprises,
                    memory.momentum,
                    prev_momentum=state.last_momentum[name],
                )

        # Store for next iteration
        state = memory.update(h, state)


def profile_with_torch_profiler(
    model: TitansGPT,
    x: torch.Tensor,
    targets: torch.Tensor,
    num_iters: int = 3,
    export_trace: bool = False,
) -> None:
    """
    Profile using torch.profiler for detailed CUDA kernel analysis.

    This shows actual CUDA kernel times, memory operations, and whether
    Triton kernels are being used.
    """
    from torch.profiler import ProfilerActivity, profile

    device = x.device
    batch_size = x.shape[0]

    print("\n" + "=" * 60)
    print("TORCH PROFILER - CUDA KERNEL ANALYSIS")
    print("=" * 60)

    # Warmup
    print("Warming up...")
    for _ in range(3):
        with torch.no_grad():
            states = model.init_memory_states(batch_size, torch.device(device))
            model(x, targets=targets, memory_states=states)

    # Profile
    print(f"Profiling {num_iters} iterations...")
    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof,
        torch.no_grad(),
    ):
        for _ in range(num_iters):
            states = model.init_memory_states(batch_size, torch.device(device))
            model(x, targets=targets, memory_states=states)

    # Print results sorted by CUDA time
    print("\n" + "=" * 60)
    print("TOP CUDA OPERATIONS (by total CUDA time)")
    print("=" * 60)
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30,
        )
    )

    # Print results sorted by CPU time
    print("\n" + "=" * 60)
    print("TOP CPU OPERATIONS (by total CPU time)")
    print("=" * 60)
    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=20,
        )
    )

    # Export Chrome trace if requested
    if export_trace:
        trace_path = "trace.json"
        prof.export_chrome_trace(trace_path)
        print(f"\nChrome trace exported to: {trace_path}")
        print("Open chrome://tracing and load the file to visualize.")

    # Summary stats
    print("\n" + "=" * 60)
    print("KERNEL SUMMARY")
    print("=" * 60)

    # Helper to get CUDA time (attribute name varies by PyTorch version)
    def get_cuda_time(event) -> float:
        for attr in ["cuda_time_total", "self_cuda_time_total", "device_time_total"]:
            if hasattr(event, attr):
                val = getattr(event, attr)
                if val is not None:
                    return val
        return 0.0

    def simplify_kernel_name(name: str) -> str:
        """Extract readable kernel name from verbose CUDA/Triton names."""
        # Custom Triton kernels
        if "_fused_weight_update_kernel" in name:
            return "triton: fused_weight_update"
        if "_momentum_update_kernel" in name:
            return "triton: momentum_update"
        if "triton_tem_fused" in name:
            return "triton: flex_attention"
        if "triton" in name.lower():
            return f"triton: {name[:40]}"

        # CUDA GEMM kernels
        if "sgemm" in name.lower():
            # Extract shape info like "128x64"
            for part in name.split("_"):
                if "x" in part and part.replace("x", "").isdigit():
                    return f"cuda: sgemm_{part}"
            return "cuda: sgemm"
        if "gemmk1_kernel" in name:
            return "cuda: gemmk1 (large matmul)"

        # Common PyTorch kernels
        if "SoftMax" in name:
            return "cuda: softmax"
        if "layer_norm" in name:
            return "cuda: layer_norm"
        if "CatArray" in name:
            return "cuda: cat"
        if "elementwise" in name:
            if "add" in name.lower():
                return "cuda: elementwise_add"
            if "mul" in name.lower():
                return "cuda: elementwise_mul"
            if "copy" in name.lower():
                return "cuda: elementwise_copy"
            return "cuda: elementwise"
        if "gather" in name:
            return "cuda: gather"
        if "Memcpy" in name:
            return "cuda: memcpy_d2d"

        # Truncate long names
        if len(name) > 50:
            return name[:47] + "..."
        return name

    # Collect kernel stats
    kernel_stats: dict[str, float] = {}
    for e in prof.key_averages():
        cuda_time = get_cuda_time(e)
        if cuda_time > 0:
            simple_name = simplify_kernel_name(e.key)
            kernel_stats[simple_name] = kernel_stats.get(simple_name, 0) + cuda_time

    # Sort by time and print
    sorted_kernels = sorted(kernel_stats.items(), key=lambda x: -x[1])
    total_cuda = sum(kernel_stats.values())

    print(f"\n{'Kernel':<40} {'Time (ms)':<12} {'%':<8}")
    print("-" * 60)
    for name, time_us in sorted_kernels[:15]:
        time_ms = time_us / 1000
        pct = 100 * time_us / total_cuda if total_cuda > 0 else 0
        print(f"{name:<40} {time_ms:<12.2f} {pct:<8.1f}")
    print("-" * 60)
    print(f"{'Total CUDA time':<40} {total_cuda / 1000:<12.2f}")

    # Highlight custom kernels
    custom_kernels = [k for k in sorted_kernels if k[0].startswith("triton:")]
    if custom_kernels:
        print("\n✓ Custom Triton kernels active:")
        for name, time_us in custom_kernels:
            print(f"  {name}: {time_us / 1000:.2f}ms")


def profile_memory_update_detailed(
    model: TitansGPT,
    x: torch.Tensor,
    num_iters: int = 3,
) -> None:
    """Profile just the memory update operation with torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    device = x.device
    b, _t = x.shape

    # Find memory block
    mem_block = None
    for block in model.transformer["h"]:  # type: ignore[union-attr]
        if block.has_memory:
            mem_block = block
            break

    if mem_block is None:
        print("No memory layer found!")
        return

    memory = mem_block.memory
    seg_len = model.config.segment_len
    h = torch.randn(b, seg_len, model.config.n_embd, device=device)

    print("\n" + "=" * 60)
    print("MEMORY UPDATE DETAILED PROFILE")
    print("=" * 60)

    # Warmup
    state = memory.init_state(b, device)
    for _ in range(3):
        with torch.no_grad():
            state = memory.update(h, state)

    # Profile
    state = memory.init_state(b, device)
    with (
        profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof,
        torch.no_grad(),
    ):
        for _ in range(num_iters):
            state = memory.update(h, state)

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=25,
        )
    )


def run_simple_profiler(model: TitansGPT, x: torch.Tensor, targets: torch.Tensor) -> None:
    """Run the simple timer-based profiler."""
    device = x.device
    batch_size = x.shape[0]
    seq_len = x.shape[1]

    # Profile forward pass
    print("\nProfiling forward pass...")
    timer = Timer()
    with torch.no_grad():
        profile_forward_pass(model, x, timer, num_iters=10)

    print("\nFORWARD PASS BREAKDOWN")
    timer.report()

    # Profile memory operations
    print("\nProfiling memory operations...")
    mem_timer = Timer()
    with torch.no_grad():
        profile_memory_ops(model, x, mem_timer, num_iters=10)

    print("\nMEMORY OPERATIONS BREAKDOWN")
    mem_timer.report()

    # Overall timing
    print("\n" + "=" * 60)
    print("OVERALL FORWARD PASS TIMING")
    print("=" * 60)

    def run_forward() -> torch.Tensor | None:
        states = model.init_memory_states(batch_size, torch.device(device))
        _logits, loss, _ = model(x, targets=targets, memory_states=states)
        return loss

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            run_forward()

    # Time
    torch.cuda.synchronize()
    start = time.perf_counter()
    num_runs = 10
    with torch.no_grad():
        for _ in range(num_runs):
            run_forward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Average forward pass: {1000 * elapsed / num_runs:.2f} ms")
    print(f"Throughput: {batch_size * seq_len * num_runs / elapsed:.0f} tokens/sec")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile TitansGPT")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Use torch.profiler for detailed CUDA kernel analysis",
    )
    parser.add_argument(
        "--memory-only",
        action="store_true",
        help="Profile only memory update operation (with --detailed)",
    )
    parser.add_argument(
        "--export-trace",
        action="store_true",
        help="Export Chrome trace file (with --detailed)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for profiling (default: 10)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for profiling (default: 512)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Config matching training
    config = TitansConfig(
        block_size=512,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        segment_len=128,
        num_persist_mem=4,
        num_longterm_mem=16,
    )

    model = TitansGPT(config).to(device)
    model.eval()

    # Test input
    x = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
    targets = x.clone()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(3):
            model(x, targets=targets)

    if args.detailed:
        if args.memory_only:
            profile_memory_update_detailed(model, x, num_iters=5)
        else:
            profile_with_torch_profiler(
                model, x, targets, num_iters=3, export_trace=args.export_trace
            )
    else:
        run_simple_profiler(model, x, targets)


if __name__ == "__main__":
    main()

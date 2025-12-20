"""
Perplexity evaluation for TitansGPT.

This script evaluates perplexity at different positions in sequences
to demonstrate the benefit of the Titans memory mechanism.

Key metrics:
- Overall perplexity
- Perplexity by position (early/middle/late in sequence)
- Comparison with baseline (no memory context)

Usage:
    uv run python -m nanogpt_titans.eval_perplexity --checkpoint=out-titans/ckpt.pt --dataset=wikitext103
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

from nanogpt_titans.model import TitansConfig, TitansGPT
from nanogpt_titans.train import TrainConfig  # noqa: F401 - needed for checkpoint unpickling

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_model(checkpoint_path: str, device: str) -> TitansGPT:
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config")
    if model_config is None:
        model_config = TitansConfig()

    model = TitansGPT(model_config)
    state_dict = checkpoint["model"]

    # Handle compiled model prefix
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def evaluate_perplexity_by_position(
    model: TitansGPT,
    data: NDArray[np.uint16],
    block_size: int,
    batch_size: int,
    device: torch.device,
    ctx: torch.amp.autocast | nullcontext,  # type: ignore[type-arg]
    num_batches: int = 100,
) -> dict[str, Any]:
    """
    Evaluate perplexity at different positions in sequences.

    This is the key metric for demonstrating Titans memory benefit:
    - Early tokens: Memory hasn't accumulated much
    - Late tokens: Memory should help predict better

    Returns:
        Dictionary with perplexity metrics
    """
    segment_len = model.config.segment_len
    num_segments = block_size // segment_len

    # Track losses by segment position - explicitly typed
    segment_losses: dict[int, list[float]] = {i: [] for i in range(num_segments)}
    total_losses: list[float] = []

    for _ in tqdm(range(num_batches), desc="Evaluating"):
        # Get random starting positions
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))

        x = torch.stack([
            torch.from_numpy(data[i:i + block_size].astype(np.int64))
            for i in ix
        ]).to(device)

        y = torch.stack([
            torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
            for i in ix
        ]).to(device)

        with torch.no_grad(), ctx:
            # Process and get per-token losses
            memory_states = model.init_memory_states(batch_size, device)

            all_losses = []
            for seg_idx in range(num_segments):
                start = seg_idx * segment_len
                end = start + segment_len

                seg_x = x[:, start:end]
                seg_y = y[:, start:end]

                # Get embeddings
                tok_emb = model.transformer["wte"](seg_x)
                pos = torch.arange(start, end, dtype=torch.long, device=device)
                pos_emb = model.transformer["wpe"](pos)
                h = model.transformer["drop"](tok_emb + pos_emb)

                # Process through blocks with memory
                new_states: list[Any] = []
                blocks = model.transformer["h"]  # type: ignore[index]
                for i, block in enumerate(blocks):  # type: ignore[arg-type]
                    h, new_state = block(h, memory_states[i])
                    new_states.append(new_state)
                memory_states = new_states

                h = model.transformer["ln_f"](h)
                logits = model.lm_head(h)

                # Compute per-token loss for this segment
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    seg_y.view(-1),
                    reduction='none'
                ).view(batch_size, segment_len)

                segment_losses[seg_idx].append(loss.mean().item())
                all_losses.append(loss)

            # Overall loss
            total_loss = torch.cat(all_losses, dim=1).mean()
            total_losses.append(total_loss.item())

    # Compute perplexities with explicit typing
    overall_ppl = math.exp(float(np.mean(total_losses)))
    overall_loss = float(np.mean(total_losses))

    by_segment: dict[int, dict[str, float]] = {}
    position_ppls: dict[str, list[float]] = {"early": [], "middle": [], "late": []}

    for seg_idx in range(num_segments):
        avg_loss = float(np.mean(segment_losses[seg_idx]))
        ppl = math.exp(avg_loss)
        by_segment[seg_idx] = {"loss": avg_loss, "perplexity": ppl}

        # Categorize by position
        if seg_idx < num_segments // 3:
            position_ppls["early"].append(ppl)
        elif seg_idx < 2 * num_segments // 3:
            position_ppls["middle"].append(ppl)
        else:
            position_ppls["late"].append(ppl)

    # Average by position
    by_position: dict[str, float] = {}
    for pos in ["early", "middle", "late"]:
        if position_ppls[pos]:
            by_position[pos] = float(np.mean(position_ppls[pos]))
        else:
            by_position[pos] = 0.0

    return {
        "overall_perplexity": overall_ppl,
        "overall_loss": overall_loss,
        "by_segment": by_segment,
        "by_position": by_position,
    }


def print_results(results: dict[str, Any], model_name: str = "Model") -> None:
    """Print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"PERPLEXITY EVALUATION: {model_name}")
    print("=" * 60)

    print(f"\nOverall Perplexity: {results['overall_perplexity']:.2f}")
    print(f"Overall Loss: {results['overall_loss']:.4f}")

    print("\nPerplexity by Position:")
    print(f"  Early segments:  {results['by_position']['early']:.2f}")
    print(f"  Middle segments: {results['by_position']['middle']:.2f}")
    print(f"  Late segments:   {results['by_position']['late']:.2f}")

    improvement = (results['by_position']['early'] - results['by_position']['late']) / results['by_position']['early'] * 100
    print(f"\n  Improvement (early->late): {improvement:.1f}%")

    print("\nPerplexity by Segment:")
    for seg_idx, data in results["by_segment"].items():
        bar = "█" * int(data["perplexity"] / 10)
        print(f"  Segment {seg_idx:2d}: {data['perplexity']:6.2f} {bar}")


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TitansGPT perplexity")
    parser.add_argument("--checkpoint", type=str, default="out-titans/ckpt.pt")
    parser.add_argument("--dataset", type=str, default="wikitext103",
                       choices=["shakespeare", "wikitext103", "openwebtext"])
    parser.add_argument("--block_size", type=int, default=None,
                       help="Override block size (default: use model's)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
    parser.add_argument("--dtype", type=str, default="float16")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    ptdtype = dtype_map.get(args.dtype, torch.float16)
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load data
    data_dir = Path("data") / args.dataset
    val_path = data_dir / "val.bin"

    if not val_path.exists():
        print(f"Data not found at {val_path}")
        print(f"Run: uv run python -m nanogpt_titans.prepare_data {args.dataset}")
        return

    data = np.memmap(val_path, dtype=np.uint16, mode="r")
    print(f"Loaded validation data: {len(data):,} tokens")

    # Load model
    model = load_model(args.checkpoint, args.device)
    block_size = args.block_size or model.config.block_size

    print(f"Model block_size: {model.config.block_size}")
    print(f"Model segment_len: {model.config.segment_len}")
    print(f"Num longterm memory: {model.config.num_longterm_mem}")
    print(f"Num persistent memory: {model.config.num_persist_mem}")

    # Evaluate
    results = evaluate_perplexity_by_position(
        model=model,
        data=data,
        block_size=block_size,
        batch_size=args.batch_size,
        device=device,
        ctx=ctx,
        num_batches=args.num_batches,
    )

    print_results(results, "TitansGPT")

    # Key insight for verification
    print("\n" + "=" * 60)
    print("TITANS VERIFICATION")
    print("=" * 60)
    print("""
If Titans memory is working correctly, you should see:
- Late segment perplexity < Early segment perplexity
- The improvement increases with sequence length

Compare with a baseline (train with --num_longterm_mem=0) to see
the benefit of the memory mechanism.
""")


if __name__ == "__main__":
    main()

"""
Perplexity evaluation for TitansGPT and Qwen-Titans.

This script evaluates perplexity at different positions in sequences
to demonstrate the benefit of the Titans memory mechanism.

Key metrics:
- Overall perplexity
- Perplexity by position (early/middle/late in sequence)
- Comparison with baseline (no memory context)

Usage:
    # TitansGPT (nanoGPT-based)
    uv run python -m nanogpt_titans.eval_perplexity --checkpoint=out-titans/ckpt.pt --dataset=wikitext103

    # Qwen-Titans
    uv run python -m nanogpt_titans.eval_perplexity --qwen --model_name=Qwen/Qwen2-0.5B --titans_state=out-qwen-titans/titans_state_final.pt

    # Base Qwen (no memory, for comparison)
    uv run python -m nanogpt_titans.eval_perplexity --qwen --model_name=Qwen/Qwen2-0.5B
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
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
                    logits.reshape(-1, logits.size(-1)),
                    seg_y.reshape(-1),
                    reduction='none'
                ).reshape(batch_size, segment_len)

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

        # Categorize by position (handle small segment counts)
        if num_segments <= 2:
            # With 1-2 segments, just use early/late
            if seg_idx == 0:
                position_ppls["early"].append(ppl)
            else:
                position_ppls["late"].append(ppl)
        elif num_segments <= 4:
            # With 3-4 segments, split evenly
            if seg_idx < num_segments // 2:
                position_ppls["early"].append(ppl)
            else:
                position_ppls["late"].append(ppl)
        else:
            # Standard thirds for 5+ segments
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

    early_ppl = results['by_position']['early']
    late_ppl = results['by_position']['late']
    if early_ppl > 0:
        improvement = (early_ppl - late_ppl) / early_ppl * 100
        print(f"\n  Improvement (early->late): {improvement:.1f}%")
    else:
        print("\n  Improvement: N/A (no early segment data)")

    print("\nPerplexity by Segment:")
    for seg_idx, data in results["by_segment"].items():
        bar = "█" * int(data["perplexity"] / 10)
        print(f"  Segment {seg_idx:2d}: {data['perplexity']:6.2f} {bar}")


# =============================================================================
# Qwen-Titans Support
# =============================================================================


def load_qwen_model(
    model_name: str,
    titans_state: str | None,
    memory_layers: list[int],
    segment_len: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[nn.Module, Any, int]:
    """
    Load Qwen model, optionally with Titans memory.

    Returns:
        Tuple of (model, tokenizer, segment_len)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanogpt_titans.qwen_titans import (
        TitansQwenConfig,
        patch_qwen_with_titans,
        load_titans_state,
    )

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use SDPA for faster attention on CUDA
    attn_impl = "sdpa" if "cuda" in device else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_impl,
    )

    if titans_state:
        # Patch with Titans memory
        titans_config = TitansQwenConfig.from_qwen_config(
            model.config,
            segment_len=segment_len,
            memory_layers=memory_layers,
        )
        print(f"Patching with Titans memory at layers {memory_layers}...")
        model = patch_qwen_with_titans(model, titans_config)
        load_titans_state(model, titans_state)

    model.eval()
    return model, tokenizer, segment_len


def evaluate_qwen_perplexity_by_position(
    model: nn.Module,
    tokenizer: Any,
    texts: list[str],
    segment_len: int,
    device: torch.device,
    ctx: Any,
    use_titans: bool = False,
) -> dict[str, Any]:
    """
    Evaluate perplexity at different positions for Qwen models.

    Args:
        model: Qwen model (with or without Titans)
        tokenizer: Tokenizer
        texts: List of text samples to evaluate
        segment_len: Segment length for processing
        device: Device to use
        ctx: Autocast context
        use_titans: Whether model has Titans memory

    Returns:
        Dictionary with perplexity metrics
    """
    from nanogpt_titans.qwen_titans import TitansStateManager

    if use_titans:
        state_manager = TitansStateManager(model)

    # Track losses by position
    position_losses: dict[str, list[float]] = {"early": [], "middle": [], "late": []}
    total_losses: list[float] = []

    skipped_short = 0
    for text in tqdm(texts, desc="Evaluating"):
        # Tokenize - use longer max_length for proper eval
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        input_ids = tokens["input_ids"].to(device)
        seq_len = input_ids.size(1)

        if seq_len < segment_len * 2:
            skipped_short += 1
            continue  # Skip short sequences

        num_segments = seq_len // segment_len

        if use_titans:
            state_manager.reset()
            state_manager.init_states(1, device)

        segment_losses = []

        with torch.no_grad(), ctx:
            for seg_idx in range(num_segments):
                start = seg_idx * segment_len
                end = min(start + segment_len, seq_len - 1)

                seg_input = input_ids[:, start:end]
                seg_labels = input_ids[:, start + 1:end + 1]

                if seg_input.size(1) != seg_labels.size(1):
                    continue

                if use_titans:
                    state_manager.sync_to_layers()

                outputs = model(seg_input, labels=seg_labels, use_cache=False)
                loss = outputs.loss.item()

                if use_titans:
                    state_manager.sync_from_layers()

                # Skip NaN losses
                if not math.isnan(loss):
                    segment_losses.append((seg_idx, loss))

        if not segment_losses:
            continue

        # Categorize by position
        num_segs = len(segment_losses)
        for i, (seg_idx, loss) in enumerate(segment_losses):
            total_losses.append(loss)

            if num_segs <= 2:
                if i == 0:
                    position_losses["early"].append(loss)
                else:
                    position_losses["late"].append(loss)
            else:
                if i < num_segs // 3:
                    position_losses["early"].append(loss)
                elif i < 2 * num_segs // 3:
                    position_losses["middle"].append(loss)
                else:
                    position_losses["late"].append(loss)

    # Compute perplexities
    if not total_losses:
        return {"error": f"No valid losses computed. Skipped {skipped_short} short sequences (need >= {segment_len * 2} tokens)"}

    overall_loss = float(np.mean(total_losses))
    overall_ppl = math.exp(overall_loss)

    by_position: dict[str, float] = {}
    for pos in ["early", "middle", "late"]:
        if position_losses[pos]:
            avg_loss = float(np.mean(position_losses[pos]))
            by_position[pos] = math.exp(avg_loss)
        else:
            by_position[pos] = 0.0

    return {
        "overall_perplexity": overall_ppl,
        "overall_loss": overall_loss,
        "by_position": by_position,
        "num_samples": len(texts),
        "num_valid_losses": len(total_losses),
    }


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TitansGPT or Qwen-Titans perplexity")
    # Model selection
    parser.add_argument("--qwen", action="store_true", help="Use Qwen-Titans instead of TitansGPT")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B", help="Qwen model name")
    parser.add_argument("--titans_state", type=str, default=None, help="Path to Titans state for Qwen")
    parser.add_argument("--memory_layers", type=str, default="12", help="Memory layer indices (comma-separated)")
    parser.add_argument("--segment_len", type=int, default=256, help="Segment length for Qwen")

    # TitansGPT args
    parser.add_argument("--checkpoint", type=str, default="out-titans/ckpt.pt")

    # Common args
    parser.add_argument("--dataset", type=str, default="wikitext",
                       choices=["shakespeare", "wikitext103", "openwebtext", "wikitext"])
    parser.add_argument("--block_size", type=int, default=None,
                       help="Override block size (default: use model's)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples for Qwen eval")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()
    memory_layers = [int(x.strip()) for x in args.memory_layers.split(",")]

    # Setup
    device = torch.device(args.device)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    ptdtype = dtype_map.get(args.dtype, torch.bfloat16)
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # =========================================================================
    # Qwen-Titans evaluation
    # =========================================================================
    if args.qwen:
        model, tokenizer, segment_len = load_qwen_model(
            model_name=args.model_name,
            titans_state=args.titans_state,
            memory_layers=memory_layers,
            segment_len=args.segment_len,
            device=args.device,
            dtype=ptdtype,
        )

        use_titans = args.titans_state is not None
        model_name = f"Qwen-Titans ({args.model_name})" if use_titans else f"Base Qwen ({args.model_name})"

        # Load text data from HuggingFace datasets
        print(f"\nLoading {args.dataset} dataset...")
        from datasets import load_dataset

        if args.dataset == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
            # Concatenate texts to create long sequences
            # WikiText articles are short, so we join them
            all_text = "\n\n".join([t for t in dataset["text"] if t.strip()])
            
            # Split into chunks of ~8K tokens (~32K chars)
            min_chars = segment_len * 8 * 4  # 8 segments worth
            texts = []
            for i in range(0, len(all_text), min_chars):
                chunk = all_text[i:i + min_chars]
                if len(chunk) > segment_len * 2 * 4:  # At least 2 segments
                    texts.append(chunk)
                if len(texts) >= args.num_samples:
                    break
            print(f"Created {len(texts)} long text chunks (~{min_chars} chars each)")
        else:
            # Fallback to local data if available
            data_dir = Path("data") / args.dataset
            val_path = data_dir / "val.bin"
            if val_path.exists():
                data = np.memmap(val_path, dtype=np.uint16, mode="r")
                # Convert to text - use longer chunks for proper eval
                chunk_size = segment_len * 4  # 4 segments worth
                texts = [tokenizer.decode(data[i:i+chunk_size].tolist()) 
                         for i in range(0, min(len(data), args.num_samples * chunk_size), chunk_size)]
            else:
                print(f"Dataset {args.dataset} not found. Using wikitext.")
                dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
                min_chars = segment_len * 2 * 4
                texts = [t for t in dataset["text"] if len(t) > min_chars][:args.num_samples]

        print(f"Evaluating on {len(texts)} samples...")

        results = evaluate_qwen_perplexity_by_position(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            segment_len=segment_len,
            device=device,
            ctx=ctx,
            use_titans=use_titans,
        )

        # Print results
        print(f"\n{'=' * 60}")
        print(f"PERPLEXITY EVALUATION: {model_name}")
        print("=" * 60)

        if "error" in results:
            print(f"Error: {results['error']}")
            return

        print(f"\nOverall Perplexity: {results['overall_perplexity']:.2f}")
        print(f"Overall Loss: {results['overall_loss']:.4f}")
        print(f"Samples evaluated: {results['num_samples']}")
        print(f"Valid losses: {results['num_valid_losses']}")

        print("\nPerplexity by Position:")
        for pos in ["early", "middle", "late"]:
            ppl = results['by_position'].get(pos, 0)
            if ppl > 0:
                print(f"  {pos.capitalize():8s}: {ppl:.2f}")

        early_ppl = results['by_position'].get('early', 0)
        late_ppl = results['by_position'].get('late', 0)
        if early_ppl > 0 and late_ppl > 0:
            improvement = (early_ppl - late_ppl) / early_ppl * 100
            print(f"\n  Improvement (early->late): {improvement:.1f}%")
            if improvement > 0:
                print("  ✓ Memory is helping with later positions")
            else:
                print("  ✗ Memory not showing benefit (may need more training)")

        return

    # =========================================================================
    # TitansGPT evaluation (original)
    # =========================================================================
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

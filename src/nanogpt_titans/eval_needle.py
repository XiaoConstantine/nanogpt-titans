"""
Needle-in-Haystack evaluation for TitansGPT.

Tests the model's ability to retrieve information from long contexts.
This is where Titans should significantly outperform vanilla transformers.

Usage:
    uv run python -m nanogpt_titans.eval_needle --checkpoint=out-titans/ckpt.pt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tiktoken
import torch
from tqdm import tqdm

from nanogpt_titans.model import TitansConfig, TitansGPT
from nanogpt_titans.train import TrainConfig  # noqa: F401 - needed for checkpoint loading


def load_model(
    checkpoint_path: str,
    device: str,
) -> tuple[TitansGPT, tiktoken.Encoding]:
    """Load model from checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        msg = f"Checkpoint not found: {ckpt_path}"
        raise FileNotFoundError(msg)

    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config")
    if model_config is None:
        model_config = TitansConfig()

    model = TitansGPT(model_config)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    return model, enc


def evaluate_needle_retrieval(
    model: TitansGPT,
    enc: tiktoken.Encoding,
    samples: list[dict[str, Any]],
    device: str,
    max_gen_tokens: int = 20,
) -> dict[str, Any]:
    """
    Evaluate model on needle-in-haystack retrieval.

    Args:
        model: TitansGPT model
        enc: Tokenizer
        samples: List of needle-in-haystack samples
        device: Device to run on
        max_gen_tokens: Max tokens to generate for answer

    Returns:
        Dictionary with evaluation metrics
    """
    # Use explicit typed structures to help type checker
    correct_count = 0
    by_position: dict[str, dict[str, int]] = {
        "early": {"total": 0, "correct": 0},
        "middle": {"total": 0, "correct": 0},
        "late": {"total": 0, "correct": 0},
    }
    by_length: dict[int, dict[str, int]] = {}
    sample_results: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Evaluating"):
        prompt = sample["prompt"]
        needle = sample["needle"]
        position = sample["position"]
        target_length = sample["target_length"]

        # Tokenize
        input_ids = enc.encode_ordinary(prompt)

        # Check if context fits
        if len(input_ids) > model.config.block_size:
            print(f"Skipping sample {sample['id']}: too long ({len(input_ids)} > {model.config.block_size})")
            continue

        x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]

        # Generate
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=max_gen_tokens, temperature=0.1, top_k=10)

        # Decode generated part only
        generated_ids = y[0, len(input_ids) :].tolist()
        generated_text = enc.decode(generated_ids).strip()

        # Check if needle is in generated text
        is_correct = needle.upper() in generated_text.upper()

        # Update results
        if is_correct:
            correct_count += 1

        by_position[position]["total"] += 1
        if is_correct:
            by_position[position]["correct"] += 1

        if target_length not in by_length:
            by_length[target_length] = {"total": 0, "correct": 0}
        by_length[target_length]["total"] += 1
        if is_correct:
            by_length[target_length]["correct"] += 1

        sample_results.append(
            {
                "id": sample["id"],
                "needle": needle,
                "generated": generated_text[:100],
                "correct": is_correct,
                "position": position,
                "length": target_length,
            }
        )

    # Calculate accuracies
    total_samples = len(samples)
    accuracy = correct_count / total_samples if total_samples > 0 else 0.0

    by_position_with_acc: dict[str, dict[str, float | int]] = {}
    for pos, data in by_position.items():
        total = data["total"]
        correct = data["correct"]
        by_position_with_acc[pos] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    by_length_with_acc: dict[int, dict[str, float | int]] = {}
    for length, data in by_length.items():
        total = data["total"]
        correct = data["correct"]
        by_length_with_acc[length] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
        }

    return {
        "total": total_samples,
        "correct": correct_count,
        "accuracy": accuracy,
        "by_position": by_position_with_acc,
        "by_length": by_length_with_acc,
        "samples": sample_results,
    }


def print_results(results: dict[str, Any]) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("NEEDLE-IN-HAYSTACK EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nOverall Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")

    print("\nAccuracy by Needle Position:")
    for pos, data in results["by_position"].items():
        if data["total"] > 0:
            print(f"  {pos:8s}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")

    print("\nAccuracy by Context Length:")
    for length in sorted(results["by_length"].keys()):
        data = results["by_length"][length]
        if data["total"] > 0:
            print(f"  {length:5d} tokens: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")

    print("\nSample Results:")
    for sample in results["samples"][:5]:
        status = "correct" if sample["correct"] else "WRONG"
        print(f"  [{status:7s}] Needle: {sample['needle']}, Generated: {sample['generated'][:50]}...")


def main() -> None:
    """Entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TitansGPT on needle-in-haystack")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="out-titans/ckpt.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/needle/needle_haystack.json",
        help="Path to needle-in-haystack data",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (None = all)",
    )
    parser.add_argument(
        "--max_gen_tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate for answer",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Run: uv run python -m nanogpt_titans.prepare_data needle")
        return

    with data_path.open() as f:
        samples = json.load(f)

    if args.max_samples:
        samples = samples[: args.max_samples]

    print(f"Loaded {len(samples)} samples")

    # Load model
    model, enc = load_model(args.checkpoint, args.device)

    # Evaluate
    results = evaluate_needle_retrieval(
        model=model,
        enc=enc,
        samples=samples,
        device=args.device,
        max_gen_tokens=args.max_gen_tokens,
    )

    # Print results
    print_results(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

"""
Needle-in-Haystack evaluation for Titans-enhanced Qwen models.

Tests the model's ability to retrieve information from long contexts.
This is where Titans memory should significantly outperform vanilla transformers.

Usage:
    # Evaluate Titans-enhanced Qwen
    uv run python -m nanogpt_titans.eval_qwen_needle --model_name Qwen/Qwen2-1.5B --titans

    # Compare with base Qwen (no memory)
    uv run python -m nanogpt_titans.eval_qwen_needle --model_name Qwen/Qwen2-1.5B

    # Load trained Titans state
    uv run python -m nanogpt_titans.eval_qwen_needle \
        --model_name Qwen/Qwen2-1.5B \
        --titans \
        --titans_state out-qwen-titans/titans_state_final.pt
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    raise ImportError("transformers is required. Install with: uv add transformers") from e

from nanogpt_titans.qwen_titans import (
    TitansQwenConfig,
    TitansStateManager,
    load_titans_state,
    patch_qwen_with_titans,
    titans_generate,
)


@dataclass
class NeedleConfig:
    """Configuration for needle-in-haystack evaluation."""

    # Context settings
    context_lengths: list[int] = None  # Will be set in __post_init__
    needle_depths: list[float] = None  # Will be set in __post_init__
    num_samples_per_combo: int = 3

    # Generation settings
    max_gen_tokens: int = 50
    temperature: float = 0.1

    # Needle templates
    needle_template: str = "The secret password is: {secret}"
    question_template: str = "\n\nQuestion: What is the secret password?\nAnswer:"
    secrets: list[str] = None  # Will be set in __post_init__

    def __post_init__(self):
        if self.context_lengths is None:
            self.context_lengths = [512, 1024, 2048, 4096]
        if self.needle_depths is None:
            self.needle_depths = [0.1, 0.25, 0.5, 0.75, 0.9]
        if self.secrets is None:
            self.secrets = [
                "ALPHA7X",
                "BETA42Z",
                "GAMMA99",
                "DELTA33",
                "OMEGA88",
                "SIGMA55",
                "THETA77",
                "LAMBDA22",
            ]


def generate_haystack(tokenizer, target_tokens: int) -> str:
    """
    Generate filler text (haystack) of approximately target_tokens length.

    Uses repetitive but varied text that's realistic but doesn't contain
    information that could be confused with the needle.
    """
    filler_sentences = [
        "The weather today is quite pleasant with clear skies.",
        "Many people enjoy walking in the park during autumn.",
        "Technology continues to advance at a rapid pace.",
        "Reading books is a great way to expand knowledge.",
        "Music has been an important part of human culture.",
        "The ocean covers most of the Earth's surface.",
        "Mountains provide habitats for many unique species.",
        "Cities around the world have diverse architectures.",
        "Science helps us understand the natural world.",
        "Art expresses human creativity in many forms.",
        "History teaches us about past civilizations.",
        "Languages evolve and change over time.",
        "Sports bring communities together.",
        "Food traditions vary across cultures.",
        "Transportation has transformed modern life.",
    ]

    # Estimate tokens per sentence (rough approximation)
    avg_tokens_per_sentence = 10

    # Generate more sentences than needed
    num_sentences = (target_tokens // avg_tokens_per_sentence) + 10
    text_parts = []

    for i in range(num_sentences):
        sentence = filler_sentences[i % len(filler_sentences)]
        text_parts.append(sentence)

    text = " ".join(text_parts)

    # Trim to approximate target length
    tokens = tokenizer.encode(text)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        text = tokenizer.decode(tokens)

    return text


def create_needle_sample(
    tokenizer,
    target_length: int,
    needle_depth: float,
    secret: str,
    config: NeedleConfig,
) -> dict[str, Any]:
    """
    Create a single needle-in-haystack sample.

    Args:
        tokenizer: Tokenizer for encoding
        target_length: Target context length in tokens
        needle_depth: Where to place needle (0.0 = start, 1.0 = end)
        secret: The secret value to hide
        config: Needle configuration

    Returns:
        Dictionary with prompt, needle, position, etc.
    """
    needle = config.needle_template.format(secret=secret)
    question = config.question_template

    # Reserve space for needle and question
    needle_tokens = len(tokenizer.encode(needle))
    question_tokens = len(tokenizer.encode(question))
    haystack_tokens = target_length - needle_tokens - question_tokens - 20  # buffer

    if haystack_tokens < 100:
        haystack_tokens = 100

    # Generate haystack
    haystack = generate_haystack(tokenizer, haystack_tokens)

    # Split haystack at needle position
    haystack_parts = haystack.split(". ")
    insert_idx = int(len(haystack_parts) * needle_depth)
    insert_idx = max(1, min(insert_idx, len(haystack_parts) - 1))

    # Insert needle
    before = ". ".join(haystack_parts[:insert_idx]) + ". "
    after = ". ".join(haystack_parts[insert_idx:])
    full_context = before + needle + " " + after + question

    # Determine position label
    if needle_depth <= 0.33:
        position = "early"
    elif needle_depth <= 0.66:
        position = "middle"
    else:
        position = "late"

    return {
        "prompt": full_context,
        "needle": secret,
        "position": position,
        "depth": needle_depth,
        "target_length": target_length,
        "actual_length": len(tokenizer.encode(full_context)),
    }


def generate_samples(
    tokenizer,
    config: NeedleConfig,
) -> list[dict[str, Any]]:
    """Generate all needle-in-haystack samples."""
    samples = []
    sample_id = 0

    for context_len in config.context_lengths:
        for depth in config.needle_depths:
            for i in range(config.num_samples_per_combo):
                secret = config.secrets[(sample_id + i) % len(config.secrets)]
                sample = create_needle_sample(tokenizer, context_len, depth, secret, config)
                sample["id"] = sample_id
                samples.append(sample)
                sample_id += 1

    random.shuffle(samples)
    return samples


@torch.no_grad()
def evaluate_needle(
    model,
    tokenizer,
    samples: list[dict[str, Any]],
    config: NeedleConfig,
    device: torch.device,
    use_titans: bool = False,
    state_manager: TitansStateManager | None = None,
    segment_len: int = 512,
) -> dict[str, Any]:
    """
    Evaluate model on needle-in-haystack samples.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        samples: List of needle samples
        config: Needle configuration
        device: Device to run on
        use_titans: Whether to use Titans generation
        state_manager: Titans state manager (if use_titans)
        segment_len: Segment length for Titans

    Returns:
        Dictionary with evaluation metrics
    """
    correct_count = 0
    by_position: dict[str, dict[str, int]] = {
        "early": {"total": 0, "correct": 0},
        "middle": {"total": 0, "correct": 0},
        "late": {"total": 0, "correct": 0},
    }
    by_length: dict[int, dict[str, int]] = {}
    by_depth: dict[float, dict[str, int]] = {}
    sample_results: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Evaluating"):
        prompt = sample["prompt"]
        needle = sample["needle"]
        position = sample["position"]
        target_length = sample["target_length"]
        depth = sample["depth"]

        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        # Generate
        if use_titans and state_manager is not None:
            output_ids = titans_generate(
                model,
                input_ids,
                max_new_tokens=config.max_gen_tokens,
                temperature=config.temperature,
                state_manager=state_manager,
                segment_len=segment_len,
                reset_memory=True,
            )
        else:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=config.max_gen_tokens,
                temperature=config.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode generated part only
        generated_ids = output_ids[0, input_ids.size(1) :].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Check if needle is in generated text (case-insensitive)
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

        if depth not in by_depth:
            by_depth[depth] = {"total": 0, "correct": 0}
        by_depth[depth]["total"] += 1
        if is_correct:
            by_depth[depth]["correct"] += 1

        sample_results.append(
            {
                "id": sample["id"],
                "needle": needle,
                "generated": generated_text[:100],
                "correct": is_correct,
                "position": position,
                "depth": depth,
                "length": target_length,
            }
        )

    # Calculate accuracies
    total_samples = len(samples)
    accuracy = correct_count / total_samples if total_samples > 0 else 0.0

    def add_accuracy(d: dict) -> dict:
        return {
            **d,
            "accuracy": d["correct"] / d["total"] if d["total"] > 0 else 0.0,
        }

    return {
        "total": total_samples,
        "correct": correct_count,
        "accuracy": accuracy,
        "by_position": {k: add_accuracy(v) for k, v in by_position.items()},
        "by_length": {k: add_accuracy(v) for k, v in sorted(by_length.items())},
        "by_depth": {k: add_accuracy(v) for k, v in sorted(by_depth.items())},
        "samples": sample_results,
    }


def print_results(results: dict[str, Any], title: str = "EVALUATION RESULTS") -> None:
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"\nOverall Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})"
    )

    print("\nAccuracy by Needle Position:")
    for pos, data in results["by_position"].items():
        if data["total"] > 0:
            print(f"  {pos:8s}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")

    print("\nAccuracy by Context Length:")
    for length, data in results["by_length"].items():
        if data["total"] > 0:
            print(
                f"  {length:5d} tokens: {data['accuracy']:.1%} ({data['correct']}/{data['total']})"
            )

    print("\nAccuracy by Needle Depth:")
    for depth, data in results["by_depth"].items():
        if data["total"] > 0:
            print(
                f"  {depth:.0%} depth: {data['accuracy']:.1%} ({data['correct']}/{data['total']})"
            )

    print("\nSample Results (first 5):")
    for sample in results["samples"][:5]:
        status = "OK" if sample["correct"] else "WRONG"
        print(
            f"  [{status:5s}] Needle: {sample['needle']}, Generated: {sample['generated'][:50]}..."
        )


def main() -> None:
    """Entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Qwen model on needle-in-haystack")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--titans", action="store_true", help="Use Titans memory")
    parser.add_argument(
        "--titans_state", type=str, default=None, help="Path to trained Titans state"
    )
    parser.add_argument(
        "--memory_layers", type=str, default="14", help="Comma-separated layer indices for Titans"
    )

    # Titans config
    parser.add_argument("--segment_len", type=int, default=512)
    parser.add_argument("--num_persist_mem", type=int, default=4)
    parser.add_argument("--num_longterm_mem", type=int, default=16)

    # Evaluation config
    parser.add_argument(
        "--context_lengths",
        type=str,
        default="512,1024,2048",
        help="Comma-separated context lengths",
    )
    parser.add_argument(
        "--needle_depths",
        type=str,
        default="0.1,0.25,0.5,0.75,0.9",
        help="Comma-separated needle depths",
    )
    parser.add_argument("--num_samples", type=int, default=3, help="Samples per length/depth combo")
    parser.add_argument("--max_gen_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.1)

    # Output
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")

    # System
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    # Setup dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"Device: {device}, dtype: {dtype}")

    # Load model
    print(f"\nLoading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
    )

    # Patch with Titans if requested
    state_manager = None
    if args.titans:
        print("\nPatching model with Titans memory...")
        memory_layers = [int(x.strip()) for x in args.memory_layers.split(",")]

        titans_config = TitansQwenConfig.from_qwen_config(
            model.config,
            segment_len=args.segment_len,
            num_persist_mem=args.num_persist_mem,
            num_longterm_mem=args.num_longterm_mem,
            memory_layers=memory_layers,
        )

        model = patch_qwen_with_titans(model, titans_config)

        # Load trained state if provided
        if args.titans_state:
            print(f"Loading Titans state from {args.titans_state}")
            load_titans_state(model, args.titans_state)

        state_manager = TitansStateManager(model)

    model.to(device)
    model.eval()

    # Create needle config
    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    needle_depths = [float(x.strip()) for x in args.needle_depths.split(",")]

    needle_config = NeedleConfig(
        context_lengths=context_lengths,
        needle_depths=needle_depths,
        num_samples_per_combo=args.num_samples,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
    )

    # Generate samples
    print("\nGenerating needle-in-haystack samples...")
    samples = generate_samples(tokenizer, needle_config)
    print(f"Generated {len(samples)} samples")

    # Evaluate
    print("\nEvaluating...")
    t0 = time.time()
    results = evaluate_needle(
        model,
        tokenizer,
        samples,
        needle_config,
        device,
        use_titans=args.titans,
        state_manager=state_manager,
        segment_len=args.segment_len,
    )
    t1 = time.time()

    # Print results
    title = "TITANS-QWEN RESULTS" if args.titans else "BASE QWEN RESULTS"
    print_results(results, title)
    print(f"\nEvaluation time: {t1 - t0:.1f}s")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        results["metadata"] = {
            "model_name": args.model_name,
            "titans": args.titans,
            "titans_state": args.titans_state,
            "segment_len": args.segment_len,
            "device": str(device),
            "dtype": args.dtype,
        }

        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

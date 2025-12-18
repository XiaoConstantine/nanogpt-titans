"""
Sample from a trained TitansGPT model.

Usage:
    uv run python -m nanogpt_titans.sample --checkpoint=out-titans/ckpt.pt
    uv run python -m nanogpt_titans.sample --prompt="To be or not to be"
"""

from __future__ import annotations

from pathlib import Path

import tiktoken
import torch

from nanogpt_titans.model import TitansConfig, TitansGPT


def sample(
    checkpoint_path: str,
    prompt: str = "\n",
    num_samples: int = 1,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 200,
    device: str = "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu",
) -> list[str]:
    """
    Generate samples from a trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        prompt: Starting prompt text
        num_samples: Number of samples to generate
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        device: Device to run on

    Returns:
        List of generated text samples
    """
    # Load checkpoint
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        msg = f"Checkpoint not found: {ckpt_path}"
        raise FileNotFoundError(msg)

    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Load model config
    model_config = checkpoint.get("model_config")
    if model_config is None:
        # Fallback to default config
        print("No model config in checkpoint, using defaults")
        model_config = TitansConfig()

    # Create model
    model = TitansGPT(model_config)

    # Load weights
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Encode prompt
    start_ids = enc.encode_ordinary(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Generate
    samples = []
    with torch.no_grad():
        for i in range(num_samples):
            print(f"Generating sample {i + 1}/{num_samples}...")

            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            generated = enc.decode(y[0].tolist())
            samples.append(generated)
            print("-" * 50)
            print(generated)
            print("-" * 50)

    return samples


def main() -> None:
    """Entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Sample from TitansGPT")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="out-titans/ckpt.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="\n",
        help="Starting prompt",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=200,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    sample(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()

"""
Data preparation scripts for training TitansGPT.

Supports:
- Shakespeare (tiny dataset for quick experiments)
- WikiText-103 (standard LM benchmark, ~100M tokens)
- OpenWebText (larger dataset for serious training)

Usage:
    uv run python -m nanogpt_titans.prepare_data shakespeare
    uv run python -m nanogpt_titans.prepare_data wikitext103
    uv run python -m nanogpt_titans.prepare_data openwebtext
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tiktoken


def prepare_shakespeare(data_dir: Path) -> None:
    """
    Prepare the Shakespeare dataset.

    Downloads the tiny shakespeare dataset and tokenizes it.
    """
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)

    # Download shakespeare
    input_file_path = data_dir / "input.txt"
    if not input_file_path.exists():
        print("Downloading Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, input_file_path)
        print(f"Downloaded to {input_file_path}")

    with input_file_path.open(encoding="utf-8") as f:
        data = f.read()

    n = len(data)
    print(f"Dataset size: {n:,} characters")

    # Train/val split
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # Encode with tiktoken (GPT-2 tokenizer)
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    # Save to binary files
    train_ids_np = np.array(train_ids, dtype=np.uint16)
    val_ids_np = np.array(val_ids, dtype=np.uint16)

    train_ids_np.tofile(data_dir / "train.bin")
    val_ids_np.tofile(data_dir / "val.bin")

    print(f"Saved to {data_dir}/train.bin and {data_dir}/val.bin")


def prepare_wikitext103(data_dir: Path, num_proc: int = 4) -> None:
    """
    Prepare the WikiText-103 dataset.

    WikiText-103 is a standard language modeling benchmark with ~100M tokens
    from Wikipedia articles. Good for testing long-range dependencies.

    Requires: pip install datasets
    """
    from datasets import load_dataset

    data_dir.mkdir(parents=True, exist_ok=True)

    print("Loading WikiText-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    enc = tiktoken.get_encoding("gpt2")

    def process(example: dict) -> dict:
        text = example["text"]
        if not text.strip():
            return {"ids": [], "len": 0}
        ids = enc.encode_ordinary(text)
        return {"ids": ids, "len": len(ids)}

    print("Tokenizing dataset...")
    tokenized = dataset.map(  # type: ignore[call-arg]
        process,
        remove_columns=["text"],
        desc="Tokenizing",  # type: ignore[unknown-argument]
        num_proc=num_proc,  # type: ignore[unknown-argument]
    )

    # Save train, validation, and test splits
    for split in ["train", "validation", "test"]:
        if split not in tokenized:
            continue

        dset = tokenized[split]
        # Filter empty examples
        dset = dset.filter(lambda x: x["len"] > 0)  # type: ignore[union-attr]

        arr_len = sum(dset["len"])
        print(f"{split} has {arr_len:,} tokens")

        # Map validation -> val for consistency
        filename = "val.bin" if split == "validation" else f"{split}.bin"
        filepath = data_dir / filename

        arr = np.memmap(filepath, dtype=np.uint16, mode="w+", shape=(arr_len,))

        idx = 0
        for item in dset:
            ids = item["ids"]
            arr[idx : idx + len(ids)] = ids
            idx += len(ids)

        arr.flush()
        print(f"Saved to {filepath}")

    print(f"\nWikiText-103 prepared in {data_dir}")
    print("This dataset has natural document boundaries - good for testing memory!")


def prepare_openwebtext(data_dir: Path, num_proc: int = 8) -> None:
    """
    Prepare the OpenWebText dataset.

    Uses HuggingFace datasets to download and process.
    Requires: pip install datasets
    """
    from datasets import load_dataset

    data_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OpenWebText dataset (this may take a while)...")
    # Use the updated dataset path (old 'openwebtext' is deprecated)
    dataset = load_dataset("Skylion007/openwebtext", num_proc=num_proc, trust_remote_code=True)

    # Split train into train and val
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)  # type: ignore[union-attr]
    split_dataset["val"] = split_dataset.pop("test")

    # Tokenize
    enc = tiktoken.get_encoding("gpt2")

    def process(example: dict) -> dict:
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    print("Tokenizing dataset...")
    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=num_proc,
    )

    # Save to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} has {arr_len:,} tokens")

        filename = data_dir / f"{split}.bin"
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))

        total_batches = 1024
        idx = 0
        for batch_idx in range(total_batches):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        print(f"Saved to {filename}")


def prepare_needle_haystack(data_dir: Path, num_samples: int = 100) -> None:
    """
    Prepare needle-in-haystack evaluation data.

    Creates synthetic data where a "needle" (secret info) is hidden
    in a "haystack" (filler text), and the model must retrieve it.
    """
    import json
    import random

    data_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    # Filler text for haystack
    filler_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm.",
        "Actions speak louder than words.",
        "Beauty is in the eye of the beholder.",
        "Better late than never.",
        "Birds of a feather flock together.",
        "Don't count your chickens before they hatch.",
        "Every cloud has a silver lining.",
        "Fortune favors the bold.",
        "Honesty is the best policy.",
        "Knowledge is power.",
        "Laughter is the best medicine.",
    ]

    # Generate samples with different needle positions and haystack lengths
    samples = []
    needle_positions = ["early", "middle", "late"]
    haystack_lengths = [256, 512, 1024, 2048]

    for i in range(num_samples):
        # Random needle (secret code)
        colors = ["RED", "BLUE", "GREEN", "YELLOW", "PURPLE", "ORANGE"]
        animals = ["ELEPHANT", "TIGER", "DOLPHIN", "EAGLE", "WOLF", "BEAR"]
        needle = f"{random.choice(colors)} {random.choice(animals)}"

        # Random config
        position = random.choice(needle_positions)
        target_length = random.choice(haystack_lengths)

        # Build haystack
        needle_text = f"The secret code is: {needle}."
        question = "Question: What was the secret code?\nAnswer:"

        # Calculate filler needed
        needle_tokens = len(enc.encode_ordinary(needle_text))
        question_tokens = len(enc.encode_ordinary(question))
        filler_tokens_needed = target_length - needle_tokens - question_tokens

        # Generate filler
        filler = ""
        current_tokens = 0
        while current_tokens < filler_tokens_needed:
            sentence = random.choice(filler_sentences) + " "
            filler += sentence
            current_tokens = len(enc.encode_ordinary(filler))

        # Position needle
        filler_parts = filler.split(". ")
        num_parts = len(filler_parts)

        if position == "early":
            insert_idx = num_parts // 10
        elif position == "middle":
            insert_idx = num_parts // 2
        else:  # late
            insert_idx = num_parts * 9 // 10

        filler_parts.insert(insert_idx, needle_text)  # type: ignore[arg-type]
        haystack = ". ".join(filler_parts)

        # Create prompt
        prompt = f"{haystack}\n\n{question}"

        samples.append(
            {
                "id": i,
                "prompt": prompt,
                "needle": needle,
                "position": position,
                "target_length": target_length,
                "actual_tokens": len(enc.encode_ordinary(prompt)),
            }
        )

    # Save
    output_file = data_dir / "needle_haystack.json"
    with output_file.open("w") as f:
        json.dump(samples, f, indent=2)

    print(f"Generated {num_samples} needle-in-haystack samples")
    print(f"Saved to {output_file}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare data for TitansGPT training")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["shakespeare", "wikitext103", "openwebtext", "needle"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory for data",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of processes for parallel processing",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples for needle-in-haystack",
    )

    args = parser.parse_args()
    base_dir = Path(args.data_dir)

    if args.dataset == "shakespeare":
        prepare_shakespeare(base_dir / "shakespeare")
    elif args.dataset == "wikitext103":
        prepare_wikitext103(base_dir / "wikitext103", num_proc=args.num_proc)
    elif args.dataset == "openwebtext":
        prepare_openwebtext(base_dir / "openwebtext", num_proc=args.num_proc)
    elif args.dataset == "needle":
        prepare_needle_haystack(base_dir / "needle", num_samples=args.num_samples)


if __name__ == "__main__":
    main()

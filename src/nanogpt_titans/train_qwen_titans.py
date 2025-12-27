"""
Training script for HOPE-Titans Qwen models.

Trains the memory components on top of a frozen Qwen2 base model.
Uses HOPE-style gated residual integration.

Phase 3 Diagnostics monitored:
- Gate Mean: Should increase from 0.1 → 0.5 as memory becomes useful
- Internal Loss: Should decrease
- LM Loss: Should decrease
- CMS Level Weights: Should be distributed (not all on one level)

Usage:
    uv run python -m nanogpt_titans.train_qwen_titans --model_name Qwen/Qwen2-0.5B

Example:
    uv run python -m nanogpt_titans.train_qwen_titans \
        --model_name Qwen/Qwen2-0.5B \
        --memory_layers 12 \
        --learning_rate 1e-4 \
        --batch_size 2 \
        --use_cms
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Try to import transformers (required for Qwen)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    raise ImportError(
        "transformers is required for Qwen training. Install with: uv add transformers"
    ) from e

from nanogpt_titans.qwen_titans import (
    TitansQwenConfig,
    TitansStateManager,
    freeze_base_model,
    patch_qwen_with_titans,
    save_titans_state,
    get_titans_layers,
    get_gate_statistics,
    get_internal_losses,
)
import torch.nn.functional as F


@dataclass
class QwenTitansTrainConfig:
    """Training configuration for HOPE-Titans Qwen."""

    # Model
    model_name: str = "Qwen/Qwen2-0.5B"
    memory_layers: list[int] = field(default_factory=lambda: [12])

    # Titans config
    segment_len: int = 512
    num_persist_mem: int = 4
    num_longterm_mem: int = 16
    adaptive_memory: bool = True

    # HOPE features
    use_cms: bool = True  # Continuum Memory System
    num_cms_levels: int = 3
    use_self_mod_proj: bool = True
    use_self_mod_gate: bool = True
    use_warm_start: bool = False  # Start without for simplicity
    use_internal_loss: bool = True
    internal_loss_weight: float = 0.01  # Reduced: raw loss is 100s-1000s

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 1000
    warmup_steps: int = 100
    eval_interval: int = 100
    save_interval: int = 500
    max_length: int = 2048

    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"

    # I/O
    output_dir: str = "out-qwen-titans"
    resume_from: Optional[str] = None

    # System
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))
    dtype: str = field(default_factory=lambda: (
        "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16" if torch.cuda.is_available()
        else "float32"
    ))

    # Logging
    wandb_log: bool = False
    wandb_project: str = "qwen-titans"
    wandb_run_name: str = "run"


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for text in texts:
            # Tokenize
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            if encodings["input_ids"].size(1) >= 100:  # Skip very short texts
                self.examples.append(encodings["input_ids"].squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def collate_fn(batch, pad_token_id: int):
    """Collate function with padding."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad to max length in batch
    max_len = max(x.size(0) for x in input_ids)

    padded_input_ids = []
    padded_labels = []

    for inp, lab in zip(input_ids, labels):
        pad_len = max_len - inp.size(0)
        if pad_len > 0:
            inp = torch.cat([inp, torch.full((pad_len,), pad_token_id, dtype=inp.dtype)])
            lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)])  # -100 = ignore in loss
        padded_input_ids.append(inp)
        padded_labels.append(lab)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
    }


def get_lr(step: int, config: QwenTitansTrainConfig) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps

    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def collect_diagnostics(model, titans_layers) -> dict:
    """
    Collect Phase 3 diagnostic metrics.

    Monitors:
    - gate_mean: Should increase from 0.1 → 0.5
    - gate_bias: Gate bias term
    - cms_weights: CMS level weights (should be distributed)
    """
    diagnostics = {}

    # Gate statistics
    gate_stats = get_gate_statistics(model)
    if gate_stats:
        mean_gates = [s.get("mean_gate", 0) for s in gate_stats.values()]
        gate_biases = [s.get("gate_bias", 0) for s in gate_stats.values()]
        diagnostics["gate_mean"] = sum(mean_gates) / len(mean_gates) if mean_gates else 0
        diagnostics["gate_bias"] = sum(gate_biases) / len(gate_biases) if gate_biases else 0

    # CMS level weights (if using CMS)
    for layer in titans_layers:
        if hasattr(layer.memory, "level_weights"):
            weights = F.softmax(layer.memory.level_weights, dim=0)
            for i in range(len(weights)):
                diagnostics[f"cms_weight_{i}"] = weights[i].item()
            break

    return diagnostics


def train(config: QwenTitansTrainConfig) -> None:
    """Main training loop."""
    print(f"Training Titans-Qwen with config:")
    print(f"  Model: {config.model_name}")
    print(f"  Memory layers: {config.memory_layers}")
    print(f"  Segment length: {config.segment_len}")
    print(f"  Device: {config.device}")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device and dtype
    device = torch.device(config.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(config.dtype, torch.float32)

    # Load model and tokenizer
    print(f"\nLoading {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SDPA is faster than eager attention and works with Titans
    # (Titans uses gated residual, doesn't modify attention masks)
    attn_impl = "sdpa" if torch.cuda.is_available() else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    print(f"Using attention implementation: {attn_impl}")

    # Create Titans config with HOPE features
    titans_config = TitansQwenConfig.from_qwen_config(
        model.config,
        segment_len=config.segment_len,
        num_persist_mem=config.num_persist_mem,
        num_longterm_mem=config.num_longterm_mem,
        adaptive_memory=config.adaptive_memory,
        memory_layers=config.memory_layers,
        # HOPE features
        use_cms=config.use_cms,
        num_cms_levels=config.num_cms_levels,
        use_self_mod_proj=config.use_self_mod_proj,
        use_self_mod_gate=config.use_self_mod_gate,
        use_warm_start=config.use_warm_start,
        use_internal_loss=config.use_internal_loss,
        internal_loss_weight=config.internal_loss_weight,
    )

    # Patch model with Titans memory (HOPE-style gated residual)
    print("\nPatching model with HOPE-Titans memory...")
    model = patch_qwen_with_titans(model, titans_config)

    # Freeze base model (only Titans components trainable)
    print("\nFreezing base model parameters...")
    param_stats = freeze_base_model(model)

    # Get Titans layers for diagnostics
    titans_layers = get_titans_layers(model)

    # Move to device
    model.to(device)

    # Load dataset
    print(f"\nLoading dataset: {config.dataset_name}...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(config.dataset_name, config.dataset_config)
        train_texts = [ex["text"] for ex in dataset["train"] if ex["text"].strip()]
        val_texts = [ex["text"] for ex in dataset["validation"] if ex["text"].strip()]
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Using dummy data for testing...")
        train_texts = ["This is a test sentence. " * 100] * 100
        val_texts = ["This is a validation sentence. " * 100] * 10

    train_dataset = TextDataset(train_texts, tokenizer, config.max_length)
    val_dataset = TextDataset(val_texts, tokenizer, config.max_length)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=0,
    )

    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Create state manager
    state_manager = TitansStateManager(model)

    # Gradient scaler for mixed precision (only works on CUDA)
    scaler = torch.amp.GradScaler(
        enabled=(config.dtype == "float16" and device.type == "cuda")
    )

    # Calculate number of segments per sequence for proper loss scaling
    num_segments = max(1, config.max_length // config.segment_len)

    # Starting step (may be overwritten by checkpoint)
    start_step = 0

    # Load checkpoint if resuming
    if config.resume_from:
        print(f"\nResuming from checkpoint: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if "scaler_state_dict" in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"Resumed from step {start_step}, best_val_loss: {best_val_loss:.4f}")

    # Training loop
    print(f"\nStarting training for {config.max_steps} steps...")
    model.train()

    step = start_step
    if start_step == 0:
        best_val_loss = float("inf")
    train_iter = iter(train_loader)

    t0 = time.time()
    log_history = []
    total_internal_loss = 0.0

    while step < config.max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        B, T = input_ids.shape

        # Initialize memory states for this batch
        state_manager.reset()
        state_manager.init_states(B, device)

        # Gradient accumulation
        total_loss = 0.0
        num_micro_segments = 0  # Track actual segments for accurate loss averaging

        for micro_step in range(config.gradient_accumulation_steps):
            # Count segments in this sequence
            seq_segments = (T + config.segment_len - 1) // config.segment_len

            # Process in segments
            for start in range(0, T, config.segment_len):
                end = min(start + config.segment_len, T)
                seg_input = input_ids[:, start:end]
                seg_labels = labels[:, start:end]

                # Sync memory states
                state_manager.sync_to_layers()

                # Forward pass
                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    outputs = model(
                        seg_input,
                        labels=seg_labels,
                        use_cache=False,
                    )
                    loss = outputs.loss

                    # Add internal loss from HOPE memory
                    if config.use_internal_loss:
                        internal_losses = get_internal_losses(model)
                        internal_loss = sum(internal_losses) if internal_losses else torch.tensor(0.0, device=device)
                        if internal_losses:
                            # Check for NaN in internal loss
                            if torch.isnan(internal_loss):
                                print(f"WARNING: NaN detected in internal_loss at step {step}")
                                internal_loss = torch.tensor(0.0, device=device)
                            else:
                                loss = loss + config.internal_loss_weight * internal_loss
                                total_internal_loss += internal_loss.item()

                    # Check for NaN in main loss - skip this segment if NaN
                    if torch.isnan(loss):
                        print(f"WARNING: NaN detected in loss at step {step}, segment {start}:{end}, skipping")
                        state_manager.sync_from_layers()
                        continue  # Skip backward for this segment

                    # Scale loss ONCE for gradient accumulation across both micro-steps AND segments
                    # Total divisor: accumulation steps × segments per sequence
                    total_divisor = config.gradient_accumulation_steps * seq_segments
                    scaled_loss = loss / total_divisor

                # Backward
                scaler.scale(scaled_loss).backward()
                total_loss += outputs.loss.item()  # Track unscaled loss for logging
                num_micro_segments += 1

                # Sync memory states from layers
                state_manager.sync_from_layers()

            # Get next batch for accumulation
            if micro_step < config.gradient_accumulation_steps - 1:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                B, T = input_ids.shape  # CRITICAL: Update B and T for new batch before state init

                # Reset memory for new sequences
                state_manager.reset()
                state_manager.init_states(B, device)

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Update learning rate
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Logging - show normalized loss (average per segment)
        avg_loss = total_loss / max(1, num_micro_segments)
        avg_internal = total_internal_loss / max(1, num_micro_segments)
        if step % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Collect diagnostics
            diagnostics = collect_diagnostics(model, titans_layers)

            # Log entry
            log_entry = {
                "step": step,
                "loss": avg_loss,
                "internal_loss": avg_internal,
                "lr": lr,
                "time_ms": dt * 1000,
                **diagnostics,
            }
            log_history.append(log_entry)

            # Print with diagnostics
            gate_str = f", gate={diagnostics.get('gate_mean', 0):.4f}" if 'gate_mean' in diagnostics else ""
            print(f"step {step}: loss {avg_loss:.4f}, int_loss {avg_internal:.4f}{gate_str}, lr {lr:.2e}, time {dt*1000:.1f}ms")

            # Log CMS weights every 50 steps
            if config.use_cms and step % 50 == 0 and "cms_weight_0" in diagnostics:
                cms_str = ", ".join(f"{diagnostics.get(f'cms_weight_{i}', 0):.3f}" for i in range(config.num_cms_levels))
                print(f"         CMS weights: [{cms_str}]")

            total_internal_loss = 0.0

        # Evaluation
        if step > 0 and step % config.eval_interval == 0:
            model.eval()
            val_loss = evaluate(model, val_dataset, tokenizer, state_manager, config, device, dtype)
            print(f"step {step}: val_loss {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, step, val_loss, config,
                    output_dir / "best.pt", scaler, best_val_loss
                )

            model.train()

        # Save checkpoint
        if step > 0 and step % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, step, avg_loss, config,
                output_dir / f"ckpt_{step}.pt", scaler, best_val_loss
            )
            save_titans_state(model, str(output_dir / f"titans_state_{step}.pt"))

        step += 1

    # Final save
    save_checkpoint(
        model, optimizer, step, avg_loss, config,
        output_dir / "final.pt", scaler, best_val_loss
    )
    save_titans_state(model, str(output_dir / "titans_state_final.pt"))

    # Save training log
    import json
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"Saved training log to {log_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best val loss: {best_val_loss:.4f}")

    if log_history:
        first = log_history[0]
        last = log_history[-1]
        print(f"\nPhase 3 Summary:")
        print(f"  Loss: {first['loss']:.4f} → {last['loss']:.4f}")
        print(f"  Internal Loss: {first.get('internal_loss', 0):.4f} → {last.get('internal_loss', 0):.4f}")
        print(f"  Gate Mean: {first.get('gate_mean', 0):.4f} → {last.get('gate_mean', 0):.4f}")


@torch.no_grad()
def evaluate(
    model,
    val_dataset: TextDataset,
    tokenizer,
    state_manager: TitansStateManager,
    config: QwenTitansTrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Evaluate model on validation set."""
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        B, T = input_ids.shape

        # Initialize memory
        state_manager.reset()
        state_manager.init_states(B, device)

        # Process in segments, tracking loss per token
        batch_loss = 0.0
        batch_tokens = 0
        for start in range(0, T, config.segment_len):
            end = min(start + config.segment_len, T)
            seg_input = input_ids[:, start:end]
            seg_labels = labels[:, start:end]
            seg_len = end - start

            state_manager.sync_to_layers()

            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                outputs = model(seg_input, labels=seg_labels, use_cache=False)
                seg_loss = outputs.loss.item()
                
                # Check for NaN
                if torch.isnan(torch.tensor(seg_loss)):
                    print(f"WARNING: NaN detected in validation loss")
                    continue
                
                # Weight by actual segment length (last segment may be shorter)
                batch_loss += seg_loss * seg_len * B
                batch_tokens += seg_len * B

            state_manager.sync_from_layers()

        # Normalize batch loss by tokens
        if batch_tokens > 0:
            total_loss += batch_loss
            total_tokens += batch_tokens

    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def save_checkpoint(
    model,
    optimizer,
    step: int,
    loss: float,
    config: QwenTitansTrainConfig,
    path: Path,
    scaler: Optional[torch.amp.GradScaler] = None,
    best_val_loss: float = float("inf"),
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "best_val_loss": best_val_loss,
        "config": config,
        "titans_layer_indices": model._titans_layer_indices,
    }
    if scaler is not None and scaler.is_enabled():
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def main() -> None:
    """Entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Titans-enhanced Qwen model")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B")
    parser.add_argument("--memory_layers", type=str, default="14",
                        help="Comma-separated layer indices")

    # Titans arguments
    parser.add_argument("--segment_len", type=int, default=512)
    parser.add_argument("--num_persist_mem", type=int, default=4)
    parser.add_argument("--num_longterm_mem", type=int, default=16)
    parser.add_argument("--adaptive_memory", action="store_true", default=True)
    parser.add_argument("--no_adaptive_memory", action="store_false", dest="adaptive_memory")

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=2048)

    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")

    # HOPE feature arguments
    parser.add_argument("--use_cms", action="store_true", default=True,
                        help="Use Continuum Memory System")
    parser.add_argument("--no_cms", action="store_true",
                        help="Disable CMS")
    parser.add_argument("--num_cms_levels", type=int, default=3)
    parser.add_argument("--use_self_mod_proj", action="store_true", default=True)
    parser.add_argument("--no_self_mod_proj", action="store_true")
    parser.add_argument("--use_self_mod_gate", action="store_true", default=True)
    parser.add_argument("--no_self_mod_gate", action="store_true")
    parser.add_argument("--use_warm_start", action="store_true", default=False)
    parser.add_argument("--use_internal_loss", action="store_true", default=True)
    parser.add_argument("--no_internal_loss", action="store_true")
    parser.add_argument("--internal_loss_weight", type=float, default=0.01)

    # I/O arguments
    parser.add_argument("--output_dir", type=str, default="out-qwen-titans")
    parser.add_argument("--resume_from", type=str, default=None)

    # System arguments
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)

    args = parser.parse_args()

    # Parse memory_layers
    memory_layers = [int(x.strip()) for x in args.memory_layers.split(",")]

    # Create config
    config_kwargs = {
        "model_name": args.model_name,
        "memory_layers": memory_layers,
        "segment_len": args.segment_len,
        "num_persist_mem": args.num_persist_mem,
        "num_longterm_mem": args.num_longterm_mem,
        "adaptive_memory": args.adaptive_memory,
        # HOPE features
        "use_cms": args.use_cms and not args.no_cms,
        "num_cms_levels": args.num_cms_levels,
        "use_self_mod_proj": args.use_self_mod_proj and not args.no_self_mod_proj,
        "use_self_mod_gate": args.use_self_mod_gate and not args.no_self_mod_gate,
        "use_warm_start": args.use_warm_start,
        "use_internal_loss": args.use_internal_loss and not args.no_internal_loss,
        "internal_loss_weight": args.internal_loss_weight,
        # Training
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "max_length": args.max_length,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "output_dir": args.output_dir,
        "resume_from": args.resume_from,
    }

    if args.device:
        config_kwargs["device"] = args.device
    if args.dtype:
        config_kwargs["dtype"] = args.dtype

    config = QwenTitansTrainConfig(**config_kwargs)
    train(config)


if __name__ == "__main__":
    main()

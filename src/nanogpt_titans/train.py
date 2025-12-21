"""
Training script for TitansGPT.
Based on nanoGPT's train.py with modifications for memory state handling.

Usage:
    uv run python -m nanogpt_titans.train [options]

Example:
    uv run python -m nanogpt_titans.train --dataset=shakespeare --batch_size=8
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from nanogpt_titans.model import TitansConfig, TitansGPT


@dataclass
class TrainConfig:
    """Training configuration."""

    # I/O
    out_dir: str = "out-titans"
    eval_interval: int = 250
    log_interval: int = 10
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = False
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # wandb logging
    wandb_log: bool = False
    wandb_project: str = "titans"
    wandb_run_name: str = "run"

    # data
    dataset: str = "shakespeare"
    gradient_accumulation_steps: int = 4
    batch_size: int = 4
    block_size: int = 256
    use_packing: bool = False  # Enable padding-free packing

    # model
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    segment_len: int = 128
    num_persist_mem: int = 4
    num_longterm_mem: int = 16
    memory_lr: float = 0.01
    memory_momentum: float = 0.9
    memory_decay: float = 0.001
    dropout: float = 0.0
    bias: bool = False

    # optimizer
    learning_rate: float = 6e-4
    max_iters: int = 5000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # learning rate schedule
    decay_lr: bool = True
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 6e-5

    # system
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype: str = (
        "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    )
    compile: bool = False  # PyTorch 2.0 compile

    # derived fields (computed in __post_init__)
    tokens_per_iter: int = field(init=False)

    def __post_init__(self) -> None:
        self.tokens_per_iter = self.gradient_accumulation_steps * self.batch_size * self.block_size


def get_batch(
    split: str,
    config: TrainConfig,
    data_dir: Path,
    device: torch.device,
    device_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data from the dataset."""
    data_file = data_dir / f"{split}.bin"
    data = np.memmap(data_file, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + config.block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + config.block_size].astype(np.int64)) for i in ix]
    )

    if device_type == "cuda":
        # Pin memory for async transfer
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


def estimate_flops_per_iter(config: TrainConfig, num_params: int) -> float:
    """
    Estimate FLOPS per training iteration for Titans model.

    Standard transformer: ~6 * N * tokens (2N forward, 4N backward)
    Titans adds memory overhead from:
    - Memory retrieval (MLP forward): ~2 * hidden * embd per token
    - Gradient computation (einsum): ~4 * hidden * embd per token
    - Momentum updates: negligible

    Args:
        config: Training configuration
        num_params: Number of model parameters

    Returns:
        Estimated FLOPS per iteration
    """
    tokens = config.tokens_per_iter

    # Standard transformer FLOPS (forward + backward)
    transformer_flops = 6 * num_params * tokens

    # Titans memory overhead (only on memory layer, per segment)
    num_segments = config.block_size // config.segment_len
    hidden_dim = config.n_embd * 2  # memory_expansion default
    memory_flops_per_token = 6 * hidden_dim * config.n_embd  # MLP forward + gradient

    # Memory overhead: applied once per segment per batch
    # tokens_per_segment = segment_len, so total memory ops = tokens / segment_len * segment_len = tokens
    memory_flops = memory_flops_per_token * tokens

    return transformer_flops + memory_flops


def get_peak_tflops(device_type: str, dtype: str) -> float:
    """Get theoretical peak TFLOPS for common GPUs."""
    # BF16/FP16 tensor core peaks (approximate)
    gpu_peaks = {
        "L4": 242,      # NVIDIA L4
        "T4": 65,       # NVIDIA T4
        "A100": 312,    # NVIDIA A100 (BF16)
        "H100": 990,    # NVIDIA H100 (BF16)
        "RTX4090": 330, # RTX 4090
    }

    if device_type != "cuda":
        return 100.0  # Placeholder for non-CUDA

    # Try to detect GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        for key, peak in gpu_peaks.items():
            if key in gpu_name:
                return peak

    # Default to L4 if unknown
    return 242.0


def get_lr(it: int, config: TrainConfig) -> float:
    """Learning rate scheduler with warmup and cosine decay."""
    # Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters

    # After decay_iters, return min_lr
    if it > config.lr_decay_iters:
        return config.min_lr

    # Cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: TitansGPT,
    config: TrainConfig,
    data_dir: Path,
    device: torch.device,
    device_type: str,
    ctx: torch.amp.autocast | nullcontext,
) -> dict[str, float]:
    """Estimate loss on train and val splits."""
    out: dict[str, float] = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(split, config, data_dir, device, device_type)
            with ctx:
                _logits, loss, _states = model(x, targets=y)
            if loss is not None:
                losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


def train(config: TrainConfig) -> None:
    """Main training loop."""
    # Setup
    Path(config.out_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(config.device)
    device_type = "cuda" if "cuda" in config.device else "cpu"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    ptdtype = dtype_map.get(config.dtype, torch.float16)
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # Data
    data_dir = Path("data") / config.dataset
    if not data_dir.exists():
        msg = f"Dataset directory {data_dir} not found. Run data preparation script first."
        raise FileNotFoundError(msg)

    # Setup packed data loaders if enabled
    packed_train_loader = None
    if config.use_packing:
        from nanogpt_titans.packed_data import PackedDataLoader

        print("Using padding-free packing for training")
        packed_train_loader = PackedDataLoader(
            data_path=data_dir / "train.bin",
            block_size=config.block_size,
            batch_size=config.batch_size,
            device=device,
            shuffle=True,
        )
        _packed_val_loader = PackedDataLoader(
            data_path=data_dir / "val.bin",
            block_size=config.block_size,
            batch_size=config.batch_size,
            device=device,
            shuffle=False,
        )

    # Model
    model_config = TitansConfig(
        block_size=config.block_size,
        vocab_size=50304,  # GPT-2 vocab
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias,
        segment_len=config.segment_len,
        num_persist_mem=config.num_persist_mem,
        num_longterm_mem=config.num_longterm_mem,
        memory_lr=config.memory_lr,
        memory_momentum=config.memory_momentum,
        memory_decay=config.memory_decay,
    )

    iter_num = 0
    best_val_loss = 1e9

    if config.init_from == "scratch":
        print("Initializing model from scratch")
        model = TitansGPT(model_config)
    elif config.init_from == "resume":
        print(f"Resuming from {config.out_dir}")
        ckpt_path = Path(config.out_dir) / "ckpt.pt"
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = TitansGPT(model_config)
        state_dict = checkpoint["model"]
        # Fix state dict if compiled
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    else:
        msg = f"Unknown init_from: {config.init_from}"
        raise ValueError(msg)

    model.to(device)

    # Optimizer
    scaler = torch.amp.GradScaler(enabled=(config.dtype == "float16"))
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type,
    )

    if config.init_from == "resume" and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Compile
    if config.compile:
        print("Compiling model...")
        model = torch.compile(model)  # type: ignore[assignment]

    # Logging
    if config.wandb_log:
        import wandb

        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config))

    # Training loop
    if config.use_packing:
        # Get initial packed batch
        packed_batch = packed_train_loader.get_batch()
        x, y = packed_batch.input_ids, packed_batch.targets
        position_ids, packed_mask = packed_batch.position_ids, packed_batch.attention_mask
    else:
        x, y = get_batch("train", config, data_dir, device, device_type)
        position_ids, packed_mask = None, None

    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    # Pre-compute FLOPS estimation for MFU calculation
    num_params = model.get_num_params() if hasattr(model, "get_num_params") else sum(p.numel() for p in model.parameters())
    flops_per_iter = estimate_flops_per_iter(config, num_params)
    peak_tflops = get_peak_tflops(device_type, config.dtype)

    print(f"Starting training for {config.max_iters} iterations")
    print(f"Tokens per iteration: {config.tokens_per_iter:,}")
    print(f"Estimated TFLOPS/iter: {flops_per_iter / 1e12:.2f}, Peak GPU: {peak_tflops:.0f} TFLOPS")
    if config.use_packing:
        print("Padding-free packing enabled - maximizing GPU utilization")

    while iter_num < config.max_iters:
        # Learning rate schedule
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluation
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, config, data_dir, device, device_type, ctx)  # type: ignore[arg-type]
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            if config.wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": losses["train"],
                        "val/loss": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    }
                )

            if losses["val"] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": model.state_dict(),  # type: ignore[union-attr]
                        "optimizer": optimizer.state_dict(),
                        "model_config": model_config,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"Saving checkpoint to {config.out_dir}")
                    torch.save(checkpoint, Path(config.out_dir) / "ckpt.pt")

        if config.eval_only:
            break

        # Forward + backward with gradient accumulation
        # Initialize memory states once per optimizer step (reuse across micro-steps)
        memory_states = model.init_memory_states(config.batch_size, device)  # type: ignore[union-attr]

        for _micro_step in range(config.gradient_accumulation_steps):
            with ctx:
                # Pass memory states and get updated states
                # Use packed data if enabled (includes position_ids and attention mask)
                _logits, loss, memory_states = model(
                    x,
                    targets=y,
                    memory_states=memory_states,
                    position_ids=position_ids,
                    packed_mask=packed_mask,
                )
                loss = loss / config.gradient_accumulation_steps

            # Backward before fetching next batch to overlap GPU compute with CPU data loading
            scaler.scale(loss).backward()

            # Get next batch (overlaps with async GPU ops from backward)
            if config.use_packing:
                packed_batch = packed_train_loader.get_batch()
                x, y = packed_batch.input_ids, packed_batch.targets
                position_ids = packed_batch.position_ids
                packed_mask = packed_batch.attention_mask
            else:
                x, y = get_batch("train", config, data_dir, device, device_type)
                position_ids, packed_mask = None, None

            # Reset memory in-place for new batch (avoids reallocation)
            model.reset_memory_states(memory_states)  # type: ignore[union-attr]

        # Gradient clipping
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)  # type: ignore[union-attr]

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % config.log_interval == 0:
            lossf = loss.item() * config.gradient_accumulation_steps
            if local_iter_num >= 5:
                # MFU = achieved TFLOPS / peak TFLOPS
                achieved_tflops = flops_per_iter / dt / 1e12
                mfu = achieved_tflops / peak_tflops
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            utilization = ""
            if config.use_packing:
                util_pct = packed_batch.total_tokens / (config.batch_size * config.block_size) * 100
                utilization = f", util {util_pct:.1f}%"
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
                f"mfu {running_mfu * 100:.2f}%{utilization}"
            )

        iter_num += 1
        local_iter_num += 1

    print("Training complete!")


def main() -> None:
    """Entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Train TitansGPT")

    # Add arguments for all config fields
    parser.add_argument("--out_dir", type=str, default="out-titans")
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--always_save_checkpoint", action="store_true")
    parser.add_argument("--init_from", type=str, default="scratch", choices=["scratch", "resume"])
    parser.add_argument("--wandb_log", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="titans")
    parser.add_argument("--wandb_run_name", type=str, default="run")
    parser.add_argument("--dataset", type=str, default="shakespeare")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--use_packing", action="store_true", help="Enable padding-free packing")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--segment_len", type=int, default=64)
    parser.add_argument("--num_persist_mem", type=int, default=4)
    parser.add_argument("--num_longterm_mem", type=int, default=16)
    parser.add_argument("--memory_lr", type=float, default=0.01)
    parser.add_argument("--memory_momentum", type=float, default=0.9)
    parser.add_argument("--memory_decay", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--decay_lr", action="store_true", default=True)
    parser.add_argument("--no_decay_lr", action="store_false", dest="decay_lr")
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--lr_decay_iters", type=int, default=5000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16",
    )
    parser.add_argument("--compile", action="store_true")

    args = parser.parse_args()
    config = TrainConfig(**vars(args))

    train(config)


if __name__ == "__main__":
    main()

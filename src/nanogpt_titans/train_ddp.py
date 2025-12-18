"""
DDP Training script for TitansGPT - supports multi-GPU training.

For Kaggle with 2x T4 GPUs:
    torchrun --standalone --nproc_per_node=2 -m nanogpt_titans.train_ddp

For single GPU:
    python -m nanogpt_titans.train_ddp
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nanogpt_titans.model import TitansConfig, TitansGPT


@dataclass
class TrainConfig:
    """Training configuration."""

    # I/O
    out_dir: str = "out-titans"
    eval_interval: int = 500
    log_interval: int = 10
    eval_iters: int = 100
    always_save_checkpoint: bool = False
    init_from: str = "scratch"

    # wandb
    wandb_log: bool = False
    wandb_project: str = "titans"
    wandb_run_name: str = "run"

    # data
    dataset: str = "wikitext103"
    gradient_accumulation_steps: int = 8
    batch_size: int = 8  # per GPU
    block_size: int = 1024

    # model
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    segment_len: int = 64
    num_persist_mem: int = 4
    num_longterm_mem: int = 16
    memory_lr: float = 0.01
    memory_momentum: float = 0.9
    memory_decay: float = 0.001
    dropout: float = 0.0
    bias: bool = False

    # optimizer
    learning_rate: float = 6e-4
    max_iters: int = 10000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # lr schedule
    decay_lr: bool = True
    warmup_iters: int = 200
    lr_decay_iters: int = 10000
    min_lr: float = 6e-5

    # system
    device: str = "cuda"
    dtype: str = "float16"  # float16 for T4 (no bfloat16 support)
    compile: bool = False

    # DDP
    backend: str = "nccl"

    # derived
    tokens_per_iter: int = field(init=False)

    def __post_init__(self) -> None:
        self.tokens_per_iter = self.gradient_accumulation_steps * self.batch_size * self.block_size


# DDP setup
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    dist.init_process_group(backend="nccl")  # type: ignore[call-arg]
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    seed_offset = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(
    split: str,
    config: TrainConfig,
    data_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data."""
    data_file = data_dir / f"{split}.bin"
    data = np.memmap(data_file, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i + config.block_size].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + config.block_size].astype(np.int64))
        for i in ix
    ])

    x = x.pin_memory().to(device, non_blocking=True)
    y = y.pin_memory().to(device, non_blocking=True)

    return x, y


def get_lr(it: int, config: TrainConfig) -> float:
    """Learning rate with warmup and cosine decay."""
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: TitansGPT | DDP[TitansGPT],
    config: TrainConfig,
    data_dir: Path,
    ctx: Any,
) -> dict[str, float]:
    """Estimate loss on train and val splits."""
    out: dict[str, float] = {}
    raw_model: TitansGPT = model.module if isinstance(model, DDP) else model  # type: ignore[assignment]
    raw_model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(split, config, data_dir)
            with ctx:
                _, loss, _ = raw_model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    raw_model.train()
    return out


def train(config: TrainConfig) -> None:
    """Main training loop with DDP support."""
    if master_process:
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda"
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    ptdtype = dtype_map.get(config.dtype, torch.float16)
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Data
    data_dir = Path("data") / config.dataset
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}")

    # Model
    model_config = TitansConfig(
        block_size=config.block_size,
        vocab_size=50304,
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
        if master_process:
            print("Initializing model from scratch")
        model = TitansGPT(model_config)
    elif config.init_from == "resume":
        if master_process:
            print(f"Resuming from {config.out_dir}")
        ckpt_path = Path(config.out_dir) / "ckpt.pt"
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = TitansGPT(model_config)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    else:
        raise ValueError(f"Unknown init_from: {config.init_from}")

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
    compiled_model: TitansGPT | Any = model
    if config.compile:
        if master_process:
            print("Compiling model...")
        compiled_model = torch.compile(model)  # type: ignore[assignment]

    # DDP wrap
    wrapped_model: TitansGPT | DDP[TitansGPT] | Any = compiled_model
    if ddp:
        wrapped_model = DDP(compiled_model, device_ids=[ddp_local_rank])  # type: ignore[arg-type]

    raw_model: TitansGPT = model  # Keep reference to unwrapped model

    # Logging
    if config.wandb_log and master_process:
        import wandb
        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config))

    # Training
    x, y = get_batch("train", config, data_dir)
    t0 = time.time()
    local_iter_num = 0

    if master_process:
        print(f"Starting training for {config.max_iters} iterations")
        print(f"DDP: {ddp}, World size: {ddp_world_size}")
        print(f"Tokens per iter: {config.tokens_per_iter * ddp_world_size:,} (across all GPUs)")

    while iter_num < config.max_iters:
        # LR schedule
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Eval
        if iter_num % config.eval_interval == 0 and master_process:
            losses = estimate_loss(wrapped_model, config, data_dir, ctx)  # type: ignore[arg-type]
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if config.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                })

            if losses["val"] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": model_config,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"Saving checkpoint to {config.out_dir}")
                    torch.save(checkpoint, Path(config.out_dir) / "ckpt.pt")

        # Forward + backward
        memory_states = raw_model.init_memory_states(config.batch_size, device)  # type: ignore[arg-type]

        for micro_step in range(config.gradient_accumulation_steps):
            if ddp:
                wrapped_model.require_backward_grad_sync = (micro_step == config.gradient_accumulation_steps - 1)  # type: ignore[union-attr]

            with ctx:
                _, loss, memory_states = raw_model(x, targets=y, memory_states=memory_states)
                loss = loss / config.gradient_accumulation_steps

            x, y = get_batch("train", config, data_dir)
            memory_states = raw_model.init_memory_states(config.batch_size, device)  # type: ignore[arg-type]

            scaler.scale(loss).backward()

        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), config.grad_clip)  # type: ignore[union-attr]

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % config.log_interval == 0 and master_process:
            lossf = loss.item() * config.gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, lr {lr:.2e}")

        iter_num += 1
        local_iter_num += 1

    if ddp:
        dist.destroy_process_group()  # type: ignore[attr-defined]

    if master_process:
        print("Training complete!")


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train TitansGPT with DDP")

    parser.add_argument("--out_dir", type=str, default="out-titans")
    parser.add_argument("--dataset", type=str, default="wikitext103")
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_iters", type=int, default=100)
    parser.add_argument("--init_from", type=str, default="scratch")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--segment_len", type=int, default=64)
    parser.add_argument("--num_persist_mem", type=int, default=4)
    parser.add_argument("--num_longterm_mem", type=int, default=16)
    parser.add_argument("--memory_lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--lr_decay_iters", type=int, default=10000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--wandb_log", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="titans")
    parser.add_argument("--wandb_run_name", type=str, default="titans-ddp")

    args = parser.parse_args()

    config = TrainConfig(
        out_dir=args.out_dir,
        dataset=args.dataset,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        eval_iters=args.eval_iters,
        init_from=args.init_from,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        segment_len=args.segment_len,
        num_persist_mem=args.num_persist_mem,
        num_longterm_mem=args.num_longterm_mem,
        memory_lr=args.memory_lr,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        max_iters=args.max_iters,
        weight_decay=args.weight_decay,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        min_lr=args.min_lr,
        dtype=args.dtype,
        compile=args.compile,
        wandb_log=args.wandb_log,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    train(config)


if __name__ == "__main__":
    main()

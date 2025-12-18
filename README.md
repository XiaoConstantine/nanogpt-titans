# nanoGPT-Titans

Implementation of **Titans: Learning to Memorize at Test Time** ([arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) built on top of [nanoGPT](https://github.com/karpathy/nanoGPT).

Titans introduces a neural long-term memory module that **learns at test time**, enabling effective context windows of 2M+ tokens while maintaining accuracy advantages over standard Transformers.

## Key Features

- **Memory as Context (MAC)** architecture - combines short-term attention with long-term neural memory
- **Test-time learning** - memory updates based on "surprise" (gradient magnitude)
- **Efficient** - linear scaling with context length instead of quadratic
- Clean, hackable implementation in ~600 lines of PyTorch

## Installation

Requires Python 3.13+.

```bash
# Clone the repo
git clone https://github.com/XiaoConstantine/nanogpt-titans.git
cd nanogpt-titans

# Install with uv
uv sync

# Or install dev dependencies
uv sync --all-extras
```

## Quick Start

### 1. Prepare Data

```bash
# Shakespeare (tiny, for quick experiments)
uv run python -m nanogpt_titans.prepare_data shakespeare

# OpenWebText (large, for serious training)
uv run python -m nanogpt_titans.prepare_data openwebtext
```

### 2. Train

```bash
# Train on Shakespeare
uv run python -m nanogpt_titans.train --dataset=shakespeare --max_iters=5000

# With custom config
uv run python -m nanogpt_titans.train \
    --dataset=shakespeare \
    --n_layer=6 \
    --n_head=6 \
    --n_embd=384 \
    --segment_len=64 \
    --num_longterm_mem=16 \
    --batch_size=8 \
    --max_iters=5000
```

### 3. Sample

```bash
uv run python -m nanogpt_titans.sample \
    --checkpoint=out-titans/ckpt.pt \
    --prompt="To be or not to be" \
    --max_new_tokens=200
```

### 4. Evaluate Needle-in-Haystack

```bash
# Prepare evaluation data
uv run python -m nanogpt_titans.prepare_data needle

# Run evaluation
uv run python -m nanogpt_titans.eval_needle --checkpoint=out-titans/ckpt.pt
```

## Architecture

```
Token Embeddings → Segment → [MAC Block × N] → LayerNorm → LM Head
                      │
                      ▼
              ┌────────────────────────────┐
              │ Query Neural Memory        │
              │ Concat [memory, persistent,│
              │         current_segment]   │
              │ Full Attention             │
              │ + Residual                 │
              │ MLP + Residual             │
              │ Update Memory (surprise)   │
              └────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| `NeuralMemory` | Long-term memory that updates at test time via surprise-based learning |
| `TitansBlock` | MAC block combining memory retrieval + attention + MLP |
| `TitansGPT` | Full model with segmented processing |

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segment_len` | 64 | Tokens per segment |
| `num_persist_mem` | 4 | Persistent (task) memory tokens |
| `num_longterm_mem` | 16 | Long-term memory tokens |
| `memory_lr` | 0.01 | Learning rate for test-time updates |
| `memory_momentum` | 0.9 | Momentum for surprise smoothing |
| `memory_decay` | 0.001 | Forgetting factor |

## Usage as Library

```python
from nanogpt_titans import TitansConfig, TitansGPT

# Configure model
config = TitansConfig(
    block_size=2048,
    vocab_size=50304,
    n_layer=6,
    n_head=6,
    n_embd=384,
    segment_len=64,
    num_longterm_mem=16,
)

# Create model
model = TitansGPT(config)

# Forward pass
import torch
x = torch.randint(0, 50304, (1, 512))
logits, loss, memory_states = model(x, targets=x)

# Generation
prompt = torch.randint(0, 50304, (1, 10))
generated = model.generate(prompt, max_new_tokens=100)
```

## Development

```bash
# Run linting
uv run ruff check src/

# Run formatting
uv run ruff format src/

# Run type checking
uv run ty check src/

# Run tests
uv run pytest
```

## References

- **Paper**: [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) (Behrouz, Zhong, Mirrokni, 2024)
- **nanoGPT**: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- **titans-pytorch**: [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch)

## License

MIT

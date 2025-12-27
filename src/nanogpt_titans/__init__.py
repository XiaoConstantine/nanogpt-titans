"""
nanoGPT-Titans: Learning to Memorize at Test Time

Implementation of the Titans architecture from https://arxiv.org/abs/2501.00663
built on top of nanoGPT.
"""

from nanogpt_titans.model import (
    MLP,
    CausalSelfAttention,
    LayerNorm,
    MemoryState,
    NeuralMemory,
    TitansBlock,
    TitansConfig,
    TitansGPT,
)

# Qwen integration (HOPE-Titans)
from nanogpt_titans import qwen_titans

__version__ = "0.1.0"
__all__ = [
    # Core Titans
    "MLP",
    "CausalSelfAttention",
    "LayerNorm",
    "MemoryState",
    "NeuralMemory",
    "TitansBlock",
    "TitansConfig",
    "TitansGPT",
    # Qwen integration
    "qwen_titans",
]

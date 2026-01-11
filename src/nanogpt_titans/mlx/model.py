"""
NanoGPT-style Model for MLX - Train from Scratch.

A clean, simple GPT implementation in MLX for training from scratch.
Following the nanoGPT philosophy: minimal, readable, educational.

Architecture:
    Token Embedding → Position Embedding → Transformer Blocks → LM Head
                                               ↓
                          [Attention → MLP] × n_layer

Usage:
    config = GPTConfig(vocab_size=50257, n_layer=6, n_head=6, n_embd=384)
    model = GPT(config)
    logits = model(tokens)  # [batch, seq_len, vocab_size]
"""

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


@dataclass
class GPTConfig:
    """
    GPT Model Configuration.

    Presets:
        - gpt2-small:  n_layer=12, n_head=12, n_embd=768  (~124M params)
        - gpt2-medium: n_layer=24, n_head=16, n_embd=1024 (~350M params)
        - nano:        n_layer=6,  n_head=6,  n_embd=384  (~10M params)
        - small:       n_layer=8,  n_head=8,  n_embd=512  (~45M params)
        - medium:      n_layer=12, n_head=12, n_embd=768  (~124M params)
    """
    # Model architecture
    vocab_size: int = 50257       # GPT-2 vocab size (or use tokenizer.vocab_size)
    block_size: int = 1024        # Maximum sequence length
    n_layer: int = 6              # Number of transformer layers
    n_head: int = 6               # Number of attention heads
    n_embd: int = 384             # Embedding dimension
    dropout: float = 0.0          # Dropout rate (0.0 for small datasets)
    bias: bool = True             # Use bias in Linear and LayerNorm

    # TITANS memory integration (optional)
    use_memory: bool = False      # Whether to use TITANS memory
    memory_layer: int = -1        # Which layer gets memory (-1 = middle)

    @classmethod
    def nano(cls, vocab_size: int = 50257) -> "GPTConfig":
        """~10M params - good for learning."""
        return cls(vocab_size=vocab_size, n_layer=6, n_head=6, n_embd=384)

    @classmethod
    def small(cls, vocab_size: int = 50257) -> "GPTConfig":
        """~45M params - capable but trainable on laptop."""
        return cls(vocab_size=vocab_size, n_layer=8, n_head=8, n_embd=512)

    @classmethod
    def medium(cls, vocab_size: int = 50257) -> "GPTConfig":
        """~124M params - GPT-2 small equivalent."""
        return cls(vocab_size=vocab_size, n_layer=12, n_head=12, n_embd=768)

    @classmethod
    def large(cls, vocab_size: int = 50257) -> "GPTConfig":
        """~250M params - the 'more capable than you think' size."""
        return cls(vocab_size=vocab_size, n_layer=16, n_head=16, n_embd=1024)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    The math:
        Q, K, V = x @ W_q, x @ W_k, x @ W_v       # Project to queries, keys, values
        attn = softmax(Q @ K.T / sqrt(d_k))       # Attention weights
        attn = attn * causal_mask                  # Only attend to past
        out = attn @ V                             # Weighted sum of values
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Key, Query, Value projections combined
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array, return_attention: bool = False):
        B, T, C = x.shape

        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention: [B, T, n_head, head_dim]
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        # Attention: [B, n_head, T, T]
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        # Causal mask: can only attend to positions <= current position
        mask = mx.triu(mx.full((T, T), float('-inf')), k=1)
        attn_scores = attn_scores + mask

        attn_weights = mx.softmax(attn_scores, axis=-1)
        attn = self.attn_dropout(attn_weights)

        # Apply attention to values
        y = attn @ v  # [B, n_head, T, head_dim]

        # Reshape back: [B, T, C]
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        if return_attention:
            return y, attn_weights  # [B, n_head, T, T]
        return y


class MLP(nn.Module):
    """
    Feed-forward network with GELU activation.

    The math:
        hidden = GELU(x @ W1 + b1)
        output = hidden @ W2 + b2
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Standard 4x expansion
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = nn.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block: LayerNorm → Attention → LayerNorm → MLP

    Uses pre-norm (LayerNorm before attention/MLP) like GPT-2.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def __call__(self, x: mx.array, return_attention: bool = False):
        # Residual connections
        if return_attention:
            attn_out, attn_weights = self.attn(self.ln_1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


class GPT(nn.Module):
    """
    GPT Language Model.

    Architecture:
        1. Token embedding: vocab_size → n_embd
        2. Position embedding: block_size → n_embd
        3. Transformer blocks: n_layer × (Attention + MLP)
        4. Final LayerNorm
        5. Output head: n_embd → vocab_size (tied with token embedding)

    The forward pass:
        tokens → embeddings → transformer blocks → logits
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = [Block(config) for _ in range(config.n_layer)]

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)

        # Output head (weight tied to wte in loss computation)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: lm_head.weight = wte.weight
        # In MLX, we'll handle this by using wte.weight directly in forward

        # Initialize weights
        self._init_weights()

        # Report size
        n_params = self.get_num_params()
        print(f"GPT initialized: {n_params/1e6:.2f}M parameters")
        print(f"  Layers: {config.n_layer}, Heads: {config.n_head}, Dim: {config.n_embd}")

    def _init_weights(self) -> None:
        """Initialize weights with small random values."""
        def init_linear(module):
            if isinstance(module, nn.Linear):
                # Small std for stability
                module.weight = mx.random.normal(module.weight.shape) * 0.02
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias = mx.zeros(module.bias.shape)

        # Note: MLX modules don't have apply(), so we initialize inline

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count parameters."""
        n_params = sum(p.size for _, p in tree_flatten(self.parameters()))
        if non_embedding:
            n_params -= self.wpe.weight.size
        return n_params

    def __call__(self, idx: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            idx: Token indices [batch, seq_len]

        Returns:
            logits: Predicted token logits [batch, seq_len, vocab_size]
        """
        B, T = idx.shape

        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Token embeddings
        tok_emb = self.wte(idx)  # [B, T, n_embd]

        # Position embeddings
        pos = mx.arange(0, T)
        pos_emb = self.wpe(pos)  # [T, n_embd]

        # Combine
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Output logits (using tied weights)
        # logits = x @ self.wte.weight.T  # Weight tying
        logits = self.lm_head(x)  # Or use separate head

        return logits

    def get_attention_maps(self, idx: mx.array) -> list:
        """
        Get attention weights from all layers.

        Args:
            idx: Token indices [batch, seq_len]

        Returns:
            List of attention weights, one per layer.
            Each is [batch, n_heads, seq_len, seq_len]
        """
        B, T = idx.shape

        # Token + position embeddings
        tok_emb = self.wte(idx)
        pos = mx.arange(0, T)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        # Collect attention from each layer
        attention_maps = []
        for block in self.blocks:
            x, attn_weights = block(x, return_attention=True)
            attention_maps.append(attn_weights)

        return attention_maps

    def generate(
        self,
        idx: mx.array,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> mx.array:
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting token indices [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens

        Returns:
            Extended token indices [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]

            # Get logits
            logits = self(idx_cond)

            # Take last position
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = mx.topk(logits, min(top_k, logits.shape[-1]))
                logits = mx.where(logits < v[:, [-1]], float('-inf'), logits)

            # Sample
            probs = mx.softmax(logits, axis=-1)
            idx_next = mx.random.categorical(probs)
            idx_next = idx_next.reshape(-1, 1)

            # Append
            idx = mx.concatenate([idx, idx_next], axis=1)

        return idx


class GPTWithMemory(GPT):
    """
    GPT with TITANS Memory integration.

    Adds neural memory at specified layer(s) for test-time learning.
    """

    def __init__(self, config: GPTConfig, titans_layer=None):
        super().__init__(config)

        self.titans_layer = titans_layer
        self.memory_layer_idx = config.memory_layer if config.memory_layer >= 0 else config.n_layer // 2

        if titans_layer:
            print(f"  TITANS memory at layer {self.memory_layer_idx}")

    def __call__(self, idx: mx.array, memory_state=None):
        """Forward with optional memory."""
        B, T = idx.shape

        # Embeddings
        tok_emb = self.wte(idx)
        pos = mx.arange(0, T)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        new_memory_state = memory_state

        # Transformer blocks with optional memory
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Apply TITANS memory at specified layer
            if self.titans_layer and i == self.memory_layer_idx:
                if memory_state is None:
                    memory_state = self.titans_layer.init_memory_state(B)
                x, new_memory_state, _ = self.titans_layer(x, memory_state)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, new_memory_state


def create_model(
    size: str = "nano",
    vocab_size: int = 50257,
    use_memory: bool = False,
    **kwargs,
) -> GPT:
    """
    Create a GPT model by size.

    Args:
        size: One of "nano", "small", "medium", "large"
        vocab_size: Vocabulary size
        use_memory: Whether to add TITANS memory
        **kwargs: Override any config options

    Returns:
        Configured GPT model
    """
    configs = {
        "nano": GPTConfig.nano,
        "small": GPTConfig.small,
        "medium": GPTConfig.medium,
        "large": GPTConfig.large,
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    config = configs[size](vocab_size=vocab_size)

    # Apply overrides
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    if use_memory:
        config.use_memory = True
        return GPTWithMemory(config)

    return GPT(config)


# Quick test
if __name__ == "__main__":
    print("Testing GPT model...")

    # Create nano model
    config = GPTConfig.nano(vocab_size=1000)
    model = GPT(config)

    # Test forward
    tokens = mx.array([[1, 2, 3, 4, 5]])
    logits = model(tokens)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {logits.shape}")

    # Test generation
    generated = model.generate(tokens, max_new_tokens=10, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")

    print("\nGPT model test passed!")

"""
Configuration for Titans-enhanced Qwen models.

Bridges HuggingFace Qwen2Config to TitansConfig parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass
class TitansQwenConfig:
    """
    Configuration for Titans memory integration with Qwen models.

    Bridges Qwen2Config parameters to what NeuralMemory expects,
    while adding Titans-specific configuration.

    Attributes:
        n_embd: Hidden size (from Qwen config)
        block_size: Max position embeddings (from Qwen config)
        n_layer: Number of layers (from Qwen config)
        n_head: Number of attention heads (from Qwen config)
        bias: Whether to use bias in projections
        segment_len: Tokens per segment for memory processing
        memory_depth: Number of layers in memory MLP
        memory_expansion: Hidden dim multiplier for memory MLP
        num_persist_mem: Number of persistent memory tokens
        num_longterm_mem: Number of retrieved memory tokens
        memory_lr: Learning rate for memory updates (when not adaptive)
        memory_momentum: Momentum for memory updates (when not adaptive)
        memory_decay: Decay factor for memory (when not adaptive)
        memory_layers: Which layer indices to enhance with memory
        adaptive_memory: Use learned per-token lr/momentum/decay
        memory_lr_max: Max learning rate for adaptive mode
    """

    # Derived from Qwen2Config
    n_embd: int = 1536  # Qwen2-1.5B hidden_size
    block_size: int = 32768  # Qwen2 max_position_embeddings
    n_layer: int = 28  # Qwen2-1.5B num_hidden_layers
    n_head: int = 12  # Qwen2-1.5B num_attention_heads
    vocab_size: int = 151936  # Qwen2 vocab size
    bias: bool = False  # Qwen uses no bias in most places

    # Titans-specific configuration
    segment_len: int = 512  # Larger than nanoGPT default due to Qwen's context
    memory_depth: int = 2  # MLP layers in neural memory
    memory_expansion: int = 2  # Hidden dim = n_embd * expansion
    num_persist_mem: int = 4  # Persistent memory tokens
    num_longterm_mem: int = 16  # Retrieved memory tokens per segment
    memory_lr: float = 0.01  # Learning rate (used when adaptive=False)
    memory_momentum: float = 0.9  # Momentum (used when adaptive=False)
    memory_decay: float = 0.001  # Decay factor (used when adaptive=False)
    memory_layers: list[int] = field(default_factory=lambda: [14])  # Middle layer
    adaptive_memory: bool = True  # Use learned per-token parameters
    memory_lr_max: float = 0.01  # Max lr for adaptive mode
    dropout: float = 0.0  # Dropout rate (usually 0 for inference)

    # HOPE-inspired configuration (from Nested Learning paper)
    # Uses gated residual integration (no sequence length change)

    # Titans variant: "hope" (recommended), "mag", or "mal"
    # HOPE: gated residual integration (fixed for pre-trained models) - RECOMMENDED
    # MAG: memory modulates attention via multiplicative gating (deprecated)
    # MAL: memory as preprocessing layer (deprecated)
    # Note: "mac" is an alias for "hope" for backwards compatibility
    titans_variant: str = "hope"  # Default to HOPE (the fixed implementation)

    # Continuum Memory System (multi-frequency memory)
    use_cms: bool = True  # Use multi-frequency memory levels
    num_cms_levels: int = 3  # Number of frequency levels
    cms_update_frequencies: list[int] = field(
        default_factory=lambda: [1, 4, 16]
    )  # Update every N segments

    # Self-modifying components (online adaptation via delta rule)
    # NOTE: use_self_mod_gate=False enables the new PositionDependentGate
    use_self_mod_proj: bool = False  # Disabled: standard projection is more stable
    use_self_mod_gate: bool = False  # Disabled: use PositionDependentGate instead
    self_mod_lr: float = 0.001  # Learning rate for delta rule updates
    gate_init_bias: float = 0.0  # sigmoid(0) = 0.5, allows gradient flow

    # Warm start (initialize memory from input instead of random)
    use_warm_start: bool = False  # Disabled: adds complexity, not proven to help
    warm_start_prefix_len: int = 32  # Tokens to use for warm start
    warm_start_layers: int = 2  # Transformer layers in warm start encoder

    # Internal loss (CRITICAL: teaches memory projections to store/retrieve patterns)
    # Must be True for memory to learn during training!
    # The internal loss trains key_proj, value_proj, query_proj to produce meaningful mappings
    use_internal_loss: bool = True  # ENABLED by default - critical for memory learning
    internal_loss_weight: float = 0.1  # Stronger signal for memory training

    @classmethod
    def from_qwen_config(
        cls,
        qwen_config: PretrainedConfig,
        **overrides,
    ) -> TitansQwenConfig:
        """
        Create TitansQwenConfig from a HuggingFace Qwen2Config.

        Args:
            qwen_config: HuggingFace model config (Qwen2Config)
            **overrides: Override any Titans-specific parameters

        Returns:
            TitansQwenConfig with Qwen dimensions and Titans defaults

        Example:
            >>> from transformers import AutoConfig
            >>> qwen_config = AutoConfig.from_pretrained("Qwen/Qwen2-1.5B")
            >>> titans_config = TitansQwenConfig.from_qwen_config(qwen_config)
            >>> print(titans_config.n_embd)  # 1536
        """
        # Extract Qwen dimensions
        base_kwargs = {
            "n_embd": qwen_config.hidden_size,
            "block_size": qwen_config.max_position_embeddings,
            "n_layer": qwen_config.num_hidden_layers,
            "n_head": qwen_config.num_attention_heads,
            "vocab_size": qwen_config.vocab_size,
            "bias": getattr(qwen_config, "attention_bias", False),
        }

        # Default memory_layers to middle layer if not specified
        if "memory_layers" not in overrides:
            middle_layer = qwen_config.num_hidden_layers // 2
            overrides["memory_layers"] = [middle_layer]

        # Merge with overrides
        base_kwargs.update(overrides)

        return cls(**base_kwargs)

    def to_titans_config(self):
        """
        Convert to TitansConfig for use with NeuralMemory.

        Returns:
            TitansConfig compatible with nanogpt_titans.model.NeuralMemory
        """
        from nanogpt_titans.model import TitansConfig

        return TitansConfig(
            block_size=self.block_size,
            vocab_size=self.vocab_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias,
            segment_len=self.segment_len,
            memory_depth=self.memory_depth,
            memory_expansion=self.memory_expansion,
            num_persist_mem=self.num_persist_mem,
            num_longterm_mem=self.num_longterm_mem,
            memory_lr=self.memory_lr,
            memory_momentum=self.memory_momentum,
            memory_decay=self.memory_decay,
            adaptive_memory=self.adaptive_memory,
            memory_lr_max=self.memory_lr_max,
        )

    @property
    def prefix_len(self) -> int:
        """Total prefix length (memory + persistent tokens)."""
        return self.num_longterm_mem + self.num_persist_mem

    def estimate_memory_params(self) -> dict[str, int]:
        """
        Estimate number of trainable parameters for memory module.

        Returns:
            Dict with parameter counts by component
        """
        hidden_dim = self.n_embd * self.memory_expansion

        params = {
            "persist_mem": self.num_persist_mem * self.n_embd,
            "query_proj": self.n_embd * self.n_embd,
            "key_proj": self.n_embd * self.n_embd,
            "value_proj": self.n_embd * self.n_embd,
            "out_proj": self.n_embd * self.n_embd,
            "memory_mlp_layer0": self.n_embd * hidden_dim,
            "memory_mlp_layer1": hidden_dim * self.n_embd,
            "init_query": self.n_embd,
        }

        if self.adaptive_memory:
            params["to_lr"] = self.n_embd
            params["to_momentum"] = self.n_embd
            params["to_decay"] = self.n_embd

        if self.bias:
            params["biases"] = (
                self.n_embd * 4  # projections
                + hidden_dim  # mlp layer 0
                + self.n_embd  # mlp layer 1
            )

        params["total"] = sum(params.values())
        params["total_per_layer"] = params["total"]
        params["total_all_layers"] = params["total"] * len(self.memory_layers)

        return params

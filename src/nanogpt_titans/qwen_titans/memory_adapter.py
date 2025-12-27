"""
Memory adapter for Qwen models.

Wraps the NeuralMemory class from nanogpt_titans to work with Qwen dimensions,
providing a clean interface for the decoder layer integration.

Also includes HOPE-inspired components:
- SelfModifyingLinear: Delta rule weight updates during forward pass
- SelfModifyingGate: Learned gate that opens as memory becomes useful
- ContinuumMemorySystem: Multi-frequency memory MLPs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from nanogpt_titans.model import MemoryState, NeuralMemory
from nanogpt_titans.qwen_titans.config import TitansQwenConfig


# =============================================================================
# Self-Modifying Components (from Nested Learning / HOPE)
# =============================================================================


class SelfModifyingLinear(nn.Module):
    """
    Linear layer with delta rule weight updates during forward pass.

    Implements online adaptation: weights are updated based on input to reduce
    reconstruction error, allowing the projection to adapt to current context.

    Update rule: W -= lr * (W @ x @ x^T) / ||x||^2

    This is a simplified Hebbian/delta rule that moves weights toward
    identity-like behavior on the current input subspace.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        lr: float = 0.001,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        Forward with optional delta rule update.

        Args:
            x: Input tensor [B, T, in_features]
            update: Whether to apply delta rule update

        Returns:
            Output tensor [B, T, out_features]
        """
        # Standard linear forward
        out = F.linear(x, self.weight, self.bias)

        if update and self.training:
            # Delta rule update: W -= lr * (W @ x @ x^T) / ||x||^2
            # Averaged over batch and sequence
            with torch.no_grad():
                # x: [B, T, C] -> [B*T, C]
                x_flat = x.reshape(-1, x.size(-1))
                # Compute outer product averaged over samples
                # x^T @ x / N gives covariance-like term
                norm_sq = (x_flat**2).sum() / x_flat.size(0) + 1e-8
                # Simplified update: move toward identity on input subspace
                delta = self.lr * (self.weight @ x_flat.T @ x_flat) / norm_sq / x_flat.size(0)
                self.weight.data -= delta

        return out


class SelfModifyingGate(nn.Module):
    """
    Learned gate that controls memory contribution.

    Starts near 0 (conservative) and opens as training learns that memory is useful.
    The gate uses delta rule to adapt during forward pass, becoming more confident
    about memory quality over time.

    Output: gate_value * memory_output + (1 - gate_value) * residual
    """

    def __init__(
        self,
        dim: int,
        init_bias: float = -2.0,
        lr: float = 0.001,
    ) -> None:
        """
        Args:
            dim: Hidden dimension
            init_bias: Initial bias (sigmoid(-2) ≈ 0.12 for conservative start)
            lr: Learning rate for delta rule updates
        """
        super().__init__()
        self.lr = lr

        # Project from dim to scalar gate per position
        self.gate_proj = nn.Linear(dim, 1, bias=True)
        # Initialize bias for conservative gate
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, init_bias)

        # Track gate statistics for monitoring
        self.register_buffer("_mean_gate", torch.tensor(0.0))
        self.register_buffer("_gate_count", torch.tensor(0))

    def forward(
        self,
        memory_out: torch.Tensor,
        residual: torch.Tensor,
        update: bool = True,
    ) -> torch.Tensor:
        """
        Apply gated combination of memory and residual.

        Args:
            memory_out: Memory module output [B, T, C]
            residual: Original hidden states [B, T, C]
            update: Whether to apply delta rule adaptation

        Returns:
            Gated combination [B, T, C]
        """
        # Compute gate from residual (what the model "expected")
        gate_logits = self.gate_proj(residual)  # [B, T, 1]
        gate = torch.sigmoid(gate_logits)  # [B, T, 1]

        # Update statistics
        with torch.no_grad():
            self._mean_gate = (
                self._mean_gate * self._gate_count + gate.mean()
            ) / (self._gate_count + 1)
            self._gate_count += 1

        # Gated combination
        output = gate * memory_out + (1 - gate) * residual

        # Delta rule: if memory_out is useful (reduces error), increase gate
        if update and self.training:
            with torch.no_grad():
                # Measure if memory helps (correlation between gate and improvement)
                # This is a simplified heuristic
                error_with_mem = (output - residual).pow(2).mean()
                error_without = (memory_out - residual).pow(2).mean()

                # If memory reduces reconstruction error, nudge gate up
                if error_with_mem < error_without:
                    self.gate_proj.bias.data += self.lr * 0.1

        return output

    @property
    def mean_gate_value(self) -> float:
        """Get average gate value for monitoring."""
        return self._mean_gate.item()


# =============================================================================
# Continuum Memory System (Multi-frequency memory)
# =============================================================================


@dataclass
class ContinuumMemoryState:
    """State for multi-level memory system."""

    level_states: list[MemoryState]  # One state per frequency level
    segment_counts: list[int]  # Segments processed at each level


class ContinuumMemorySystem(nn.Module):
    """
    Multi-frequency memory system from Nested Learning.

    Different memory levels update at different rates:
    - Level 0: Updates every segment (fast, working memory)
    - Level 1: Updates every 4 segments (medium, episodic)
    - Level 2: Updates every 16 segments (slow, semantic)

    This allows capturing patterns at multiple timescales without
    interference between fast and slow dynamics.
    """

    def __init__(
        self,
        config: TitansQwenConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.num_levels = config.num_cms_levels
        self.update_frequencies = config.cms_update_frequencies

        # Create a memory module for each level
        self.memories = nn.ModuleList([
            NeuralMemoryAdapter(config) for _ in range(self.num_levels)
        ])

        # Projection to combine outputs from all levels
        self.combine_proj = nn.Linear(
            config.n_embd * self.num_levels,
            config.n_embd,
            bias=False,
        )

        # Learnable weights for combining levels
        self.level_weights = nn.Parameter(torch.ones(self.num_levels) / self.num_levels)

    def init_state(self, batch_size: int, device: torch.device) -> ContinuumMemoryState:
        """Initialize state for all memory levels."""
        return ContinuumMemoryState(
            level_states=[mem.init_state(batch_size, device) for mem in self.memories],
            segment_counts=[0] * self.num_levels,
        )

    def reset_state(self, state: ContinuumMemoryState) -> None:
        """Reset all memory levels."""
        for mem, level_state in zip(self.memories, state.level_states):
            mem.reset_state(level_state)
        state.segment_counts = [0] * self.num_levels

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: ContinuumMemoryState,
    ) -> torch.Tensor:
        """
        Retrieve from all memory levels and combine.

        Args:
            hidden_states: Current segment [B, T, C]
            state: Multi-level memory state

        Returns:
            Combined memory output [B, num_longterm_mem, C]
        """
        weights = F.softmax(self.level_weights, dim=0)

        # Fused retrieval: stack outputs and apply weights in one operation
        # This reduces Python loop overhead and enables better GPU utilization
        level_outputs = torch.stack([
            mem(hidden_states, level_state)
            for mem, level_state in zip(self.memories, state.level_states)
        ], dim=0)  # [num_levels, B, num_longterm_mem, C]

        # Apply weights: [num_levels, 1, 1, 1] * [num_levels, B, T, C] -> weighted sum
        weights_expanded = weights.view(-1, 1, 1, 1)
        combined = (weights_expanded * level_outputs).sum(dim=0)  # [B, num_longterm_mem, C]

        return combined

    def update(
        self,
        hidden_states: torch.Tensor,
        state: ContinuumMemoryState,
    ) -> ContinuumMemoryState:
        """
        Update memory levels based on their frequencies.

        Args:
            hidden_states: Current segment output [B, T, C]
            state: Current multi-level state

        Returns:
            Updated state
        """
        new_level_states = []
        new_segment_counts = []

        for i, (mem, level_state, freq) in enumerate(
            zip(self.memories, state.level_states, self.update_frequencies)
        ):
            count = state.segment_counts[i] + 1

            # Update only if segment count is multiple of frequency
            if count % freq == 0:
                new_state = mem.update(hidden_states, level_state)
            else:
                new_state = level_state

            new_level_states.append(new_state)
            new_segment_counts.append(count)

        return ContinuumMemoryState(
            level_states=new_level_states,
            segment_counts=new_segment_counts,
        )


# =============================================================================
# Warm Start Encoder
# =============================================================================


class WarmStartEncoder(nn.Module):
    """
    Initialize memory from input tokens instead of random.

    Uses a small transformer to encode prefix tokens into initial memory state,
    giving the memory a "warm start" based on the actual input context.
    """

    def __init__(
        self,
        config: TitansQwenConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.prefix_len = config.warm_start_prefix_len
        self.n_embd = config.n_embd

        # Simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=config.n_embd * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.warm_start_layers,
        )

        # Project to memory initialization
        self.to_memory_init = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Encode prefix tokens to memory initialization.

        Args:
            hidden_states: Input hidden states [B, T, C]

        Returns:
            Memory initialization vector [B, C]
        """
        # Take first prefix_len tokens
        prefix = hidden_states[:, : self.prefix_len]  # [B, prefix_len, C]

        # Encode
        encoded = self.encoder(prefix)  # [B, prefix_len, C]

        # Pool to single vector (mean pooling)
        pooled = encoded.mean(dim=1)  # [B, C]

        # Project to memory init
        mem_init = self.to_memory_init(pooled)  # [B, C]

        return mem_init


# =============================================================================
# Deep Momentum (Learned gradient compression)
# =============================================================================


class DeepMomentumUpdate(nn.Module):
    """
    Learned gradient compression for memory updates.

    Instead of using fixed momentum (β=0.9), this learns to compress
    gradients through an MLP, allowing problem-specific adaptation.

    From Nested Learning paper: the momentum term itself can be learned,
    providing better gradient compression than hand-designed rules.
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: int = 2,
    ) -> None:
        """
        Args:
            dim: Dimension of gradients/momentum
            hidden_mult: Hidden layer multiplier
        """
        super().__init__()

        hidden_dim = dim * hidden_mult

        # MLP to combine gradient and previous momentum
        self.momentum_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

        # Initialize final layer small for stable start
        with torch.no_grad():
            self.momentum_mlp[-1].weight.data *= 0.01
            self.momentum_mlp[-1].bias.data.zero_()

        # Learnable mixing coefficient (how much to trust learned vs simple)
        self.mix_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def forward(
        self,
        gradient: torch.Tensor,
        prev_momentum: torch.Tensor,
        beta: float = 0.9,
    ) -> torch.Tensor:
        """
        Compute new momentum from gradient and previous momentum.

        Args:
            gradient: Current gradient [B, dim] or [dim]
            prev_momentum: Previous momentum [B, dim] or [dim]
            beta: Fallback momentum coefficient

        Returns:
            New momentum tensor
        """
        # Simple momentum baseline
        simple_momentum = beta * prev_momentum + (1 - beta) * gradient

        # Learned momentum
        combined = torch.cat([gradient, prev_momentum], dim=-1)
        learned_momentum = self.momentum_mlp(combined)

        # Mix based on learned alpha
        alpha = torch.sigmoid(self.mix_alpha)
        new_momentum = alpha * learned_momentum + (1 - alpha) * simple_momentum

        return new_momentum


class NeuralMemoryAdapter(nn.Module):
    """
    Adapter wrapping NeuralMemory for Qwen2 integration.

    Handles:
    1. Translating TitansQwenConfig to TitansConfig
    2. Managing memory state lifecycle
    3. Providing clean forward/update interface for decoder layer

    The underlying NeuralMemory expects TitansConfig, but we want to configure
    it using TitansQwenConfig for a cleaner Qwen-specific interface.

    Example:
        >>> config = TitansQwenConfig(n_embd=1536, num_longterm_mem=16)
        >>> adapter = NeuralMemoryAdapter(config)
        >>> state = adapter.init_state(batch_size=2, device=torch.device('cuda'))
        >>> x = torch.randn(2, 512, 1536)  # [B, T, C]
        >>> retrieved = adapter(x, state)  # [B, 16, 1536]
        >>> new_state = adapter.update(x, state)
    """

    def __init__(self, config: TitansQwenConfig) -> None:
        super().__init__()

        self.config = config

        # Convert to TitansConfig and create NeuralMemory
        titans_config = config.to_titans_config()
        self.memory = NeuralMemory(titans_config)

        # Cache dimensions for easy access
        self.n_embd = config.n_embd
        self.num_longterm_mem = config.num_longterm_mem
        self.segment_len = config.segment_len

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """
        Initialize fresh memory state for a batch.

        Args:
            batch_size: Number of sequences in batch
            device: Device to create tensors on

        Returns:
            MemoryState with initialized MLP weights per batch item
        """
        return self.memory.init_state(batch_size, device)

    def reset_state(self, state: MemoryState) -> None:
        """
        Reset memory state in-place (avoids reallocation).

        Args:
            state: Existing MemoryState to reset
        """
        self.memory.reset_state(state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: MemoryState,
    ) -> torch.Tensor:
        """
        Retrieve from memory (uses previous segment for causality).

        Args:
            hidden_states: Current segment hidden states [B, T, C]
            state: Current memory state (contains previous segment's output)

        Returns:
            Memory context [B, num_longterm_mem, C] to prepend to attention
        """
        return self.memory(hidden_states, state)

    def update(
        self,
        hidden_states: torch.Tensor,
        state: MemoryState,
    ) -> MemoryState:
        """
        Update memory with current segment's hidden states.

        This implements the test-time learning: memory MLP weights are updated
        based on the "surprise" (gradient of MSE loss between MLP(key) and value).

        Args:
            hidden_states: Current segment output [B, T, C]
            state: Current memory state

        Returns:
            New memory state with updated weights
        """
        return self.memory.update(hidden_states, state)

    def set_aggregated_update(self, enabled: bool) -> None:
        """
        Enable/disable aggregated gradient updates.

        When enabled (default if available): 500x memory reduction,
        slight accuracy tradeoff.

        Args:
            enabled: Whether to use aggregated updates
        """
        self.memory.set_aggregated_update(enabled)

    @property
    def update_enabled(self) -> bool:
        """Whether memory updates are enabled."""
        return getattr(self.memory, "_use_aggregated_update", True)

    def get_param_count(self) -> int:
        """Get total parameter count for this memory module."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_param_count(self) -> int:
        """Get trainable parameter count for this memory module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

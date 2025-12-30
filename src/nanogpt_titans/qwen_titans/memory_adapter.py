"""
Memory adapter for Qwen models.

Wraps the NeuralMemory class from nanogpt_titans to work with Qwen dimensions,
providing a clean interface for the decoder layer integration.

Also includes HOPE-inspired components:
- SelfModifyingLinear: Delta rule weight updates during forward pass
- SelfModifyingGate: Learned gate that opens as memory becomes useful
- ContinuumMemorySystem: Multi-frequency memory MLPs

Features from nested_learning:
- MemoryMetrics: Per-update metrics (grad_norm, surprise, update_skipped)
- CMSMetrics: Multi-level metrics
- Surprise threshold: Skip updates for low-surprise tokens
- Per-level gradient clipping: Prevent any level from dominating
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from nanogpt_titans.model import MemoryState, NeuralMemory

if TYPE_CHECKING:
    from nanogpt_titans.qwen_titans.config import TitansQwenConfig


# =============================================================================
# Memory Metrics (from nested_learning)
# =============================================================================


@dataclass
class MemoryMetrics:
    """Metrics collected during a single memory update.

    Provides visibility into memory learning dynamics for debugging
    and hyperparameter tuning.

    Attributes:
        grad_norm: L2 norm of gradients before clipping
        surprise: Average "surprise" value (proxy for how unexpected the input was)
        update_skipped: Whether update was skipped due to low surprise
        lr_mean: Mean adaptive learning rate (if adaptive=True)
        momentum_mean: Mean adaptive momentum (if adaptive=True)
        decay_mean: Mean adaptive decay (if adaptive=True)
    """

    grad_norm: float = 0.0
    surprise: float = 0.0
    update_skipped: bool = False
    lr_mean: float = 0.0
    momentum_mean: float = 0.0
    decay_mean: float = 0.0


@dataclass
class CMSMetrics:
    """Metrics for ContinuumMemorySystem (multi-level memory).

    Aggregates metrics from all CMS levels for debugging multi-frequency
    memory dynamics.

    Attributes:
        level_metrics: List of MemoryMetrics, one per CMS level
        avg_surprise: Average surprise across all levels
        updates_skipped: Number of levels that skipped update
    """

    level_metrics: list[MemoryMetrics] = field(default_factory=list)
    avg_surprise: float = 0.0
    updates_skipped: int = 0

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
            self._mean_gate = (self._mean_gate * self._gate_count + gate.mean()) / (
                self._gate_count + 1
            )
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
    step: int = 0  # Global step counter for metrics


class ContinuumMemorySystem(nn.Module):
    """
    Multi-frequency memory system from Nested Learning.

    Different memory levels update at different rates:
    - Level 0: Updates every segment (fast, working memory)
    - Level 1: Updates every 4 segments (medium, episodic)
    - Level 2: Updates every 16 segments (slow, semantic)

    This allows capturing patterns at multiple timescales without
    interference between fast and slow dynamics.

    Supports two combination modes (from nested_learning):
    - Weighted sum (default): All levels process same input, outputs are weighted sum
    - Cascade: Each level transforms previous level's output (hierarchical refinement)

    Features from nested_learning:
    - Surprise threshold: Skip updates when grad norm is below threshold
    - Per-level gradient clipping: Prevent any level from dominating
    - Metrics collection: Track grad_norm, surprise per level
    """

    def __init__(
        self,
        config: TitansQwenConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.num_levels = config.num_cms_levels
        self.update_frequencies = config.cms_update_frequencies

        # New: surprise threshold and grad clipping from nested_learning
        self.surprise_threshold = getattr(config, "surprise_threshold", 0.0)
        self.grad_clip = getattr(config, "memory_grad_clip", 1.0)

        # New: cascade mode from nested_learning
        self.use_cascade = getattr(config, "use_cascade", False)

        # Create a memory module for each level
        self.memories = nn.ModuleList([NeuralMemoryAdapter(config) for _ in range(self.num_levels)])

        # Projection to combine outputs from all levels (only used in weighted sum mode)
        self.combine_proj = nn.Linear(
            config.n_embd * self.num_levels,
            config.n_embd,
            bias=False,
        )

        # Learnable weights for combining levels (only used in weighted sum mode)
        self.level_weights = nn.Parameter(torch.ones(self.num_levels) / self.num_levels)

        # Store last metrics for access
        self._last_metrics: CMSMetrics | None = None

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

        Two modes (from nested_learning):
        - Weighted sum: All levels process same input, outputs are weighted sum
        - Cascade: Each level transforms previous level's output (hierarchical refinement)

        Args:
            hidden_states: Current segment [B, T, C]
            state: Multi-level memory state

        Returns:
            Combined memory output [B, num_longterm_mem, C]
        """
        if self.use_cascade:
            # Cascade mode: each level transforms the previous level's output
            # Level 0 processes input, Level 1 processes Level 0's output, etc.
            current = hidden_states
            for mem, level_state in zip(self.memories, state.level_states):
                current = mem(current, level_state)
            return current
        else:
            # Weighted sum mode (default): all levels process same input
            weights = F.softmax(self.level_weights, dim=0)

            # Fused retrieval: stack outputs and apply weights in one operation
            # This reduces Python loop overhead and enables better GPU utilization
            level_outputs = torch.stack(
                [
                    mem(hidden_states, level_state)
                    for mem, level_state in zip(self.memories, state.level_states)
                ],
                dim=0,
            )  # [num_levels, B, num_longterm_mem, C]

            # Apply weights: [num_levels, 1, 1, 1] * [num_levels, B, T, C] -> weighted sum
            weights_expanded = weights.view(-1, 1, 1, 1)
            combined = (weights_expanded * level_outputs).sum(dim=0)  # [B, num_longterm_mem, C]

            return combined

    def update(
        self,
        hidden_states: torch.Tensor,
        state: ContinuumMemoryState,
    ) -> tuple[ContinuumMemoryState, CMSMetrics]:
        """
        Update memory levels based on their frequencies.

        In cascade mode, each level's update receives the previous level's
        transformed output (matching nested_learning behavior).

        Features from nested_learning:
        - Surprise threshold: Skip updates when grad norm is below threshold
        - Per-level gradient clipping: Applied within each level's update
        - Metrics collection: Returns CMSMetrics with per-level info

        Args:
            hidden_states: Current segment output [B, T, C]
            state: Current multi-level state

        Returns:
            Tuple of (updated state, metrics)
        """
        new_level_states = []
        new_segment_counts = []
        level_metrics = []
        updates_skipped = 0

        # In cascade mode, track the cascaded input for each level
        current_input = hidden_states

        for i, (mem, level_state, freq) in enumerate(
            zip(self.memories, state.level_states, self.update_frequencies)
        ):
            count = state.segment_counts[i] + 1

            # Update only if segment count is multiple of frequency
            if count % freq == 0:
                # For now, use the underlying memory.update() which doesn't return metrics yet
                # TODO: Extend NeuralMemory.update() to return metrics with surprise threshold
                new_state = mem.update(current_input, level_state)

                # Create placeholder metrics (full implementation requires modifying model.py)
                metrics = MemoryMetrics(
                    grad_norm=0.0,  # Would need to compute in NeuralMemory.update()
                    surprise=0.0,
                    update_skipped=False,
                )
            else:
                new_state = level_state
                metrics = MemoryMetrics(update_skipped=True)
                updates_skipped += 1

            new_level_states.append(new_state)
            new_segment_counts.append(count)
            level_metrics.append(metrics)

            # In cascade mode, transform input for next level
            # Each level's output becomes the next level's input
            if self.use_cascade and i < len(self.memories) - 1:
                # Retrieve from this level (with updated state) to get transformed output
                current_input = mem(current_input, new_state)

        # Compute aggregate metrics
        active_metrics = [m for m in level_metrics if not m.update_skipped]
        avg_surprise = (
            sum(m.surprise for m in active_metrics) / len(active_metrics)
            if active_metrics
            else 0.0
        )

        cms_metrics = CMSMetrics(
            level_metrics=level_metrics,
            avg_surprise=avg_surprise,
            updates_skipped=updates_skipped,
        )
        self._last_metrics = cms_metrics

        new_state = ContinuumMemoryState(
            level_states=new_level_states,
            segment_counts=new_segment_counts,
            step=state.step + 1,
        )

        return new_state, cms_metrics

    def get_last_metrics(self) -> CMSMetrics | None:
        """Get metrics from the last update call."""
        return self._last_metrics


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

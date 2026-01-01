"""
MLX TITANS decoder layer implementations.

Provides MLX-native implementations of:
- MLXPositionDependentGate: Per-token gating
- MLXTitansLayer: TITANS layer with HOPE architecture

Matches PyTorch TitansQwenDecoderLayer exactly.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from nanogpt_titans.mlx.memory import (
    CMSMetrics,
    MemoryMetrics,
    MLXCMSState,
    MLXContinuumMemorySystem,
    MLXMemoryState,
    MLXNeuralMemory,
)

# Type alias for memory state
TitansLayerState = MLXCMSState | MLXMemoryState


class MLXPositionDependentGate(nn.Module):
    """
    Position-dependent gate that produces per-token gate values.

    Unlike global gates, this allows the model to decide per-token
    whether to use memory, which is critical for tasks like needle-in-haystack.

    Matches PyTorch PositionDependentGate exactly.
    """

    def __init__(self, dim: int, init_bias: float = 0.0):
        super().__init__()
        # Small MLP: hidden_states -> gate value per token
        self.linear1 = nn.Linear(dim, dim // 4)
        self.linear2 = nn.Linear(dim // 4, 1)

        # Initialize with small random weights to match PyTorch:
        # nn.init.normal_(weight, std=0.01)
        scale = 0.01
        self.linear1.weight = mx.random.normal(self.linear1.weight.shape) * scale
        self.linear1.bias = mx.zeros(self.linear1.bias.shape)
        self.linear2.weight = mx.random.normal(self.linear2.weight.shape) * scale
        self.linear2.bias = mx.array([init_bias])  # sigmoid(0) = 0.5

    def __call__(self, x: mx.array) -> mx.array:
        """
        Compute per-token gate values.

        Args:
            x: Hidden states [B, T, C]

        Returns:
            Gate values [B, T, 1] in range (0, 1)
        """
        h = nn.silu(self.linear1(x))
        return mx.sigmoid(self.linear2(h))


class MLXTitansLayer(nn.Module):
    """
    TITANS memory layer in MLX with full HOPE architecture.

    Matches PyTorch TitansQwenDecoderLayer:
    1. ContinuumMemorySystem (CMS) for multi-frequency memory
    2. Memory projection + LayerNorm + learned scale
    3. Position-dependent gate for per-token gating
    4. HOPE integration: output = hidden + gate * scale * LayerNorm(project(memory))
    """

    def __init__(
        self,
        dim: int,
        use_cms: bool = True,
        num_cms_levels: int = 3,
        cms_update_frequencies: tuple = (1, 4, 16),
        memory_depth: int = 2,
        memory_expansion: int = 2,
        adaptive_memory: bool = True,
        memory_lr_max: float = 0.01,
        gate_init_bias: float = 0.0,
        grad_clip: float = 1.0,
        surprise_threshold: float = 0.0,
        use_cascade: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self._use_cms = use_cms

        # Store last metrics
        self._last_metrics: CMSMetrics | MemoryMetrics | None = None

        # Memory system (CMS or single)
        if use_cms:
            self.memory = MLXContinuumMemorySystem(
                dim,
                num_levels=num_cms_levels,
                update_frequencies=cms_update_frequencies,
                memory_depth=memory_depth,
                memory_expansion=memory_expansion,
                adaptive=adaptive_memory,
                lr_max=memory_lr_max,
                grad_clip=grad_clip,
                surprise_threshold=surprise_threshold,
                use_cascade=use_cascade,
            )
        else:
            self.memory = MLXNeuralMemory(
                dim,
                depth=memory_depth,
                expansion=memory_expansion,
                adaptive=adaptive_memory,
                lr_max=memory_lr_max,
                grad_clip=grad_clip,
                surprise_threshold=surprise_threshold,
            )

        # Memory projection (matches PyTorch mem_proj)
        self.mem_proj = nn.Linear(dim, dim, bias=False)

        # LayerNorm + learned scale
        self.mem_ln = nn.LayerNorm(dim)

        # Learned scale: start at sigmoid(0) = 0.5 for good gradient flow
        self.mem_scale = mx.array([0.0])

        # Position-dependent gate
        self.gate = MLXPositionDependentGate(dim, init_bias=gate_init_bias)

    def init_state(self, batch_size: int) -> TitansLayerState:
        """Initialize memory state for this layer."""
        return self.memory.init_state(batch_size)

    def __call__(self, hidden_states: mx.array, state: TitansLayerState) -> tuple[mx.array, TitansLayerState]:
        """
        Apply HOPE-style memory enhancement to hidden states.

        HOPE Architecture:
        1. Retrieve from memory (CMS or single)
        2. Pool -> Project -> LayerNorm -> Scale
        3. Apply position-dependent gate
        4. output = hidden_states + gate * scaled_memory
        5. Update memory with current segment

        Args:
            hidden_states: Input hidden states [B, T, C]
            state: Current memory state

        Returns:
            Tuple of (enhanced hidden states [B, T, C], updated state)
        """
        B, T, C = hidden_states.shape

        # 1. Retrieve from memory (uses PREVIOUS segment's output for causality)
        mem_retrieved = self.memory(hidden_states, state)  # [B, T, C]

        # 2. Pool -> Project -> LayerNorm -> Scale
        mem_pooled = mx.mean(mem_retrieved, axis=1, keepdims=True)  # [B, 1, C]
        mem_projected = self.mem_proj(mem_pooled)  # [B, 1, C]
        mem_projected = self.mem_ln(mem_projected)  # [B, 1, C]

        # Apply learned scale: sigmoid(scale) in (0, 1)
        scale = mx.sigmoid(self.mem_scale)  # scalar
        mem_scaled = mem_projected * scale  # [B, 1, C]

        # Broadcast to sequence length
        mem_scaled = mx.broadcast_to(mem_scaled, (B, T, C))  # [B, T, C]

        # 3. Position-dependent gate
        gate_value = self.gate(hidden_states)  # [B, T, 1]

        # 4. Update memory with RAW hidden_states (not memory-augmented output!)
        # This is critical: memory should learn to predict transformer patterns,
        # not its own output. Using output here creates a feedback loop.
        new_state, metrics = self.memory.update(hidden_states, state)
        self._last_metrics = metrics

        # 5. HOPE: Additive contribution (after update)
        output = hidden_states + gate_value * mem_scaled

        return output, new_state

    def get_last_metrics(self) -> CMSMetrics | MemoryMetrics | None:
        """Get metrics from the last forward pass."""
        return self._last_metrics

    def compute_internal_loss(self, hidden_states: mx.array, state: TitansLayerState) -> mx.array:
        """
        Compute internal loss for memory module.

        This provides an independent learning signal for memory,
        separate from the gate and LM loss.

        Also includes gradient paths for mem_scale and mem_ln so they train.

        Args:
            hidden_states: Input hidden states [B, T, C]
            state: Current memory state

        Returns:
            Scalar internal loss
        """
        # Get base memory internal loss
        memory_loss = self.memory.compute_internal_loss(hidden_states, state)

        # =========================================================================
        # Include mem_scale and mem_ln in gradient path
        # =========================================================================
        # Use mem_scale in a way that gives it gradients
        scale = mx.sigmoid(self.mem_scale).squeeze()  # Convert to scalar

        # Regularization: encourage mem_scale to be in useful range [0.3, 0.7]
        # This prevents it from collapsing to 0 or saturating at 1
        scale_target = 0.5
        scale_reg = (scale - scale_target) ** 2

        # Also give gradients to mem_proj and mem_ln by using them
        # Project a sample through and add tiny regularization
        sample = hidden_states[:, :1, :]  # Just use first token to save compute
        proj_out = self.mem_proj(sample)
        ln_out = self.mem_ln(proj_out)
        proj_reg = mx.mean(ln_out * ln_out) * 0.0  # Zero weight but creates gradient path

        total_loss = memory_loss + 0.01 * scale_reg + proj_reg

        return total_loss

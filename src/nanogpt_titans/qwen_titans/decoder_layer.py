"""
Titans-enhanced Qwen2 decoder layer.

Uses HOPE-style gated residual integration for test-time learning memory.
This approach preserves sequence length and works with pre-trained models.

Key fixes for pre-trained model compatibility:
1. LayerNorm + learned scale after memory projection (fixes scale mismatch)
2. Position-dependent gate (per-token, not global)
3. Disabled/minimal internal loss by default
4. Conservative initialization that starts as near-identity
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from nanogpt_titans.qwen_titans.memory_adapter import (
    ContinuumMemoryState,
    ContinuumMemorySystem,
    NeuralMemoryAdapter,
    SelfModifyingGate,
    SelfModifyingLinear,
    WarmStartEncoder,
)

if TYPE_CHECKING:
    from nanogpt_titans.model import MemoryState
    from nanogpt_titans.qwen_titans.config import TitansQwenConfig


class PositionDependentGate(nn.Module):
    """
    Position-dependent gate that produces per-token gate values.

    Unlike global gates, this allows the model to decide per-token
    whether to use memory, which is critical for tasks like needle-in-haystack.
    """

    def __init__(self, dim: int, init_bias: float = -2.0) -> None:
        super().__init__()
        # Small MLP: hidden_states -> gate value per token
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
        )
        # Initialize with small random weights (not zeros!) to allow gradient flow
        # Zero weights create dead gradients - the gate can never learn
        nn.init.normal_(self.gate_mlp[0].weight, std=0.01)
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.normal_(self.gate_mlp[2].weight, std=0.01)
        nn.init.constant_(self.gate_mlp[2].bias, init_bias)  # sigmoid(-2) ≈ 0.12

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token gate values.

        Args:
            hidden_states: [B, T, C]

        Returns:
            gate values: [B, T, 1] in range (0, 1)
        """
        return torch.sigmoid(self.gate_mlp(hidden_states))


class TitansQwenDecoderLayer(nn.Module):
    """
    Titans-enhanced Qwen2 decoder layer using gated residual integration.

    Architecture (HOPE-style, fixed for pre-trained models):
    1. Forward through original Qwen2DecoderLayer (unchanged)
    2. Retrieve from memory (uses M_{t-1} for causality)
    3. Project memory + LayerNorm + learned scale (fixes scale mismatch)
    4. Apply position-dependent gated residual (per-token gates)
    5. Update memory with segment output

    Key fixes for pre-trained compatibility:
    - LayerNorm + learned scale ensures memory matches hidden distribution
    - Position-dependent gate allows per-token memory usage decisions
    - Conservative initialization starts as near-identity
    - Internal loss disabled by default (set weight to 0)

    Example:
        >>> original_layer = model.model.layers[14]
        >>> titans_layer = TitansQwenDecoderLayer(original_layer, 14, config)
        >>> model.model.layers[14] = titans_layer
    """

    def __init__(
        self,
        original_layer: nn.Module,
        layer_idx: int,
        titans_config: TitansQwenConfig,
    ) -> None:
        """
        Initialize Titans-enhanced decoder layer.

        Args:
            original_layer: The original Qwen2DecoderLayer to wrap
            layer_idx: Index of this layer in the model
            titans_config: Titans configuration for memory parameters
        """
        super().__init__()

        self.layer_idx = layer_idx
        self.config = titans_config

        # Keep original layer unchanged
        self.original_layer = original_layer

        # Choose memory system based on config
        if titans_config.use_cms:
            self.memory = ContinuumMemorySystem(titans_config)
            self._use_cms = True
        else:
            self.memory = NeuralMemoryAdapter(titans_config)
            self._use_cms = False

        # Memory projection: memory output -> hidden state space
        if titans_config.use_self_mod_proj:
            self.mem_proj = SelfModifyingLinear(
                titans_config.n_embd,
                titans_config.n_embd,
                bias=False,
                lr=titans_config.self_mod_lr,
            )
        else:
            self.mem_proj = nn.Linear(
                titans_config.n_embd,
                titans_config.n_embd,
                bias=False,
            )

        # === FIX 1: LayerNorm + learned scale for distribution alignment ===
        # This fixes the scale mismatch (memory norm ~0.03 vs hidden norm ~212)
        self.mem_ln = nn.LayerNorm(titans_config.n_embd)
        # Learned scale: start at sigmoid(0) = 0.5 for stronger gradients
        # Conservative init (-2.0 -> 0.12) caused gradient starvation
        self.mem_scale = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        # === FIX 2: Position-dependent gate (per-token, not global) ===
        # This allows the model to decide per-token whether to use memory
        if titans_config.use_self_mod_gate:
            # Keep self-modifying gate but make it position-dependent
            self.gate = SelfModifyingGate(
                titans_config.n_embd,
                init_bias=titans_config.gate_init_bias,
                lr=titans_config.self_mod_lr,
            )
            self._use_self_mod_gate = True
        else:
            # New position-dependent gate
            self.gate = PositionDependentGate(
                titans_config.n_embd,
                init_bias=titans_config.gate_init_bias,
            )
            self._use_self_mod_gate = False

        # Warm start encoder (optional)
        if titans_config.use_warm_start:
            self.warm_start = WarmStartEncoder(titans_config)
        else:
            self.warm_start = None

        # Flag to control memory updates
        self.update_memory = True

        # Flag to completely disable memory (for no-op baseline testing)
        self.memory_enabled = True

        # Store memory state during forward
        self._current_memory_state: MemoryState | ContinuumMemoryState | None = None
        self._updated_memory_state: MemoryState | ContinuumMemoryState | None = None

        # Internal loss for self-supervised memory signal
        self._internal_loss: torch.Tensor | None = None

        # Cache isinstance checks for faster forward pass
        self._use_self_mod_proj = isinstance(self.mem_proj, SelfModifyingLinear)

        # Copy attributes that Qwen's internal code expects
        if hasattr(original_layer, "self_attn"):
            self.self_attn = original_layer.self_attn
        if hasattr(original_layer, "mlp"):
            self.mlp = original_layer.mlp
        if hasattr(original_layer, "input_layernorm"):
            self.input_layernorm = original_layer.input_layernorm
        if hasattr(original_layer, "post_attention_layernorm"):
            self.post_attention_layernorm = original_layer.post_attention_layernorm
        if hasattr(original_layer, "attention_type"):
            self.attention_type = original_layer.attention_type

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        memory_state: MemoryState | ContinuumMemoryState | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward with gated memory integration.

        1. Run original layer forward (unchanged)
        2. Retrieve from memory
        3. Project + LayerNorm + scale (fixes distribution mismatch)
        4. Apply position-dependent gated residual
        5. Update memory

        Args:
            hidden_states: Input hidden states [B, T, C]
            attention_mask: Attention mask (passed through unchanged)
            position_ids: Position IDs (passed through unchanged)
            past_key_values: KV cache (passed through)
            output_attentions: Whether to output attention weights
            use_cache: Whether to use/return KV cache
            cache_position: Cache position indices
            position_embeddings: Pre-computed rotary embeddings
            memory_state: Titans memory state
            **kwargs: Additional arguments for original layer

        Returns:
            Tuple of (hidden_states, ...) matching original layer format
        """
        B, T, _C = hidden_states.shape
        device = hidden_states.device

        # 1. Forward through original layer (UNCHANGED - preserves pre-trained behavior)
        layer_outputs = self.original_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Extract output
        if isinstance(layer_outputs, tuple):
            attn_output = layer_outputs[0]
            other_outputs = layer_outputs[1:]
        else:
            attn_output = layer_outputs
            other_outputs = ()

        # === NO-OP BASELINE: If memory disabled, just return original output ===
        if not self.memory_enabled:
            self._updated_memory_state = None
            self._internal_loss = None
            if other_outputs:
                return (attn_output, *other_outputs)
            return attn_output

        # Use provided state or stored state
        if memory_state is None:
            memory_state = self._current_memory_state

        # Initialize memory state if needed
        if memory_state is None:
            memory_state = self.memory.init_state(B, device)

            # Warm start: initialize memory from input
            if self.warm_start is not None and self.config.warm_start_prefix_len <= T:
                _ = self.warm_start(hidden_states)
                # TODO: Use warm start output to initialize memory state

        # 2. Retrieve from memory (causal - uses M_{t-1})
        mem_retrieved = self.memory(hidden_states, memory_state)  # [B, num_longterm_mem, C]

        # 3. Project memory + LayerNorm + scale (FIX for distribution alignment)
        # Pool -> project -> normalize -> scale -> broadcast
        mem_pooled = mem_retrieved.mean(dim=1, keepdim=True)  # [B, 1, C]

        # Apply projection (self-modifying or standard)
        if self._use_self_mod_proj:
            mem_projected = self.mem_proj(mem_pooled, update=self.training)
        else:
            mem_projected = self.mem_proj(mem_pooled)

        # === FIX 1: LayerNorm + learned scale ===
        # LayerNorm ensures memory has similar distribution to hidden states
        mem_projected = self.mem_ln(mem_projected)
        # Learned scale controls memory contribution strength (starts small)
        scale = torch.sigmoid(self.mem_scale)  # (0, 1)
        mem_projected = mem_projected * scale

        # Broadcast to sequence length
        mem_projected = mem_projected.expand(-1, T, -1)  # [B, T, C]

        # 4. Apply position-dependent gated residual
        if self._use_self_mod_gate:
            # Self-modifying gate (legacy path)
            output = self.gate(mem_projected, attn_output, update=self.training)
        else:
            # === FIX 2: Position-dependent gate ===
            # Gate is computed per-token from hidden states
            gate_value = self.gate(attn_output)  # [B, T, 1]
            output = attn_output + gate_value * mem_projected

        # 5. Compute internal loss (memory's own prediction error)
        # NOTE: This should be heavily down-weighted or disabled (weight <= 1e-4)
        # The internal loss can dominate LM loss and train memory for wrong objective
        if self.config.use_internal_loss and self.training:
            self._internal_loss = self._compute_memory_surprise(hidden_states)

        # 6. Update memory with RAW attn_output (not memory-augmented output!)
        # This is critical: memory should learn to predict transformer patterns,
        # not its own output. Using output here creates a feedback loop.
        if self.update_memory:
            update_result = self.memory.update(attn_output, memory_state)
            # Handle both old and new return types (CMS returns tuple)
            if isinstance(update_result, tuple):
                new_memory_state, _metrics = update_result
            else:
                new_memory_state = update_result
        else:
            new_memory_state = memory_state

        # Store for external access
        self._updated_memory_state = new_memory_state

        # Return in HuggingFace-compatible format
        if other_outputs:
            return (output, *other_outputs)
        return attn_output if not self.memory_enabled else output

    def set_memory_state(self, state: MemoryState | ContinuumMemoryState | None) -> None:
        """Set the memory state to use for next forward pass."""
        self._current_memory_state = state

    def get_memory_state(self) -> MemoryState | ContinuumMemoryState | None:
        """Get the memory state after last forward pass."""
        return self._updated_memory_state

    def get_internal_loss(self) -> torch.Tensor | None:
        """Get the internal loss for this layer (if enabled)."""
        return self._internal_loss

    def enable_memory_updates(self, enabled: bool = True) -> None:
        """Enable or disable memory updates during forward."""
        self.update_memory = enabled

    def get_gate_statistics(self) -> dict:
        """Get gate statistics for monitoring."""
        stats = {
            "mem_scale": torch.sigmoid(self.mem_scale).item(),
        }
        if isinstance(self.gate, SelfModifyingGate):
            stats["mean_gate"] = self.gate.mean_gate_value
            stats["gate_bias"] = self.gate.gate_proj.bias.item()
        elif isinstance(self.gate, PositionDependentGate):
            stats["gate_bias"] = self.gate.gate_mlp[2].bias.item()
        else:
            stats["gate_bias"] = self.gate[0].bias.item()
        return stats

    def set_memory_enabled(self, enabled: bool) -> None:
        """
        Enable or disable memory completely (for no-op baseline testing).

        When disabled, the layer acts exactly like the original Qwen layer.
        """
        self.memory_enabled = enabled

    def set_noop_mode(self) -> None:
        """
        Set layer to complete no-op mode for baseline testing.

        This ensures the wrapper doesn't affect model output at all.
        Use this to verify the wrapper itself doesn't cause regression.
        """
        self.memory_enabled = False
        self.update_memory = False

    def _compute_memory_surprise(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute memory's own prediction error: ||M(k) - v||^2 + regularization

        This is the correct internal loss from Titans paper.
        Memory predicts what IT would store, not what the transformer computes.

        Also includes gradient paths for mem_scale, mem_ln, and adaptive params
        so they receive training signal.

        Args:
            hidden_states: Input to memory [B, T, C]

        Returns:
            Scalar loss tensor
        """

        # Get the underlying NeuralMemory (handle CMS case)
        if self._use_cms:
            # For CMS, compute surprise for first (fastest) memory level
            mem = self.memory.memories[0].memory
        else:
            mem = self.memory.memory

        # Detach input - we don't want LM gradients flowing through this
        x = hidden_states.detach()

        # Clip input magnitude for numerical stability
        max_input_norm = 10.0
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        x = x / x_norm * torch.clamp(x_norm, max=max_input_norm)

        # Project to keys and values (what memory would store)
        keys = mem.key_proj(x)  # [B, T, C]
        values = mem.value_proj(x)  # [B, T, C]

        # Clip projected values for stability
        keys = torch.clamp(keys, min=-max_input_norm, max=max_input_norm)
        values = torch.clamp(values, min=-max_input_norm, max=max_input_norm)

        # Get memory's prediction for these keys
        # Note: memory_mlp expects [B, T, C] and outputs [B, T, C]
        predictions = mem.memory_mlp(keys)  # [B, T, C]

        # Clip predictions for stability
        predictions = torch.clamp(predictions, min=-max_input_norm, max=max_input_norm)

        # Compute MSE loss - this is what memory CAN learn to minimize
        recon_loss = F.mse_loss(predictions, values)

        # =========================================================================
        # Include mem_scale and mem_ln in gradient path (matches MLX version)
        # =========================================================================
        # Regularization: encourage mem_scale to be in useful range [0.3, 0.7]
        scale = torch.sigmoid(self.mem_scale)
        scale_target = 0.5
        scale_reg = (scale - scale_target) ** 2

        # Give gradients to mem_proj and mem_ln by using them
        sample = hidden_states[:, :1, :]  # Just use first token to save compute
        proj_out = self.mem_proj(sample)
        ln_out = self.mem_ln(proj_out)
        proj_reg = (ln_out**2).mean() * 0.0  # Zero weight but creates gradient path

        # =========================================================================
        # Adaptive parameter training (matches MLX version)
        # =========================================================================
        adaptive_reg = torch.tensor(0.0, device=hidden_states.device)
        if hasattr(mem, "to_lr") and mem.to_lr is not None:
            # L2 regularization on adaptive outputs - gives direct gradients
            lr_out = mem.to_lr(x[:, :1, :])
            mom_out = mem.to_momentum(x[:, :1, :])
            decay_out = mem.to_decay(x[:, :1, :])

            adaptive_reg = ((lr_out**2).mean() + (mom_out**2).mean() + (decay_out**2).mean()) * 0.001

        # Also train query_proj
        query_reg = torch.tensor(0.0, device=hidden_states.device)
        if hasattr(mem, "query_proj"):
            query_out = mem.query_proj(x[:, :1, :])
            query_reg = (query_out**2).mean() * 0.001

        total_loss = recon_loss + 0.01 * scale_reg + proj_reg + adaptive_reg + query_reg

        # Additional safety: if loss is NaN or Inf, return zero
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        return total_loss

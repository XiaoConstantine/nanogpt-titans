"""
Titans-enhanced Qwen2 decoder layer.

Uses HOPE-style gated residual integration for test-time learning memory.
This approach preserves sequence length and works with pre-trained models.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any, Union

import torch
from torch import nn

from nanogpt_titans.model import MemoryState
from nanogpt_titans.qwen_titans.config import TitansQwenConfig
from nanogpt_titans.qwen_titans.memory_adapter import (
    NeuralMemoryAdapter,
    ContinuumMemorySystem,
    ContinuumMemoryState,
    SelfModifyingLinear,
    SelfModifyingGate,
    WarmStartEncoder,
)


class TitansQwenDecoderLayer(nn.Module):
    """
    Titans-enhanced Qwen2 decoder layer using gated residual integration.

    Architecture (HOPE-style):
    1. Forward through original Qwen2DecoderLayer (unchanged)
    2. Retrieve from memory (uses M_{t-1} for causality)
    3. Project memory to match hidden state dimensions
    4. Apply gated residual: output = attn_output + gate * mem_projection
    5. Update memory with segment output

    Key benefits:
    - No sequence length change (avoids mask/position issues)
    - Gate starts near 0 (conservative, doesn't hurt base model)
    - Gate learns to open as memory becomes useful
    - Compatible with pre-trained attention weights

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

        # Self-modifying projection: memory output -> hidden state space
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

        # Self-modifying gate for memory integration
        if titans_config.use_self_mod_gate:
            self.gate = SelfModifyingGate(
                titans_config.n_embd,
                init_bias=titans_config.gate_init_bias,
                lr=titans_config.self_mod_lr,
            )
        else:
            # Simple learned gate without delta rule
            self.gate = nn.Sequential(
                nn.Linear(titans_config.n_embd, 1, bias=True),
                nn.Sigmoid(),
            )
            # Initialize for conservative gate
            nn.init.zeros_(self.gate[0].weight)
            nn.init.constant_(self.gate[0].bias, titans_config.gate_init_bias)

        # Warm start encoder (optional)
        if titans_config.use_warm_start:
            self.warm_start = WarmStartEncoder(titans_config)
        else:
            self.warm_start = None

        # Flag to control memory updates
        self.update_memory = True

        # Store memory state during forward
        self._current_memory_state: Optional[Union[MemoryState, ContinuumMemoryState]] = None
        self._updated_memory_state: Optional[Union[MemoryState, ContinuumMemoryState]] = None

        # Internal loss for self-supervised memory signal
        self._internal_loss: Optional[torch.Tensor] = None

        # Cache isinstance checks for faster forward pass
        self._use_self_mod_proj = isinstance(self.mem_proj, SelfModifyingLinear)
        self._use_self_mod_gate = isinstance(self.gate, SelfModifyingGate)
        self._use_cms = titans_config.use_cms

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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        memory_state: Optional[Union[MemoryState, ContinuumMemoryState]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward with gated memory integration.

        1. Run original layer forward (unchanged)
        2. Retrieve from memory
        3. Project and gate memory contribution
        4. Add to output via gated residual

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
        B, T, C = hidden_states.shape
        device = hidden_states.device

        # Use provided state or stored state
        if memory_state is None:
            memory_state = self._current_memory_state

        # Initialize memory state if needed
        if memory_state is None:
            memory_state = self.memory.init_state(B, device)

            # Warm start: initialize memory from input
            if self.warm_start is not None and T >= self.config.warm_start_prefix_len:
                _ = self.warm_start(hidden_states)
                # TODO: Use warm start output to initialize memory state

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

        # 2. Retrieve from memory (causal - uses M_{t-1})
        mem_retrieved = self.memory(hidden_states, memory_state)  # [B, num_longterm_mem, C]

        # 3. Project memory to full sequence length (fused operations)
        # Pool -> project -> broadcast in one path
        mem_pooled = mem_retrieved.mean(dim=1, keepdim=True)  # [B, 1, C]

        # Apply projection (self-modifying or standard)
        if self._use_self_mod_proj:
            mem_projected = self.mem_proj(mem_pooled, update=self.training)
        else:
            mem_projected = self.mem_proj(mem_pooled)

        # Clamp to prevent extreme values (also handles NaN/Inf)
        mem_projected = torch.clamp(mem_projected, min=-100.0, max=100.0)

        # Broadcast to sequence length
        mem_projected = mem_projected.expand(-1, T, -1)  # [B, T, C]

        # 4. Apply gated residual
        if self._use_self_mod_gate:
            output = self.gate(mem_projected, attn_output, update=self.training)
        else:
            # Simple gate
            gate_value = self.gate(attn_output)  # [B, T, 1]
            output = attn_output + gate_value * mem_projected

        # 5. Compute internal loss (memory's own prediction error)
        # This is the Titans paper approach: ||M(k) - v||^2
        # Memory predicts its own stored values, NOT transformer output
        if self.config.use_internal_loss and self.training:
            self._internal_loss = self._compute_memory_surprise(hidden_states)

        # 6. Update memory with segment output
        if self.update_memory:
            new_memory_state = self.memory.update(output, memory_state)
        else:
            new_memory_state = memory_state

        # Store for external access
        self._updated_memory_state = new_memory_state

        # Return in HuggingFace-compatible format
        if other_outputs:
            return (output,) + other_outputs
        return output

    def set_memory_state(self, state: Optional[Union[MemoryState, ContinuumMemoryState]]) -> None:
        """Set the memory state to use for next forward pass."""
        self._current_memory_state = state

    def get_memory_state(self) -> Optional[Union[MemoryState, ContinuumMemoryState]]:
        """Get the memory state after last forward pass."""
        return self._updated_memory_state

    def get_internal_loss(self) -> Optional[torch.Tensor]:
        """Get the internal loss for this layer (if enabled)."""
        return self._internal_loss

    def enable_memory_updates(self, enabled: bool = True) -> None:
        """Enable or disable memory updates during forward."""
        self.update_memory = enabled

    def get_gate_statistics(self) -> dict:
        """Get gate statistics for monitoring."""
        if isinstance(self.gate, SelfModifyingGate):
            return {
                "mean_gate": self.gate.mean_gate_value,
                "gate_bias": self.gate.gate_proj.bias.item(),
            }
        else:
            return {"gate_bias": self.gate[0].bias.item()}

    def _compute_memory_surprise(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute memory's own prediction error: ||M(k) - v||^2

        This is the correct internal loss from Titans paper.
        Memory predicts what IT would store, not what the transformer computes.

        Args:
            hidden_states: Input to memory [B, T, C]

        Returns:
            Scalar loss tensor
        """
        import torch.nn.functional as F

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
        keys = mem.key_proj(x)      # [B, T, C]
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
        surprise_loss = F.mse_loss(predictions, values)

        # Additional safety: if loss is NaN or Inf, return zero
        if torch.isnan(surprise_loss) or torch.isinf(surprise_loss):
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        return surprise_loss

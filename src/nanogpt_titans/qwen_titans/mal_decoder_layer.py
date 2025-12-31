"""
MAL (Memory as Layer) Titans-enhanced Qwen2 decoder layer.

Implements the MAL variant from the Titans paper where memory operates
as a PREPROCESSING layer before attention (sequential composition).

Key difference from other variants:
- MAC: output = Attn([memory || input])  (memory as context)
- MAG: output = Attn(input) * Memory(input)  (memory as gate)
- MAL: output = Attn(input + Memory(input))  (memory as layer)

Benefits of MAL:
1. Simplest architecture - sequential composition
2. Memory enhances input before attention processes it
3. Fastest inference (no parallel branches)
4. Most compatible with attention patterns
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
)

if TYPE_CHECKING:
    from nanogpt_titans.model import MemoryState
    from nanogpt_titans.qwen_titans.config import TitansQwenConfig


class MALQwenDecoderLayer(nn.Module):
    """
    MAL (Memory as Layer) enhanced Qwen2 decoder layer.

    Architecture:
        1. Retrieve from memory using input as query
        2. Project and add memory to input (residual)
        3. Pass enhanced input through original attention layer
        4. Update memory based on output

    Key equation: output = Attn(input + gate * MemoryProj(Memory(input)))

    The memory enhancement happens BEFORE attention, allowing attention
    to operate on memory-augmented representations.

    Example:
        >>> original_layer = model.model.layers[14]
        >>> mal_layer = MALQwenDecoderLayer(original_layer, 14, config)
        >>> model.model.layers[14] = mal_layer
    """

    def __init__(
        self,
        original_layer: nn.Module,
        layer_idx: int,
        titans_config: TitansQwenConfig,
    ) -> None:
        """
        Initialize MAL-enhanced decoder layer.

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

        # Memory output projection (memory dim -> hidden dim)
        self.mem_proj = nn.Linear(titans_config.n_embd, titans_config.n_embd, bias=False)

        # Learnable gate for memory contribution (starts small)
        # This controls how much memory influences the input before attention
        self.gate = nn.Parameter(torch.tensor(0.1))

        # Flag to control memory updates
        self.update_memory = True

        # Store memory state during forward
        self._current_memory_state: MemoryState | ContinuumMemoryState | None = None
        self._updated_memory_state: MemoryState | ContinuumMemoryState | None = None

        # Internal loss for self-supervised memory signal
        self._internal_loss: torch.Tensor | None = None

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
        Forward with MAL-style memory preprocessing.

        1. Retrieve from memory using input as query
        2. Add gated memory contribution to input
        3. Pass enhanced input through attention
        4. Update memory

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
        B, _T, _C = hidden_states.shape
        device = hidden_states.device

        # Use provided state or stored state
        if memory_state is None:
            memory_state = self._current_memory_state

        # Initialize memory state if needed
        if memory_state is None:
            memory_state = self.memory.init_state(B, device)

        # ===== STEP 1: Memory Retrieval =====
        # Retrieve from memory (causal - uses M_{t-1})
        mem_retrieved = self.memory(hidden_states, memory_state)  # [B, num_longterm_mem, C]

        # Pool memory to match input shape
        mem_pooled = mem_retrieved.mean(dim=1, keepdim=True)  # [B, 1, C]

        # Project memory
        mem_proj = self.mem_proj(mem_pooled)  # [B, 1, C]

        # ===== STEP 2: Enhance Input with Memory =====
        # Add gated memory contribution to input (broadcast across sequence)
        # gate is scalar, starts small (0.1) to not disrupt pre-trained model
        enhanced_input = hidden_states + self.gate * mem_proj

        # ===== STEP 3: Attention on Enhanced Input =====
        # Forward through original layer with memory-enhanced input
        layer_outputs = self.original_layer(
            enhanced_input,
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
            output = layer_outputs[0]
            other_outputs = layer_outputs[1:]
        else:
            output = layer_outputs
            other_outputs = ()

        # ===== STEP 4: Memory Update =====
        # Update with RAW hidden_states (not memory-enhanced or attention output!)
        # Memory should learn to predict transformer patterns, not its own output.
        if self.update_memory:
            new_memory_state = self.memory.update(hidden_states, memory_state)
        else:
            new_memory_state = memory_state

        # Store for external access
        self._updated_memory_state = new_memory_state

        # Compute internal loss if enabled
        if self.config.use_internal_loss and self.training:
            self._internal_loss = self._compute_memory_surprise(hidden_states)

        # Return in HuggingFace-compatible format
        if other_outputs:
            return (output, *other_outputs)
        return output

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
        return {
            "gate_value": self.gate.item(),
            "mean_gate": self.gate.item(),  # For compatibility with diagnostics
        }

    def _compute_memory_surprise(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute memory's own prediction error: ||M(k) - v||^2

        Args:
            hidden_states: Input to memory [B, T, C]

        Returns:
            Scalar loss tensor
        """
        # Get the underlying NeuralMemory (handle CMS case)
        mem = self.memory.memories[0].memory if self._use_cms else self.memory.memory

        # Detach input
        x = hidden_states.detach()

        # Clip for stability
        max_norm = 10.0
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-6)
        x = x / x_norm * torch.clamp(x_norm, max=max_norm)

        # Project to keys and values
        keys = mem.key_proj(x)
        values = mem.value_proj(x)

        keys = torch.clamp(keys, min=-max_norm, max=max_norm)
        values = torch.clamp(values, min=-max_norm, max=max_norm)

        # Get memory's prediction
        predictions = mem.memory_mlp(keys)
        predictions = torch.clamp(predictions, min=-max_norm, max=max_norm)

        # MSE loss
        surprise_loss = F.mse_loss(predictions, values)

        if torch.isnan(surprise_loss) or torch.isinf(surprise_loss):
            return torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        return surprise_loss

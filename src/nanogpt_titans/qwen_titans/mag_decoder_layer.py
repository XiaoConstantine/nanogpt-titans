"""
MAG (Memory as Gate) Titans-enhanced Qwen2 decoder layer.

Implements the MAG variant from the Titans paper where memory and attention
operate in PARALLEL branches combined via multiplicative gating.

Key difference from MAC:
- MAC: output = Attn([memory || input])  (memory as context)
- MAG: output = Attn(input) ⊗ σ(Memory(input))  (memory as gate)

Benefits of MAG:
1. Avoids gate collapse - multiplicative interaction can't be "turned off"
2. Parallel computation - attention and memory run simultaneously
3. Memory MODULATES attention rather than competing with it
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from nanogpt_titans.model import MemoryState
from nanogpt_titans.qwen_titans.config import TitansQwenConfig
from nanogpt_titans.qwen_titans.memory_adapter import (
    ContinuumMemoryState,
    ContinuumMemorySystem,
    NeuralMemoryAdapter,
)


class MAGQwenDecoderLayer(nn.Module):
    """
    MAG (Memory as Gate) enhanced Qwen2 decoder layer.

    Architecture:
        1. Attention and Memory run in PARALLEL on input
        2. Memory output is normalized to [0, 2] range (centered at 1)
        3. Attention output is element-wise multiplied by memory modulation
        4. Memory updated based on attention output

    Key equation: output = attn_output ⊗ memory_gate

    Where memory_gate = 1 + tanh(LayerNorm(Memory(input)))
    - Range [0, 2]: values < 1 suppress, > 1 amplify
    - Centered at 1: default is pass-through (no change)
    - Can't collapse to 0: minimum modulation is ~0

    Example:
        >>> original_layer = model.model.layers[14]
        >>> mag_layer = MAGQwenDecoderLayer(original_layer, 14, config)
        >>> model.model.layers[14] = mag_layer
    """

    def __init__(
        self,
        original_layer: nn.Module,
        layer_idx: int,
        titans_config: TitansQwenConfig,
    ) -> None:
        """
        Initialize MAG-enhanced decoder layer.

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

        # Memory output projection -> modulation signal
        # Projects from memory dim to hidden dim
        self.memory_to_gate = nn.Sequential(
            nn.Linear(titans_config.n_embd, titans_config.n_embd, bias=False),
            nn.LayerNorm(titans_config.n_embd),
        )

        # Learnable scale for modulation strength
        # Starts small (0.1) so initial modulation is subtle
        self.modulation_scale = nn.Parameter(torch.tensor(0.1))

        # Learnable bias to control default modulation center
        # Starts at 0 so default gate is 1 (pass-through)
        self.modulation_bias = nn.Parameter(torch.tensor(0.0))

        # Flag to control memory updates
        self.update_memory = True

        # Store memory state during forward
        self._current_memory_state: Optional[Union[MemoryState, ContinuumMemoryState]] = None
        self._updated_memory_state: Optional[Union[MemoryState, ContinuumMemoryState]] = None

        # Internal loss for self-supervised memory signal
        self._internal_loss: Optional[torch.Tensor] = None

        # Copy attributes that Qwen's internal code expects
        if hasattr(original_layer, "self_attn"):
            self.self_attn = original_layer.self_attn
        if hasattr(original_layer, "mlp"):
            self.mlp = original_layer.mlp
        if hasattr(original_layer, "input_layernorm"):
            self.input_layernorm = original_layer.input_layernorm
        if hasattr(original_layer, "post_attention_layernorm"):
            self.post_attention_layernorm = original_layer.post_attention_layernorm

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
        Forward with MAG-style multiplicative memory gating.

        1. Run attention on input (original layer)
        2. Run memory on input (parallel)
        3. Compute memory gate: gate = 1 + scale * tanh(memory_proj)
        4. Apply multiplicative modulation: output = attn_out * gate
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
        B, T, C = hidden_states.shape
        device = hidden_states.device

        # Use provided state or stored state
        if memory_state is None:
            memory_state = self._current_memory_state

        # Initialize memory state if needed
        if memory_state is None:
            memory_state = self.memory.init_state(B, device)

        # ===== PARALLEL BRANCH 1: Attention =====
        # Forward through original layer (UNCHANGED - preserves pre-trained behavior)
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

        # Extract attention output
        if isinstance(layer_outputs, tuple):
            attn_output = layer_outputs[0]
            other_outputs = layer_outputs[1:]
        else:
            attn_output = layer_outputs
            other_outputs = ()

        # ===== PARALLEL BRANCH 2: Memory =====
        # Retrieve from memory (causal - uses M_{t-1})
        mem_retrieved = self.memory(hidden_states, memory_state)  # [B, num_longterm_mem, C]

        # Pool memory to single vector and broadcast
        mem_pooled = mem_retrieved.mean(dim=1, keepdim=True)  # [B, 1, C]

        # Project to gate signal
        mem_gate_raw = self.memory_to_gate(mem_pooled)  # [B, 1, C]

        # ===== MULTIPLICATIVE GATING =====
        # Gate formula: gate = 1 + scale * tanh(raw + bias)
        # - tanh bounds to [-1, 1]
        # - scale controls modulation strength (starts small)
        # - bias shifts the center
        # - +1 centers at pass-through (gate=1 means no change)
        # Result: gate in range [1-scale, 1+scale], can't collapse to 0

        gate = 1.0 + self.modulation_scale * torch.tanh(mem_gate_raw + self.modulation_bias)

        # Broadcast gate to full sequence
        gate = gate.expand(-1, T, -1)  # [B, T, C]

        # Apply multiplicative modulation
        output = attn_output * gate

        # ===== Memory Update =====
        if self.update_memory:
            new_memory_state = self.memory.update(output, memory_state)
        else:
            new_memory_state = memory_state

        # Store for external access
        self._updated_memory_state = new_memory_state

        # Compute internal loss if enabled
        if self.config.use_internal_loss and self.training:
            self._internal_loss = self._compute_memory_surprise(hidden_states)

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
        """Get modulation statistics for monitoring."""
        return {
            "modulation_scale": self.modulation_scale.item(),
            "modulation_bias": self.modulation_bias.item(),
            # Mean gate is always ~1.0 due to centering, so report scale instead
            "mean_gate": 1.0 + self.modulation_scale.item() * torch.tanh(
                torch.tensor(self.modulation_bias.item())
            ).item(),
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
        if self._use_cms:
            mem = self.memory.memories[0].memory
        else:
            mem = self.memory.memory

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

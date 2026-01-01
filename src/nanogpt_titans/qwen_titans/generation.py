"""
Generation utilities for Titans-enhanced Qwen models.

Provides memory state management and generation functions that properly
handle the test-time learning memory across segments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from nanogpt_titans.qwen_titans.patcher import get_titans_layers

if TYPE_CHECKING:
    from nanogpt_titans.model import MemoryState


class TitansStateManager:
    """
    Manages memory states for Titans-enhanced Qwen models.

    Handles:
    1. Initializing memory states for all Titans layers
    2. Persisting states across forward calls
    3. Resetting states for new sequences
    4. Syncing states between layers and this manager

    Example:
        >>> manager = TitansStateManager(model)
        >>> manager.init_states(batch_size=2, device=torch.device('cuda'))
        >>> # Forward passes will use and update states
        >>> output = model(input_ids)
        >>> manager.sync_from_layers()  # Get updated states
        >>> manager.reset()  # For new sequence
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize state manager for a Titans-enhanced model.

        Args:
            model: Model patched with patch_qwen_with_titans
        """
        self.model = model
        self.titans_layers = get_titans_layers(model)
        self._states: dict[int, MemoryState] = {}

        if not self.titans_layers:
            print("Warning: No Titans layers found in model")

    def init_states(self, batch_size: int, device: torch.device) -> None:
        """
        Initialize fresh memory states for all Titans layers.

        Call this at the start of processing a new batch of sequences.

        Args:
            batch_size: Number of sequences in batch
            device: Device to create tensors on
        """
        for layer in self.titans_layers:
            state = layer.memory.init_state(batch_size, device)
            self._states[layer.layer_idx] = state
            layer.set_memory_state(state)

    def sync_to_layers(self) -> None:
        """Push current states to Titans layers before forward pass."""
        for layer in self.titans_layers:
            if layer.layer_idx in self._states:
                layer.set_memory_state(self._states[layer.layer_idx])

    def sync_from_layers(self) -> None:
        """Pull updated states from Titans layers after forward pass."""
        for layer in self.titans_layers:
            state = layer.get_memory_state()
            if state is not None:
                self._states[layer.layer_idx] = state

    def reset(self) -> None:
        """Reset all memory states (call before new sequence)."""
        self._states.clear()
        for layer in self.titans_layers:
            layer.set_memory_state(None)

    def get_state(self, layer_idx: int) -> MemoryState | None:
        """Get current memory state for a specific layer."""
        return self._states.get(layer_idx)

    def set_state(self, layer_idx: int, state: MemoryState) -> None:
        """Set memory state for a specific layer."""
        self._states[layer_idx] = state

    def get_all_states(self) -> dict[int, MemoryState]:
        """Get all memory states (for checkpointing)."""
        return self._states.copy()

    def load_states(self, states: dict[int, MemoryState]) -> None:
        """Load memory states (from checkpoint)."""
        self._states = states.copy()
        self.sync_to_layers()


def titans_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    reset_memory: bool = True,
    state_manager: TitansStateManager | None = None,
    segment_len: int | None = None,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Generate text with Titans memory state management.

    Unlike standard HuggingFace generate(), this function properly handles
    memory states across segments and autoregressive generation steps.

    Args:
        model: Titans-enhanced Qwen model
        input_ids: Input token IDs [B, T]
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature (1.0 = no change)
        top_k: Top-k sampling (None = disabled)
        top_p: Nucleus sampling threshold (None = disabled)
        reset_memory: Reset memory at start (True for new sequences)
        state_manager: Existing state manager (creates new one if None)
        segment_len: Segment length for memory (uses config default if None)
        pad_token_id: Padding token ID (for batch generation)
        eos_token_id: End-of-sequence token ID (stops generation)

    Returns:
        Generated token IDs including input [B, T + max_new_tokens]

    Example:
        >>> model = patch_qwen_with_titans(model, config)
        >>> input_ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids
        >>> output_ids = titans_generate(model, input_ids, max_new_tokens=50)
        >>> text = tokenizer.decode(output_ids[0])
    """
    B = input_ids.size(0)
    device = input_ids.device

    # Get or create state manager
    if state_manager is None:
        state_manager = TitansStateManager(model)

    # Get segment length from config
    if segment_len is None:
        if hasattr(model, "_titans_config"):
            segment_len = model._titans_config.segment_len
        else:
            segment_len = 512  # Default

    # Reset memory for new sequences
    if reset_memory:
        state_manager.reset()
        state_manager.init_states(B, device)
    else:
        state_manager.sync_to_layers()

    # Process input in segments to build up memory
    input_len = input_ids.size(1)

    with torch.no_grad():
        # Process input segments
        for start in range(0, input_len, segment_len):
            end = min(start + segment_len, input_len)
            segment = input_ids[:, start:end]

            # Forward pass updates memory states internally
            state_manager.sync_to_layers()
            _ = model(segment, use_cache=False)
            state_manager.sync_from_layers()

    # Autoregressive generation
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Use last segment_len tokens as context
            context = generated[:, -segment_len:] if generated.size(1) > segment_len else generated

            # Forward with memory
            state_manager.sync_to_layers()
            outputs = model(context, use_cache=False)
            state_manager.sync_from_layers()

            # Get logits for last position
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter back to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    return generated


def process_with_memory(
    model: nn.Module,
    input_ids: torch.Tensor,
    state_manager: TitansStateManager | None = None,
    segment_len: int | None = None,
    return_hidden_states: bool = False,
) -> tuple:
    """
    Process input sequence with memory, returning logits and optionally hidden states.

    Useful for training or evaluation where you need the full logits
    over the input sequence.

    Args:
        model: Titans-enhanced Qwen model
        input_ids: Input token IDs [B, T]
        state_manager: Existing state manager (creates new one if None)
        segment_len: Segment length (uses config default if None)
        return_hidden_states: Whether to return hidden states

    Returns:
        Tuple of (logits, state_manager, [hidden_states])
        - logits: [B, T, vocab_size]
        - state_manager: Updated state manager
        - hidden_states: Optional list of hidden states per segment
    """
    B = input_ids.size(0)
    T = input_ids.size(1)
    device = input_ids.device

    # Get or create state manager
    if state_manager is None:
        state_manager = TitansStateManager(model)
        state_manager.init_states(B, device)

    # Get segment length
    if segment_len is None:
        segment_len = model._titans_config.segment_len if hasattr(model, "_titans_config") else 512

    all_logits = []
    all_hidden_states = [] if return_hidden_states else None

    # Process in segments
    for start in range(0, T, segment_len):
        end = min(start + segment_len, T)
        segment = input_ids[:, start:end]

        # Sync states to layers
        state_manager.sync_to_layers()

        # Forward pass
        outputs = model(
            segment,
            use_cache=False,
            output_hidden_states=return_hidden_states,
        )

        # Sync states from layers
        state_manager.sync_from_layers()

        # Collect outputs
        all_logits.append(outputs.logits)
        if return_hidden_states and hasattr(outputs, "hidden_states"):
            all_hidden_states.append(outputs.hidden_states)

    # Concatenate logits
    logits = torch.cat(all_logits, dim=1)  # [B, T, vocab_size]

    if return_hidden_states:
        return logits, state_manager, all_hidden_states
    return logits, state_manager

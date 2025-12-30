"""
MLX TITANS training utilities.

Provides training loop components for MLX backend:
- CombinedModel: Wrapper combining base model with TITANS layer
- Training utilities: loss functions, gradient filtering, LR scheduling
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

from nanogpt_titans.mlx.decoder_layer import MLXTitansLayer, TitansLayerState

if TYPE_CHECKING:
    from nanogpt_titans.mlx.config import MLXTitansConfig


class CombinedModel(nn.Module):
    """
    Combined model wrapper that includes both base model and TITANS layers.

    This wrapper ensures gradients flow through the entire computation graph,
    matching PyTorch's behavior where gradients propagate through frozen layers.

    Supports multiple TITANS layers at different positions in the transformer.
    """

    def __init__(
        self,
        base_model,
        titans_layers: dict[int, MLXTitansLayer],
        use_internal_loss: bool = False,
        internal_loss_weight: float = 1e-4,
    ):
        """
        Initialize combined model with multiple TITANS layers.

        Args:
            base_model: The base language model
            titans_layers: Dict mapping layer_idx -> MLXTitansLayer
            use_internal_loss: Whether to compute internal loss
            internal_loss_weight: Weight for internal loss
        """
        super().__init__()
        self.base_model = base_model
        self.titans_layers = titans_layers
        self.memory_layer_indices = sorted(titans_layers.keys())
        self.use_internal_loss = use_internal_loss
        self.internal_loss_weight = internal_loss_weight

        # Get inner model reference
        self._inner_model = base_model.model if hasattr(base_model, "model") else base_model

        # Memory states per layer (initialized on first forward)
        self._memory_states: dict[int, TitansLayerState | None] = dict.fromkeys(self.memory_layer_indices)

        # Store internal loss from last forward (for logging)
        self._last_internal_loss: mx.array | None = None

    def init_memory_state(self, batch_size: int):
        """Initialize memory state for all TITANS layers."""
        for idx in self.memory_layer_indices:
            self._memory_states[idx] = self.titans_layers[idx].init_state(batch_size)

    def reset_memory_state(self):
        """Reset memory state to None (will be re-initialized on next forward)."""
        self._memory_states = dict.fromkeys(self.memory_layer_indices)

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass through base model with TITANS integration and memory updates."""
        B = input_ids.shape[0]

        # Initialize memory states if needed
        for idx in self.memory_layer_indices:
            if self._memory_states[idx] is None:
                self._memory_states[idx] = self.titans_layers[idx].init_state(B)

        # Get embeddings
        h = self._inner_model.embed_tokens(input_ids)

        # Create attention mask
        T = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        mask = mask.astype(h.dtype)

        # Track hidden states and states for internal loss
        h_at_memory_layers: dict[int, mx.array] = {}
        states_before_update: dict[int, TitansLayerState] = {}

        # Forward through transformer layers with TITANS integration
        for i, layer in enumerate(self._inner_model.layers):
            h = layer(h, mask=mask, cache=None)

            # Apply TITANS at specified layers (with memory state update)
            if i in self.memory_layer_indices:
                # Store hidden states and state BEFORE memory update for internal loss
                if self.use_internal_loss:
                    h_at_memory_layers[i] = h
                    states_before_update[i] = self._memory_states[i]
                h, self._memory_states[i] = self.titans_layers[i](h, self._memory_states[i])

        # Compute internal loss if enabled (sum across all layers)
        if self.use_internal_loss and h_at_memory_layers:
            internal_losses = []
            for idx in self.memory_layer_indices:
                if idx in h_at_memory_layers and idx in states_before_update:
                    loss = self.titans_layers[idx].compute_internal_loss(
                        h_at_memory_layers[idx], states_before_update[idx]
                    )
                    internal_losses.append(loss)
            if internal_losses:
                self._last_internal_loss = mx.mean(mx.stack(internal_losses))
            else:
                self._last_internal_loss = None
        else:
            self._last_internal_loss = None

        # Final layer norm
        h = self._inner_model.norm(h)

        # Project to vocabulary
        if hasattr(self.base_model, "lm_head"):
            logits = self.base_model.lm_head(h)
        elif hasattr(self._inner_model, "lm_head"):
            logits = self._inner_model.lm_head(h)
        else:
            logits = h @ self._inner_model.embed_tokens.weight.T

        return logits

    def get_internal_loss(self) -> mx.array | None:
        """Get the internal loss from the last forward pass."""
        return self._last_internal_loss

    # Convenience properties for single-layer backward compatibility
    @property
    def titans_layer(self) -> MLXTitansLayer:
        """Get the first TITANS layer (for backward compatibility)."""
        return self.titans_layers[self.memory_layer_indices[0]]

    def get_layer_stats(self) -> dict[int, dict[str, float]]:
        """Get stats for all TITANS layers."""
        stats = {}
        for idx in self.memory_layer_indices:
            layer = self.titans_layers[idx]
            stats[idx] = {
                "mem_scale": float(mx.sigmoid(layer.mem_scale).item()),
                "gate_bias": float(layer.gate.linear2.bias.item()),
            }
        return stats


def compute_gate_regularization(titans_layer: MLXTitansLayer, min_value: float = 0.15) -> mx.array:
    """
    Compute regularization loss to prevent gate from collapsing below min_value.

    Matches PyTorch compute_gate_regularization() exactly.
    Penalty: max(0, min_value - gate)^2

    Args:
        titans_layer: The TITANS layer containing the gate
        min_value: Minimum gate value to maintain

    Returns:
        Regularization loss (scalar)
    """
    # Get gate bias and compute expected gate value
    gate_bias = titans_layer.gate.linear2.bias[0]
    expected_gate = mx.sigmoid(gate_bias)

    # Soft penalty: max(0, min_value - gate)^2
    diff = min_value - expected_gate
    penalty = mx.maximum(diff, mx.array(0.0)) ** 2

    return penalty


def compute_multi_layer_gate_regularization(
    titans_layers: dict[int, MLXTitansLayer], min_value: float = 0.15
) -> mx.array:
    """
    Compute gate regularization for multiple TITANS layers.

    Args:
        titans_layers: Dict mapping layer_idx -> MLXTitansLayer
        min_value: Minimum gate value to maintain

    Returns:
        Mean regularization loss across all layers
    """
    if not titans_layers:
        return mx.array(0.0)

    penalties = []
    for layer in titans_layers.values():
        penalties.append(compute_gate_regularization(layer, min_value))

    return mx.mean(mx.stack(penalties))


def create_loss_fn(
    _combined_model: CombinedModel, gate_min_value: float = 0.0, gate_reg_weight: float = 0.0
):
    """Create a loss function for the combined model."""

    def loss_fn(combined_model: CombinedModel, input_ids: mx.array, target_ids: mx.array):
        logits = combined_model(input_ids)

        # Language modeling loss
        _B, _T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = target_ids.reshape(-1)
        lm_loss = mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))

        # Add internal loss if enabled
        total_loss = lm_loss
        if combined_model.use_internal_loss:
            internal_loss = combined_model.get_internal_loss()
            if internal_loss is not None:
                total_loss = lm_loss + combined_model.internal_loss_weight * internal_loss

        # Add gate regularization if enabled (prevents gate from collapsing)
        if gate_min_value > 0 and gate_reg_weight > 0:
            gate_reg = compute_multi_layer_gate_regularization(
                combined_model.titans_layers, gate_min_value
            )
            total_loss = total_loss + gate_reg_weight * gate_reg

        return total_loss

    return loss_fn


def filter_titans_grads(grads: dict[str, Any]) -> dict[str, Any]:
    """
    Filter gradients to only include TITANS layer gradients.

    OPTIMIZATION: Direct key lookup instead of string matching when possible.

    Args:
        grads: Full gradient dict from value_and_grad

    Returns:
        Filtered dict with only titans_layer gradients
    """
    # Fast path: direct key lookup
    if "titans_layers" in grads:
        return {"titans_layers": grads["titans_layers"]}
    if "titans_layer" in grads:
        return {"titans_layer": grads["titans_layer"]}

    # Fallback: filter by keyword (slower)
    filtered = {}
    for key, value in grads.items():
        key_lower = key.lower()
        if "titans" in key_lower or "memory" in key_lower or "gate" in key_lower:
            filtered[key] = value
    return filtered if filtered else grads


def get_lr(step: int, config: MLXTitansConfig, use_linear: bool = False) -> float:
    """
    Learning rate schedule: Linear warmup followed by cosine decay.

    Args:
        step: Current training step
        config: Training configuration
        use_linear: If True, use linear warmup only

    Returns:
        Current learning rate
    """
    warmup_steps = max(config.warmup_steps, 1)

    if step < warmup_steps:
        return config.learning_rate * step / warmup_steps

    if use_linear:
        return config.learning_rate
    else:
        progress = (step - warmup_steps) / max(1, config.max_steps - warmup_steps)
        return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def create_masked_grads(
    grads: dict[str, Any], keep_gate_scale: bool, path: str = "", freeze_gate: bool = False
) -> dict[str, Any]:
    """
    Create grads with zeros for non-target params.

    Args:
        grads: Gradient dictionary
        keep_gate_scale: If True, keep gate/scale grads; else keep memory grads
        path: Current path in gradient tree (for recursion)
        freeze_gate: If True, zero out gate gradients (for warmup)

    Returns:
        Masked gradient dictionary
    """
    if isinstance(grads, dict):
        return {
            k: create_masked_grads(v, keep_gate_scale, f"{path}.{k}" if path else k, freeze_gate)
            for k, v in grads.items()
        }
    elif isinstance(grads, mx.array):
        is_gate = "gate" in path
        is_scale_adaptive = (
            "mem_scale" in path
            or "mem_ln" in path
            or "level_weights" in path
            or "to_lr" in path
            or "to_momentum" in path
            or "to_decay" in path
        )
        is_gate_scale = is_gate or is_scale_adaptive

        # During gate warmup, freeze gate params
        if freeze_gate and is_gate:
            return mx.zeros_like(grads)

        if keep_gate_scale:
            return grads if is_gate_scale else mx.zeros_like(grads)
        else:
            return mx.zeros_like(grads) if is_gate_scale else grads
    else:
        return grads


def scale_grads_recursive(grad_tree: Any, factor: float) -> Any:
    """Recursively scale gradients by a constant factor."""
    if isinstance(grad_tree, dict):
        return {k: scale_grads_recursive(v, factor) for k, v in grad_tree.items()}
    elif isinstance(grad_tree, mx.array):
        return grad_tree * factor
    else:
        return grad_tree


def accumulate_grads(accum_grads: dict | None, new_grads: dict) -> dict:
    """Add new gradients to accumulated gradients."""
    if accum_grads is None:
        return new_grads

    def add_grads(a, b):
        if isinstance(a, dict):
            return {k: add_grads(a[k], b[k]) for k in a}
        elif isinstance(a, mx.array):
            return a + b
        else:
            return a

    return add_grads(accum_grads, new_grads)


def create_titans_layer_from_model(model, config: MLXTitansConfig) -> MLXTitansLayer:
    """Create a TITANS layer matching model dimensions."""
    # Get hidden size from model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        sample_layer = model.model.layers[0]
        if hasattr(sample_layer, "hidden_size"):
            dim = sample_layer.hidden_size
        elif hasattr(sample_layer.self_attn, "hidden_size"):
            dim = sample_layer.self_attn.hidden_size
        else:
            dim = sample_layer.self_attn.q_proj.weight.shape[0]
    else:
        dim = 896  # Default for Qwen2-0.5B

    return MLXTitansLayer(
        dim=dim,
        use_cms=config.use_cms,
        num_cms_levels=config.num_cms_levels,
        cms_update_frequencies=config.cms_update_frequencies,
        memory_depth=config.memory_depth,
        memory_expansion=config.memory_expansion,
        adaptive_memory=config.adaptive_memory,
        memory_lr_max=config.memory_lr_max,
        gate_init_bias=config.gate_init_bias,
    )

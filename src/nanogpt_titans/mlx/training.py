"""
MLX TITANS training utilities.

Provides training loop components for MLX backend:
- CombinedModel: Wrapper combining base model with TITANS layer
- Training utilities: loss functions, gradient filtering, LR scheduling
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from nanogpt_titans.mlx.config import MLXTitansConfig
from nanogpt_titans.mlx.decoder_layer import MLXTitansLayer, TitansLayerState


class CombinedModel(nn.Module):
    """
    Combined model wrapper that includes both base model and TITANS layer.

    This wrapper ensures gradients flow through the entire computation graph,
    matching PyTorch's behavior where gradients propagate through frozen layers.
    """

    def __init__(
        self,
        base_model,
        titans_layer: MLXTitansLayer,
        memory_layer_idx: int
    ):
        super().__init__()
        self.base_model = base_model
        self.titans_layer = titans_layer
        self.memory_layer_idx = memory_layer_idx

        # Get inner model reference
        self._inner_model = base_model.model if hasattr(base_model, 'model') else base_model

        # Memory state (initialized on first forward)
        self._memory_state: Optional[TitansLayerState] = None

    def init_memory_state(self, batch_size: int):
        """Initialize memory state for the TITANS layer."""
        self._memory_state = self.titans_layer.init_state(batch_size)

    def reset_memory_state(self):
        """Reset memory state to None (will be re-initialized on next forward)."""
        self._memory_state = None

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Forward pass through base model with TITANS integration and memory updates."""
        B = input_ids.shape[0]

        # Initialize memory state if needed
        if self._memory_state is None:
            self._memory_state = self.titans_layer.init_state(B)

        # Get embeddings
        h = self._inner_model.embed_tokens(input_ids)

        # Create attention mask
        T = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        mask = mask.astype(h.dtype)

        # Forward through transformer layers with TITANS integration
        for i, layer in enumerate(self._inner_model.layers):
            h = layer(h, mask=mask, cache=None)

            # Apply TITANS at the specified layer (with memory state update)
            if i == self.memory_layer_idx:
                h, self._memory_state = self.titans_layer(h, self._memory_state)

        # Final layer norm
        h = self._inner_model.norm(h)

        # Project to vocabulary
        if hasattr(self.base_model, 'lm_head'):
            logits = self.base_model.lm_head(h)
        elif hasattr(self._inner_model, 'lm_head'):
            logits = self._inner_model.lm_head(h)
        else:
            logits = h @ self._inner_model.embed_tokens.weight.T

        return logits


def create_loss_fn(combined_model: CombinedModel):
    """Create a loss function for the combined model."""
    def loss_fn(combined_model, input_ids: mx.array, target_ids: mx.array):
        logits = combined_model(input_ids)

        # Language modeling loss
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        targets_flat = target_ids.reshape(-1)
        loss = mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))

        return loss

    return loss_fn


def filter_titans_grads(grads: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter gradients to only include TITANS layer gradients.

    Args:
        grads: Full gradient dict from value_and_grad

    Returns:
        Filtered dict with only titans_layer gradients
    """
    if 'titans_layer' in grads:
        return {'titans_layer': grads['titans_layer']}
    else:
        filtered = {}
        for key, value in grads.items():
            if 'titans' in key.lower() or 'memory' in key.lower() or 'gate' in key.lower():
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
    grads: Dict[str, Any],
    keep_gate_scale: bool,
    path: str = "",
    freeze_gate: bool = False
) -> Dict[str, Any]:
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
        is_gate = 'gate' in path
        is_scale_adaptive = ('mem_scale' in path or 'mem_ln' in path or
                            'level_weights' in path or 'to_lr' in path or
                            'to_momentum' in path or 'to_decay' in path)
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


def accumulate_grads(accum_grads: Optional[Dict], new_grads: Dict) -> Dict:
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
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        sample_layer = model.model.layers[0]
        if hasattr(sample_layer, 'hidden_size'):
            dim = sample_layer.hidden_size
        elif hasattr(sample_layer.self_attn, 'hidden_size'):
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

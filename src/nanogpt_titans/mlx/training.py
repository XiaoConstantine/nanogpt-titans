"""
MLX TITANS training utilities.

Provides training loop components for MLX backend:
- CombinedModel: Wrapper combining base model with TITANS layer
- Training utilities: loss functions, gradient filtering, LR scheduling
- Teach signal: auxiliary gradient from logit residuals (from nested_learning)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

from nanogpt_titans.mlx.decoder_layer import MLXTitansLayer, TitansLayerState

if TYPE_CHECKING:
    from nanogpt_titans.mlx.config import MLXTitansConfig


def compute_teach_signal(logits: mx.array, targets: mx.array, lm_head_weight: mx.array) -> mx.array:
    """Compute teaching signal: (softmax(logits) - one_hot(targets)) @ lm_head_weight."""
    B, T, V = logits.shape

    # Softmax to get probabilities
    probs = mx.softmax(logits, axis=-1)  # [B, T, V]

    # Create one-hot targets
    # MLX doesn't have one_hot, so we construct it manually
    one_hot = mx.zeros((B, T, V))
    # Use scatter-like operation via indexing
    batch_idx = mx.arange(B)[:, None].astype(mx.int32)  # [B, 1]
    seq_idx = mx.arange(T)[None, :].astype(mx.int32)  # [1, T]
    batch_idx = mx.broadcast_to(batch_idx, (B, T))  # [B, T]
    seq_idx = mx.broadcast_to(seq_idx, (B, T))  # [B, T]

    # Create one-hot by setting the target positions to 1
    # This is equivalent to F.one_hot in PyTorch
    targets_int = targets.astype(mx.int32)
    one_hot = mx.zeros((B * T, V))
    flat_indices = mx.arange(B * T)
    flat_targets = targets_int.reshape(-1)
    # Use scatter: one_hot[flat_indices, flat_targets] = 1
    one_hot = one_hot.at[flat_indices, flat_targets].add(mx.ones(B * T))
    one_hot = one_hot.reshape(B, T, V)

    # Compute residual: what the model got wrong
    residual = probs - one_hot  # [B, T, V]

    # Project back to hidden dimension
    # residual @ lm_head_weight gives [B, T, H]
    # Note: lm_head_weight is typically [V, H] (output embedding)
    teach_signal = mx.matmul(residual, lm_head_weight)  # [B, T, H]

    return teach_signal


def apply_teach_signal_to_memory(
    teach_signal: mx.array,
    _titans_layer: MLXTitansLayer,
    state: TitansLayerState,
    signal_weight: float = 0.1,
) -> TitansLayerState:
    """Apply teaching signal to memory weights (currently returns state unchanged)."""
    # Scale down the teach signal (kept for future implementation)
    _ = teach_signal * signal_weight

    # For now, we incorporate the teach signal into the memory via its update path
    # The memory will see the teach signal as part of the input during its next update
    # This is a simplified integration - the full nested_learning approach modifies
    # the memory weights directly using the projected teach signal

    # The proper integration would be:
    # 1. Project teach_signal through memory's key/value projections
    # 2. Compute gradient w.r.t. memory weights
    # 3. Apply additional weight update

    # For this implementation, we return the state unchanged and let the
    # loss function handle teach signal integration through compute_internal_loss_with_teach

    return state


class CombinedModel(nn.Module):
    """Wrapper combining base model with TITANS layers for gradient flow."""

    def __init__(
        self,
        base_model,
        titans_layers: dict[int, MLXTitansLayer],
        use_internal_loss: bool = False,
        internal_loss_weight: float = 1e-4,
        use_teach_signal: bool = False,
        teach_signal_weight: float = 0.1,
    ):
        super().__init__()
        self.base_model = base_model
        self.use_internal_loss = use_internal_loss
        self.internal_loss_weight = internal_loss_weight
        self.use_teach_signal = use_teach_signal
        self.teach_signal_weight = teach_signal_weight

        # Get inner model reference
        self._inner_model = base_model.model if hasattr(base_model, "model") else base_model

        # Independent memory mode: separate TITANS layer per position
        self.titans_layers = titans_layers
        self.memory_layer_indices = sorted(titans_layers.keys())
        # Memory states per layer (initialized on first forward)
        self._memory_states: dict[int, TitansLayerState | None] = dict.fromkeys(self.memory_layer_indices)

        # Store internal loss from last forward (for logging)
        self._last_internal_loss: mx.array | None = None

        # Store teach signal from last forward (for loss computation)
        self._last_teach_signal: mx.array | None = None
        self._last_h_at_memory_layers: dict[int, mx.array] = {}

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
                # Independent memory mode: separate TITANS layer per position
                if self.use_internal_loss or self.use_teach_signal:
                    h_at_memory_layers[i] = h
                    states_before_update[i] = self._memory_states[i]
                h, self._memory_states[i] = self.titans_layers[i](h, self._memory_states[i])

        # Store for teach signal computation
        self._last_h_at_memory_layers = h_at_memory_layers

        # Compute internal loss if enabled (sum across all layers)
        if self.use_internal_loss and h_at_memory_layers:
            internal_losses = []
            # Independent mode: compute internal loss per TITANS layer
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
            lm_head_weight = self.base_model.lm_head.weight
        elif hasattr(self._inner_model, "lm_head"):
            logits = self._inner_model.lm_head(h)
            lm_head_weight = self._inner_model.lm_head.weight
        else:
            logits = h @ self._inner_model.embed_tokens.weight.T
            lm_head_weight = self._inner_model.embed_tokens.weight

        # Store lm_head_weight for teach signal computation
        self._lm_head_weight = lm_head_weight

        return logits

    def compute_teach_signal_loss(self, logits: mx.array, targets: mx.array) -> mx.array | None:
        """Compute teaching signal loss from logit residuals."""
        if not self.use_teach_signal or not self._last_h_at_memory_layers:
            return None

        # Compute teach signal: (softmax(logits) - one_hot(targets)) @ lm_head_weight
        teach_signal = compute_teach_signal(logits, targets, self._lm_head_weight)

        # Compute MSE between teach signal and hidden states at memory layers
        # This encourages the memory to predict what will surprise the model
        teach_losses = []
        for _idx, h in self._last_h_at_memory_layers.items():
            # teach_signal is [B, T, H], h is [B, T, H]
            # We want memory to learn to anticipate surprises
            diff = teach_signal - h
            teach_loss = mx.mean(diff * diff)
            teach_losses.append(teach_loss)

        if teach_losses:
            return mx.mean(mx.stack(teach_losses)) * self.teach_signal_weight
        return None

    def get_internal_loss(self) -> mx.array | None:
        """Get the internal loss from the last forward pass."""
        return self._last_internal_loss

    # Convenience properties for single-layer backward compatibility
    @property
    def titans_layer(self) -> MLXTitansLayer | None:
        """Get the first TITANS layer (for backward compatibility)."""
        return self.titans_layers[self.memory_layer_indices[0]]

    def get_layer_stats(self) -> dict[int, dict[str, float]]:
        """Get stats for all TITANS layers."""
        stats = {}
        # Independent mode: get stats from TITANS layers
        for idx in self.memory_layer_indices:
            layer = self.titans_layers[idx]
            stats[idx] = {
                "mem_scale": float(mx.sigmoid(layer.mem_scale).item()),
                "gate_bias": float(layer.gate.linear2.bias.item()),
            }
        return stats


def compute_gate_regularization(titans_layer: MLXTitansLayer, min_value: float = 0.15) -> mx.array:
    """Compute regularization loss: max(0, min_value - gate)^2."""
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
    """Compute mean gate regularization across all TITANS layers."""
    if not titans_layers:
        return mx.array(0.0)

    penalties = []
    for layer in titans_layers.values():
        penalties.append(compute_gate_regularization(layer, min_value))

    return mx.mean(mx.stack(penalties))


def create_loss_fn(_combined_model: CombinedModel, gate_min_value: float = 0.0, gate_reg_weight: float = 0.0):
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

        # Add teach signal loss if enabled (from nested_learning)
        if combined_model.use_teach_signal:
            teach_loss = combined_model.compute_teach_signal_loss(logits, target_ids)
            if teach_loss is not None:
                total_loss = total_loss + teach_loss

        # Add gate regularization if enabled (prevents gate from collapsing)
        if gate_min_value > 0 and gate_reg_weight > 0:
            gate_reg = compute_multi_layer_gate_regularization(combined_model.titans_layers, gate_min_value)
            total_loss = total_loss + gate_reg_weight * gate_reg

        return total_loss

    return loss_fn


def filter_titans_grads(grads: dict[str, Any]) -> dict[str, Any]:
    """Filter gradients to only include TITANS layer gradients."""
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
    """Learning rate schedule: linear warmup + cosine decay."""
    warmup_steps = max(config.warmup_steps, 1)

    if step < warmup_steps:
        return config.learning_rate * step / warmup_steps

    if use_linear:
        return config.learning_rate
    progress = (step - warmup_steps) / max(1, config.max_steps - warmup_steps)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def create_masked_grads(
    grads: dict[str, Any], keep_gate_scale: bool, path: str = "", freeze_gate: bool = False
) -> dict[str, Any]:
    """Create grads with zeros for non-target params (gate/scale vs memory)."""
    if isinstance(grads, dict):
        return {
            k: create_masked_grads(v, keep_gate_scale, f"{path}.{k}" if path else k, freeze_gate)
            for k, v in grads.items()
        }
    if isinstance(grads, mx.array):
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
        return mx.zeros_like(grads) if is_gate_scale else grads
    return grads


def scale_grads_recursive(grad_tree: Any, factor: float) -> Any:
    """Recursively scale gradients by a constant factor."""
    if isinstance(grad_tree, dict):
        return {k: scale_grads_recursive(v, factor) for k, v in grad_tree.items()}
    if isinstance(grad_tree, mx.array):
        return grad_tree * factor
    return grad_tree


def accumulate_grads(accum_grads: dict | None, new_grads: dict) -> dict:
    """Add new gradients to accumulated gradients."""
    if accum_grads is None:
        return new_grads

    def add_grads(a, b):
        if isinstance(a, dict):
            return {k: add_grads(a[k], b[k]) for k in a}
        if isinstance(a, mx.array):
            return a + b
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
        grad_clip=config.memory_grad_clip,
        surprise_threshold=config.surprise_threshold,
        use_cascade=config.use_cascade,
    )


def online_eval(
    combined_model: CombinedModel,
    input_ids: mx.array,
    chunk_size: int = 128,
    return_all_logits: bool = False,
) -> tuple[mx.array, list[dict]]:
    """Online evaluation with chunked memorization - updates memory after each chunk."""
    B, T = input_ids.shape
    chunk_metrics = []
    all_logits = [] if return_all_logits else None

    # Initialize memory state
    combined_model.init_memory_state(B)

    # Process sequence in expanding chunks
    for chunk_end in range(chunk_size, T + 1, chunk_size):
        # Get current chunk (with full history)
        chunk_ids = input_ids[:, :chunk_end]

        # Forward pass through combined model
        logits = combined_model(chunk_ids)

        # Collect metrics for this chunk
        layer_stats = combined_model.get_layer_stats()
        chunk_metrics.append(
            {
                "chunk_end": chunk_end,
                "layer_stats": layer_stats,
            }
        )

        # Store logits if requested
        if return_all_logits:
            all_logits.append(logits)

        # Force evaluation to update memory state
        mx.eval(logits)

    # Handle remaining tokens if T is not divisible by chunk_size
    remaining = T % chunk_size
    if remaining > 0 and chunk_size < T:
        logits = combined_model(input_ids)
        mx.eval(logits)

        layer_stats = combined_model.get_layer_stats()
        chunk_metrics.append(
            {
                "chunk_end": T,
                "layer_stats": layer_stats,
            }
        )

        if return_all_logits:
            all_logits.append(logits)

    # Final forward pass on full sequence (with trained memory)
    final_logits = combined_model(input_ids)

    if return_all_logits:
        return all_logits, chunk_metrics
    return final_logits, chunk_metrics


def online_generate(
    combined_model: CombinedModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    chunk_size: int = 128,
) -> tuple[str, list[dict]]:
    """Generate text token by token, updating memory as the sequence grows."""
    # Encode prompt
    input_ids = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    B, _T = input_ids.shape

    # Initialize memory
    combined_model.init_memory_state(B)

    generation_metrics = []
    generated_tokens = []

    for i in range(max_new_tokens):
        # Forward pass
        logits = combined_model(input_ids)

        # Get next token logits
        next_logits = logits[:, -1, :]  # [B, V]

        # Apply temperature
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-p sampling
        probs = mx.softmax(next_logits, axis=-1)
        sorted_probs = mx.sort(probs, axis=-1)[:, ::-1]
        cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

        # Create mask for top-p
        mask = cumsum_probs <= top_p
        # Ensure at least one token is selected
        mask = mask.at[:, 0].set(True)

        # Sample from filtered distribution
        sorted_indices = mx.argsort(probs, axis=-1)[:, ::-1]
        filtered_probs = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / mx.sum(filtered_probs, axis=-1, keepdims=True)

        # Multinomial sample (simplified - just argmax for deterministic)
        if temperature == 0:
            next_token = mx.argmax(probs, axis=-1, keepdims=True)
        else:
            # Simple sampling via cumsum trick
            u = mx.random.uniform(shape=(B, 1))
            cumsum = mx.cumsum(filtered_probs, axis=-1)
            next_idx = mx.argmax((cumsum >= u).astype(mx.int32), axis=-1, keepdims=True)
            next_token = mx.take_along_axis(sorted_indices, next_idx, axis=-1)

        # Append token
        generated_tokens.append(int(next_token[0, 0].item()))
        input_ids = mx.concatenate([input_ids, next_token], axis=1)

        # Update memory periodically
        if (i + 1) % chunk_size == 0:
            layer_stats = combined_model.get_layer_stats()
            generation_metrics.append(
                {
                    "token_idx": i + 1,
                    "layer_stats": layer_stats,
                }
            )

        mx.eval(input_ids)

        # Check for EOS
        if hasattr(tokenizer, "eos_token_id") and next_token[0, 0].item() == tokenizer.eos_token_id:
            break

    # Decode generated text
    all_token_ids = [int(t.item()) for t in input_ids[0]]
    generated_text = tokenizer.decode(all_token_ids)

    return generated_text, generation_metrics

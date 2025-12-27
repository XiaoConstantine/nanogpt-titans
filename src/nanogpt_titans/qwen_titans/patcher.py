"""
Model patching utilities for Titans-Qwen integration.

Provides functions to patch a HuggingFace Qwen2 model with Titans memory layers
using gated residual integration (HOPE-style).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from nanogpt_titans.qwen_titans.config import TitansQwenConfig
from nanogpt_titans.qwen_titans.decoder_layer import TitansQwenDecoderLayer


def patch_qwen_with_titans(
    model: nn.Module,
    titans_config: Optional[TitansQwenConfig] = None,
    layer_indices: Optional[list[int]] = None,
) -> nn.Module:
    """
    Patch a Qwen2 model with Titans memory layers.

    Replaces specified decoder layers with Titans-enhanced layers
    that add test-time learning memory via gated residual integration.

    Args:
        model: Pre-trained Qwen2ForCausalLM or similar model
        titans_config: Titans configuration (if None, creates from model config)
        layer_indices: Which layers to enhance with memory (default: middle layer)

    Returns:
        Modified model with Titans layers (modifies in-place and returns)

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B")
        >>> config = TitansQwenConfig.from_qwen_config(model.config)
        >>> model = patch_qwen_with_titans(model, config, layer_indices=[14])
    """
    # Create config if not provided
    if titans_config is None:
        titans_config = TitansQwenConfig.from_qwen_config(model.config)

    # Use config's memory_layers if layer_indices not specified
    if layer_indices is None:
        layer_indices = titans_config.memory_layers

    # Access the decoder layers
    # Handle different model structures (Qwen2ForCausalLM, Qwen2Model, etc.)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError(
            f"Cannot find decoder layers in model of type {type(model)}. "
            "Expected model.model.layers or model.layers"
        )

    # Validate layer indices
    num_layers = len(layers)
    for idx in layer_indices:
        if idx < 0 or idx >= num_layers:
            raise ValueError(
                f"Layer index {idx} out of range. Model has {num_layers} layers."
            )

    # Replace specified layers with Titans-enhanced versions
    for idx in layer_indices:
        original_layer = layers[idx]

        # Check if already a Titans layer
        if isinstance(original_layer, TitansQwenDecoderLayer):
            print(f"Layer {idx} is already a Titans layer, skipping")
            continue

        # Get device and dtype from original layer
        device = next(original_layer.parameters()).device
        dtype = next(original_layer.parameters()).dtype

        titans_layer = TitansQwenDecoderLayer(
            original_layer=original_layer,
            layer_idx=idx,
            titans_config=titans_config,
        )

        # Move new Titans components to same device/dtype as original layer
        titans_layer.memory.to(device=device, dtype=dtype)
        titans_layer.mem_proj.to(device=device, dtype=dtype)
        titans_layer.gate.to(device=device, dtype=dtype)
        if titans_layer.warm_start is not None:
            titans_layer.warm_start.to(device=device, dtype=dtype)

        layers[idx] = titans_layer
        print(f"Replaced layer {idx} with TitansQwenDecoderLayer")

    # Store metadata on model for later access
    model._titans_layer_indices = layer_indices
    model._titans_config = titans_config

    return model


def freeze_base_model(model: nn.Module) -> dict[str, int]:
    """
    Freeze all base model parameters, leaving only Titans memory trainable.

    After calling this function, these parameters are trainable:
    - memory.* (NeuralMemory / ContinuumMemory parameters)
    - mem_proj.* (memory projection)
    - gate.* (memory gate)
    - warm_start.* (warm start encoder, if enabled)

    Args:
        model: Model that has been patched with patch_qwen_with_titans

    Returns:
        Dict with parameter counts (total, trainable, frozen, titans)

    Example:
        >>> model = patch_qwen_with_titans(model, config)
        >>> stats = freeze_base_model(model)
        >>> print(f"Trainable: {stats['trainable']:,} ({stats['percent']:.2f}%)")
    """
    if not hasattr(model, "_titans_layer_indices"):
        raise ValueError(
            "Model must be patched with patch_qwen_with_titans before freezing"
        )

    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Get the layers container
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = model.layers

    # Unfreeze Titans-specific parameters
    titans_params = 0

    for idx in model._titans_layer_indices:
        layer = layers[idx]

        if not isinstance(layer, TitansQwenDecoderLayer):
            print(f"Warning: Layer {idx} is not a TitansQwenDecoderLayer")
            continue

        # Unfreeze all Titans components (memory, gate, projection, warm start)
        # Memory parameters
        for param in layer.memory.parameters():
            param.requires_grad = True
            titans_params += param.numel()

        # Memory projection
        for param in layer.mem_proj.parameters():
            param.requires_grad = True
            titans_params += param.numel()

        # Gate
        for param in layer.gate.parameters():
            param.requires_grad = True
            titans_params += param.numel()

        # Warm start encoder (if present)
        if layer.warm_start is not None:
            for param in layer.warm_start.parameters():
                param.requires_grad = True
                titans_params += param.numel()

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    stats = {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "titans": titans_params,
        "percent": 100.0 * trainable / total if total > 0 else 0,
    }

    print(f"Parameter freeze stats:")
    print(f"  Total:       {total:,}")
    print(f"  Trainable:   {trainable:,} ({stats['percent']:.2f}%)")
    print(f"  Frozen:      {frozen:,}")
    print(f"  Titans mem:  {titans_params:,}")

    return stats


def get_trainable_param_count(model: nn.Module) -> int:
    """Get count of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_titans_layers(model: nn.Module) -> list[TitansQwenDecoderLayer]:
    """
    Get all Titans layers from a patched model.

    Args:
        model: Model patched with patch_qwen_with_titans

    Returns:
        List of TitansQwenDecoderLayer instances
    """
    if not hasattr(model, "_titans_layer_indices"):
        return []

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = model.layers

    return [
        layers[idx]
        for idx in model._titans_layer_indices
        if isinstance(layers[idx], TitansQwenDecoderLayer)
    ]


def enable_memory_updates(model: nn.Module, enabled: bool = True) -> None:
    """
    Enable or disable memory updates for all Titans layers.

    Disabling can speed up inference when memory doesn't need to be updated.

    Args:
        model: Model patched with patch_qwen_with_titans
        enabled: Whether to enable updates
    """
    for layer in get_titans_layers(model):
        layer.enable_memory_updates(enabled)


def get_internal_losses(model: nn.Module) -> list[torch.Tensor]:
    """
    Get internal losses from all Titans layers.

    These can be added to the main loss for self-supervised memory training.

    Args:
        model: Model patched with patch_qwen_with_titans

    Returns:
        List of internal loss tensors (one per Titans layer with internal loss enabled)
    """
    losses = []
    for layer in get_titans_layers(model):
        loss = layer.get_internal_loss()
        if loss is not None:
            losses.append(loss)
    return losses


def get_gate_statistics(model: nn.Module) -> dict[int, dict]:
    """
    Get gate statistics from all Titans layers.

    Useful for monitoring how much the memory is being used.

    Args:
        model: Model patched with patch_qwen_with_titans

    Returns:
        Dict mapping layer index to gate statistics
    """
    stats = {}
    for layer in get_titans_layers(model):
        stats[layer.layer_idx] = layer.get_gate_statistics()
    return stats


def save_titans_state(model: nn.Module, path: str) -> None:
    """
    Save only the Titans-specific parameters.

    Useful for saving fine-tuned memory without the full base model.

    Args:
        model: Model patched with patch_qwen_with_titans
        path: Path to save state dict
    """
    titans_state = {}

    for layer in get_titans_layers(model):
        layer_key = f"layer_{layer.layer_idx}"

        # Memory parameters
        for name, param in layer.memory.named_parameters():
            titans_state[f"{layer_key}.memory.{name}"] = param.data

        # Projection parameters
        for name, param in layer.mem_proj.named_parameters():
            titans_state[f"{layer_key}.mem_proj.{name}"] = param.data

        # Gate parameters
        for name, param in layer.gate.named_parameters():
            titans_state[f"{layer_key}.gate.{name}"] = param.data

        # Warm start parameters (if present)
        if layer.warm_start is not None:
            for name, param in layer.warm_start.named_parameters():
                titans_state[f"{layer_key}.warm_start.{name}"] = param.data

    # Also save config
    if hasattr(model, "_titans_config"):
        titans_state["config"] = {
            "layer_indices": model._titans_layer_indices,
            "n_embd": model._titans_config.n_embd,
            "num_longterm_mem": model._titans_config.num_longterm_mem,
            "segment_len": model._titans_config.segment_len,
            "use_cms": model._titans_config.use_cms,
            "use_self_mod_proj": model._titans_config.use_self_mod_proj,
            "use_self_mod_gate": model._titans_config.use_self_mod_gate,
            "use_warm_start": model._titans_config.use_warm_start,
        }

    torch.save(titans_state, path)
    print(f"Saved Titans state to {path}")


def load_titans_state(model: nn.Module, path: str) -> None:
    """
    Load Titans-specific parameters.

    Args:
        model: Model patched with patch_qwen_with_titans
        path: Path to saved state dict
    """
    titans_state = torch.load(path, weights_only=False)

    for layer in get_titans_layers(model):
        layer_key = f"layer_{layer.layer_idx}"

        # Memory parameters
        for name, param in layer.memory.named_parameters():
            key = f"{layer_key}.memory.{name}"
            if key in titans_state:
                # Convert dtype to match model parameter
                param.data.copy_(titans_state[key].to(param.dtype))

        # Projection parameters
        for name, param in layer.mem_proj.named_parameters():
            key = f"{layer_key}.mem_proj.{name}"
            if key in titans_state:
                param.data.copy_(titans_state[key].to(param.dtype))

        # Gate parameters
        for name, param in layer.gate.named_parameters():
            key = f"{layer_key}.gate.{name}"
            if key in titans_state:
                param.data.copy_(titans_state[key].to(param.dtype))

        # Warm start parameters (if present)
        if layer.warm_start is not None:
            for name, param in layer.warm_start.named_parameters():
                key = f"{layer_key}.warm_start.{name}"
                if key in titans_state:
                    param.data.copy_(titans_state[key].to(param.dtype))

    print(f"Loaded Titans state from {path}")

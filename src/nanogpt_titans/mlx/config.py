"""
Configuration for MLX TITANS training.

Compatible with PyTorch TitansQwenConfig for unified training workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MLXTitansConfig:
    """Configuration for MLX TITANS training - matches PyTorch TitansQwenConfig."""

    model_name: str = "Qwen/Qwen2-0.5B"
    memory_layer: int = 12

    # Memory config
    segment_len: int = 512
    memory_depth: int = 2
    memory_expansion: int = 2
    num_longterm_mem: int = 16

    # CMS (Continuum Memory System) config
    use_cms: bool = True
    num_cms_levels: int = 3
    cms_update_frequencies: Tuple[int, ...] = (1, 4, 16)  # Fast, medium, slow

    # Gate config - match PyTorch: sigmoid(0) = 0.5 for gradient flow
    gate_init_bias: float = 0.0

    # Adaptive memory parameters (matches PyTorch model.py)
    adaptive_memory: bool = True
    memory_lr_max: float = 0.01  # Max learning rate for adaptive mode

    # Training - matches PyTorch defaults
    learning_rate: float = 1e-4
    gate_lr_scale: float = 50.0  # Gate/scale params get higher LR
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 100
    warmup_steps: int = 100
    max_length: int = 1024

    # Gate warmup: freeze gate for first N steps to let memory learn first
    gate_warmup_steps: int = 0

    # Output
    output_dir: str = "out-mlx-titans"


def config_to_mlx(pytorch_config) -> MLXTitansConfig:
    """
    Convert PyTorch TitansQwenConfig to MLXTitansConfig.

    Args:
        pytorch_config: TitansQwenConfig or QwenTitansTrainConfig

    Returns:
        MLXTitansConfig with equivalent settings
    """
    # Handle both TitansQwenConfig and QwenTitansTrainConfig
    memory_layer = getattr(pytorch_config, 'memory_layers', [12])
    if isinstance(memory_layer, list):
        memory_layer = memory_layer[0] if memory_layer else 12

    return MLXTitansConfig(
        model_name=getattr(pytorch_config, 'model_name', "Qwen/Qwen2-0.5B"),
        memory_layer=memory_layer,
        segment_len=getattr(pytorch_config, 'segment_len', 512),
        memory_depth=getattr(pytorch_config, 'memory_depth', 2),
        memory_expansion=getattr(pytorch_config, 'memory_expansion', 2),
        num_longterm_mem=getattr(pytorch_config, 'num_longterm_mem', 16),
        use_cms=getattr(pytorch_config, 'use_cms', True),
        num_cms_levels=getattr(pytorch_config, 'num_cms_levels', 3),
        cms_update_frequencies=tuple(getattr(pytorch_config, 'cms_update_frequencies', [1, 4, 16])),
        gate_init_bias=getattr(pytorch_config, 'gate_init_bias', 0.0),
        adaptive_memory=getattr(pytorch_config, 'adaptive_memory', True),
        memory_lr_max=getattr(pytorch_config, 'memory_lr_max', 0.01),
        learning_rate=getattr(pytorch_config, 'learning_rate', 1e-4),
        gate_lr_scale=getattr(pytorch_config, 'gate_lr_scale', 50.0),
        batch_size=getattr(pytorch_config, 'batch_size', 2),
        gradient_accumulation_steps=getattr(pytorch_config, 'gradient_accumulation_steps', 4),
        max_steps=getattr(pytorch_config, 'max_steps', 100),
        warmup_steps=getattr(pytorch_config, 'warmup_steps', 100),
        max_length=getattr(pytorch_config, 'max_length', 1024),
        gate_warmup_steps=getattr(pytorch_config, 'gate_warmup_steps', 0),
        output_dir=getattr(pytorch_config, 'output_dir', "out-mlx-titans"),
    )

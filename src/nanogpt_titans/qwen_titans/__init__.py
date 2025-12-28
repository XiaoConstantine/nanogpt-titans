"""
Qwen-Titans Integration Module.

Provides test-time learning memory capabilities for Qwen2 models
using the Titans architecture from nanogpt-titans.

Usage:
    from nanogpt_titans.qwen_titans import (
        TitansQwenConfig,
        patch_qwen_with_titans,
        freeze_base_model,
    )

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B")
    config = TitansQwenConfig.from_qwen_config(model.config)
    model = patch_qwen_with_titans(model, config)
    freeze_base_model(model)
"""

from nanogpt_titans.qwen_titans.config import TitansQwenConfig
from nanogpt_titans.qwen_titans.memory_adapter import NeuralMemoryAdapter
from nanogpt_titans.qwen_titans.decoder_layer import TitansQwenDecoderLayer
from nanogpt_titans.qwen_titans.mag_decoder_layer import MAGQwenDecoderLayer
from nanogpt_titans.qwen_titans.patcher import (
    patch_qwen_with_titans,
    freeze_base_model,
    get_trainable_param_count,
    get_titans_layers,
    get_gate_statistics,
    get_internal_losses,
    enable_memory_updates,
    save_titans_state,
    load_titans_state,
)
from nanogpt_titans.qwen_titans.generation import (
    TitansStateManager,
    titans_generate,
)

__all__ = [
    "TitansQwenConfig",
    "NeuralMemoryAdapter",
    "TitansQwenDecoderLayer",
    "MAGQwenDecoderLayer",
    "patch_qwen_with_titans",
    "freeze_base_model",
    "get_trainable_param_count",
    "get_titans_layers",
    "get_gate_statistics",
    "get_internal_losses",
    "enable_memory_updates",
    "save_titans_state",
    "load_titans_state",
    "TitansStateManager",
    "titans_generate",
]

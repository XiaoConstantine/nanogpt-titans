"""
MLX TITANS Module.

Provides MLX-native implementations of TITANS memory architecture
for Apple Silicon optimization.

Usage:
    from nanogpt_titans.mlx import (
        MLXTitansConfig,
        MLXTitansLayer,
        MLXNeuralMemory,
        MLXContinuumMemorySystem,
        CombinedModel,
    )

    # Create TITANS layer
    titans_layer = MLXTitansLayer(dim=896, use_cms=True)

    # Initialize memory state
    state = titans_layer.init_state(batch_size=2)

    # Forward pass
    output, new_state = titans_layer(hidden_states, state)
"""

# Check MLX availability
import importlib.util

_MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None

if _MLX_AVAILABLE:
    from nanogpt_titans.mlx.config import MLXTitansConfig, config_to_mlx
    from nanogpt_titans.mlx.decoder_layer import (
        MLXPositionDependentGate,
        MLXTitansLayer,
        TitansLayerState,
    )
    from nanogpt_titans.mlx.memory import (
        CMSMetrics,
        MemoryMetrics,
        MLXCMSState,
        MLXContinuumMemorySystem,
        MLXMemoryState,
        MLXNeuralMemory,
    )
    from nanogpt_titans.mlx.training import (
        CombinedModel,
        accumulate_grads,
        apply_teach_signal_to_memory,
        compute_teach_signal,
        create_loss_fn,
        create_masked_grads,
        create_titans_layer_from_model,
        filter_titans_grads,
        get_lr,
        online_eval,
        online_generate,
        scale_grads_recursive,
    )

    __all__ = [
        "_MLX_AVAILABLE",
        "CMSMetrics",
        "CombinedModel",
        "MLXCMSState",
        "MLXContinuumMemorySystem",
        "MLXMemoryState",
        "MLXNeuralMemory",
        "MLXPositionDependentGate",
        "MLXTitansConfig",
        "MLXTitansLayer",
        "MemoryMetrics",
        "TitansLayerState",
        "accumulate_grads",
        "apply_teach_signal_to_memory",
        "compute_teach_signal",
        "config_to_mlx",
        "create_loss_fn",
        "create_masked_grads",
        "create_titans_layer_from_model",
        "filter_titans_grads",
        "get_lr",
        "online_eval",
        "online_generate",
        "scale_grads_recursive",
    ]
else:
    __all__ = ["_MLX_AVAILABLE"]


def is_available() -> bool:
    """Check if MLX backend is available."""
    return _MLX_AVAILABLE

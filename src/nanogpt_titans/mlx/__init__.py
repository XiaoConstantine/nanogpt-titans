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
_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    import mlx.nn as nn
    _MLX_AVAILABLE = True
except ImportError:
    pass

if _MLX_AVAILABLE:
    from nanogpt_titans.mlx.config import MLXTitansConfig, config_to_mlx
    from nanogpt_titans.mlx.memory import (
        MLXMemoryState,
        MLXCMSState,
        MLXNeuralMemory,
        MLXContinuumMemorySystem,
    )
    from nanogpt_titans.mlx.decoder_layer import (
        MLXPositionDependentGate,
        MLXTitansLayer,
        TitansLayerState,
    )
    from nanogpt_titans.mlx.training import (
        CombinedModel,
        create_loss_fn,
        filter_titans_grads,
        get_lr,
        create_masked_grads,
        scale_grads_recursive,
        accumulate_grads,
        create_titans_layer_from_model,
    )

    __all__ = [
        # Config
        "MLXTitansConfig",
        "config_to_mlx",
        # Memory
        "MLXMemoryState",
        "MLXCMSState",
        "MLXNeuralMemory",
        "MLXContinuumMemorySystem",
        # Decoder layer
        "MLXPositionDependentGate",
        "MLXTitansLayer",
        "TitansLayerState",
        # Training
        "CombinedModel",
        "create_loss_fn",
        "filter_titans_grads",
        "get_lr",
        "create_masked_grads",
        "scale_grads_recursive",
        "accumulate_grads",
        "create_titans_layer_from_model",
        # Availability flag
        "_MLX_AVAILABLE",
    ]
else:
    __all__ = ["_MLX_AVAILABLE"]


def is_available() -> bool:
    """Check if MLX backend is available."""
    return _MLX_AVAILABLE

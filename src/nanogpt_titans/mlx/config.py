"""MLX TITANS training config (compatible with PyTorch TitansQwenConfig)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MLXTitansConfig:
    """MLX TITANS config."""

    # Training mode
    train_from_scratch: bool = False  # True = train GPT from scratch, False = fine-tune Qwen

    # For fine-tuning mode (train_from_scratch=False)
    model_name: str = "Qwen/Qwen2-0.5B"
    memory_layers: list[int] = field(default_factory=lambda: [12])  # Can be multiple layers

    # For from-scratch mode (train_from_scratch=True)
    model_size: str = "nano"  # "nano" (10M), "small" (45M), "medium" (124M), "large" (250M)
    vocab_size: int = 50257   # GPT-2 tokenizer vocab size
    n_layer: int = 6          # Override: number of layers
    n_head: int = 6           # Override: number of attention heads
    n_embd: int = 384         # Override: embedding dimension
    block_size: int = 1024    # Maximum sequence length

    # Memory config
    segment_len: int = 512
    memory_depth: int = 2
    memory_expansion: int = 2
    num_longterm_mem: int = 16

    # CMS (Continuum Memory System) config
    use_cms: bool = True
    num_cms_levels: int = 3
    cms_update_frequencies: tuple[int, ...] = (1, 4, 16)  # Fast, medium, slow
    # Cascade mode: each level transforms previous level's output (hierarchical refinement)
    # vs weighted sum mode: all levels process same input, outputs are weighted sum
    use_cascade: bool = False

    # CMS warmup/jitter (from nested_learning LevelSpec)
    # Warmup: number of steps before each level starts updating
    cms_warmup_steps: tuple[int, ...] = (0, 0, 0)  # Per-level warmup (e.g., (0, 10, 50))
    # Jitter: random variation in update timing (0 = no jitter, 0.1 = ±10% variation)
    cms_jitter: float = 0.0

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

    # Gate regularization: prevent gate from collapsing below min_value (like PyTorch)
    gate_min_value: float = 0.15  # Matches PyTorch default
    gate_reg_weight: float = 1.0  # Regularization weight

    # Internal loss (CRITICAL: teaches template weights to store/retrieve patterns)
    # Must be True for memory to learn during training!
    use_internal_loss: bool = True
    internal_loss_weight: float = 0.1  # Increased from 1e-4 for stronger signal

    # Output
    output_dir: str = "out-mlx-titans"

    # Performance optimization: evaluate after each micro-step
    # This prevents MLX's lazy evaluation from building huge graphs
    # Default True for best performance; set False for debugging
    eager_eval: bool = True

    # Backbone unfreezing: train layers adjacent to TITANS layers
    # Per "Titans Revisited": frozen backbone causes mismatch with evolving memory
    # Setting > 0 unfreezes N layers before/after each TITANS layer
    unfreeze_backbone_layers: int = 0  # 0 = fully frozen, 1-2 recommended if memory not helping

    # Surprise threshold: skip memory updates when surprise (grad norm) is below threshold
    # From nested_learning: prevents memory pollution from predictable tokens
    surprise_threshold: float = 0.0  # 0.0 = disabled, try 0.01-0.1

    # Per-level gradient clipping for CMS
    # Prevents any single level from dominating updates
    memory_grad_clip: float = 1.0  # Clip grad norm per memory level

    # Teach signal: auxiliary gradient from logit residuals (nested_learning extension)
    # Supplements internal loss with gradient approximation from LM predictions
    use_teach_signal: bool = False
    teach_signal_weight: float = 0.1

    # Dataset configuration
    dataset: str = "wikitext"  # "wikitext", "fineweb-edu", "slimpajama"
    min_doc_length: int = 100  # Minimum document length in tokens (2048 recommended for memory training)
    max_doc_length: int = 8192  # Maximum document length in tokens
    streaming: bool = True  # Stream dataset (required for large datasets like fineweb-edu)
    num_examples: int = 0  # Number of examples to use (0 = unlimited/streaming)


def config_to_mlx(pytorch_config) -> MLXTitansConfig:
    """
    Convert PyTorch TitansQwenConfig to MLXTitansConfig.

    Args:
        pytorch_config: TitansQwenConfig or QwenTitansTrainConfig

    Returns:
        MLXTitansConfig with equivalent settings
    """
    # Handle both TitansQwenConfig and QwenTitansTrainConfig
    memory_layers = getattr(pytorch_config, "memory_layers", [12])
    if not isinstance(memory_layers, list):
        memory_layers = [memory_layers]

    return MLXTitansConfig(
        model_name=getattr(pytorch_config, "model_name", "Qwen/Qwen2-0.5B"),
        memory_layers=memory_layers,
        segment_len=getattr(pytorch_config, "segment_len", 512),
        memory_depth=getattr(pytorch_config, "memory_depth", 2),
        memory_expansion=getattr(pytorch_config, "memory_expansion", 2),
        num_longterm_mem=getattr(pytorch_config, "num_longterm_mem", 16),
        use_cms=getattr(pytorch_config, "use_cms", True),
        num_cms_levels=getattr(pytorch_config, "num_cms_levels", 3),
        cms_update_frequencies=tuple(getattr(pytorch_config, "cms_update_frequencies", [1, 4, 16])),
        use_cascade=getattr(pytorch_config, "use_cascade", False),
        gate_init_bias=getattr(pytorch_config, "gate_init_bias", 0.0),
        adaptive_memory=getattr(pytorch_config, "adaptive_memory", True),
        memory_lr_max=getattr(pytorch_config, "memory_lr_max", 0.01),
        learning_rate=getattr(pytorch_config, "learning_rate", 1e-4),
        gate_lr_scale=getattr(pytorch_config, "gate_lr_scale", 50.0),
        batch_size=getattr(pytorch_config, "batch_size", 2),
        gradient_accumulation_steps=getattr(pytorch_config, "gradient_accumulation_steps", 4),
        max_steps=getattr(pytorch_config, "max_steps", 100),
        warmup_steps=getattr(pytorch_config, "warmup_steps", 100),
        max_length=getattr(pytorch_config, "max_length", 1024),
        gate_warmup_steps=getattr(pytorch_config, "gate_warmup_steps", 0),
        output_dir=getattr(pytorch_config, "output_dir", "out-mlx-titans"),
        use_internal_loss=getattr(pytorch_config, "use_internal_loss", True),
        internal_loss_weight=getattr(pytorch_config, "internal_loss_weight", 0.1),
        unfreeze_backbone_layers=getattr(pytorch_config, "unfreeze_backbone_layers", 0),
        surprise_threshold=getattr(pytorch_config, "surprise_threshold", 0.0),
        memory_grad_clip=getattr(pytorch_config, "memory_grad_clip", 1.0),
        use_teach_signal=getattr(pytorch_config, "use_teach_signal", False),
        teach_signal_weight=getattr(pytorch_config, "teach_signal_weight", 0.1),
        cms_warmup_steps=tuple(getattr(pytorch_config, "cms_warmup_steps", (0, 0, 0))),
        cms_jitter=getattr(pytorch_config, "cms_jitter", 0.0),
    )

#!/usr/bin/env python3
"""
MLX-optimized training for TITANS memory on Apple Silicon.

This implementation matches the PyTorch HOPE architecture exactly:
1. ContinuumMemorySystem (CMS) - Multi-frequency memory (3 levels)
2. PositionDependentGate - Per-token gating
3. NeuralMemory - Memory MLP with key/value/query projections
4. HOPE Integration: output = hidden + gate * scale * LayerNorm(project(memory))

Usage:
    uv run python -m nanogpt_titans.train_mlx --model_name Qwen/Qwen2-0.5B --max_steps 500
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Configure logger for debug output
logger = logging.getLogger(__name__)

# Import from MLX module
from nanogpt_titans.mlx import (
    CombinedModel,
    MLXTitansConfig,
    MLXTitansLayer,
    accumulate_grads,
    create_loss_fn,
    create_masked_grads,
    filter_titans_grads,
    get_lr,
    scale_grads_recursive,
)
from nanogpt_titans.mlx.visualizer import Visualizer, generate_sample
from nanogpt_titans.mlx.inspector import Inspector
from nanogpt_titans.mlx.model import GPT, GPTConfig, GPTWithMemory

# For loading HuggingFace models (fine-tuning mode)
try:
    from mlx_lm import load as mlx_load
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False
    mlx_load = None

# For training from scratch (tiktoken GPT-2 tokenizer)
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    tiktoken = None


def load_tokenizer_scratch():
    """Load GPT-2 tokenizer for from-scratch training."""
    if not HAS_TIKTOKEN:
        raise ImportError("tiktoken required for from-scratch training: uv add tiktoken")
    return tiktoken.get_encoding("gpt2")


def create_model_scratch(config: MLXTitansConfig):
    """Create GPT model from scratch (nanoGPT style)."""
    # Get vocab size from tokenizer
    tokenizer = load_tokenizer_scratch()
    vocab_size = tokenizer.n_vocab

    # Create config based on size preset or explicit values
    size_presets = {
        "nano": GPTConfig.nano,
        "small": GPTConfig.small,
        "medium": GPTConfig.medium,
        "large": GPTConfig.large,
    }

    if config.model_size in size_presets:
        gpt_config = size_presets[config.model_size](vocab_size=vocab_size)
    else:
        # Use explicit config values
        gpt_config = GPTConfig(
            vocab_size=vocab_size,
            block_size=config.block_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
        )

    # Override block_size from training config
    gpt_config.block_size = max(config.segment_len * 2, gpt_config.block_size)

    # Create model with optional TITANS memory
    if config.use_cms or config.memory_layers:
        gpt_config.use_memory = True
        gpt_config.memory_layer = config.memory_layers[0] if config.memory_layers else -1
        model = GPTWithMemory(gpt_config)
    else:
        model = GPT(gpt_config)

    return model, tokenizer


def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer using mlx-lm."""
    print(f"Loading {model_name} with MLX...")
    model, tokenizer = mlx_load(model_name)
    return model, tokenizer


def load_training_data(
    tokenizer,
    config: MLXTitansConfig,
) -> mx.array:
    """
    Load and tokenize training data from various datasets.

    Supports:
    - wikitext: Small dataset for testing (default)
    - fineweb-edu: Large educational dataset (recommended for TITANS)
    - slimpajama: Medium-sized diverse dataset

    Args:
        tokenizer: Tokenizer for encoding text
        config: Training config with dataset settings

    Returns:
        mx.array of all tokens concatenated
    """
    from datasets import load_dataset

    dataset_name = config.dataset.lower()
    min_tokens = config.min_doc_length
    max_tokens = config.max_doc_length
    num_examples = config.num_examples

    print(f"\nLoading dataset: {dataset_name}")
    print(f"  Min document length: {min_tokens} tokens")
    print(f"  Max document length: {max_tokens} tokens")

    all_tokens = []
    docs_processed = 0
    docs_filtered = 0

    if dataset_name == "wikitext":
        # Small dataset for testing
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        texts = [ex["text"] for ex in dataset["train"] if ex["text"].strip()]

        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) >= min_tokens:
                all_tokens.extend(tokens[:max_tokens])
                docs_processed += 1
            else:
                docs_filtered += 1

            if num_examples > 0 and docs_processed >= num_examples:
                break

    elif dataset_name == "fineweb-edu":
        # Large educational dataset - stream to avoid memory issues
        print("  Streaming FineWeb-Edu (this may take a moment to start)...")

        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
        )

        # Shuffle for randomness
        dataset = dataset.shuffle(seed=42, buffer_size=10000)

        # Target tokens based on training needs
        # For 50K steps with segment_len=1024 and grad_accum=4: ~200M tokens
        target_tokens = config.max_steps * config.gradient_accumulation_steps * config.segment_len * 2
        print(f"  Target tokens: {target_tokens:,}")

        for example in dataset:
            text = example.get("text", "")
            if not text or len(text) < 200:  # Quick pre-filter
                continue

            tokens = tokenizer.encode(text)

            if len(tokens) >= min_tokens:
                all_tokens.extend(tokens[:max_tokens])
                docs_processed += 1

                # Progress update every 100 docs
                if docs_processed % 100 == 0:
                    print(f"  Processed {docs_processed} docs, {len(all_tokens):,} tokens...")

            else:
                docs_filtered += 1

            # Stop conditions
            if num_examples > 0 and docs_processed >= num_examples:
                break
            if len(all_tokens) >= target_tokens:
                print("  Reached target token count")
                break

    elif dataset_name == "slimpajama":
        # Medium-sized diverse dataset
        print("  Streaming SlimPajama...")

        dataset = load_dataset(
            "cerebras/SlimPajama-627B",
            split="train",
            streaming=True,
        )

        dataset = dataset.shuffle(seed=42, buffer_size=10000)

        target_tokens = config.max_steps * config.gradient_accumulation_steps * config.segment_len * 2

        for example in dataset:
            text = example.get("text", "")
            if not text:
                continue

            tokens = tokenizer.encode(text)

            if len(tokens) >= min_tokens:
                all_tokens.extend(tokens[:max_tokens])
                docs_processed += 1

                if docs_processed % 100 == 0:
                    print(f"  Processed {docs_processed} docs, {len(all_tokens):,} tokens...")

            else:
                docs_filtered += 1

            if num_examples > 0 and docs_processed >= num_examples:
                break
            if len(all_tokens) >= target_tokens:
                break

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'wikitext', 'fineweb-edu', or 'slimpajama'")

    print("\nDataset loaded:")
    print(f"  Documents processed: {docs_processed:,}")
    print(f"  Documents filtered (too short): {docs_filtered:,}")
    print(f"  Total tokens: {len(all_tokens):,}")

    if min_tokens >= 2048:
        print(f"  (Using min_doc_length={min_tokens} for memory training - good!)")
    elif min_tokens < 1024:
        print(f"  WARNING: min_doc_length={min_tokens} is low for memory training.")
        print("  Consider --min_doc_length 2048 for better memory utilization.")

    return mx.array(all_tokens)


def get_model_dim(model) -> int:
    """Get hidden dimension from model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        sample_layer = model.model.layers[0]
        if hasattr(sample_layer, "hidden_size"):
            return sample_layer.hidden_size
        if hasattr(sample_layer.self_attn, "hidden_size"):
            return sample_layer.self_attn.hidden_size
        return sample_layer.self_attn.q_proj.weight.shape[0]
    return 896  # Default for Qwen2-0.5B


def create_titans_layers(model, config: MLXTitansConfig) -> dict[int, MLXTitansLayer]:
    """Create TITANS layers for all specified layer indices."""
    dim = get_model_dim(model)
    num_layers = len(model.model.layers)

    titans_layers = {}
    for layer_idx in config.memory_layers:
        # Clamp to valid range
        idx = min(layer_idx, num_layers - 1)
        titans_layers[idx] = MLXTitansLayer(
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

    return titans_layers


def get_layers_to_unfreeze(memory_layers: list, num_backbone_layers: int, radius: int) -> set:
    """Return backbone layer indices to unfreeze around TITANS layer positions."""
    if radius <= 0:
        return set()

    layers_to_unfreeze = set()
    for mem_idx in memory_layers:
        # Unfreeze layers before and after the TITANS insertion point
        for offset in range(-radius, radius + 1):
            layer_idx = mem_idx + offset
            if 0 <= layer_idx < num_backbone_layers:
                layers_to_unfreeze.add(layer_idx)

    return layers_to_unfreeze


def train(config: MLXTitansConfig):
    """Main MLX training loop - supports both from-scratch and fine-tuning."""
    print("=" * 60)
    if config.train_from_scratch:
        print("NanoGPT-style Training FROM SCRATCH")
        print("=" * 60)
        print(f"Model size: {config.model_size}")
        print(f"Architecture: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim")
    else:
        print("MLX TITANS Training (Fine-tuning Mode)")
        print("=" * 60)
        print(f"Base model: {config.model_name}")
        print(f"Memory layers: {config.memory_layers}")
    print(f"CMS enabled: {config.use_cms}")
    if config.use_cms:
        print(f"CMS levels: {config.num_cms_levels}")
        print(f"CMS frequencies: {config.cms_update_frequencies}")
        print(f"CMS cascade mode: {config.use_cascade}")
    print(f"Max steps: {config.max_steps}")
    print(f"Internal loss: {config.use_internal_loss} (weight={config.internal_loss_weight})")
    if config.surprise_threshold > 0:
        print(f"Surprise threshold: {config.surprise_threshold}")
    if config.memory_grad_clip > 0:
        print(f"Memory grad clip: {config.memory_grad_clip}")
    if not config.train_from_scratch and config.unfreeze_backbone_layers > 0:
        print(f"Backbone unfreezing: {config.unfreeze_backbone_layers} layers near TITANS")
    print()

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer for real-time training inspection
    viz = Visualizer(
        output_dir=str(output_dir / "viz"),
        plot_every=10,
    )
    inspector = Inspector(
        verbose=False,
        save_history=True,
        history_dir=str(output_dir / "inspection"),
    )
    print(f"Visualization enabled: {output_dir}/viz/")

    # Load or create model
    if config.train_from_scratch:
        print("\nCreating model from scratch...")
        model, tokenizer = create_model_scratch(config)
        # For from-scratch, we train ALL parameters
        titans_layers = {}  # No separate TITANS layers - memory is built into GPTWithMemory
        unfrozen_backbone_indices = set()  # All params trained
    else:
        print("\nLoading pre-trained model...")
        model, tokenizer = load_model_and_tokenizer(config.model_name)

        # Create TITANS layers (independent memory per layer)
        titans_layers = create_titans_layers(model, config)
        num_titans_layers = len(titans_layers)
        print(f"Created {num_titans_layers} TITANS layer(s) at indices: {sorted(titans_layers.keys())}")

        # Count parameters
        titans_params = sum(p.size for layer in titans_layers.values() for _, p in tree_flatten(layer.parameters()))
        print(f"TITANS trainable params: {titans_params:,} ({titans_params // num_titans_layers:,} per layer)")

        # Get number of backbone layers for unfreezing logic
        num_backbone_layers = len(model.model.layers)

        # Determine which backbone layers to unfreeze (if any)
        unfrozen_backbone_indices = get_layers_to_unfreeze(
            config.memory_layers, num_backbone_layers, config.unfreeze_backbone_layers
        )

    # Separate param groups for different LRs
    def count_param_groups_from_module(module):
        """Count gate/scale vs memory params for any module."""
        gate_scale_count = 0
        memory_count = 0

        for name, param in tree_flatten(module.parameters()):
            name_str = ".".join(str(k) for k in name) if isinstance(name, tuple) else str(name)
            if (
                "gate" in name_str
                or "mem_scale" in name_str
                or "mem_ln" in name_str
                or "level_weights" in name_str
                or "to_lr" in name_str
                or "to_momentum" in name_str
                or "to_decay" in name_str
            ):
                gate_scale_count += param.size
            else:
                memory_count += param.size

        return gate_scale_count, memory_count

    gate_scale_count, memory_count = 0, 0
    for layer in titans_layers.values():
        g, m = count_param_groups_from_module(layer)
        gate_scale_count += g
        memory_count += m
    if titans_layers:
        print(f"Gate/scale/CMS params: {gate_scale_count:,}")
        print(f"Memory params: {memory_count:,}")

    # For from-scratch mode, count total params
    if config.train_from_scratch:
        total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
        print(f"Total trainable params: {total_params:,} ({total_params/1e6:.2f}M)")
        num_backbone_layers = 0  # Not applicable

    # Setup optimizers based on mode
    if config.train_from_scratch:
        # From-scratch: single optimizer for all params
        optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=0.01)
        optimizer_memory = None
        optimizer_gate = None
        optimizer_backbone = None
        print(f"\nOptimizer: AdamW, LR={config.learning_rate:.2e}")
    else:
        # Fine-tuning: separate optimizers for different param groups
        if unfrozen_backbone_indices:
            print(f"Unfreezing backbone layers: {sorted(unfrozen_backbone_indices)}")
            unfrozen_backbone_params = sum(
                p.size for idx in unfrozen_backbone_indices for _, p in tree_flatten(model.model.layers[idx].parameters())
            )
            print(f"Unfrozen backbone params: {unfrozen_backbone_params:,}")

        gate_lr = config.learning_rate * config.gate_lr_scale
        backbone_lr = config.learning_rate * 0.1

        optimizer = None  # Not used in fine-tuning mode
        optimizer_memory = optim.AdamW(learning_rate=config.learning_rate, weight_decay=0.0)
        optimizer_gate = optim.AdamW(learning_rate=gate_lr, weight_decay=0.0)
        optimizer_backbone = (
            optim.AdamW(learning_rate=backbone_lr, weight_decay=0.01) if unfrozen_backbone_indices else None
        )

        print("\nOptimizer learning rates:")
        print(f"  Memory params: {config.learning_rate:.2e}")
        print(f"  Gate/scale params: {gate_lr:.2e} ({config.gate_lr_scale}x)")
        if unfrozen_backbone_indices:
            print(f"  Backbone params: {backbone_lr:.2e} (0.1x base LR)")

    # Load training data using configurable dataset
    try:
        all_tokens = load_training_data(tokenizer, config)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Falling back to dummy data for testing...")
        dummy_text = "This is a test sentence for training. " * 100
        tokens = tokenizer.encode(dummy_text)
        all_tokens = mx.array(tokens * 100)  # Repeat to get enough data
        print(f"Total tokens (dummy): {len(all_tokens):,}")

    # Training loop
    print("\nStarting training...")
    log_history = []

    # Determine memory layer indices
    num_layers = len(model.model.layers)
    memory_layer_indices = sorted(titans_layers.keys())
    print(f"Applying TITANS at layers {memory_layer_indices} of {num_layers} total")

    # Create combined model wrapper
    combined_model = CombinedModel(
        model,
        titans_layers=titans_layers,
        use_internal_loss=config.use_internal_loss,
        internal_loss_weight=config.internal_loss_weight,
    )
    if config.use_internal_loss:
        print(f"Internal loss enabled with weight={config.internal_loss_weight}")

    # Create loss function with gate regularization
    loss_fn = create_loss_fn(
        combined_model, gate_min_value=config.gate_min_value, gate_reg_weight=config.gate_reg_weight
    )
    loss_and_grad_fn = nn.value_and_grad(combined_model, loss_fn)

    # Note: mx.compile doesn't work well with nn.value_and_grad wrapper
    # The optimization gains from async_eval and batched loss accumulation are sufficient

    # Helper function to evaluate gradient tree (prevents graph accumulation)
    def _eval_grad_tree(tree):
        """Evaluate all arrays in a gradient tree structure."""
        arrays = []

        def collect(t):
            if isinstance(t, dict):
                for v in t.values():
                    collect(v)
            elif isinstance(t, mx.array):
                arrays.append(t)

        collect(tree)
        if arrays:
            mx.eval(*arrays)

    if config.gate_min_value > 0:
        print(f"Gate regularization enabled: min_value={config.gate_min_value}, weight={config.gate_reg_weight}")

    t0 = time.time()
    step_times = []  # Track step times for averaging

    for step in range(config.max_steps):
        step_start = time.perf_counter()

        # Reset memory state at start of each step
        combined_model.reset_memory_state()

        # Gradient accumulation loop
        accumulated_grads = None
        accumulated_backbone_grads = None
        total_loss = mx.array(0.0)  # Keep as mx.array to avoid .item() in loop

        for micro_step in range(config.gradient_accumulation_steps):
            batch_idx = step * config.gradient_accumulation_steps + micro_step
            start_idx = (batch_idx * config.segment_len) % (len(all_tokens) - config.segment_len - 1)
            batch = all_tokens[start_idx : start_idx + config.segment_len + 1]

            input_ids = batch[:-1].reshape(1, -1)
            target_ids = batch[1:].reshape(1, -1)

            loss, full_grads = loss_and_grad_fn(combined_model, input_ids, target_ids)
            titans_grads = filter_titans_grads(full_grads)

            # Debug: show gradient structure at step 0
            if step == 0 and micro_step == 0 and unfrozen_backbone_indices:
                logger.debug("full_grads keys: %s", list(full_grads.keys()))
                if "base_model" in full_grads:
                    base_grads = full_grads["base_model"]
                    logger.debug("base_model keys: %s", list(base_grads.keys()) if isinstance(base_grads, dict) else type(base_grads))
                    if isinstance(base_grads, dict) and "model" in base_grads:
                        model_grads = base_grads["model"]
                        logger.debug("model keys: %s", list(model_grads.keys()) if isinstance(model_grads, dict) else type(model_grads))
                        if "layers" in model_grads:
                            layers_grads = model_grads["layers"]
                            logger.debug("layers type: %s, len: %s", type(layers_grads), len(layers_grads) if hasattr(layers_grads, '__len__') else 'N/A')
                            logger.debug("unfrozen_backbone_indices: %s", sorted(unfrozen_backbone_indices))
                            # Check if layers is list (indexed by position) or dict
                            if isinstance(layers_grads, list):
                                logger.debug("layers is LIST - will index by position")
                                found = [idx for idx in unfrozen_backbone_indices if idx < len(layers_grads) and layers_grads[idx] is not None]
                                logger.debug("unfrozen layers with grads: %s", found)
                            else:
                                logger.debug("layers is DICT with keys: %s", list(layers_grads.keys())[:5])
                else:
                    logger.debug("'base_model' NOT in full_grads - backbone unfreezing may not work!")

            # Extract backbone gradients if unfreezing is enabled
            backbone_grads = None
            if unfrozen_backbone_indices and "base_model" in full_grads:
                base_grads = full_grads.get("base_model", {})
                model_grads = base_grads.get("model", {})
                layers_grads = model_grads.get("layers", [])
                if layers_grads:
                    # Handle both list (MLX) and dict (potential future) formats
                    if isinstance(layers_grads, list):
                        backbone_grads = {
                            idx: layers_grads[idx]
                            for idx in unfrozen_backbone_indices
                            if idx < len(layers_grads) and layers_grads[idx] is not None
                        }
                    else:
                        backbone_grads = {
                            idx: layers_grads.get(str(idx), layers_grads.get(idx))
                            for idx in unfrozen_backbone_indices
                            if str(idx) in layers_grads or idx in layers_grads
                        }

            # OPTIMIZATION: Use async_eval to pipeline graph construction with computation
            # This returns control immediately, allowing next iteration to start building graph
            # while previous computation runs on GPU
            if config.eager_eval:
                mx.async_eval(loss)
                _eval_grad_tree(titans_grads)
                if backbone_grads:
                    _eval_grad_tree(backbone_grads)

            scale_factor = 1.0 / config.gradient_accumulation_steps
            titans_grads_scaled = scale_grads_recursive(titans_grads, scale_factor)

            accumulated_grads = accumulate_grads(accumulated_grads, titans_grads_scaled)

            # Accumulate backbone grads separately
            if backbone_grads:
                backbone_grads_scaled = scale_grads_recursive(backbone_grads, scale_factor)
                accumulated_backbone_grads = accumulate_grads(accumulated_backbone_grads, backbone_grads_scaled)

            # OPTIMIZATION: Collect losses without triggering eval
            # Avoid .item() in loop - batch the conversion later
            total_loss = total_loss + loss  # Keep as mx.array

        # OPTIMIZATION: Convert loss to float only once after accumulation
        avg_loss = float((total_loss / config.gradient_accumulation_steps).item())

        # Get current learning rate
        lr = get_lr(step, config)
        lr_gate = lr * config.gate_lr_scale

        # Extract TITANS gradients
        if "titans_layers" in accumulated_grads:
            titans_grads = accumulated_grads["titans_layers"]
        elif "titans_layer" in accumulated_grads:
            titans_grads = accumulated_grads["titans_layer"]
        else:
            titans_grads = accumulated_grads

        # Gate warmup: freeze gate params
        gate_warmup_active = step < config.gate_warmup_steps

        # Apply optimizers
        # OPTIMIZATION: Batch updates - set LR once, collect grads, then update
        params_to_eval = []

        # Set learning rates once (avoid repeated assignments)
        optimizer_memory.learning_rate = lr
        optimizer_gate.learning_rate = lr_gate

        t_opt_start = time.perf_counter()

        # Update each TITANS layer separately
        all_memory_grads = {}
        all_gate_grads = {}

        for layer_idx, _layer in titans_layers.items():
            # Get gradients for this layer
            if isinstance(titans_grads, dict) and str(layer_idx) in titans_grads:
                layer_grads = titans_grads[str(layer_idx)]
            elif isinstance(titans_grads, dict) and layer_idx in titans_grads:
                layer_grads = titans_grads[layer_idx]
            else:
                layer_grads = titans_grads  # Single layer case

            # Pre-compute masked grads
            all_memory_grads[layer_idx] = create_masked_grads(layer_grads, keep_gate_scale=False)
            all_gate_grads[layer_idx] = create_masked_grads(
                layer_grads, keep_gate_scale=True, freeze_gate=gate_warmup_active
            )

        # Apply all updates
        for layer_idx, layer in titans_layers.items():
            optimizer_memory.update(layer, all_memory_grads[layer_idx])
            optimizer_gate.update(layer, all_gate_grads[layer_idx])
            params_to_eval.extend(tree_flatten(layer.parameters()))

        # Update backbone layers if unfreezing is enabled
        backbone_updated_count = 0
        if unfrozen_backbone_indices and accumulated_backbone_grads and optimizer_backbone:
            lr_backbone = lr * 0.1  # Lower LR for backbone
            optimizer_backbone.learning_rate = lr_backbone

            for idx in unfrozen_backbone_indices:
                # Try both int and string keys for compatibility
                layer_grads = accumulated_backbone_grads.get(idx, accumulated_backbone_grads.get(str(idx)))
                if layer_grads is not None:
                    backbone_layer = model.model.layers[idx]
                    optimizer_backbone.update(backbone_layer, layer_grads)
                    params_to_eval.extend(tree_flatten(backbone_layer.parameters()))
                    backbone_updated_count += 1

            # Debug: show backbone update count at step 0
            if step == 0:
                logger.debug("Updated %d/%d backbone layers", backbone_updated_count, len(unfrozen_backbone_indices))

        _t_opt = time.perf_counter() - t_opt_start  # Keep for debugging

        # OPTIMIZATION: Release intermediate references before eval to reduce peak memory
        # This allows MLX to free gradient arrays before computing parameter updates
        del all_memory_grads, all_gate_grads
        del accumulated_grads
        if accumulated_backbone_grads is not None:
            del accumulated_backbone_grads

        # Eval all updated parameters
        t_eval_start = time.perf_counter()
        mx.eval(*[p for _, p in params_to_eval])
        _t_eval = time.perf_counter() - t_eval_start  # Keep for debugging

        # Track step time
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)

        # Logging
        if step % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Debug gradients at key steps
            if step == 0 or step == 50 or step == 90:
                logger.debug("step %d gradients:", step)
                for path, g in tree_flatten(titans_grads):
                    name = ".".join(str(x) for x in path) if isinstance(path, tuple) else str(path)
                    if "mem_scale" in name or "linear2.bias" in name or "to_lr" in name:
                        grad_val = float(mx.mean(mx.abs(g)).item()) if g.size > 1 else float(g.item())
                        logger.debug("  %s: grad_mean_abs=%.6f", name, grad_val)
                logger.debug("  LR memory=%.6f, LR gate=%.6f", lr, lr_gate)
                logger.debug("  Gate warmup active: %s", gate_warmup_active)

            # Collect stats from all layers
            layer_stats = combined_model.get_layer_stats()
            first_layer_idx = combined_model.memory_layer_indices[0]

            # Use first layer for primary stats (backward compatible)
            mem_scale = layer_stats[first_layer_idx]["mem_scale"]
            gate_bias = layer_stats[first_layer_idx]["gate_bias"]
            gate_mean = 1 / (1 + math.exp(-gate_bias))

            # Get CMS weights if enabled
            cms_weights = {}
            if config.use_cms and titans_layers and first_layer_idx in titans_layers:
                # Get weights from first TITANS layer
                first_layer = titans_layers[first_layer_idx]
                if first_layer._use_cms:
                    weights = mx.softmax(first_layer.memory.level_weights)
                    for i in range(config.num_cms_levels):
                        cms_weights[f"cms_weight_{i}"] = float(weights[i].item())

            # Get internal loss if enabled
            internal_loss_val = 0.0
            if config.use_internal_loss:
                il = combined_model.get_internal_loss()
                if il is not None:
                    internal_loss_val = float(il.item())

            # Build log entry with all layer stats
            log_entry = {
                "step": step,
                "loss": avg_loss,
                "mem_scale": mem_scale,
                "gate_mean": gate_mean,
                "lr": lr,
                "time_ms": dt * 1000,
                "internal_loss": internal_loss_val,
                **cms_weights,
            }

            # Add per-layer stats
            for idx, stats in layer_stats.items():
                log_entry[f"layer_{idx}_mem_scale"] = stats["mem_scale"]
                log_entry[f"layer_{idx}_gate"] = 1 / (1 + math.exp(-stats["gate_bias"]))

            log_history.append(log_entry)

            # Format output string
            cms_str = ""
            if cms_weights:
                cms_str = ", cms=[" + ", ".join(f"{v:.3f}" for v in cms_weights.values()) + "]"

            il_str = ""
            if config.use_internal_loss:
                il_str = f", int_loss={internal_loss_val:.4f}"

            # Show per-layer gates if multiple layers
            num_memory_layers = len(titans_layers) if titans_layers else 0
            if num_memory_layers > 1:
                layer_gates = [f"L{idx}:{1 / (1 + math.exp(-s['gate_bias'])):.3f}" for idx, s in layer_stats.items()]
                gates_str = f", gates=[{', '.join(layer_gates)}]"
            else:
                gates_str = f", gate={gate_mean:.3f}"

            # Timing info - note: fwd_bwd/opt show graph build time, eval shows actual compute
            # With lazy evaluation, most computation happens during eval
            avg_step_time = sum(step_times[-10:]) / len(step_times[-10:]) if step_times else 0
            steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
            # Show actual step time (more accurate than sum of components due to lazy eval)
            timing_str = f" [{avg_step_time * 1000:.0f}ms, {steps_per_sec:.1f} step/s]"

            print(
                f"step {step}: loss={avg_loss:.4f}, mem_scale={mem_scale:.3f}{gates_str}{cms_str}{il_str}, lr={lr:.2e}{timing_str}"
            )

            # === VISUALIZATION: Log metrics ===
            gate_values = {idx: 1 / (1 + math.exp(-s["gate_bias"])) for idx, s in layer_stats.items()}
            cms_list = [cms_weights.get(f"cms_weight_{i}", 0.0) for i in range(config.num_cms_levels)] if cms_weights else None

            viz.log_step(
                step=step,
                loss=avg_loss,
                mem_scale=mem_scale,
                gate_values=gate_values,
                cms_weights=cms_list,
                lr=lr,
                step_time=step_time,
                internal_loss=internal_loss_val,
            )

            # Log gradient statistics
            if titans_grads:
                viz.log_gradients(titans_grads, combined_model.parameters())

            # Log memory states
            for layer_idx, layer in titans_layers.items():
                if hasattr(layer, "memory") and hasattr(layer.memory, "mlp"):
                    if hasattr(layer.memory.mlp, "layers") and layer.memory.mlp.layers:
                        w = layer.memory.mlp.layers[0].weight
                        viz.log_memory_state(w, level=0)

            # Update plots
            viz.update_plots()

            # Generate sample every 50 steps to see learning progress
            if step > 0 and step % 50 == 0:
                prompt = "The"
                try:
                    generated = generate_sample(model, tokenizer, prompt=prompt, max_tokens=20)
                    viz.log_generation(step, prompt, generated, avg_loss)
                    print(f"  Sample: '{generated[:60]}...'")
                except Exception as e:
                    logger.debug(f"Generation failed: {e}")

            # Create inspector snapshot
            inspector.snapshot(step, avg_loss, lr)

    # Save results
    log_path = output_dir / "mlx_training_log.json"
    with log_path.open("w") as f:
        json.dump(log_history, f, indent=2)
    print(f"\nSaved training log to {log_path}")

    # Save TITANS weights
    weights_path = output_dir / "titans_weights.safetensors"
    titans_weights = {}
    # Save each TITANS layer
    for layer_idx, layer in titans_layers.items():
        for name, param in tree_flatten(layer.parameters()):
            # name is already a dotted string like 'gate.linear1.weight'
            key = f"layer_{layer_idx}.{name}"
            titans_weights[key] = param
    mx.save_safetensors(str(weights_path), titans_weights)
    print(f"Saved TITANS weights to {weights_path}")

    # Summary
    if log_history:
        first = log_history[0]
        last = log_history[-1]
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Loss: {first['loss']:.4f} → {last['loss']:.4f}")
        print(f"mem_scale: {first['mem_scale']:.3f} → {last['mem_scale']:.3f}")

        # Show per-layer gate progression
        print("Gate values per layer:")
        layer_indices = sorted(titans_layers.keys())
        for layer_idx in layer_indices:
            first_gate = first.get(f"layer_{layer_idx}_gate", first.get("gate_mean", 0))
            last_gate = last.get(f"layer_{layer_idx}_gate", last.get("gate_mean", 0))
            print(f"  Layer {layer_idx}: {first_gate:.3f} → {last_gate:.3f}")

        if config.use_cms:
            print("CMS weights:")
            for i in range(config.num_cms_levels):
                key = f"cms_weight_{i}"
                print(
                    f"  Level {i} (freq={config.cms_update_frequencies[i]}): {first.get(key, 0):.3f} → {last.get(key, 0):.3f}"
                )

    # === VISUALIZATION: Final report ===
    viz.save_final_report()
    viz.print_sample_generations(last_n=5)
    print(f"\nVisualization plots saved to: {output_dir}/viz/training_progress.png")


def train_scratch(config: MLXTitansConfig):
    """
    NanoGPT-style training from scratch.

    Simple, clean training loop for educational purposes.
    Train ALL parameters of a small GPT model from scratch.
    """
    print("=" * 60)
    print("NanoGPT-style Training FROM SCRATCH")
    print("=" * 60)
    print(f"Model size: {config.model_size}")
    print(f"Max steps: {config.max_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print()

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualizer
    viz = Visualizer(output_dir=str(output_dir / "viz"), plot_every=10)
    print(f"Visualization: {output_dir}/viz/")

    # Create model from scratch
    print("\nCreating model from scratch...")
    model, tokenizer = create_model_scratch(config)

    # Count params
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Single optimizer for all params
    optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=0.01)

    # Load data
    print("\nLoading training data...")
    all_tokens = load_training_data(tokenizer, config)

    # Training loop
    print(f"\nStarting training for {config.max_steps} steps...")
    log_history = []
    t0 = time.time()

    for step in range(config.max_steps):
        step_start = time.perf_counter()

        # Get batch
        idx = (step * config.segment_len) % (len(all_tokens) - config.segment_len - 1)
        batch = all_tokens[idx : idx + config.segment_len + 1]
        inputs = batch[:-1].reshape(1, -1)
        targets = batch[1:].reshape(1, -1)

        # Forward + loss + backward
        def loss_fn(model):
            logits, _ = model(inputs)
            # Cross-entropy loss
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="mean")
            return loss

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, grads)

        # === INSPECT DATA STRUCTURES (every 10 steps) ===
        if step % 10 == 0:
            print(f"\n{'─'*60}")
            print(f"STEP {step} - DATA STRUCTURE INSPECTION")
            print(f"{'─'*60}")

            # 1. Input tokens
            print(f"\n[INPUT]")
            print(f"  Shape: {inputs.shape} (batch, seq_len)")
            print(f"  First 10 tokens: {inputs[0, :10].tolist()}")
            print(f"  Text: '{tokenizer.decode(inputs[0, :20].tolist())[:50]}...'")

            # 2. Forward pass - inspect intermediate values
            if True:  # MLX doesn't need no_grad context
                # Embeddings
                tok_emb = model.wte(inputs)
                pos_emb = model.wpe(mx.arange(inputs.shape[1]))
                embeddings = tok_emb + pos_emb
                mx.eval(embeddings)

                print(f"\n[EMBEDDINGS]")
                print(f"  Token emb shape: {tok_emb.shape}")
                print(f"  Position emb shape: {pos_emb.shape}")
                print(f"  Combined shape: {embeddings.shape}")
                emb_np = embeddings[0, 0, :8].tolist()
                print(f"  First token, first 8 dims: [{', '.join(f'{v:.3f}' for v in emb_np)}]")

                # Logits
                logits, _ = model(inputs)
                mx.eval(logits)
                print(f"\n[LOGITS]")
                print(f"  Shape: {logits.shape} (batch, seq_len, vocab_size)")
                print(f"  Range: [{float(mx.min(logits)):.2f}, {float(mx.max(logits)):.2f}]")

                # Predictions vs targets
                probs = mx.softmax(logits[0, -1], axis=-1)
                top_prob = float(mx.max(probs))
                top_token = int(mx.argmax(probs))
                true_token = int(targets[0, -1])
                print(f"\n[PREDICTION @ last position]")
                print(f"  Predicted: '{tokenizer.decode([top_token])}' (prob={top_prob:.3f})")
                print(f"  Actual: '{tokenizer.decode([true_token])}'")
                print(f"  Correct: {'✓' if top_token == true_token else '✗'}")

            # 3. Gradient inspection
            print(f"\n[GRADIENTS]")
            grad_norms = []
            for name, g in tree_flatten(grads):
                if isinstance(g, mx.array):
                    mx.eval(g)
                    import numpy as np
                    norm = float(np.linalg.norm(np.array(g)))
                    grad_norms.append((name, norm, g.shape))

            # Sort by norm, show top 5
            grad_norms.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top 5 gradients by magnitude:")
            for name, norm, shape in grad_norms[:5]:
                print(f"    {name}: norm={norm:.4f}, shape={shape}")

            # 4. Weight norms
            print(f"\n[WEIGHTS]")
            weight_norms = []
            for name, p in tree_flatten(model.parameters()):
                if isinstance(p, mx.array):
                    mx.eval(p)
                    import numpy as np
                    norm = float(np.linalg.norm(np.array(p)))
                    weight_norms.append((name, norm, p.shape))

            # Show key weights
            for name, norm, shape in weight_norms[:5]:
                print(f"    {name}: norm={norm:.4f}, shape={shape}")

            print(f"\n[LOSS] {loss.item():.4f}")

            # === VISUALIZATIONS ===

            # 5. Attention Heatmap - shows softmax(QK^T / √d)
            print(f"\n[ATTENTION HEATMAP]")
            try:
                attention_maps = model.get_attention_maps(inputs)
                # Visualize first layer's attention
                if attention_maps:
                    mx.eval(attention_maps[0])
                    tokens_for_viz = [tokenizer.decode([int(t)]) for t in inputs[0, :32].tolist()]
                    viz.plot_attention_heatmap(
                        attention_maps[0],
                        tokens=tokens_for_viz,
                        layer_idx=0,
                        max_heads=4,
                        max_seq=32,
                    )
                    print(f"  Saved: attention_layer0.png")
                    print(f"  Shows: which tokens each position 'looks at'")
                    # Also save last layer
                    if len(attention_maps) > 1:
                        mx.eval(attention_maps[-1])
                        viz.plot_attention_heatmap(
                            attention_maps[-1],
                            tokens=tokens_for_viz,
                            layer_idx=len(attention_maps)-1,
                            max_heads=4,
                            max_seq=32,
                        )
                        print(f"  Saved: attention_layer{len(attention_maps)-1}.png")
            except Exception as e:
                print(f"  (attention viz failed: {e})")

            # 6. Gradient Flow Bar Chart - shows ∂Loss/∂W
            print(f"\n[GRADIENT FLOW BAR CHART]")
            try:
                grad_list = [(name, norm) for name, norm, _ in grad_norms]
                viz.plot_gradient_flow(grad_list, step=step)
                print(f"  Saved: gradient_flow_step{step}.png")
                print(f"  Shows: learning signal strength per layer")
            except Exception as e:
                print(f"  (gradient viz failed: {e})")

            print(f"{'─'*60}\n")

        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        loss_val = loss.item()
        step_time = time.perf_counter() - step_start

        # Logging every 10 steps
        if step % 10 == 0:
            # Log to visualizer
            viz.log_step(
                step=step,
                loss=loss_val,
                mem_scale=0.0,
                gate_values={},
                lr=config.learning_rate,
                step_time=step_time,
            )
            viz.update_plots()

            log_history.append({"step": step, "loss": loss_val})

            steps_per_sec = 1.0 / step_time if step_time > 0 else 0
            print(f"step {step}: loss={loss_val:.4f} [{step_time*1000:.0f}ms, {steps_per_sec:.1f} step/s]")

        # Generate sample every 100 steps
        if step > 0 and step % 100 == 0:
            try:
                prompt_tokens = mx.array([[tokenizer.encode("The")[0]]])
                generated = model.generate(prompt_tokens, max_new_tokens=20, temperature=0.8)
                text = tokenizer.decode(generated[0].tolist())
                viz.log_generation(step, "The", text, loss_val)
                print(f"  Sample: '{text[:60]}...'")
            except Exception as e:
                logger.debug(f"Generation failed: {e}")

    # Save results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    if log_history:
        print(f"Loss: {log_history[0]['loss']:.4f} -> {log_history[-1]['loss']:.4f}")

    # Save model
    weights_path = output_dir / "model_weights.safetensors"
    model_weights = {name: param for name, param in tree_flatten(model.parameters())}
    mx.save_safetensors(str(weights_path), model_weights)
    print(f"Saved model to {weights_path}")

    # Save log
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)

    viz.save_final_report()
    print(f"\nVisualization: {output_dir}/viz/training_progress.png")


def parse_memory_layers(s: str) -> list:
    """Parse comma-separated layer indices."""
    if not s:
        return [12]
    return [int(x.strip()) for x in s.split(",")]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLX TITANS Training")

    # Mode selection
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="Train GPT from scratch (nanoGPT style) instead of fine-tuning",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=["nano", "small", "medium", "large"],
        help="Model size for from-scratch training: nano (10M), small (45M), medium (124M), large (250M)",
    )

    # Fine-tuning options
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument(
        "--memory_layers",
        type=str,
        default="12",
        help="Comma-separated layer indices for TITANS (e.g., '6,12,18')",
    )
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gate_lr_scale", type=float, default=50.0)
    parser.add_argument("--segment_len", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="out-mlx-titans")
    parser.add_argument("--use_cms", action="store_true", default=True)
    parser.add_argument("--no_cms", action="store_false", dest="use_cms")
    parser.add_argument("--gate_warmup_steps", type=int, default=0)
    parser.add_argument("--no_adaptive", action="store_true")
    parser.add_argument(
        "--use_internal_loss",
        action="store_true",
        default=True,
        help="Enable internal loss for memory to learn (default: True)",
    )
    parser.add_argument(
        "--no_internal_loss",
        action="store_false",
        dest="use_internal_loss",
        help="Disable internal loss",
    )
    parser.add_argument(
        "--internal_loss_weight",
        type=float,
        default=0.1,
        help="Weight for internal loss (default: 0.1)",
    )
    parser.add_argument(
        "--unfreeze_backbone_layers",
        type=int,
        default=0,
        help="Number of backbone layers to unfreeze near TITANS layers (0=frozen)",
    )
    parser.add_argument(
        "--gate_min_value",
        type=float,
        default=0.15,
        help="Minimum gate value for regularization (default: 0.15, like PyTorch)",
    )
    parser.add_argument(
        "--gate_reg_weight",
        type=float,
        default=1.0,
        help="Weight for gate regularization loss (default: 1.0)",
    )
    parser.add_argument(
        "--gate_init_bias",
        type=float,
        default=-2.0,
        help="Initial gate bias (default: -2.0 → sigmoid ≈ 0.12, like PyTorch)",
    )
    parser.add_argument(
        "--no_eager_eval",
        action="store_true",
        help="Disable eager evaluation (slower but uses less memory per micro-step)",
    )
    parser.add_argument(
        "--use_cascade",
        action="store_true",
        default=False,
        help="Use cascade mode for CMS (each level transforms previous level's output)",
    )
    parser.add_argument(
        "--surprise_threshold",
        type=float,
        default=0.0,
        help="Skip memory updates when grad norm below threshold (0=disabled)",
    )
    parser.add_argument(
        "--memory_grad_clip",
        type=float,
        default=1.0,
        help="Per-level gradient clipping for CMS",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["wikitext", "fineweb-edu", "slimpajama"],
        help="Training dataset: 'wikitext' (small, for testing), 'fineweb-edu' (recommended for TITANS), 'slimpajama' (diverse)",
    )
    parser.add_argument(
        "--min_doc_length",
        type=int,
        default=100,
        help="Minimum document length in tokens. Use 2048+ for memory training (default: 100)",
    )
    parser.add_argument(
        "--max_doc_length",
        type=int,
        default=8192,
        help="Maximum document length in tokens (default: 8192)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=0,
        help="Number of examples to use (0 = auto based on max_steps)",
    )

    args = parser.parse_args()

    # Parse memory layers
    memory_layers = parse_memory_layers(args.memory_layers)

    config = MLXTitansConfig(
        # Mode
        train_from_scratch=args.from_scratch,
        model_size=args.model_size,
        # Fine-tuning
        model_name=args.model_name,
        memory_layers=memory_layers,
        # Training
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        gate_lr_scale=args.gate_lr_scale,
        segment_len=args.segment_len,
        output_dir=args.output_dir,
        use_cms=args.use_cms,
        use_cascade=args.use_cascade,
        gate_warmup_steps=args.gate_warmup_steps,
        adaptive_memory=not args.no_adaptive,
        use_internal_loss=args.use_internal_loss,
        internal_loss_weight=args.internal_loss_weight,
        gate_min_value=args.gate_min_value,
        gate_reg_weight=args.gate_reg_weight,
        gate_init_bias=args.gate_init_bias,
        eager_eval=not args.no_eager_eval,
        unfreeze_backbone_layers=args.unfreeze_backbone_layers,
        surprise_threshold=args.surprise_threshold,
        memory_grad_clip=args.memory_grad_clip,
        # Dataset config
        dataset=args.dataset,
        min_doc_length=args.min_doc_length,
        max_doc_length=args.max_doc_length,
        num_examples=args.num_examples,
    )

    # Choose training function based on mode
    if config.train_from_scratch:
        train_scratch(config)
    else:
        train(config)


if __name__ == "__main__":
    main()

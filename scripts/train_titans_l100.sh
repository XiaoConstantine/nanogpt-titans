#!/bin/bash
# L100 Training Script for Titans Validation
# 
# This script runs a proper experiment to validate Titans memory works.
# Key changes from previous runs:
# 1. Longer sequences (8K tokens) - forces memory to provide info attention can't reach
# 2. Longer training (3000 steps) - enough for memory to learn
# 3. Multiple memory layers - more capacity
# 4. Proper baseline comparison

set -e

# Configuration
MODEL="Qwen/Qwen2-0.5B"
OUTPUT_DIR="out-titans-l100-validation"
STEPS=3000
BATCH_SIZE=2
GRAD_ACCUM=16  # Effective batch = 32
MAX_LENGTH=8192  # 8K tokens - key change!
SEGMENT_LEN=512  # 16 segments per sequence
MEMORY_LAYERS="8,12,16"  # Multiple layers

echo "=============================================="
echo "Titans L100 Validation Training"
echo "=============================================="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Max length: $MAX_LENGTH tokens"
echo "Segment length: $SEGMENT_LEN"
echo "Memory layers: $MEMORY_LAYERS"
echo "Steps: $STEPS"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "=============================================="

# =============================================================================
# STEP 1: Evaluate baseline (before training)
# =============================================================================
echo ""
echo "=============================================="
echo "STEP 1: Baseline Qwen Evaluation (no memory)"
echo "=============================================="

uv run python -m nanogpt_titans.eval_perplexity \
    --qwen \
    --model_name "$MODEL" \
    --segment_len $SEGMENT_LEN \
    --num_samples 100 \
    --dtype bfloat16 \
    2>&1 | tee "$OUTPUT_DIR/baseline_eval.log"

# =============================================================================
# STEP 2: Train Titans
# =============================================================================
echo ""
echo "=============================================="
echo "STEP 2: Training Titans Memory"
echo "=============================================="

uv run python -m nanogpt_titans.train_qwen_titans \
    --model_name "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps $STEPS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_length $MAX_LENGTH \
    --segment_len $SEGMENT_LEN \
    --memory_layers $MEMORY_LAYERS \
    --learning_rate 1e-4 \
    --warmup_steps 200 \
    --eval_interval 200 \
    --save_interval 500 \
    --use_cms \
    --num_cms_levels 3 \
    --use_self_mod_proj \
    --use_self_mod_gate \
    --use_internal_loss \
    --internal_loss_weight 0.001 \
    --dataset_name "wikitext" \
    --dataset_config "wikitext-103-raw-v1" \
    --wandb_log \
    --wandb_project "titans-validation" \
    --wandb_run_name "l100-8k-3layers"

# =============================================================================
# STEP 3: Evaluate trained Titans
# =============================================================================
echo ""
echo "=============================================="
echo "STEP 3: Titans Evaluation (after training)"
echo "=============================================="

uv run python -m nanogpt_titans.eval_perplexity \
    --qwen \
    --model_name "$MODEL" \
    --titans_state "$OUTPUT_DIR/titans_state_final.pt" \
    --memory_layers "$MEMORY_LAYERS" \
    --segment_len $SEGMENT_LEN \
    --num_samples 100 \
    --dtype bfloat16 \
    2>&1 | tee "$OUTPUT_DIR/titans_eval.log"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "VALIDATION COMPLETE"
echo "=============================================="
echo ""
echo "Compare results:"
echo "  Baseline: $OUTPUT_DIR/baseline_eval.log"
echo "  Titans:   $OUTPUT_DIR/titans_eval.log"
echo ""
echo "Key metric: Titans should show LARGER improvement (early->late) than baseline"
echo ""
echo "If both show same improvement: memory learned nothing useful"
echo "If Titans shows MORE improvement: memory is working!"

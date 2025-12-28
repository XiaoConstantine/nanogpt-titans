#!/bin/bash
# L100 Training Script for Titans Validation
#
# Usage:
#   ./train_titans_l100.sh              # Full pipeline: baseline eval -> train -> final eval
#   ./train_titans_l100.sh --train-only # Skip baseline eval, just train
#   ./train_titans_l100.sh --resume     # Resume training from checkpoint
#   ./train_titans_l100.sh --eval-only  # Skip training, just evaluate existing model

set -e

# =============================================================================
# Parse arguments
# =============================================================================
TRAIN_ONLY=false
RESUME=false
EVAL_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --train-only)
            TRAIN_ONLY=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--train-only] [--resume] [--eval-only]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Configuration
# =============================================================================
MODEL="Qwen/Qwen2-0.5B"
OUTPUT_DIR="out-titans-l100-validation"
STEPS=3000
BATCH_SIZE=2
GRAD_ACCUM=16  # Effective batch = 32
MAX_LENGTH=8192  # 8K tokens - key change!
SEGMENT_LEN=512  # 16 segments per sequence
MEMORY_LAYERS="8,12,16"  # Multiple layers

# Create output directory
mkdir -p "$OUTPUT_DIR"

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
echo ""
echo "Options:"
echo "  Train only: $TRAIN_ONLY"
echo "  Resume: $RESUME"
echo "  Eval only: $EVAL_ONLY"
echo "=============================================="

# =============================================================================
# STEP 1: Evaluate baseline (before training)
# =============================================================================
if [[ "$TRAIN_ONLY" == "false" && "$RESUME" == "false" && "$EVAL_ONLY" == "false" ]]; then
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
else
    echo ""
    echo "Skipping baseline evaluation..."
fi

# =============================================================================
# STEP 2: Train Titans
# =============================================================================
if [[ "$EVAL_ONLY" == "false" ]]; then
    echo ""
    echo "=============================================="
    echo "STEP 2: Training Titans Memory"
    echo "=============================================="

    # Build training command
    # Using MAG variant (Memory as Gate) - more stable than MAC, no gate collapse
    TRAIN_CMD="uv run python -m nanogpt_titans.train_qwen_titans \
        --model_name $MODEL \
        --output_dir $OUTPUT_DIR \
        --titans_variant mag \
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
        --num_cms_levels 3 \
        --internal_loss_weight 0.01 \
        --dataset_name wikitext \
        --dataset_config wikitext-103-raw-v1"

    # Add resume flag if requested
    if [[ "$RESUME" == "true" ]]; then
        CHECKPOINT="$OUTPUT_DIR/best.pt"
        if [[ -f "$CHECKPOINT" ]]; then
            echo "Resuming from checkpoint: $CHECKPOINT"
            TRAIN_CMD="$TRAIN_CMD --resume_from $CHECKPOINT"
        else
            echo "WARNING: No checkpoint found at $CHECKPOINT, starting fresh"
        fi
    fi

    # Run training
    eval $TRAIN_CMD
else
    echo ""
    echo "Skipping training..."
fi

# =============================================================================
# STEP 3: Evaluate trained Titans
# =============================================================================
echo ""
echo "=============================================="
echo "STEP 3: Titans Evaluation (after training)"
echo "=============================================="

# Check which state file to use
if [[ -f "$OUTPUT_DIR/titans_state_final.pt" ]]; then
    TITANS_STATE="$OUTPUT_DIR/titans_state_final.pt"
elif [[ -f "$OUTPUT_DIR/titans_state_best.pt" ]]; then
    TITANS_STATE="$OUTPUT_DIR/titans_state_best.pt"
else
    # Find latest checkpoint
    TITANS_STATE=$(ls -t "$OUTPUT_DIR"/titans_state_*.pt 2>/dev/null | head -1)
fi

if [[ -z "$TITANS_STATE" || ! -f "$TITANS_STATE" ]]; then
    echo "ERROR: No Titans state file found in $OUTPUT_DIR"
    exit 1
fi

echo "Using Titans state: $TITANS_STATE"

uv run python -m nanogpt_titans.eval_perplexity \
    --qwen \
    --model_name "$MODEL" \
    --titans_state "$TITANS_STATE" \
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

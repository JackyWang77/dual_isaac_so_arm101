#!/bin/bash
# ============================================================
# Train Residual RL (Graph-Unet only)
# ============================================================
#
# Residual RL is Unet-only:
#   - Graph-Unet backbone (frozen): base action + single z
#   - Z Adapter (trainable): frozen z → RL-friendly z
#   - Residual Actor (trainable): δa ~ π(δa | obs, a_base, z_bar)
#   - Value Critic (trainable): bar head only
#
# Usage:
#   ./train_residual_rl.sh <pretrained_unet_checkpoint> [options]
#
# ============================================================

set -e

# ============================================================
# Default Arguments
# ============================================================
PRETRAINED_CHECKPOINT="${1:-}"
NUM_ENVS="${2:-2}"
MAX_ITERATIONS="${3:-500}"
STEPS_PER_ENV="${4:-130}"  # Increased from 24 to 130 to allow episodes to complete
MINI_BATCH_SIZE="${5:-64}"
NUM_EPOCHS="${6:-5}"
SEED="${7:-42}"
HEADLESS="${8:-false}"  # Default: false (enable visualization)

# Task (can be overridden)
TASK="${TASK:-SO-ARM101-Lift-Cube-v0}"

# ============================================================
# Validate Arguments
# ============================================================
if [ -z "$PRETRAINED_CHECKPOINT" ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              Residual RL Training (Graph-Unet only)           ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "❌ Error: Missing required argument: pretrained_unet_checkpoint"
    echo ""
    echo "Usage:"
    echo "  $0 <pretrained_unet_checkpoint> [num_envs] [max_iter] [steps_per_env] [mini_batch] [epochs] [seed] [headless]"
    echo ""
    echo "Examples:"
    echo "  $0 ./logs/graph_unet/lift_joint/best_model.pt"
    echo "  $0 ./logs/graph_unet/lift_joint/best_model.pt 64 500 130 64 5 42 false"
    echo ""
    exit 1
fi

if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $PRETRAINED_CHECKPOINT"
    exit 1
fi

# ============================================================
# Check Environment
# ============================================================
echo ""
echo "Checking environment..."

# Set headless flag
if [ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ]; then
    HEADLESS_FLAG="--headless"
else
    HEADLESS_FLAG=""
fi

# ============================================================
# Print Configuration
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              Residual RL (Graph-Unet only)                     ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║   Graph-Unet (frozen) → base action + single z                 ║"
echo "║   Z Adapter (trainable) → RL-friendly z                        ║"
echo "║   Residual Actor (trainable) → δa ~ π(δa | obs, a_base, z_bar) ║"
echo "║   Value Critic (trainable) → bar head only                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Task:                 $TASK"
echo "  Pretrained Graph-Unet: $PRETRAINED_CHECKPOINT"
echo "  Num envs:             $NUM_ENVS"
echo "  Max iterations:       $MAX_ITERATIONS"
echo "  Steps per env:        $STEPS_PER_ENV"
echo "  Mini-batch size:      $MINI_BATCH_SIZE"
echo "  Epochs per iter:      $NUM_EPOCHS"
echo "  Seed:                 $SEED"
echo "  Headless:             $HEADLESS"
echo ""
echo "Computed:"
echo "  Batch size:           $((NUM_ENVS * STEPS_PER_ENV))"
echo "  Updates per iter:     $((NUM_ENVS * STEPS_PER_ENV / MINI_BATCH_SIZE * NUM_EPOCHS))"
echo ""

# ============================================================
# Confirm
# ============================================================
read -p "Start training? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Aborted."
    exit 0
fi

# ============================================================
# Run Training
# ============================================================
echo ""
echo "Starting training..."
echo "========================================"

# Check for gripper model (optional)
GRIPPER_MODEL="${GRIPPER_MODEL:-}"
if [ -n "$GRIPPER_MODEL" ] && [ -f "$GRIPPER_MODEL" ]; then
    echo "  Gripper model:      $GRIPPER_MODEL"
    GRIPPER_MODEL_ARG="--gripper-model $GRIPPER_MODEL"
else
    echo "  Gripper model:      Not provided (using Graph-Unet for gripper)"
    GRIPPER_MODEL_ARG=""
fi

# Use isaaclab.sh to properly initialize Isaac Sim
python scripts/graph_dit_rl/train_graph_rl.py \
    --task "$TASK" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    $GRIPPER_MODEL_ARG \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --steps_per_env "$STEPS_PER_ENV" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --seed "$SEED" \
    --log_dir "./logs/graph_unet_rl" \
    --save_interval 50 \
    $HEADLESS_FLAG

echo ""
echo "========================================"
echo "✅ Training completed!"
echo ""
echo "Logs saved to: ./logs/graph_unet_rl/$TASK/"
echo "========================================"
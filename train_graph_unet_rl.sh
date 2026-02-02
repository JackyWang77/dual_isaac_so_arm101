#!/bin/bash
# ============================================================
# Train Graph-Unet + Residual RL Policy
# ============================================================
#
# Same method as GR-DiT but with Graph-Unet backbone:
#   - Graph-Unet (frozen): base action + single z (no layer-wise gate)
#   - Z Adapter (trainable): frozen z → RL-friendly z
#   - Residual Actor (trainable): delta_a ~ π(δa | obs, a_base, z_bar)
#   - Value Critic (trainable): bar head only (no layer heads)
#
# Usage:
#   ./train_graph_unet_rl.sh [pretrained_unet_checkpoint] [num_envs] [max_iter] ...
#
# ============================================================

set -e

PRETRAINED_CHECKPOINT="${1:-}"
NUM_ENVS="${2:-2}"
MAX_ITERATIONS="${3:-500}"
STEPS_PER_ENV="${4:-130}"
MINI_BATCH_SIZE="${5:-64}"
NUM_EPOCHS="${6:-5}"
SEED="${7:-42}"
HEADLESS="${8:-false}"

TASK="${TASK:-SO-ARM101-Lift-Cube-v0}"

if [ -z "$PRETRAINED_CHECKPOINT" ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║              Graph-Unet + Residual RL Training                 ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "❌ Error: Missing required argument: pretrained Graph-Unet checkpoint"
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

if [ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ]; then
    HEADLESS_FLAG="--headless"
else
    HEADLESS_FLAG=""
fi

echo ""
echo "Configuration:"
echo "  Task:                 $TASK"
echo "  Pretrained Graph-Unet: $PRETRAINED_CHECKPOINT"
echo "  Num envs:             $NUM_ENVS"
echo "  Max iterations:       $MAX_ITERATIONS"
echo "  Steps per env:        $STEPS_PER_ENV"
echo "  Log dir:              ./logs/graph_unet_rl"
echo ""

read -p "Start training? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting training..."
echo "========================================"

python scripts/graph_dit_rl/train_graph_rl.py \
    --task "$TASK" \
    --backbone graph_unet \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
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
echo "Logs saved to: ./logs/graph_unet_rl/$TASK/"
echo "========================================"

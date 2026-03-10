#!/bin/bash
# Baseline Residual RL training - Dual Cube Stack
# Uses UnetPolicy (MLP backbone, no graph attention) with default conservative hyperparams.
#
# Usage:
#   bash stack/train_baseline_rl.sh <pretrained_checkpoint>
#   bash stack/train_baseline_rl.sh ./logs/graph_unet_full/stack_joint/best_model.pt
#   NUM_ENVS=32 MAX_ITER=500 bash stack/train_baseline_rl.sh ./logs/.../best_model.pt
cd "$(dirname "$0")/.."
set -e

PRETRAINED_CHECKPOINT="${1:-}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERATIONS="${MAX_ITER:-200}"
STEPS_PER_ENV="${STEPS_PER_ENV:-400}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
C_DELTA_REG="${C_DELTA_REG:-2.0}"
C_ENT="${C_ENT:-0.0}"
BETA="${BETA:-1.0}"
ALPHA_INIT="${ALPHA_INIT:-0.10}"
EXPECTILE_TAU="${EXPECTILE_TAU:-0.5}"
LR="${LR:-5e-4}"
SEED="${SEED:-42}"
HEADLESS="${HEADLESS:-true}"

TASK="${TASK:-SO-ARM101-Dual-Cube-Stack-RL-v0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
USE_ADAPTIVE_ALPHA="${USE_ADAPTIVE_ALPHA:-false}"
USE_ADAPTIVE_ENTROPY="${USE_ADAPTIVE_ENTROPY:-false}"
C_ENT_BAD="${C_ENT_BAD:-0.005}"
C_ENT_GOOD="${C_ENT_GOOD:-0.0001}"
LOG_DIR="${LOG_DIR:-./logs/baseline_rl}"

if [ -z "$PRETRAINED_CHECKPOINT" ] || [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "Usage: $0 <pretrained_checkpoint>"
    echo ""
    echo "Example:"
    echo "  $0 ./logs/graph_unet_full/stack_joint/best_model.pt"
    echo ""
    echo "Environment variables:"
    echo "  NUM_ENVS=64        Number of parallel environments"
    echo "  MAX_ITER=200       Maximum RL iterations"
    echo "  STEPS_PER_ENV=160  Steps per env per iteration"
    echo "  LR=5e-4            Learning rate"
    echo "  SEED=42            Random seed"
    echo "  LOG_DIR=./logs/baseline_rl  Log directory"
    echo "  HEADLESS=true      Run headless (no GUI)"
    exit 1
fi

HEADLESS_FLAG=""
if [ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ]; then
    HEADLESS_FLAG="--headless"
fi
ADAPTIVE_ALPHA_FLAG=""
[ "$USE_ADAPTIVE_ALPHA" = "false" ] && ADAPTIVE_ALPHA_FLAG="--no_adaptive_alpha"
ADAPTIVE_ENTROPY_FLAG=""
[ "$USE_ADAPTIVE_ENTROPY" = "false" ] && ADAPTIVE_ENTROPY_FLAG="--no_adaptive_entropy"

echo "========================================"
echo "Baseline Residual RL - Dual Cube Stack"
echo "========================================"
echo "Pretrained: $PRETRAINED_CHECKPOINT"
echo "Task:       $TASK"
echo "Envs:       $NUM_ENVS"
echo "Iterations: $MAX_ITERATIONS"
echo "LR:         $LR"
echo "Seed:       $SEED"
echo "Log:        $LOG_DIR"
echo "========================================"

python scripts/graph_dit_rl/train_graph_rl.py \
    --task "$TASK" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --policy_type unet \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --steps_per_env "$STEPS_PER_ENV" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --c_delta_reg "$C_DELTA_REG" \
    --c_ent "$C_ENT" \
    --c_ent_bad "$C_ENT_BAD" \
    --c_ent_good "$C_ENT_GOOD" \
    --beta "$BETA" \
    --alpha_init "$ALPHA_INIT" \
    --expectile_tau "$EXPECTILE_TAU" \
    $ADAPTIVE_ALPHA_FLAG \
    $ADAPTIVE_ENTROPY_FLAG \
    --lr "$LR" \
    --seed "$SEED" \
    --log_dir "$LOG_DIR" \
    --save_interval "$SAVE_INTERVAL" \
    $HEADLESS_FLAG

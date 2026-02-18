#!/bin/bash
# Train UnetPolicy + Residual RL - Lift task
cd "$(dirname "$0")/.."
set -e

PRETRAINED_CHECKPOINT="${1:-}"
NUM_ENVS="${2:-64}"
MAX_ITERATIONS="${3:-200}"
STEPS_PER_ENV="${4:-160}"
MINI_BATCH_SIZE="${5:-64}"
NUM_EPOCHS="${6:-2}"
C_DELTA_REG="${7:-2.0}"
C_ENT="${8:-0.0}"
BETA="${9:-1.0}"
ALPHA_INIT="${10:-0.10}"
EXPECTILE_TAU="${11:-0.7}"
LR="${12:-5e-4}"
SEED="${13:-42}"
HEADLESS="${14:-true}"

TASK="${TASK:-SO-ARM101-Lift-Cube-v0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
USE_ADAPTIVE_ALPHA="${USE_ADAPTIVE_ALPHA:-false}"
USE_ADAPTIVE_ENTROPY="${USE_ADAPTIVE_ENTROPY:-false}"
C_ENT_BAD="${C_ENT_BAD:-0.005}"
C_ENT_GOOD="${C_ENT_GOOD:-0.0001}"

if [ -z "$PRETRAINED_CHECKPOINT" ] || [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "Usage: $0 <pretrained_checkpoint> [num_envs] [max_iter] ..."
    echo ""
    echo "Example:"
    echo "  $0 ./logs/graph_unet/lift_joint/best_model.pt"
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
echo "Training UnetPolicy + RL - Lift"
echo "Pretrained: $PRETRAINED_CHECKPOINT"
echo "========================================"

python scripts/graph_dit_rl/train_graph_rl.py \
    --task "$TASK" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
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
    --log_dir "./logs/graph_unet_rl" \
    --save_interval "$SAVE_INTERVAL" \
    $HEADLESS_FLAG

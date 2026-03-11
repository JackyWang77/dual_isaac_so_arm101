#!/bin/bash
# Train DualArmDisentangledPolicyGated + Residual RL - Dual Cube Stack task
# Backbone type is auto-detected from checkpoint (dual_arm_gated, dual_arm, graph_unet, etc.)
cd "$(dirname "$0")/.."
set -e

PRETRAINED_CHECKPOINT="${1:-}"
NUM_ENVS="${2:-128}"
MAX_ITERATIONS="${3:-100}"
STEPS_PER_ENV="${4:-405}"
MINI_BATCH_SIZE="${5:-64}"
NUM_EPOCHS="${6:-2}"
C_DELTA_REG="${7:-5.0}"
C_ENT="${8:-0.0}"
BETA="${9:-1.0}"
ALPHA_INIT="${10:-0.05}"
EXPECTILE_TAU="${11:-0.5}"
LR="${12:-1e-4}"
SEED="${13:-42}"
HEADLESS="${14:-true}"

TASK="${TASK:-SO-ARM101-Dual-Cube-Stack-RL-v0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
USE_ADAPTIVE_ALPHA="${USE_ADAPTIVE_ALPHA:-false}"
USE_ADAPTIVE_ENTROPY="${USE_ADAPTIVE_ENTROPY:-false}"
C_ENT_BAD="${C_ENT_BAD:-0.02}"
C_ENT_GOOD="${C_ENT_GOOD:-0.005}"
CRITIC_WARMUP_ITERS="${CRITIC_WARMUP_ITERS:-5}"
LOG_DIR="${LOG_DIR:-./logs/dual_arm_rl}"

if [ -z "$PRETRAINED_CHECKPOINT" ] || [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "Usage: $0 <pretrained_checkpoint> [num_envs] [max_iter] [steps_per_env] [mini_batch_size] [num_epochs] [c_delta_reg] [c_ent] [beta] [alpha_init] [expectile_tau] [lr] [seed] [headless]"
    echo ""
    echo "Example (baseline):"
    echo "  $0 ./logs/gated_small/stack_joint/best_model.pt 128 200"
    echo ""
    echo "Ablation examples:"
    echo "  $0 ./path/to/best_model.pt 128 200 160 64 2 0.5   # c_delta_reg=0.5"
    echo "  $0 ./path/to/best_model.pt 128 200 160 64 2 2.0 0.0 0.3  # beta=0.3"
    echo "  $0 ./path/to/best_model.pt 128 200 160 32  # mini_batch_size=32"
    echo ""
    echo "Environment variables:"
    echo "  TASK=SO-ARM101-Dual-Cube-Stack-RL-v0  (RL rewards, default)"
    echo "  TASK=SO-ARM101-Dual-Cube-Stack-v0     (original BC rewards)"
    echo "  LOG_DIR=./logs/ablation_xxx            (custom log directory)"
    echo "  SEED=123                               (random seed)"
    echo "  CRITIC_WARMUP_ITERS=5                  (first N iters: only train critic, default 5)"
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
echo "Training Dual Arm + Residual RL - Stack"
echo "Pretrained: $PRETRAINED_CHECKPOINT"
echo "Task: $TASK"
echo "Envs=$NUM_ENVS Iter=$MAX_ITERATIONS Steps=$STEPS_PER_ENV"
echo "Batch=$MINI_BATCH_SIZE Epochs=$NUM_EPOCHS"
echo "c_delta_reg=$C_DELTA_REG beta=$BETA c_ent=$C_ENT critic_warmup=$CRITIC_WARMUP_ITERS"
echo "Log: $LOG_DIR"
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
    --critic_warmup_iters "$CRITIC_WARMUP_ITERS" \
    --log_dir "$LOG_DIR" \
    --save_interval "$SAVE_INTERVAL" \
    $HEADLESS_FLAG

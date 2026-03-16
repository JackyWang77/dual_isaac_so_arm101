#!/bin/bash
# Train DualArmDisentangledPolicyGated + Residual RL - Dual Cube Stack task
# Backbone type is auto-detected from checkpoint (dual_arm_gated, dual_arm, graph_unet, etc.)
cd "$(dirname "$0")/.."
set -e

PRETRAINED_CHECKPOINT="${1:-}"
RESUME_CHECKPOINT="${2:-}"
NUM_ENVS="${3:-512}"
MAX_ITERATIONS="${4:-200}"
STEPS_PER_ENV="${5:-405}"
MINI_BATCH_SIZE="${6:-512}"
NUM_EPOCHS="${7:-2}"
ALPHA_INIT="${8:-0.05}"
ALPHA_MAX="${ALPHA_MAX:-0.05}"
MAX_DELTA_NORM="${MAX_DELTA_NORM:-0.12}"
EXPECTILE_TAU="${9:-0.5}"
LR="${10:-1e-4}"
SEED="${11:-42}"
HEADLESS="${12:-true}"

TASK="${TASK:-SO-ARM101-Dual-Cube-Stack-RL-v0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
USE_ADAPTIVE_ALPHA="${USE_ADAPTIVE_ALPHA:-false}"
CRITIC_WARMUP_ITERS="${CRITIC_WARMUP_ITERS:-10}"
USE_COUNTERFACTUAL_Q="${USE_COUNTERFACTUAL_Q:-true}"
COUNTERFACTUAL_LOG_TAU="${COUNTERFACTUAL_LOG_TAU:-0.5}"
LOG_DIR="${LOG_DIR:-./logs/dual_arm_rl}"
RUN_NAME="${RUN_NAME:-}"

# SAC-style adaptive parameters (all data-driven, no manual tuning needed)
USE_ADAPTIVE_DELTA_REG="${USE_ADAPTIVE_DELTA_REG:-true}"
TARGET_DELTA_NORM="${TARGET_DELTA_NORM:-0.12}"
C_DELTA_REG_INIT="${C_DELTA_REG_INIT:-10.0}"
USE_AUTO_ENTROPY="${USE_AUTO_ENTROPY:-true}"
TARGET_ENTROPY="${TARGET_ENTROPY:--6.0}"
C_ENT_INIT="${C_ENT_INIT:-0.01}"
USE_ADAPTIVE_BETA="${USE_ADAPTIVE_BETA:-true}"
TARGET_EFF_RATIO="${TARGET_EFF_RATIO:-0.4}"
BETA_INIT="${BETA_INIT:-2.0}"

if [ -z "$PRETRAINED_CHECKPOINT" ] || [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "Usage: $0 <pretrained_checkpoint> [resume_checkpoint] [num_envs] [max_iter] ..."
    echo "  resume_checkpoint: optional, RL checkpoint to continue training (policy_iter_X.pt or policy_final.pt)"
    echo ""
    echo "Example (baseline):"
    echo "  $0 ./logs/gated_small/stack_joint/best_model.pt 128 200"
    echo ""
    echo "Example (resume from iter 100 to 200):"
    echo "  $0 ./logs/gated_small/stack_joint/best_model.pt ./logs/dual_arm_rl/xxx/policy_iter_100.pt 128 200"
    echo ""
    echo "Environment variables:"
    echo "  TASK=SO-ARM101-Dual-Cube-Stack-RL-v0  (RL rewards, default)"
    echo "  LOG_DIR=./logs/ablation_xxx            (custom log directory)"
    echo "  SEED=123                               (random seed)"
    echo "  CRITIC_WARMUP_ITERS=10                 (first N iters: only train critic)"
    echo "  TARGET_DELTA_NORM=0.25                 (target ||δ|| for adaptive delta_reg)"
    echo "  TARGET_ENTROPY=-6.0                    (target entropy for auto entropy)"
    exit 1
fi

HEADLESS_FLAG=""
if [ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ]; then
    HEADLESS_FLAG="--headless"
fi
ADAPTIVE_ALPHA_FLAG=""
[ "$USE_ADAPTIVE_ALPHA" = "false" ] && ADAPTIVE_ALPHA_FLAG="--no_adaptive_alpha"
COUNTERFACTUAL_Q_FLAG=""
[ "$USE_COUNTERFACTUAL_Q" = "true" ] && COUNTERFACTUAL_Q_FLAG="--use_counterfactual_q --counterfactual_log_tau $COUNTERFACTUAL_LOG_TAU"

# SAC-style adaptive flags
ADAPTIVE_DELTA_REG_FLAG=""
if [ "$USE_ADAPTIVE_DELTA_REG" = "true" ]; then
    ADAPTIVE_DELTA_REG_FLAG="--use_adaptive_delta_reg --target_delta_norm $TARGET_DELTA_NORM --c_delta_reg_init $C_DELTA_REG_INIT"
else
    ADAPTIVE_DELTA_REG_FLAG="--no_adaptive_delta_reg"
fi
AUTO_ENTROPY_FLAG=""
if [ "$USE_AUTO_ENTROPY" = "true" ]; then
    AUTO_ENTROPY_FLAG="--use_auto_entropy --target_entropy $TARGET_ENTROPY --c_ent_init $C_ENT_INIT"
else
    AUTO_ENTROPY_FLAG="--no_auto_entropy"
fi
ADAPTIVE_BETA_FLAG=""
if [ "$USE_ADAPTIVE_BETA" = "true" ]; then
    ADAPTIVE_BETA_FLAG="--use_adaptive_beta --target_eff_ratio $TARGET_EFF_RATIO --beta_init $BETA_INIT"
else
    ADAPTIVE_BETA_FLAG="--no_adaptive_beta"
fi

echo "========================================"
echo "Training Dual Arm + Residual RL - Stack"
echo "Pretrained: $PRETRAINED_CHECKPOINT"
[ -n "$RESUME_CHECKPOINT" ] && echo "Resume from: $RESUME_CHECKPOINT"
echo "Task: $TASK"
echo "Envs=$NUM_ENVS Iter=$MAX_ITERATIONS Steps=$STEPS_PER_ENV"
echo "Batch=$MINI_BATCH_SIZE Epochs=$NUM_EPOCHS"
echo "critic_warmup=$CRITIC_WARMUP_ITERS counterfactual_q=$USE_COUNTERFACTUAL_Q"
echo "[Adaptive] delta_reg=$USE_ADAPTIVE_DELTA_REG (target_δ=$TARGET_DELTA_NORM c_init=$C_DELTA_REG_INIT)"
echo "[Adaptive] entropy=$USE_AUTO_ENTROPY (target_H=$TARGET_ENTROPY c_init=$C_ENT_INIT)"
echo "[Adaptive] beta=$USE_ADAPTIVE_BETA (target_eff=$TARGET_EFF_RATIO β_init=$BETA_INIT)"
echo "[Adaptive] alpha=$USE_ADAPTIVE_ALPHA (α_init=$ALPHA_INIT α_max=$ALPHA_MAX max_δ=$MAX_DELTA_NORM)"
echo "Log: $LOG_DIR"
echo "========================================"

RUN_NAME_ARGS=""
[ -n "$RUN_NAME" ] && RUN_NAME_ARGS="--run_name $RUN_NAME"

RESUME_ARGS=""
[ -n "$RESUME_CHECKPOINT" ] && [ -f "$RESUME_CHECKPOINT" ] && RESUME_ARGS="--resume $RESUME_CHECKPOINT"

python scripts/graph_dit_rl/train_graph_rl.py \
    --task "$TASK" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    $RESUME_ARGS \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --steps_per_env "$STEPS_PER_ENV" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --c_delta_reg "$C_DELTA_REG_INIT" \
    --c_ent "$C_ENT_INIT" \
    --beta "$BETA_INIT" \
    --alpha_init "$ALPHA_INIT" \
    --alpha_max "$ALPHA_MAX" \
    --max_delta_norm "$MAX_DELTA_NORM" \
    --expectile_tau "$EXPECTILE_TAU" \
    $ADAPTIVE_ALPHA_FLAG \
    $COUNTERFACTUAL_Q_FLAG \
    $ADAPTIVE_DELTA_REG_FLAG \
    $AUTO_ENTROPY_FLAG \
    $ADAPTIVE_BETA_FLAG \
    --lr "$LR" \
    --seed "$SEED" \
    --critic_warmup_iters "$CRITIC_WARMUP_ITERS" \
    --log_dir "$LOG_DIR" \
    --save_interval "$SAVE_INTERVAL" \
    $RUN_NAME_ARGS \
    $HEADLESS_FLAG

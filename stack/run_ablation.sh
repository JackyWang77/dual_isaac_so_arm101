#!/bin/bash
# Ablation experiments for dual arm stack RL fine-tuning
# Total: 10 experiments (baseline already trained separately)
#
# Usage:
#   ./stack/run_ablation.sh <pretrained_checkpoint> [num_envs] [max_iterations]
#
# Example:
#   ./stack/run_ablation.sh ./logs/gated_small/stack_joint/best_model.pt 10 200
#
cd "$(dirname "$0")/.."

CKPT="${1:-}"
NUM_ENVS="${2:-128}"
MAX_ITER="${3:-100}"
HEADLESS="${HEADLESS:-true}"

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "Usage: $0 <pretrained_checkpoint> [num_envs=64] [max_iterations=100]"
    exit 1
fi

# Baseline config (aligned with train_graph_unet_rl.sh defaults)
BASE_STEPS=405
BASE_BATCH=64
BASE_EPOCHS=2
BASE_REG=2.0
BASE_BETA=1.0
BASE_ENT=0.01
BASE_ALPHA=0.05
BASE_TAU=0.5
BASE_LR=1e-4

TOTAL=10
DONE=0
FAILED=0

run_experiment() {
    local name="$1"
    shift
    DONE=$((DONE + 1))
    echo ""
    echo "================================================================"
    echo "  EXPERIMENT [$DONE/$TOTAL]: $name"
    echo "================================================================"
    LOG_DIR="./logs/ablation/${name}" \
    ./stack/train_graph_unet_rl.sh "$CKPT" "$NUM_ENVS" "$MAX_ITER" "$@" \
        || { echo "[FAILED] $name"; FAILED=$((FAILED + 1)); }
}

# ================================================================
# 1. ACTION REGULARIZATION SWEEP (c_delta_reg)
# ================================================================
run_experiment "reg_0.5" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    0.5 "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR" 42 "$HEADLESS"

run_experiment "reg_4.0" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    4.0 "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR" 42 "$HEADLESS"

# ================================================================
# 2. AWR BETA SWEEP
# ================================================================
run_experiment "beta_0.3" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" 0.3 \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR" 42 "$HEADLESS"

run_experiment "beta_2.0" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" 2.0 \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR" 42 "$HEADLESS"

# ================================================================
# 3. ENTROPY SWEEP (baseline=fixed c_ent=0.0, ablations=adaptive)
# ================================================================
USE_ADAPTIVE_ENTROPY=true C_ENT_BAD=0.02 C_ENT_GOOD=0.005 \
run_experiment "entropy_adaptive_mild" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR" 42 "$HEADLESS"

USE_ADAPTIVE_ENTROPY=true C_ENT_BAD=0.005 C_ENT_GOOD=0.0025 \
run_experiment "entropy_adaptive_strong" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR" 42 "$HEADLESS"

# ================================================================
# 4. BATCH SIZE SWEEP
# ================================================================
run_experiment "batch_32" \
    "$BASE_STEPS" 32 "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR" 42 "$HEADLESS"

run_experiment "batch_128" \
    "$BASE_STEPS" 128 "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR" 42 "$HEADLESS"

# ================================================================
# 5. GRADIENT UPDATES (NUM_EPOCHS) SWEEP
# ================================================================
run_experiment "epochs_1" \
    "$BASE_STEPS" "$BASE_BATCH" 1 \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR" 42 "$HEADLESS"

run_experiment "epochs_5" \
    "$BASE_STEPS" "$BASE_BATCH" 5 \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR" 42 "$HEADLESS"

echo ""
echo "================================================================"
echo "  ALL $TOTAL EXPERIMENTS DONE ($((TOTAL - FAILED)) succeeded, $FAILED failed)"
echo "  Results in: ./logs/ablation/"
echo "================================================================"

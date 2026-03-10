#!/bin/bash
# Ablation experiments for dual arm stack RL fine-tuning
# Total: 11 experiments (1 baseline + 2×5 ablations)
#
# Usage:
#   ./stack/run_ablation.sh <pretrained_checkpoint> [num_envs] [max_iterations]
#
# Example:
#   ./stack/run_ablation.sh ./logs/gated_small/stack_joint/best_model.pt 128 200
#
cd "$(dirname "$0")/.."
set -e

CKPT="${1:-}"
NUM_ENVS="${2:-128}"
MAX_ITER="${3:-200}"

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "Usage: $0 <pretrained_checkpoint> [num_envs=128] [max_iterations=200]"
    exit 1
fi

# Baseline config
BASE_STEPS=400
BASE_BATCH=64
BASE_EPOCHS=2
BASE_REG=2.0
BASE_BETA=1.0
BASE_ENT=0.0
BASE_ALPHA=0.10
BASE_TAU=0.5
BASE_LR=5e-4

run_experiment() {
    local name="$1"
    shift
    echo ""
    echo "================================================================"
    echo "  EXPERIMENT: $name"
    echo "================================================================"
    LOG_DIR="./logs/ablation/${name}" \
    ./stack/train_graph_unet_rl.sh "$CKPT" "$NUM_ENVS" "$MAX_ITER" "$@"
}

# ================================================================
# 1. BASELINE (fixed entropy c_ent=0.0, c_delta_reg=2.0, beta=1.0, batch=64, epochs=2)
# ================================================================
run_experiment "baseline" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR"

# ================================================================
# 2. ACTION REGULARIZATION SWEEP (c_delta_reg)
# ================================================================
run_experiment "reg_0.5" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    0.5 "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR"

run_experiment "reg_4.0" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    4.0 "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR"

# ================================================================
# 3. AWR BETA SWEEP
# ================================================================
run_experiment "beta_0.3" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" 0.3 \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR"

run_experiment "beta_2.0" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" 2.0 \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR"

# ================================================================
# 4. ENTROPY SWEEP (baseline=fixed c_ent=0.0, ablations=adaptive)
#    - adaptive_mild: c_ent_bad=0.02, c_ent_good=0.005
#    - adaptive_strong: c_ent_bad=0.05, c_ent_good=0.001
# ================================================================
USE_ADAPTIVE_ENTROPY=true C_ENT_BAD=0.02 C_ENT_GOOD=0.005 \
run_experiment "entropy_adaptive_mild" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR"

USE_ADAPTIVE_ENTROPY=true C_ENT_BAD=0.05 C_ENT_GOOD=0.001 \
run_experiment "entropy_adaptive_strong" \
    "$BASE_STEPS" "$BASE_BATCH" "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR"

# ================================================================
# 5. BATCH SIZE SWEEP
# ================================================================
run_experiment "batch_32" \
    "$BASE_STEPS" 32 "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR"

run_experiment "batch_256" \
    "$BASE_STEPS" 256 "$BASE_EPOCHS" \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR"

# ================================================================
# 6. GRADIENT UPDATES (NUM_EPOCHS) SWEEP
# ================================================================
run_experiment "epochs_1" \
    "$BASE_STEPS" "$BASE_BATCH" 1 \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR"

run_experiment "epochs_5" \
    "$BASE_STEPS" "$BASE_BATCH" 5 \
    "$BASE_REG" "$BASE_ENT" "$BASE_BETA" \
    "$BASE_ALPHA" "$BASE_TAU" "$BASE_LR"

echo ""
echo "================================================================"
echo "  ALL 11 EXPERIMENTS COMPLETE"
echo "  Results in: ./logs/ablation/"
echo "================================================================"

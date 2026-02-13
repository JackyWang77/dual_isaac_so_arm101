#!/bin/bash
# ============================================================
# Train Graph-Unet (Full Graph) + Residual RL Policy
# ============================================================
#
# Architecture:
#   - GraphUnetPolicy (frozen): Full graph encoder (GRU edge + Graph Attention
#     + Edge-Conditioned Modulation) + U-Net → base action + z
#   - Z Adapter (trainable): frozen z → RL-friendly z
#   - Residual Actor (trainable): delta_a ~ π(δa | obs, a_base, z_bar)
#   - Value Critic (trainable): bar head only (no layer heads)
#
# Key difference from train_unet_rl.sh:
#   UnetPolicy:      Node → MLP → pool → z (no graph attention)
#   GraphUnetPolicy: Node → MLP → GraphAttention×N → pool → z (richer z)
#
# Key Metrics to Watch:
#   - SR (Success Rate): target 78% → 90%+
#   - EV (Explained Variance): >0 meaningful, >0.5 good
#   - Delta: 0.01~0.1 normal
#
# Usage:
#   ./train_graph_unet_rl.sh <pretrained_checkpoint> [options...]
#
# ============================================================

set -e

# ============================================================
# Parse Arguments
# ============================================================
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
EXPECTILE_TAU="${11:-0.5}"
LR="${12:-5e-4}"
SEED="${13:-42}"
HEADLESS="${14:-true}"

TASK="${TASK:-SO-ARM101-Lift-Cube-v0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
USE_ADAPTIVE_ALPHA="${USE_ADAPTIVE_ALPHA:-false}"
USE_ADAPTIVE_ENTROPY="${USE_ADAPTIVE_ENTROPY:-false}"
C_ENT_BAD="${C_ENT_BAD:-0.005}"
C_ENT_GOOD="${C_ENT_GOOD:-0.0001}"

# ============================================================
# Help / Validation
# ============================================================
if [ -z "$PRETRAINED_CHECKPOINT" ]; then
    echo "================================================================"
    echo "  Graph-Unet (Full Graph) + Residual RL Training"
    echo "================================================================"
    echo ""
    echo "Error: Missing required argument: pretrained GraphUnetPolicy checkpoint"
    echo ""
    echo "Usage:"
    echo "  $0 <checkpoint> [num_envs] [max_iter] [steps] [batch] [epochs] [c_delta] [c_ent] [beta] [alpha_init] [expectile_tau] [lr] [seed] [headless]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint    Pretrained GraphUnetPolicy checkpoint (required)"
    echo "  num_envs      Number of parallel environments (default: 64)"
    echo "  max_iter      Maximum training iterations (default: 200)"
    echo "  steps         Steps per env per iteration (default: 160)"
    echo "  batch         Mini-batch size (default: 64)"
    echo "  epochs        Epochs per iteration (default: 2)"
    echo "  c_delta       Delta regularization weight (default: 2.0)"
    echo "  c_ent         Entropy coefficient (default: 0.0)"
    echo "  beta          AWR beta (default: 1.0)"
    echo "  alpha_init    Residual alpha init (default: 0.10)"
    echo "  expectile_tau Expectile tau for value loss (default: 0.5)"
    echo "  lr            Learning rate (default: 5e-4)"
    echo "  seed          Random seed (default: 42)"
    echo "  headless      Run headless (default: true)"
    echo ""
    echo "Examples:"
    echo "  $0 ./logs/graph_unet_full/lift_joint/best_model.pt"
    echo "  $0 ./logs/graph_unet_full/lift_joint/best_model.pt 64 500"
    echo ""
    exit 1
fi

if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $PRETRAINED_CHECKPOINT"
    exit 1
fi

# ============================================================
# Headless Flag / Adaptive Alpha Flag
# ============================================================
if [ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ]; then
    HEADLESS_FLAG="--headless"
else
    HEADLESS_FLAG=""
fi
ADAPTIVE_ALPHA_FLAG=""
[ "$USE_ADAPTIVE_ALPHA" = "false" ] && ADAPTIVE_ALPHA_FLAG="--no_adaptive_alpha"
ADAPTIVE_ENTROPY_FLAG=""
[ "$USE_ADAPTIVE_ENTROPY" = "false" ] && ADAPTIVE_ENTROPY_FLAG="--no_adaptive_entropy"
DEBUG_ALPHA_FLAG=""
[ "${DEBUG_ALPHA:-false}" = "true" ] && DEBUG_ALPHA_FLAG="--debug_alpha"

# ============================================================
# Display Configuration
# ============================================================
echo ""
echo "================================================================"
echo "  Graph-Unet (Full Graph) + Residual RL Training"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  |- Task:                    $TASK"
echo "  |- Pretrained GraphUnet:    $PRETRAINED_CHECKPOINT"
echo "  |- Policy type:             graph_unet (GRU + Graph Attention + Edge Modulation)"
echo "  |- Num envs:                $NUM_ENVS"
echo "  |- Max iterations:          $MAX_ITERATIONS"
echo "  |- Steps per env:           $STEPS_PER_ENV"
echo "  |- Mini-batch size:         $MINI_BATCH_SIZE"
echo "  |- Epochs per iter:         $NUM_EPOCHS"
echo "  |- Delta regularization:    $C_DELTA_REG"
echo "  |- Entropy coef (c_ent):    $C_ENT"
echo "  |- Adaptive entropy:        $USE_ADAPTIVE_ENTROPY (bad=$C_ENT_BAD, good=$C_ENT_GOOD)"
echo "  |- AWR beta:                $BETA"
echo "  |- Alpha init:              $ALPHA_INIT"
echo "  |- Adaptive alpha:          $USE_ADAPTIVE_ALPHA"
echo "  |- Expectile tau:           $EXPECTILE_TAU"
echo "  |- Learning rate:           $LR"
echo "  |- Seed:                    $SEED"
echo "  |- Headless:                $HEADLESS"
echo "  |- Save interval:           $SAVE_INTERVAL"
echo "  |- Log dir:                 ./logs/graph_unet_full_rl/$TASK/"
echo ""

read -p "Start training? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting training..."
echo "================================================================"
echo ""

# ============================================================
# Run Training
# ============================================================
python scripts/graph_dit_rl/train_graph_rl.py \
    --task "$TASK" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --policy_type graph_unet \
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
    $DEBUG_ALPHA_FLAG \
    --lr "$LR" \
    --seed "$SEED" \
    --log_dir "./logs/graph_unet_full_rl" \
    --save_interval "$SAVE_INTERVAL" \
    $HEADLESS_FLAG

# ============================================================
# Done
# ============================================================
echo ""
echo "================================================================"
echo "Training completed!"
echo ""
echo "Results saved to: ./logs/graph_unet_full_rl/$TASK/"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir=./logs/graph_unet_full_rl/$TASK/"
echo ""
echo "Key metrics in TensorBoard:"
echo "  - main/success_rate       - Success rate per rollout"
echo "  - main/success_rate_100   - Success rate (last 100 episodes)"
echo "  - main/explained_variance - Critic quality (should be > 0)"
echo "  - main/delta_norm_mean    - RL residual magnitude"
echo "================================================================"

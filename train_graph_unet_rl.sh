#!/bin/bash
# ============================================================
# Train Graph-Unet + Residual RL Policy (FIXED VERSION)
# ============================================================
#
# Graph-Unet + Residual RL Architecture:
#   - Graph-Unet (frozen): base action + single z (no layer-wise gate)
#   - Z Adapter (trainable): frozen z → RL-friendly z
#   - Residual Actor (trainable): delta_a ~ π(δa | obs, a_base, z_bar)
#   - Value Critic (trainable): bar head only (no layer heads)
#
# Key Metrics to Watch:
#   - SR (Success Rate): 目标 78% → 90%+
#   - EV (Explained Variance): >0 才有意义，>0.5 较好
#   - Δ (Delta): 0.01~0.1 正常，太大说明 RL 乱改
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
NUM_ENVS="${2:-128}"           # 折中：64太少，512太卡，128 是 L4 甜点位
MAX_ITERATIONS="${3:-1000}"
STEPS_PER_ENV="${4:-150}"      # 150步=3s，一次采完 Episode；超时=判负
MINI_BATCH_SIZE="${5:-2048}"   # 128*150=19200，切成 2048 合适
NUM_EPOCHS="${6:-5}"
C_DELTA_REG="${7:-1.0}"
C_ENT="${8:-0.01}"
BETA="${9:-0.05}"              # 保持精英筛选
LR="${10:-3e-4}"
SEED="${11:-42}"
HEADLESS="${12:-true}"

TASK="${TASK:-SO-ARM101-Lift-Cube-v0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

# ============================================================
# Help / Validation
# ============================================================
if [ -z "$PRETRAINED_CHECKPOINT" ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║        Graph-Unet + Residual RL Training (FIXED)              ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "❌ Error: Missing required argument: pretrained Graph-Unet checkpoint"
    echo ""
    echo "Usage:"
    echo "  $0 <checkpoint> [num_envs] [max_iter] [steps] [batch] [epochs] [c_delta] [c_ent] [beta] [lr] [seed] [headless]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint    Pretrained Graph-Unet checkpoint (required)"
    echo "  num_envs      Number of parallel environments (default: 128)"
    echo "  max_iter      Maximum training iterations (default: 1000)"
    echo "  steps         Steps per env per iteration (default: 150, full episode; 超时=判负)"
    echo "  batch         Mini-batch size (default: 2048)"
    echo "  epochs        Epochs per iteration (default: 5)"
    echo "  c_delta       Delta regularization weight (default: 1.0)"
    echo "  c_ent         Entropy coefficient (default: 0.01)"
    echo "  beta          AWR beta: w=exp(adv/beta) (default: 0.05, 精英筛选)"
    echo "  lr            Learning rate (default: 3e-4)"
    echo "  seed          Random seed (default: 42)"
    echo "  headless      Run headless (default: true)"
    echo ""
    echo "Environment Variables:"
    echo "  TASK          Task name (default: SO-ARM101-Lift-Cube-RL-v0, Position+Rotation)"
    echo "  SAVE_INTERVAL Save interval (default: 20)"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 ./logs/graph_unet/lift_joint/best_model.pt"
    echo ""
    echo "  # With 64 envs, 500 iterations"
    echo "  $0 ./logs/graph_unet/lift_joint/best_model.pt 64 500"
    echo ""
    echo "  # Full control"
    echo "  $0 ./logs/graph_unet/lift_joint/best_model.pt 128 1000 150 2048 5 1.0 0.01 0.05 3e-4 42 true"
    echo ""
    echo "  # If EV is negative, increase c_delta_reg:"
    echo "  # If EV negative, increase c_delta_reg"
    echo "  $0 ./logs/graph_unet/lift_joint/best_model.pt 128 1000 150 2048 5 5.0 0.01 0.05"
    echo ""
    echo "Tips:"
    echo "  - Watch SR (Success Rate): should increase from ~78% to 90%+"
    echo "  - Watch EV (Explained Variance): should be > 0, ideally > 0.5"
    echo "  - If EV < 0, increase c_delta_reg to 5.0 or higher"
    echo "  - If SR drops, reduce lr or c_delta_reg"
    echo ""
    exit 1
fi

if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $PRETRAINED_CHECKPOINT"
    exit 1
fi

# ============================================================
# Headless Flag
# ============================================================
if [ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ]; then
    HEADLESS_FLAG="--headless"
else
    HEADLESS_FLAG=""
fi

# ============================================================
# Display Configuration
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        Graph-Unet + Residual RL Training (FIXED)               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  ├─ Task:                 $TASK"
echo "  ├─ Pretrained Graph-Unet: $PRETRAINED_CHECKPOINT"
echo "  ├─ Num envs:             $NUM_ENVS"
echo "  ├─ Max iterations:       $MAX_ITERATIONS"
echo "  ├─ Steps per env:        $STEPS_PER_ENV"
echo "  ├─ Mini-batch size:      $MINI_BATCH_SIZE"
echo "  ├─ Epochs per iter:      $NUM_EPOCHS"
echo "  ├─ Delta regularization: $C_DELTA_REG"
echo "  ├─ Entropy coef (c_ent): $C_ENT"
echo "  ├─ AWR beta:             $BETA"
echo "  ├─ Learning rate:        $LR"
echo "  ├─ Seed:                 $SEED"
echo "  ├─ Headless:             $HEADLESS"
echo "  ├─ Save interval:        $SAVE_INTERVAL"
echo "  └─ Log dir:              ./logs/graph_unet_rl/$TASK/"
echo ""
echo "Key Metrics to Watch:"
echo "  • SR (Success Rate):      78% → 90%+ is the goal"
echo "  • EV (Explained Variance): >0 means Critic is learning"
echo "  • Δ (Delta norm):         0.01~0.1 is normal"
echo ""

read -p "Start training? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting training..."
echo "════════════════════════════════════════════════════════════════"
echo ""

# ============================================================
# Run Training
# ============================================================
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
    --beta "$BETA" \
    --lr "$LR" \
    --seed "$SEED" \
    --log_dir "./logs/graph_unet_rl" \
    --save_interval "$SAVE_INTERVAL" \
    $HEADLESS_FLAG

# ============================================================
# Done
# ============================================================
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Training completed!"
echo ""
echo "Results saved to: ./logs/graph_unet_rl/$TASK/"
echo ""
echo "To view TensorBoard:"
echo "  tensorboard --logdir=./logs/graph_unet_rl/$TASK/"
echo ""
echo "Key metrics in TensorBoard:"
echo "  • main/success_rate       - Success rate per rollout"
echo "  • main/success_rate_100   - Success rate (last 100 episodes)"
echo "  • main/explained_variance - Critic quality (should be > 0)"
echo "  • main/delta_norm_mean    - RL residual magnitude"
echo "════════════════════════════════════════════════════════════════"

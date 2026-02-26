#!/bin/bash
# Play/Test UnetPolicy (MLP encoder) - Dual Cube Stack task
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-}"

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -t ./logs/graph_unet/stack_joint*/*/best_model.pt 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Usage: $0 [checkpoint_path]"
    echo "  Auto-detects from ./logs/graph_unet/stack_joint*/"
    exit 1
fi

echo "========================================"
echo "Playing UnetPolicy (MLP) - Stack"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"

# 与 play_graph_unet.sh 对齐
NUM_ENVS="${NUM_ENVS:-1}"
NUM_EPISODES="${NUM_EPISODES:-1000}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-15}"
EXEC_HORIZON="${EXEC_HORIZON:-10}"
EMA="${EMA:-0.5}"

python scripts/graph_unet/play.py \
    --task SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-Play-v0 \
    --checkpoint "$CHECKPOINT" \
    --policy_type unet \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --episode_length_s "$EPISODE_LENGTH_S" \
    --num_diffusion_steps 15 \
    --exec_horizon "$EXEC_HORIZON" \
    --ema "$EMA" \
    # --headless true

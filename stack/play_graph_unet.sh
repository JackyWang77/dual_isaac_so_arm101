#!/bin/bash
# Play/Test GraphUnetPolicy - Dual Cube Stack task
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-}"

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -t ./logs/graph_unet_full/stack_joint*/*/best_model.pt 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Usage: $0 [checkpoint_path]"
    echo "  Auto-detects from ./logs/graph_unet_full/stack_joint*/"
    exit 1
fi

echo "========================================"
echo "Playing GraphUnetPolicy - Stack"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"

NUM_ENVS="${NUM_ENVS:-1}"
NUM_EPISODES="${NUM_EPISODES:-1000}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-8}"

python scripts/graph_unet/play.py \
    --task SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-Play-v0 \
    --checkpoint "$CHECKPOINT" \
    --policy_type graph_unet \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --episode_length_s "$EPISODE_LENGTH_S" \
    --num_diffusion_steps 10 \
    # --headless true

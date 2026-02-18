#!/bin/bash
# Play/Test GraphUnetPolicy — Lift task
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-}"

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -t ./logs/graph_unet_full/lift_joint*/*/best_model.pt 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Usage: $0 [checkpoint_path]"
    echo "  Auto-detects from ./logs/graph_unet_full/lift_joint*/"
    exit 1
fi

echo "========================================"
echo "Playing GraphUnetPolicy — Lift"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"

NUM_ENVS="${NUM_ENVS:-50}"
NUM_EPISODES="${NUM_EPISODES:-1000}"

python scripts/graph_unet/play.py \
    --task SO-ARM101-Lift-Cube-Play-v0 \
    --checkpoint "$CHECKPOINT" \
    --policy_type graph_unet \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --headless true

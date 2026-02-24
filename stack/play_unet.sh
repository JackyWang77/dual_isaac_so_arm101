#!/bin/bash
# Play/Test UnetPolicy - Dual Cube Stack task
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
echo "Playing UnetPolicy - Stack"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"

NUM_ENVS="${NUM_ENVS:-50}"
NUM_EPISODES="${NUM_EPISODES:-1000}"

python scripts/graph_unet/play.py \
    --task SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-v0 \
    --checkpoint "$CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --headless true

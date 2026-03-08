#!/bin/bash
# Play/Test dual-arm Stack (GraphUnet / RawOnly / Gated — policy auto-detected from checkpoint)
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-}"

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -t ./logs/graph_unet_full/stack_joint*/*/best_model.pt 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Usage: $0 [checkpoint_path]"
    echo "  Auto-detects from ./logs/graph_unet_full/stack_joint*/"
    echo "  Pass any dual-arm stack checkpoint (raw_only / gated / graph_unet); policy type is auto-detected."
    exit 1
fi

echo "========================================"
echo "Playing dual-arm Stack (policy from checkpoint)"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"

NUM_ENVS="${NUM_ENVS:-10}"
NUM_EPISODES="${NUM_EPISODES:-1000}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-10}"
EXEC_HORIZON="${EXEC_HORIZON:-10}"
EMA="${EMA:-0.8}"

python scripts/graph_unet/play.py \
    --task SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-Play-v0 \
    --checkpoint "$CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --episode_length_s "$EPISODE_LENGTH_S" \
    --num_diffusion_steps 15 \
    --exec_horizon "$EXEC_HORIZON" \
    --ema "$EMA" 
    # --headless true
#!/bin/bash
# Play/Test trained Graph DiT policy

echo "========================================"
echo "Testing Graph DiT Policy"
echo "========================================"

# Mode selection (default: auto-detect latest)
MODE="${1:-}"

# Base directory
BASE_DIR="./logs/graph_dit/reach_joint"

# If checkpoint path is provided directly, use it
if [ -n "$MODE" ] && [ -f "$MODE" ]; then
    LATEST_CHECKPOINT="$MODE"
elif [ -n "$MODE" ] && [ "$MODE" != "ddpm" ] && [ "$MODE" != "flow_matching" ]; then
    # If it's not a mode and not a file, treat as checkpoint path
    LATEST_CHECKPOINT="$MODE"
else
    # Auto-detect latest checkpoint
    if [ -z "$MODE" ]; then
        # Find latest from both modes
        LATEST_CHECKPOINT=$(ls -t ./logs/graph_dit/reach_joint_*/*/best_model.pt 2>/dev/null | head -1)
        if [ -z "$LATEST_CHECKPOINT" ]; then
            # Fallback to old directory structure
            LATEST_CHECKPOINT=$(ls -t ./logs/graph_dit/reach_joint/*/best_model.pt 2>/dev/null | head -1)
        fi
    else
        # Use specified mode
        LATEST_CHECKPOINT=$(ls -t ./logs/graph_dit/reach_joint_${MODE}/*/best_model.pt 2>/dev/null | head -1)
    fi
fi

if [ -z "$LATEST_CHECKPOINT" ] || [ ! -f "$LATEST_CHECKPOINT" ]; then
    echo "❌ Error: No checkpoint found!"
    echo ""
    echo "Usage:"
    echo "  $0                          # Auto-detect latest (any mode)"
    echo "  $0 ddpm                     # Use latest DDPM model"
    echo "  $0 flow_matching            # Use latest Flow Matching model"
    echo "  $0 <checkpoint_path>        # Use specific checkpoint"
    echo ""
    echo "Example:"
    echo "  $0 ./logs/graph_dit/reach_joint_ddpm/2025-01-10_15-30-45/best_model.pt"
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"
echo ""

python scripts/graph_dit/play.py \
    --task SO-ARM101-Reach-Cube-Play-v0 \
    --checkpoint "$LATEST_CHECKPOINT" \
    --num_envs 16 \
    --num_episodes 5
    # --num_diffusion_steps  # Auto-detect based on mode (DDPM: 50, Flow Matching: 10)

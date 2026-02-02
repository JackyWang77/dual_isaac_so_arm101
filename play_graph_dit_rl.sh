#!/bin/bash
# Play/Test trained Residual RL Policy (Graph-DiT + PPO fine-tuned)
# This script plays a model trained with train_graph_rl.py
#
# Usage:
#   ./play_graph_dit_rl.sh <checkpoint_path> [pretrained_checkpoint] [task]
#   ./play_graph_dit_rl.sh ./logs/gr_dit/.../policy_iter_300.pt

set -e

echo "========================================"
echo "Testing Residual RL Policy (Graph-DiT + PPO)"
echo "========================================"

# Checkpoint path (first argument, required)
CHECKPOINT="${1:-}"

# Validate checkpoint exists
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found!"
    echo ""
    echo "Usage:"
    echo "  $0 <checkpoint_path> [pretrained_checkpoint] [task]"
    echo ""
    echo "Examples:"
    echo "  $0 ./logs/gr_dit/SO-ARM101-Lift-Cube-v0/2026-01-20_14-43-40/policy_iter_300.pt"
    echo "  $0 ./logs/gr_dit/SO-ARM101-Lift-Cube-v0/2026-01-20_14-43-40/policy_iter_300.pt \\"
    echo "     ./logs/graph_dit/lift_joint_flow_matching/2026-01-20_11-46-23/best_model.pt"
    echo ""
    echo "Available checkpoints in logs/gr_dit:"
    find logs/gr_dit -name "policy_*.pt" -type f 2>/dev/null | head -5 | while read f; do
        echo "  $f"
    done
    exit 1
fi

echo "Using RL checkpoint: $CHECKPOINT"

# Pretrained Graph-DiT checkpoint (second argument, or auto-detect)
PRETRAINED_CHECKPOINT="${2:-}"

if [ -z "$PRETRAINED_CHECKPOINT" ]; then
    # Auto-detect: find latest Graph-DiT checkpoint
    PRETRAINED_CHECKPOINT=$(ls -t ./logs/graph_dit/lift_joint_flow_matching/*/best_model.pt 2>/dev/null | head -1)
    if [ -z "$PRETRAINED_CHECKPOINT" ] || [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
        echo "❌ Error: No Graph-DiT checkpoint found!"
        echo "   Please specify --pretrained_checkpoint manually"
        echo ""
        echo "Available Graph-DiT checkpoints:"
        find logs/graph_dit -name "best_model.pt" -type f 2>/dev/null | head -5 | while read f; do
            echo "  $f"
        done
        exit 1
    fi
    echo "Auto-detected Graph-DiT checkpoint: $PRETRAINED_CHECKPOINT"
else
    if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
        echo "❌ Error: Graph-DiT checkpoint not found: $PRETRAINED_CHECKPOINT"
        exit 1
    fi
    echo "Using Graph-DiT checkpoint: $PRETRAINED_CHECKPOINT"
fi

# Task name (third argument, or default)
TASK="${3:-SO-ARM101-Lift-Cube-v0}"
echo "Using task: $TASK"

# Default parameters (can be overridden via environment variables)
NUM_ENVS="${NUM_ENVS:-2}"            # Default: 64 environments for playback
NUM_EPISODES="${NUM_EPISODES:-10}"    # Default: 10 episodes

# Check if headless mode
HEADLESS_FLAG=""
if [ "${HEADLESS:-false}" = "true" ]; then
    HEADLESS_FLAG="--headless"
fi

echo ""
echo "Playback settings:"
echo "  Task: $TASK"
echo "  RL Checkpoint: $CHECKPOINT"
echo "  Graph-DiT Checkpoint: $PRETRAINED_CHECKPOINT"
echo "  Num envs: $NUM_ENVS"
echo "  Num episodes: $NUM_EPISODES"
echo "  Headless: ${HEADLESS:-false}"
echo ""

# Run the play script
python scripts/graph_dit_rl/play_graph_rl.py \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    $HEADLESS_FLAG

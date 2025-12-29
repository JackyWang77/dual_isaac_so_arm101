#!/bin/bash
# Play/Test trained Graph DiT policy
#
# Usage:
#   bash play_graph_dit.sh [task] [mode|checkpoint_path]
#   
#   task: reach (default) or lift
#   mode: ddpm, flow_matching, or checkpoint path

TASK="${1:-lift}"
ARG2="${2:-}"

echo "========================================"
echo "Testing Graph DiT Policy"
echo "Task: $TASK"
echo "========================================"

# Validate task
if [ "$TASK" != "reach" ] && [ "$TASK" != "lift" ] && [ -f "$TASK" ]; then
    # First arg is actually a checkpoint path
    LATEST_CHECKPOINT="$TASK"
    # Try to infer task from checkpoint path
    if echo "$TASK" | grep -q "reach"; then
        TASK="reach"
    elif echo "$TASK" | grep -q "lift"; then
        TASK="lift"
    else
        TASK="lift"  # Default
    fi
elif [ "$TASK" != "reach" ] && [ "$TASK" != "lift" ]; then
    echo "❌ Error: Invalid task '$TASK'. Use 'reach' or 'lift'"
    exit 1
else
    # Task is valid, check second argument
    MODE="$ARG2"
    
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
            LATEST_CHECKPOINT=$(ls -t ./logs/graph_dit/${TASK}_joint_*/*/best_model.pt 2>/dev/null | head -1)
            if [ -z "$LATEST_CHECKPOINT" ]; then
                # Fallback to old directory structure (without mode suffix)
                LATEST_CHECKPOINT=$(ls -t ./logs/graph_dit/${TASK}_joint/*/best_model.pt 2>/dev/null | head -1)
            fi
        else
            # Use specified mode
            LATEST_CHECKPOINT=$(ls -t ./logs/graph_dit/${TASK}_joint_${MODE}/*/best_model.pt 2>/dev/null | head -1)
        fi
    fi
fi

if [ -z "$LATEST_CHECKPOINT" ] || [ ! -f "$LATEST_CHECKPOINT" ]; then
    echo "❌ Error: No checkpoint found!"
    echo ""
    echo "Usage:"
    echo "  $0 [task] [mode|checkpoint]"
    echo ""
    echo "  task: reach or lift (default: lift)"
    echo "  mode: ddpm, flow_matching, or checkpoint path"
    echo ""
    echo "Examples:"
    echo "  $0 lift                              # Auto-detect latest lift checkpoint"
    echo "  $0 lift flow_matching                # Use latest lift Flow Matching model"
    echo "  $0 reach ddpm                        # Use latest reach DDPM model"
    echo "  $0 lift ./logs/graph_dit/lift_joint_flow_matching/2025-12-17_20-57-54/best_model.pt"
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"
echo ""

# Set task name based on TASK variable
if [ "$TASK" = "reach" ]; then
    TASK_NAME="SO-ARM101-Reach-Cube-Play-v0"
elif [ "$TASK" = "lift" ]; then
    TASK_NAME="SO-ARM101-Lift-Cube-Play-v0"
else
    echo "❌ Error: Unknown task: $TASK"
    exit 1
fi

python scripts/graph_dit/play.py \
    --task "$TASK_NAME" \
    --checkpoint "$LATEST_CHECKPOINT" \
    --num_envs 2 \
    --num_episodes 20
    # --num_diffusion_steps  # Auto-detect based on mode (DDPM: 50, Flow Matching: 10)

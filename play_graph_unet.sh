#!/bin/bash
# Play/Test trained Graph-Unet policy
#
# Usage:
#   bash play_graph_unet.sh [task] [mode|checkpoint_path]
#
#   task: reach (default) or lift
#   mode: flow_matching or checkpoint path

TASK="${1:-lift}"
ARG2="${2:-}"

echo "========================================"
echo "Testing Graph-Unet Policy"
echo "Task: $TASK"
echo "========================================"

# Validate task
if [ "$TASK" != "reach" ] && [ "$TASK" != "lift" ] && [ -f "$TASK" ]; then
    LATEST_CHECKPOINT="$TASK"
    if echo "$TASK" | grep -q "reach"; then
        TASK="reach"
    elif echo "$TASK" | grep -q "lift"; then
        TASK="lift"
    else
        TASK="lift"
    fi
elif [ "$TASK" != "reach" ] && [ "$TASK" != "lift" ]; then
    echo "❌ Error: Invalid task '$TASK'. Use 'reach' or 'lift'"
    exit 1
else
    MODE="$ARG2"

    if [ -n "$MODE" ] && [ -f "$MODE" ]; then
        LATEST_CHECKPOINT="$MODE"
    elif [ -n "$MODE" ] && [ "$MODE" != "flow_matching" ]; then
        LATEST_CHECKPOINT="$MODE"
    else
        if [ -z "$MODE" ]; then
            LATEST_CHECKPOINT=$(ls -t ./logs/graph_unet/${TASK}_joint_*/*/best_model.pt 2>/dev/null | head -1)
            if [ -z "$LATEST_CHECKPOINT" ]; then
                LATEST_CHECKPOINT=$(ls -t ./logs/graph_unet/${TASK}_joint/*/best_model.pt 2>/dev/null | head -1)
            fi
        else
            LATEST_CHECKPOINT=$(ls -t ./logs/graph_unet/${TASK}_joint_${MODE}/*/best_model.pt 2>/dev/null | head -1)
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
    echo "  mode: flow_matching or checkpoint path"
    echo ""
    echo "Examples:"
    echo "  $0 lift                              # Auto-detect latest lift checkpoint"
    echo "  $0 lift flow_matching                # Use latest lift Flow Matching model"
    echo "  $0 lift ./logs/graph_unet/lift_joint_flow_matching/.../best_model.pt"
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"
echo ""

if [ "$TASK" = "reach" ]; then
    TASK_NAME="SO-ARM101-Reach-Cube-Play-v0"
elif [ "$TASK" = "lift" ]; then
    TASK_NAME="SO-ARM101-Lift-Cube-Play-v0"
else
    echo "❌ Error: Unknown task: $TASK"
    exit 1
fi

# num_envs=1 + many episodes: verify success rate one-by-one (e.g. 50–200 runs)
NUM_ENVS="${NUM_ENVS:-50}"
NUM_EPISODES="${NUM_EPISODES:-1000}"

# 不传 --headless 则默认有窗口；传 --headless false 会变成 True（bool("false")=True）
python scripts/graph_unet/play.py \
    --task "$TASK_NAME" \
    --checkpoint "$LATEST_CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --headless true
#!/bin/bash
# Play/Test trained Residual RL Policy (Graph-Unet + PPO fine-tuned)
#
# Usage:
#   ./play_graph_unet_rl.sh <checkpoint_path> [pretrained_checkpoint] [task] [headless]
#
#   headless: 最后一个参数，传 true 则不显示窗口；不传则默认显示
#
# Example:
#   ./play_graph_unet_rl.sh ./logs/graph_unet_rl/.../policy_iter_300.pt           # 默认显示
#   ./play_graph_unet_rl.sh ./logs/graph_unet_rl/.../policy_iter_300.pt true      # 不显示

set -e

echo "========================================"
echo "Testing Residual RL Policy (Graph-Unet + PPO)"
echo "========================================"

# 最后一个参数如果是 true，则 headless；否则默认显示
HEADLESS_FLAG=""
args=("$@")
n=${#args[@]}
if [ $n -gt 0 ] && [ "${args[$((n-1))]}" = "true" ]; then
    HEADLESS_FLAG="--headless"
    args=("${args[@]:0:$((n-1))}")
    set -- "${args[@]}"
fi

CHECKPOINT="${1:-}"

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found!"
    echo ""
    echo "Usage:"
    echo "  $0 <checkpoint_path> [pretrained_checkpoint] [task] [true]"
    echo ""
    echo "  [true]: 最后一个参数，传 true 则不显示窗口；不传则默认显示"
    echo ""
    echo "Examples:"
    echo "  $0 ./logs/graph_unet_rl/.../policy_iter_300.pt              # 默认显示"
    echo "  $0 ./logs/graph_unet_rl/.../policy_iter_300.pt true         # 不显示"
    echo "  $0 ./logs/graph_unet_rl/.../policy_iter_300.pt ./logs/graph_unet/.../best_model.pt true"
    echo ""
    echo "Available checkpoints in logs/graph_unet_rl:"
    find logs/graph_unet_rl -name "policy_*.pt" -type f 2>/dev/null | head -5 | while read f; do
        echo "  $f"
    done
    exit 1
fi

echo "Using RL checkpoint: $CHECKPOINT"

PRETRAINED_CHECKPOINT="${2:-}"

if [ -z "$PRETRAINED_CHECKPOINT" ]; then
    PRETRAINED_CHECKPOINT=$(ls -t ./logs/graph_unet/lift_joint/best_model.pt ./logs/graph_unet/*/best_model.pt 2>/dev/null | head -1)
    if [ -z "$PRETRAINED_CHECKPOINT" ] || [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
        echo "❌ Error: No Graph-Unet checkpoint found!"
        echo "   Please specify pretrained_checkpoint manually"
        echo ""
        find logs/graph_unet -name "best_model.pt" -type f 2>/dev/null | head -5 | while read f; do
            echo "  $f"
        done
        exit 1
    fi
    echo "Auto-detected Graph-Unet checkpoint: $PRETRAINED_CHECKPOINT"
else
    if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
        echo "❌ Error: Graph-Unet checkpoint not found: $PRETRAINED_CHECKPOINT"
        exit 1
    fi
    echo "Using Graph-Unet checkpoint: $PRETRAINED_CHECKPOINT"
fi

TASK="${3:-SO-ARM101-Lift-Cube-v0}"
echo "Using task: $TASK"

NUM_ENVS="${NUM_ENVS:-50}"
NUM_EPISODES="${NUM_EPISODES:-1000}"

echo ""
echo "Playback settings:"
echo "  Task: $TASK"
echo "  RL Checkpoint: $CHECKPOINT"
echo "  Graph-Unet Checkpoint: $PRETRAINED_CHECKPOINT"
echo "  Num envs: $NUM_ENVS"
echo "  Num episodes: $NUM_EPISODES"
echo "  Headless: $([ -n "$HEADLESS_FLAG" ] && echo 'true (不显示)' || echo 'false (默认显示)')"
echo ""

python scripts/graph_dit_rl/play_graph_rl.py \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    $HEADLESS_FLAG

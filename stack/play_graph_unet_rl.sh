#!/bin/bash
# Play/Test GraphUnetPolicy + Residual RL - Dual Cube Stack task
cd "$(dirname "$0")/.."
set -e

HEADLESS_FLAG=""
args=("$@")
n=${#args[@]}
if [ $n -gt 0 ] && [ "${args[$((n-1))]}" = "true" ]; then
    HEADLESS_FLAG="--headless"
    args=("${args[@]:0:$((n-1))}")
    set -- "${args[@]}"
fi

CHECKPOINT="${1:-}"
PRETRAINED_CHECKPOINT="${2:-}"
TASK="${3:-SO-ARM101-Dual-Cube-Stack-v0}"

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Usage: $0 <rl_checkpoint> [pretrained_checkpoint] [task] [true]"
    echo ""
    echo "Example:"
    echo "  $0 ./logs/graph_unet_full_rl/.../policy_iter_200.pt"
    exit 1
fi

if [ -z "$PRETRAINED_CHECKPOINT" ]; then
    PRETRAINED_CHECKPOINT=$(ls -t ./logs/graph_unet_full/stack_joint*/*/best_model.pt 2>/dev/null | head -1)
fi

if [ -z "$PRETRAINED_CHECKPOINT" ] || [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "Error: No pretrained GraphUnetPolicy checkpoint found. Specify as 2nd arg."
    exit 1
fi

NUM_ENVS="${NUM_ENVS:-50}"
NUM_EPISODES="${NUM_EPISODES:-1000}"

echo "========================================"
echo "Playing GraphUnetPolicy + RL - Stack"
echo "RL: $CHECKPOINT"
echo "Base: $PRETRAINED_CHECKPOINT"
echo "========================================"

python scripts/graph_dit_rl/play_graph_rl.py \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --policy_type graph_unet \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    $HEADLESS_FLAG

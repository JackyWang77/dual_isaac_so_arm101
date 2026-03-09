#!/bin/bash
# Play (inference) with RL-trained dual arm stacking policy
# Loads pretrained BC backbone + RL residual policy
cd "$(dirname "$0")/.."
set -e

PRETRAINED_CHECKPOINT="${1:-}"
RL_CHECKPOINT="${2:-}"
NUM_ENVS="${3:-64}"
NUM_BATCHES="${4:-2}"

TASK="${TASK:-SO-ARM101-Dual-Cube-Stack-Play-v0}"
HEADLESS="${HEADLESS:-false}"

if [ -z "$PRETRAINED_CHECKPOINT" ] || [ -z "$RL_CHECKPOINT" ]; then
    echo "Usage: $0 <bc_checkpoint> <rl_checkpoint> [num_envs] [num_batches]"
    echo ""
    echo "Example:"
    echo "  $0 ./logs/gated_small/stack_joint/best_model.pt ./logs/dual_arm_rl/best_model.pt 64 2"
    echo ""
    echo "  bc_checkpoint: Pretrained backbone (DualArmDisentangledPolicyGated)"
    echo "  rl_checkpoint: RL fine-tuned residual policy"
    exit 1
fi

HEADLESS_FLAG=""
if [ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ]; then
    HEADLESS_FLAG="--headless"
fi

echo "========================================"
echo "Play: Dual Arm RL Stack"
echo "BC Backbone: $PRETRAINED_CHECKPOINT"
echo "RL Policy:   $RL_CHECKPOINT"
echo "Envs=$NUM_ENVS Batches=$NUM_BATCHES"
echo "========================================"

python scripts/graph_dit_rl/play_rl.py \
    --task "$TASK" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --rl_checkpoint "$RL_CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --num_batches "$NUM_BATCHES" \
    $HEADLESS_FLAG

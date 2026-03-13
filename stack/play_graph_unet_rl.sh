#!/bin/bash
# Play/Test GraphUnetPolicy + Residual RL - Dual Cube Stack (SR experiments)
# Mimics play_disentangled_graph_gated.sh: same env, same structure for fair comparison
cd "$(dirname "$0")/.."
set -e

CHECKPOINT="${1:-}"
PRETRAINED_CHECKPOINT="${2:-logs/graph_unet_full/stack_joint_t1_gripper_flow_matching/gate_graph/best_model.pt}"

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -t ./logs/dual_arm_rl/SO-ARM101-Dual-Cube-Stack-RL-v0/*/best_model.pt 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Usage: $0 [rl_checkpoint] [pretrained_checkpoint]"
    echo "  Auto-detects RL from ./logs/dual_arm_rl/SO-ARM101-Dual-Cube-Stack-RL-v0/*/best_model.pt"
    echo "  Pass RL checkpoint (1st) and Graph-Unet backbone (2nd) for success rate eval."
    exit 1
fi

if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "Error: No pretrained GraphUnetPolicy checkpoint found. Specify as 2nd arg."
    exit 1
fi

# Same structure as play_disentangled_graph_gated.sh
NUM_ENVS="${NUM_ENVS:-1}"
NUM_EPISODES="${NUM_EPISODES:-1000}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-10}"
TASK="${TASK:-SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-Play-v0}"
# Use TASK=SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-Play-Large-v0 for generalization test


echo "========================================"
echo "Playing GraphUnetPolicy + RL - Stack (SR eval)"
echo "RL: $CHECKPOINT"
echo "Base: $PRETRAINED_CHECKPOINT"
echo "Task: $TASK"
echo "========================================"

python scripts/graph_dit_rl/play_graph_rl.py \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --episode_length_s "$EPISODE_LENGTH_S"

#!/bin/bash
# Play/Test DualArmDisentangledPolicyGated - Dual Cube Stack task (gated fusion)
# Uses Large env (wider cube spawn range) for generalization test.
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-}"

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -t ./logs/disentangled_graph_gated/stack_disentangled_gated*/*/best_model.pt 2>/dev/null | head -1)
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Usage: $0 [checkpoint_path]"
    echo "  Auto-detects from ./logs/disentangled_graph_gated/stack_disentangled_gated*/"
    exit 1
fi

echo "========================================"
echo "Playing DualArmDisentangledPolicyGated - Stack (Large env, generalization)"
echo "Checkpoint: $CHECKPOINT"
echo "========================================"

NUM_ENVS="${NUM_ENVS:-1}"
NUM_EPISODES="${NUM_EPISODES:-1000}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-10}"
EXEC_HORIZON="${EXEC_HORIZON:-10}"
EMA="${EMA:-1}"

python scripts/graph_unet/play.py \
    --task SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-Play-v0 \
    --checkpoint "$CHECKPOINT" \
    --policy_type disentangled_graph_unet_gated \
    --gripper_threshold -0.25 \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --episode_length_s "$EPISODE_LENGTH_S" \
    --num_diffusion_steps 15 \
    --exec_horizon "$EXEC_HORIZON" \
    --ema "$EMA"
    # --headless true
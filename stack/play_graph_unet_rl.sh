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
    echo "  BACKBONE_ONLY=true — backbone-only eval (no residual RL), same args."
    echo "  NO_GRIPPER_HEAD=true — residual on arms, gripper from backbone (a_base) only, no RL gripper head."
    exit 1
fi

if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "Error: No pretrained GraphUnetPolicy checkpoint found. Specify as 2nd arg."
    exit 1
fi

# Same structure as play_disentangled_graph_gated.sh
HEADLESS="${HEADLESS:-true}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_EPISODES="${NUM_EPISODES:-1000}"
# Match train_graph_unet_rl.sh / cube_stack: 8s ≈ 400 steps @ 50Hz (override only if you know why)
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-10}"
# MUST match training task for comparable SR. RL-v0 = same env as train (CubeStackRLEnvCfg).
# Joint-States-Mimic-Play-v* is a different env class + obs — SR will not match training.
TASK="${TASK:-SO-ARM101-Dual-Cube-Stack-RL-v0}"
# Generalization: TASK=SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-Play-Large-v0 (different distribution)
# BACKBONE_ONLY=true — only Graph-Unet base action (no residual δ / no RL gripper head), for A/B vs full RL

BACKBONE_ONLY="${BACKBONE_ONLY:-true}"
BACKBONE_ONLY_FLAG=""
if [ "$BACKBONE_ONLY" = "true" ] || [ "$BACKBONE_ONLY" = "1" ]; then
    BACKBONE_ONLY_FLAG="--backbone_only"
fi

NO_GRIPPER_HEAD="${NO_GRIPPER_HEAD:-false}"
NO_GRIPPER_HEAD_FLAG=""
if [ "$NO_GRIPPER_HEAD" = "true" ] || [ "$NO_GRIPPER_HEAD" = "1" ]; then
    NO_GRIPPER_HEAD_FLAG="--no_gripper_head"
fi

echo "========================================"
echo "Playing GraphUnetPolicy + RL - Stack (SR eval)"
echo "RL: $CHECKPOINT"
echo "Base: $PRETRAINED_CHECKPOINT"
echo "Task: $TASK"
echo "BACKBONE_ONLY: $BACKBONE_ONLY"
echo "NO_GRIPPER_HEAD: $NO_GRIPPER_HEAD"
echo "========================================"

HEADLESS_FLAG=""
[ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ] && HEADLESS_FLAG="--headless"

python scripts/graph_dit_rl/play_graph_rl.py \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --deterministic \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --episode_length_s "$EPISODE_LENGTH_S" \
    $BACKBONE_ONLY_FLAG \
    $NO_GRIPPER_HEAD_FLAG
    # $HEADLESS_FLAG
#!/bin/bash
# Play/Test Graph-DiT policy with Gripper Model
#
# Usage:
#   ./isaaclab.sh -p play_graph_dit_gripper.sh \
#       --task SO-ARM101-Lift-Cube-Play-v0 \
#       --checkpoint ./logs/graph_dit/lift_joint_flow_matching/2026-01-12_12-35-32/best_model.pt \
#       --gripper-model ./logs/gripper_model/gripper_model.pt \
#       --num_envs 64 \
#       --num_episodes 10

# Default values
TASK="SO-ARM101-Lift-Cube-Play-v0"
CHECKPOINT=""
GRIPPER_MODEL=""
NUM_ENVS=2
NUM_EPISODES=10
DEVICE="cuda"
NUM_DIFFUSION_STEPS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --gripper-model)
            GRIPPER_MODEL="$2"
            shift 2
            ;;
        --num-envs|--num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --num-episodes|--num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num-diffusion-steps|--num_diffusion_steps)
            NUM_DIFFUSION_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            echo "Usage:"
            echo "  ./isaaclab.sh -p play_graph_dit_gripper.sh [options]"
            echo ""
            echo "Options:"
            echo "  --task TASK_NAME              Task name (default: SO-ARM101-Lift-Cube-Play-v0)"
            echo "  --checkpoint PATH             Path to Graph-DiT checkpoint (required)"
            echo "  --gripper-model PATH          Path to gripper model checkpoint (required)"
            echo "  --num-envs N (or --num_envs)  Number of parallel environments (default: 2)"
            echo "  --num-episodes N (or --num_episodes) Number of episodes to run (default: 10)"
            echo "  --device DEVICE               Device: cuda or cpu (default: cuda)"
            echo "  --num-diffusion-steps N       Number of diffusion steps (default: 2 for Flow Matching)"
            echo ""
            echo "Example:"
            echo "  ./isaaclab.sh -p play_graph_dit_gripper.sh \\"
            echo "      --task SO-ARM101-Lift-Cube-Play-v0 \\"
            echo "      --checkpoint ./logs/graph_dit/lift_joint_flow_matching/2026-01-12_12-35-32/best_model.pt \\"
            echo "      --gripper-model ./logs/gripper_model/gripper_model.pt \\"
            echo "      --num-envs 64 \\"
            echo "      --num-episodes 10"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CHECKPOINT" ]; then
    echo "❌ Error: --checkpoint is required!"
    echo ""
    echo "Usage:"
    echo "  ./isaaclab.sh -p play_graph_dit_gripper.sh --checkpoint PATH --gripper-model PATH"
    exit 1
fi

if [ -z "$GRIPPER_MODEL" ]; then
    echo "❌ Error: --gripper-model is required!"
    echo ""
    echo "Usage:"
    echo "  ./isaaclab.sh -p play_graph_dit_gripper.sh --checkpoint PATH --gripper-model PATH"
    exit 1
fi

# Check if files exist
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$GRIPPER_MODEL" ]; then
    echo "❌ Error: Gripper model file not found: $GRIPPER_MODEL"
    exit 1
fi

echo "========================================"
echo "Playing Graph-DiT Policy with Gripper Model"
echo "========================================"
echo "Task:              $TASK"
echo "Checkpoint:        $CHECKPOINT"
echo "Gripper Model:     $GRIPPER_MODEL"
echo "Num Envs:          $NUM_ENVS"
echo "Num Episodes:      $NUM_EPISODES"
echo "Device:            $DEVICE"
if [ -n "$NUM_DIFFUSION_STEPS" ]; then
    echo "Diffusion Steps:   $NUM_DIFFUSION_STEPS"
else
    echo "Diffusion Steps:   Default (2 for Flow Matching)"
fi
echo "========================================"
echo ""

# Build command
CMD="python scripts/graph_dit/play.py"
CMD="$CMD --task \"$TASK\""
CMD="$CMD --checkpoint \"$CHECKPOINT\""
CMD="$CMD --gripper-model \"$GRIPPER_MODEL\""
CMD="$CMD --num_envs $NUM_ENVS"
CMD="$CMD --num_episodes $NUM_EPISODES"
CMD="$CMD --device $DEVICE"

if [ -n "$NUM_DIFFUSION_STEPS" ]; then
    CMD="$CMD --num_diffusion_steps $NUM_DIFFUSION_STEPS"
fi

# Execute command
eval $CMD

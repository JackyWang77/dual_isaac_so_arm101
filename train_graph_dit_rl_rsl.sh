#!/bin/bash
# Train Graph DiT RL Policy using RSL-RL framework (multi-env parallel training)
# 
# This script uses the STANDARD RSL-RL training flow, just like lift task.
# It sets the pretrained_checkpoint via environment variable and calls the standard script.
#
# Note: Requires conda environment 'env_isaaclab' to be activated first!

# Check if RSL-RL module is available (indicates correct environment)
if ! python -c "from rsl_rl.modules import ActorCritic" 2>/dev/null; then
    echo "⚠️  Warning: RSL-RL modules not found!"
    echo ""
    echo "This usually means the conda environment 'env_isaaclab' is not activated."
    echo ""
    echo "Please activate it first:"
    echo "  conda activate env_isaaclab"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check arguments
if [ $# -lt 1 ]; then
    echo "❌ Error: Missing required arguments"
    echo ""
    echo "Usage:"
    echo "  $0 <pretrained_checkpoint> [num_envs] [max_iterations] [headless]"
    echo ""
    echo "Arguments:"
    echo "  pretrained_checkpoint  Path to pre-trained Graph DiT checkpoint (required)"
    echo "  num_envs              Number of parallel environments (default: 1024, adjust based on GPU)"
    echo "  max_iterations        Training iterations (default: 200)"
    echo "  headless              Run in headless mode: true/false (default: true)"
    echo ""
    echo "Examples:"
    echo "  # Use defaults (1024 envs, 200 iterations, headless)"
    echo "  $0 ./logs/graph_dit/reach_joint_flow_matching/2025-12-10_18-47-03/best_model.pt"
    echo ""
    echo "  # Custom settings"
    echo "  $0 ./logs/graph_dit/.../best_model.pt 1024 500 true"
    echo "  $0 ./logs/graph_dit/.../best_model.pt 512 200 false  # With GUI (fewer envs)"
    echo "  $0 ./logs/graph_dit/.../best_model.pt 2048 200 true  # More envs if you have a powerful GPU"
    exit 1
fi

PRETRAINED_CHECKPOINT="$1"
NUM_ENVS="${2:-1024}"  # Default to 1024 (adjust based on your GPU memory)
MAX_ITERATIONS="${3:-200}"  # Default to 200 (usually enough for PPO)
HEADLESS="${4:-true}"  # Default to headless for faster training

# Validate checkpoint exists
if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $PRETRAINED_CHECKPOINT"
    exit 1
fi

echo "========================================"
echo "Graph DiT RL Fine-Tuning (RSL-RL)"
echo "========================================"
echo ""
echo "Using STANDARD RSL-RL training flow (like lift task)"
echo ""
echo "Pre-trained checkpoint: $PRETRAINED_CHECKPOINT"
echo "Number of environments: $NUM_ENVS"
echo "Max iterations: $MAX_ITERATIONS"
echo "Headless mode: $HEADLESS"
echo ""

# Note: train_rsl_rl.py will set the pretrained_checkpoint via environment variable
# and handle the Graph DiT specific setup

# Use the Graph DiT RSL-RL training script
# This script handles Graph DiT specific setup and calls RSL-RL

# Build command arguments
CMD_ARGS=(
    --task SO-ARM101-Reach-Cube-v0
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT"
    --num_envs "$NUM_ENVS"
    --max_iterations "$MAX_ITERATIONS"
)

# Add headless flag if needed
if [ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ]; then
    CMD_ARGS+=(--headless)
fi

python scripts/graph_dit_rl/train_rsl_rl.py "${CMD_ARGS[@]}"

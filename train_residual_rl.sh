#!/bin/bash
# Train Residual RL Policy (PPO fine-tuning on top of Graph-DiT)
#
# This script trains a PPO agent to output RESIDUAL actions that correct
# the base actions from a pre-trained Graph-DiT policy.
#
# Architecture:
#   - Graph-DiT (frozen): Provides base action + scene understanding features
#   - PPO (trainable): Uses [Robot_State, Graph_Embedding] → Residual action
#   - Final action = base_action + scale * residual_action
#
# Usage:
#   ./train_residual_rl.sh <pretrained_dit_checkpoint> [num_envs] [max_iterations]

set -e

# Check conda environment
if ! python -c "from rsl_rl.modules import ActorCritic" 2>/dev/null; then
    echo "⚠️  Warning: RSL-RL modules not found!"
    echo ""
    echo "Please activate conda environment first:"
    echo "  conda activate env_isaaclab"
    exit 1
fi

# Arguments
PRETRAINED_CHECKPOINT="${1:-}"
NUM_ENVS="${2:-256}"  # Lower default! Diffusion inference needs more VRAM
MAX_ITERATIONS="${3:-300}"
HEADLESS="${4:-true}"

# Validate checkpoint
if [ -z "$PRETRAINED_CHECKPOINT" ]; then
    echo "❌ Error: Missing required argument: pretrained_dit_checkpoint"
    echo ""
    echo "Usage:"
    echo "  $0 <pretrained_dit_checkpoint> [num_envs] [max_iterations]"
    echo ""
    echo "Arguments:"
    echo "  pretrained_dit_checkpoint  Path to pre-trained Graph-DiT checkpoint (required)"
    echo "  num_envs                   Number of parallel environments (default: 256, lower for diffusion)"
    echo "  max_iterations             Maximum training iterations (default: 300)"
    echo ""
    echo "Examples:"
    echo "  $0 ./logs/graph_dit/lift_joint_flow_matching/2025-12-17_21-22-22/best_model.pt"
    echo "  $0 ./logs/graph_dit/lift_joint_flow_matching/2025-12-17_20-00-00/best_model.pt 128 500"
    exit 1
fi

if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $PRETRAINED_CHECKPOINT"
    exit 1
fi

# Set headless flag
if [ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ]; then
    HEADLESS_FLAG="--headless"
else
    HEADLESS_FLAG=""
fi

echo "========================================"
echo "Training Residual RL Policy"
echo "========================================"
echo "Pre-trained DiT: $PRETRAINED_CHECKPOINT"
echo "Num envs: $NUM_ENVS (lower for diffusion to save VRAM)"
echo "Max iterations: $MAX_ITERATIONS"
echo "Headless: $HEADLESS"
echo ""
echo "Architecture:"
echo "  Graph-DiT (frozen) → Base action + Graph Embedding"
echo "  PPO (trainable) → Residual action"
echo "  Final action = base + scale * residual"
echo "========================================"
echo ""

# Run training using the graph_dit_rl script
python scripts/graph_dit_rl/train_rsl_rl.py \
    --task SO-ARM101-Lift-Cube-ResidualRL-v0 \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    $HEADLESS_FLAG

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"

#!/bin/bash
# Play/Test trained Residual RL Policy (Graph-DiT + PPO fine-tuned)
# This script plays a model trained with train_residual_rl.sh
#
# Usage:
#   ./play_graph_dit_rl.sh                           # Auto-detect latest checkpoint
#   ./play_graph_dit_rl.sh <checkpoint_path>         # Use specific checkpoint

set -e

echo "========================================"
echo "Testing Residual RL Policy (Graph-DiT + PPO)"
echo "========================================"

# Check if RSL-RL module is available
if ! python -c "from rsl_rl.modules import ActorCritic" 2>/dev/null; then
    echo "⚠️  Warning: RSL-RL modules not found!"
    echo ""
    echo "Please activate conda environment first:"
    echo "  conda activate env_isaaclab"
    exit 1
fi

# Checkpoint path (first argument, or auto-detect latest)
CHECKPOINT="${1:-}"

# Base directory for RL training logs (matches train_residual_rl.sh output)
BASE_DIR="./logs/rsl_rl/lift_residual_rl"

# Auto-detect latest checkpoint if not provided
if [ -z "$CHECKPOINT" ]; then
    # Find latest run directory
    LATEST_RUN=$(ls -dt "$BASE_DIR"/*/ 2>/dev/null | head -1)
    
    if [ -n "$LATEST_RUN" ]; then
        # Find the highest numbered model checkpoint in the latest run
        LATEST_CHECKPOINT=$(ls -v "$LATEST_RUN"model_*.pt 2>/dev/null | tail -1)
        
        if [ -n "$LATEST_CHECKPOINT" ]; then
            CHECKPOINT="$LATEST_CHECKPOINT"
            echo "Auto-detected latest checkpoint: $CHECKPOINT"
        fi
    fi
fi

# Validate checkpoint exists
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found!"
    echo ""
    echo "Usage:"
    echo "  $0                          # Auto-detect latest checkpoint"
    echo "  $0 <checkpoint_path>        # Use specific checkpoint"
    echo ""
    echo "Example:"
    echo "  $0 ./logs/rsl_rl/lift_residual_rl/2025-12-30_19-52-29/model_299.pt"
    echo ""
    echo "Available checkpoints in $BASE_DIR:"
    for dir in $(ls -dt "$BASE_DIR"/*/ 2>/dev/null | head -5); do
        latest=$(ls -v "${dir}"model_*.pt 2>/dev/null | tail -1)
        [ -n "$latest" ] && echo "  $latest"
    done
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

# ============================================================
# CRITICAL: Extract pretrained_checkpoint path from saved config
# The Residual RL policy needs the Graph-DiT checkpoint path,
# which is stored in params/agent.yaml from training.
# ============================================================
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
AGENT_YAML="$CHECKPOINT_DIR/params/agent.yaml"

if [ -f "$AGENT_YAML" ]; then
    # Extract pretrained_checkpoint path from YAML
    PRETRAINED_CHECKPOINT=$(grep "pretrained_checkpoint:" "$AGENT_YAML" | sed 's/.*pretrained_checkpoint: //' | tr -d ' ')
    
    if [ -n "$PRETRAINED_CHECKPOINT" ] && [ -f "$PRETRAINED_CHECKPOINT" ]; then
        echo "Found Graph-DiT checkpoint: $PRETRAINED_CHECKPOINT"
        export RESIDUAL_RL_PRETRAINED_CHECKPOINT="$PRETRAINED_CHECKPOINT"
    else
        echo "⚠️  Warning: pretrained_checkpoint from config not found: $PRETRAINED_CHECKPOINT"
        echo "   Trying to find a valid checkpoint..."
        
        # Fallback: find latest Graph-DiT checkpoint
        FALLBACK_CHECKPOINT=$(ls -t ./logs/graph_dit/lift_joint_flow_matching/*/best_model.pt 2>/dev/null | head -1)
        if [ -n "$FALLBACK_CHECKPOINT" ]; then
            echo "   Using fallback: $FALLBACK_CHECKPOINT"
            export RESIDUAL_RL_PRETRAINED_CHECKPOINT="$FALLBACK_CHECKPOINT"
        else
            echo "❌ Error: No Graph-DiT checkpoint found!"
            exit 1
        fi
    fi
else
    echo "⚠️  Warning: Agent config not found: $AGENT_YAML"
    # Fallback: find latest Graph-DiT checkpoint
    FALLBACK_CHECKPOINT=$(ls -t ./logs/graph_dit/lift_joint_flow_matching/*/best_model.pt 2>/dev/null | head -1)
    if [ -n "$FALLBACK_CHECKPOINT" ]; then
        echo "   Using fallback: $FALLBACK_CHECKPOINT"
        export RESIDUAL_RL_PRETRAINED_CHECKPOINT="$FALLBACK_CHECKPOINT"
    fi
fi

echo ""

# Default parameters (can be overridden via environment variables)
NUM_ENVS="${NUM_ENVS:-4}"            # Default: 4 environments for playback

# Check if headless mode
HEADLESS_FLAG=""
if [ "${HEADLESS:-false}" = "true" ]; then
    HEADLESS_FLAG="--headless"
fi

echo "Playback settings:"
echo "  Num envs: $NUM_ENVS"
echo "  Headless: ${HEADLESS:-false}"
echo ""

# Run the play script using RSL-RL play
python scripts/rsl_rl/play.py \
    --task SO-ARM101-Lift-Cube-ResidualRL-v0 \
    --checkpoint "$CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    $HEADLESS_FLAG

#!/bin/bash
# Play/Test trained Graph DiT RL Policy (RL fine-tuned)
# This script plays a model trained with RSL-RL fine-tuning

echo "========================================"
echo "Testing Graph DiT RL Policy (RL Fine-tuned)"
echo "========================================"

# Check if RSL-RL module is available
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

# Checkpoint path (first argument, or auto-detect latest)
CHECKPOINT="${1:-}"

# Base directory for RL training logs
BASE_DIR="./logs/rsl_rl/reach_graph_dit_rl"

# Auto-detect latest checkpoint if not provided
if [ -z "$CHECKPOINT" ]; then
    # Find latest checkpoint (prefer final_checkpoint.pt, fallback to model_*.pt)
    LATEST_CHECKPOINT=$(ls -t "$BASE_DIR"/*/final_checkpoint.pt 2>/dev/null | head -1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        # Fallback to model_*.pt (RSL-RL naming)
        LATEST_CHECKPOINT=$(ls -t "$BASE_DIR"/*/model_*.pt 2>/dev/null | head -1)
    fi
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        CHECKPOINT="$LATEST_CHECKPOINT"
        echo "Auto-detected latest checkpoint: $CHECKPOINT"
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
    echo "  $0 ./logs/rsl_rl/reach_graph_dit_rl/2025-12-10_21-05-46/model_199.pt"
    echo ""
    echo "Available checkpoints:"
    ls -t "$BASE_DIR"/*/final_checkpoint.pt "$BASE_DIR"/*/model_*.pt 2>/dev/null | head -5 || echo "  (none found)"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
echo ""

# Default parameters (can be overridden via environment variables)
NUM_ENVS="${NUM_ENVS:-4}"            # Default: 4 environments for playback (fewer for faster/more responsive)
NUM_EPISODES="${NUM_EPISODES:-5}"    # Default: 5 episodes
DEVICE="${DEVICE:-cuda}"             # Default: cuda
DETERMINISTIC="${DETERMINISTIC:-true}" # Default: deterministic actions

# Check if headless mode
HEADLESS_FLAG=""
if [ "${HEADLESS:-false}" = "true" ]; then
    HEADLESS_FLAG="--headless"
fi

echo "Playback settings:"
echo "  Num envs: $NUM_ENVS"
echo "  Num episodes: $NUM_EPISODES"
echo "  Device: $DEVICE"
echo "  Deterministic: $DETERMINISTIC"
echo "  Headless: ${HEADLESS:-false}"
echo ""

# Run the play script using standard RSL-RL play (recommended for RL-trained models)
# This uses the same flow as standard Isaac Lab RL tasks
python scripts/rsl_rl/play.py \
    --task SO-ARM101-Reach-Cube-Play-v0 \
    --checkpoint "$CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    $HEADLESS_FLAG

# Alternative: Use custom play_rl.py (if you need custom behavior)
# python scripts/graph_dit_rl/play_rl.py \
#     --task SO-ARM101-Reach-Cube-Play-v0 \
#     --checkpoint "$CHECKPOINT" \
#     --num_envs "$NUM_ENVS" \
#     --num_episodes "$NUM_EPISODES" \
#     --device "$DEVICE" \
#     $([ "$DETERMINISTIC" = "true" ] && echo "--deterministic" || echo "--stochastic") \
#     $HEADLESS_FLAG

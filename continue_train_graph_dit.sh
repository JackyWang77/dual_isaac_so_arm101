#!/bin/bash
# Continue Training Graph DiT Policy

# Checkpoint path (optional - will auto-detect latest if not provided)
CHECKPOINT="${1:-}"

# Base directory for checkpoints (default to flow_matching, can be overridden)
BASE_DIR="./logs/graph_dit/lift_joint_flow_matching"

# Auto-detect latest checkpoint if not provided
if [ -z "$CHECKPOINT" ]; then
    echo "🔍 Auto-detecting latest checkpoint in $BASE_DIR..."
    
    # Find the most recent directory
    LATEST_DIR=$(ls -dt "${BASE_DIR}"/*/ 2>/dev/null | head -1)
    
    if [ -z "$LATEST_DIR" ]; then
        echo "❌ Error: No training directories found in $BASE_DIR"
        echo ""
        echo "Available directories:"
        ls -d "${BASE_DIR}"/*/ 2>/dev/null | head -5 || echo "  (none)"
        exit 1
    fi
    
    # Try to find best_model.pt first (preferred), then final_model.pt
    if [ -f "${LATEST_DIR}best_model.pt" ]; then
        CHECKPOINT="${LATEST_DIR}best_model.pt"
    elif [ -f "${LATEST_DIR}final_model.pt" ]; then
        CHECKPOINT="${LATEST_DIR}final_model.pt"
    else
        echo "❌ Error: No checkpoint found in $LATEST_DIR"
        echo ""
        echo "Available files:"
        ls -lh "${LATEST_DIR}"*.pt 2>/dev/null || echo "  (none)"
        exit 1
    fi
    
    echo "✅ Found checkpoint: $CHECKPOINT"
fi

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Extract checkpoint directory and detect mode from path
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")

# Try to detect mode from checkpoint path (lift_joint_flow_matching or lift_joint_ddpm)
if [[ "$CHECKPOINT_DIR" == *"_flow_matching"* ]]; then
    DETECTED_MODE="flow_matching"
elif [[ "$CHECKPOINT_DIR" == *"_ddpm"* ]]; then
    DETECTED_MODE="ddpm"
else
    DETECTED_MODE="flow_matching"  # Default fallback
fi

# Mode selection (default: detected from path, or flow_matching)
MODE="${2:-$DETECTED_MODE}"
# LR schedule selection (default: constant)
LR_SCHEDULE="${3:-constant}"

# Validate mode
if [ "$MODE" != "ddpm" ] && [ "$MODE" != "flow_matching" ]; then
    echo "❌ Error: Invalid mode '$MODE'"
    echo ""
    echo "Usage:"
    echo "  $0 [checkpoint_path] [ddpm|flow_matching] [constant|cosine]"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Auto-detect latest checkpoint"
    echo "  $0 <checkpoint_path>                 # Use specific checkpoint (mode auto-detected)"
    echo "  $0 <checkpoint_path> flow_matching   # Use specific checkpoint + mode"
    echo "  $0 <checkpoint_path> flow_matching cosine  # Use checkpoint + mode + lr_schedule"
    exit 1
fi

# Validate lr_schedule
if [ "$LR_SCHEDULE" != "constant" ] && [ "$LR_SCHEDULE" != "cosine" ]; then
    echo "❌ Error: Invalid lr_schedule '$LR_SCHEDULE'"
    echo "  Valid options: constant, cosine"
    exit 1
fi

echo "========================================"
echo "Continue Training Graph DiT Policy"
echo "Checkpoint: $CHECKPOINT"
echo "Mode: $MODE"
echo "LR Schedule: $LR_SCHEDULE"
echo "========================================"

# DEMO-LEVEL TRAINING: Each sample is a complete demo sequence
# batch_size now means number of demos per batch (not timesteps!)
# Each demo has ~100 timesteps, so effective batch = batch_size * 100
# skip_first_steps: Skip noisy initial actions from human demo collection

python scripts/graph_dit/train.py \
    --task SO-ARM101-Lift-Cube-v0 \
    --dataset ./datasets/lift_annotated_dataset.hdf5 \
    --obs_dim 32 \
    --action_dim 6 \
    --action_history_length 4 \
    --skip_first_steps 0 \
    --mode "$MODE" \
    --lr_schedule "$LR_SCHEDULE" \
    --epochs 5000 \
    --batch_size 32 \
    --lr 3e-4 \
    --hidden_dim 96 \
    --num_layers 3 \
    --num_heads 4 \
    --pred_horizon 16 \
    --exec_horizon 8 \
    --device cuda \
    --save_dir ./logs/graph_dit/lift_joint \
    --log_dir ./logs/graph_dit/lift_joint \
    --resume "$CHECKPOINT"
  
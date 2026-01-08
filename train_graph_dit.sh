#!/bin/bash
# Train Graph DiT Policy

# Mode selection (default: flow_matching)
MODE="${1:-flow_matching}"
# LR schedule selection (default: constant)
LR_SCHEDULE="${2:-constant}"

# Validate mode
if [ "$MODE" != "ddpm" ] && [ "$MODE" != "flow_matching" ]; then
    echo "❌ Error: Invalid mode '$MODE'"
    echo ""
    echo "Usage:"
    echo "  $0 [ddpm|flow_matching] [constant|cosine]"
    echo ""
    echo "Examples:"
    echo "  $0                        # Train with Flow Matching + constant LR (default)"
    echo "  $0 flow_matching          # Train with Flow Matching + constant LR"
    echo "  $0 flow_matching cosine   # Train with Flow Matching + cosine LR schedule"
    echo "  $0 ddpm constant          # Train with DDPM + constant LR"
    exit 1
fi

# Validate lr_schedule
if [ "$LR_SCHEDULE" != "constant" ] && [ "$LR_SCHEDULE" != "cosine" ]; then
    echo "❌ Error: Invalid lr_schedule '$LR_SCHEDULE'"
    echo "  Valid options: constant, cosine"
    exit 1
fi

echo "========================================"
echo "Training Graph DiT Policy"
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
    --action_history_length 8 \
    --skip_first_steps 0 \
    --mode "$MODE" \
    --lr_schedule "$LR_SCHEDULE" \
    --epochs 500 \
    --batch_size 32 \
    --lr 3e-4 \
    --hidden_dim 64 \
    --num_layers 2 \
    --num_heads 4 \
    --pred_horizon 20 \
    --exec_horizon 10\
    --device cuda \
    --save_dir ./logs/graph_dit/lift_joint \
    --log_dir ./logs/graph_dit/lift_joint
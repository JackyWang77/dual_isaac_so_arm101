#!/bin/bash
# Train Graph-Unet Policy (Graph encoder + U-Net 1D backbone)

# Mode selection (default: flow_matching)
MODE="${1:-flow_matching}"

# Validate mode (only flow_matching supported)
if [ "$MODE" != "flow_matching" ]; then
    echo "❌ Error: Invalid mode '$MODE'"
    echo ""
    echo "Usage:"
    echo "  $0 [flow_matching]"
    echo ""
    echo "Examples:"
    echo "  $0                        # Train with Flow Matching (default)"
    echo "  $0 flow_matching          # Train with Flow Matching"
    exit 1
fi

echo "========================================"
echo "Training Graph-Unet Policy"
echo "Mode: $MODE"
echo "LR Schedule: constant (fixed)"
echo "========================================"

python scripts/graph_unet/train.py \
    --task SO-ARM101-Lift-Cube-v0 \
    --dataset ./datasets/lift_annotated_dataset.hdf5 \
    --obs_dim 32 \
    --action_dim 6 \
    --action_history_length 10 \
    --skip_first_steps 0 \
    --mode "$MODE" \
    --lr_schedule constant \
    --epochs 1000 \
    --batch_size 16 \
    --lr 3e-4 \
    --hidden_dim 64 \
    --num_layers 2 \
    --num_heads 4 \
    --pred_horizon 20 \
    --exec_horizon 10 \
    --device cuda \
    --save_dir ./logs/graph_unet/lift_joint \
    --log_dir ./logs/graph_unet/lift_joint

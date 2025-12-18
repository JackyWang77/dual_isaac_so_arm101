#!/bin/bash
# Train Graph DiT Policy

# Mode selection (default: ddpm)
MODE="${1:-ddpm}"

# Validate mode
if [ "$MODE" != "ddpm" ] && [ "$MODE" != "flow_matching" ]; then
    echo "❌ Error: Invalid mode '$MODE'"
    echo ""
    echo "Usage:"
    echo "  $0 [ddpm|flow_matching]"
    echo ""
    echo "Examples:"
    echo "  $0              # Train with DDPM (default)"
    echo "  $0 ddpm         # Train with DDPM"
    echo "  $0 flow_matching  # Train with Flow Matching (faster inference)"
    exit 1
fi

echo "========================================"
echo "Training Graph DiT Policy"
echo "Mode: $MODE"
echo "========================================"

python scripts/graph_dit/train.py \
    --task SO-ARM101-Lift-Cube-v0 \
    --dataset ./datasets/lift_annotated_dataset.hdf5 \
    --obs_dim 32 \
    --action_dim 6 \
    --action_history_length 4 \
    --mode "$MODE" \
    --epochs 500 \
    --batch_size 256 \
    --lr 3e-4 \
    --hidden_dim 128 \
    --num_layers 3 \
    --num_heads 4 \
    --device cuda \
    --save_dir ./logs/graph_dit/lift_joint \
    --log_dir ./logs/graph_dit/lift_joint

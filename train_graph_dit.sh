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
    --task SO-ARM101-Reach-Cube-v0 \
    --dataset ./datasets/dataset.hdf5 \
    --obs_dim 32 \
    --action_dim 6 \
    --action_history_length 4 \
    --mode "$MODE" \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-4 \
    --hidden_dim 256 \
    --num_layers 6 \
    --num_heads 8 \
    --device cuda \
    --save_dir ./logs/graph_dit/reach_joint_${MODE} \
    --log_dir ./logs/graph_dit/reach_joint_${MODE}

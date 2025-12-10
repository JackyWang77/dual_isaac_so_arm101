#!/bin/bash
# Train Graph DiT Policy with improved hyperparameters using ANNOTATED dataset

# Mode selection (default: flow_matching)
MODE="${1:-flow_matching}"

# Validate mode
if [ "$MODE" != "ddpm" ] && [ "$MODE" != "flow_matching" ]; then
    echo "❌ Error: Invalid mode '$MODE'"
    echo ""
    echo "Usage:"
    echo "  $0 [ddpm|flow_matching]"
    exit 1
fi

echo "========================================"
echo "Training Graph DiT Policy (Improved Config)"
echo "Using: ANNOTATED Dataset"
echo "Mode: $MODE"
echo "========================================"
echo ""
echo "Dataset: ./datasets/reach_annotated_dataset.hdf5"
echo ""
echo "Improvements:"
echo "  - Epochs: 500 (was 200)"
echo "  - Batch size: 512 (was 256)"
echo "  - Learning rate: 5e-5 (was 1e-4)"
echo "  - Hidden dim: 512 (was 256)"
echo "  - Layers: 8 (was 6)"
echo "  - Heads: 16 (was 8)"
echo ""

python scripts/graph_dit/train.py \
    --task SO-ARM101-Reach-Cube-v0 \
    --dataset ./datasets/reach_annotated_dataset.hdf5 \
    --obs_dim 32 \
    --action_dim 6 \
    --action_history_length 4 \
    --mode "$MODE" \
    --epochs 100 \
    --batch_size 512 \
    --lr 5e-5 \
    --hidden_dim 512 \
    --num_layers 8 \
    --num_heads 16 \
    --device cuda \
    --save_dir ./logs/graph_dit/reach_joint \
    --log_dir ./logs/graph_dit/reach_joint

#!/bin/bash
# 🚑 SINGLE BATCH OVERFITTING TEST
# This script tests if the model can overfit a single demo
# Expected outcome: Loss should drop to < 0.001 if code logic is correct
# If loss stays > 0.05, there's likely a bug in model architecture or data flow

echo "========================================"
echo "🚑 SINGLE BATCH OVERFITTING TEST"
echo "========================================"
echo ""
echo "This test uses ONLY the first demo, replicated 16 times"
echo "Expected: Loss should drop to < 0.001 within 100 epochs"
echo "If loss stays > 0.05, check:"
echo "  1. Graph conditioning injection (Cross-Attention/AdaLN)"
echo "  2. Time embedding broadcasting"
echo "  3. Normalization stats (check output above)"
echo "  4. GNN layer count (should be 2, not 6+)"
echo ""

# Enable single batch test mode
export SINGLE_BATCH_TEST=true
export SINGLE_BATCH_SIZE=16
# Enable debug mode to print diagnostic info
export DEBUG_LOSS=true

# Use "low-power" config for faster debugging
python scripts/graph_dit/train.py \
    --task SO-ARM101-Lift-Cube-v0 \
    --dataset ./datasets/lift_annotated_dataset.hdf5 \
    --obs_dim 32 \
    --action_dim 6 \
    --action_history_length 4 \
    --mode flow_matching \
    --lr_schedule constant \
    --epochs 300 \
    --batch_size 16 \
    --lr 1e-3 \
    --hidden_dim 128 \
    --num_layers 4 \
    --num_heads 4 \
    --pred_horizon 16 \
    --exec_horizon 8 \
    --device cuda \
    --save_dir ./logs/graph_dit/single_batch_test \
    --log_dir ./logs/graph_dit/single_batch_test


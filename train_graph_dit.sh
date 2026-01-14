#!/bin/bash
# Train Graph DiT Policy

# Mode selection (default: flow_matching)
MODE="${1:-flow_matching}"
# LR schedule selection (default: constant)
LR_SCHEDULE="${2:-constant}"

# Validate mode (only flow_matching supported)
if [ "$MODE" != "flow_matching" ]; then
    echo "❌ Error: Invalid mode '$MODE'"
    echo ""
    echo "Usage:"
    echo "  $0 [flow_matching] [constant|cosine]"
    echo ""
    echo "Examples:"
    echo "  $0                        # Train with Flow Matching + constant LR (default)"
    echo "  $0 flow_matching          # Train with Flow Matching + constant LR"
    echo "  $0 flow_matching cosine   # Train with Flow Matching + cosine LR schedule"
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
#
# ABSOLUTE POSITION VERSION:
# - action_seq[i] = joint_pos[i + action_target_offset] (absolute position)
# - action_trajectory_seq[i] = [joint_pos[i+start_offset], ..., joint_pos[i+start_offset+pred_horizon-1]] (absolute positions)
# - action_history_seq[i] = [actions[i-history_length+1], ..., actions[i]] (absolute positions)
# - All actions are normalized using absolute position statistics

python scripts/graph_dit/train.py \
    --task SO-ARM101-Lift-Cube-v0 \
    --dataset ./datasets/lift_annotated_dataset.hdf5 \
    --obs_dim 32 \
    --action_dim 6 \
    --action_history_length 10 \
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
    --exec_horizon 10 \
    --device cuda \
    --save_dir ./logs/graph_dit/lift_joint \
    --log_dir ./logs/graph_dit/lift_joint
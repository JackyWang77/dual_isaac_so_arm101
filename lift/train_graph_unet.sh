#!/bin/bash
# Train GraphUnetPolicy (Full Graph Attention + U-Net) — Lift task
cd "$(dirname "$0")/.."

MODE="${1:-flow_matching}"
JOINT_FILM="${2:-}"

if [ "$MODE" != "flow_matching" ]; then
    echo "Usage: $0 [flow_matching] [joint]"
    exit 1
fi

EXTRA_ARGS=""
SUFFIX="lift_joint"
if [ "$JOINT_FILM" = "joint" ]; then
    EXTRA_ARGS="--use_joint_film"
    SUFFIX="lift_joint_film"
fi

echo "========================================"
echo "Training GraphUnetPolicy — Lift"
echo "Mode: $MODE | Joint FiLM: ${JOINT_FILM:-off}"
echo "Save: ./logs/graph_unet_full/$SUFFIX"
echo "========================================"

python scripts/graph_unet/train.py \
    --task SO-ARM101-Lift-Cube-v0 \
    --dataset ./datasets/lift_annotated_dataset.hdf5 \
    --policy_type graph_unet \
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
    --num_layers 1 \
    --num_heads 4 \
    --graph_edge_dim 32 \
    --pred_horizon 20 \
    --exec_horizon 10 \
    --device cuda \
    --save_dir "./logs/graph_unet_full/$SUFFIX" \
    --log_dir "./logs/graph_unet_full/$SUFFIX" \
    $EXTRA_ARGS

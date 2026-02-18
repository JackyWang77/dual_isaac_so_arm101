#!/bin/bash
# Train GraphUnetPolicy (Full Graph Attention + U-Net) - Dual Cube Stack task
cd "$(dirname "$0")/.."

MODE="${1:-flow_matching}"
JOINT_FILM="${2:-}"

if [ "$MODE" != "flow_matching" ]; then
    echo "Usage: $0 [flow_matching] [joint]"
    exit 1
fi

EXTRA_ARGS=""
SUFFIX="stack_joint"
if [ "$JOINT_FILM" = "joint" ]; then
    EXTRA_ARGS="--use_joint_film"
    SUFFIX="stack_joint_film"
fi

# Stack task: 4 nodes (2 EE + 2 objects)
NODE_CONFIGS='[{"name":"left_ee","type":0,"pos_key":"left_ee_position","ori_key":"left_ee_orientation"},{"name":"right_ee","type":0,"pos_key":"right_ee_position","ori_key":"right_ee_orientation"},{"name":"cube_1","type":1,"pos_key":"cube_1_pos","ori_key":"cube_1_ori"},{"name":"cube_2","type":1,"pos_key":"cube_2_pos","ori_key":"cube_2_ori"}]'
OBS_KEYS='["left_joint_pos","left_joint_vel","right_joint_pos","right_joint_vel","left_ee_position","left_ee_orientation","right_ee_position","right_ee_orientation","cube_1_pos","cube_1_ori","cube_2_pos","cube_2_ori","last_action_all"]'

echo "========================================"
echo "Training GraphUnetPolicy - Stack"
echo "Mode: $MODE | Joint FiLM: ${JOINT_FILM:-off}"
echo "Save: ./logs/graph_unet_full/$SUFFIX"
echo "========================================"

python scripts/graph_unet/train.py \
    --task SO-ARM101-Dual-Cube-Stack-v0 \
    --dataset ./datasets/dual_cube_stack_joint_states_mimic_dataset_100.hdf5 \
    --policy_type graph_unet \
    --obs_dim 64 \
    --action_dim 12 \
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
    --obs_keys "$OBS_KEYS" \
    --node_configs "$NODE_CONFIGS" \
    --save_dir "./logs/graph_unet_full/$SUFFIX" \
    --log_dir "./logs/graph_unet_full/$SUFFIX" \
    $EXTRA_ARGS

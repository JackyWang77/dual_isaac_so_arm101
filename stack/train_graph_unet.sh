#!/bin/bash
# Train GraphUnetPolicy (Full Graph Attention + U-Net) - Dual Cube Stack task
cd "$(dirname "$0")/.."

MODE="${1:-flow_matching}"
JOINT_FILM="${2:-}"
RESUME="${3:-}"  # Optional: path to checkpoint to resume, or use env RESUME_CHECKPOINT
RESUME_CHECKPOINT="${RESUME:-$RESUME_CHECKPOINT}"

if [ "$MODE" != "flow_matching" ]; then
    echo "Usage: $0 [flow_matching] [joint] [resume_checkpoint]"
    echo "  Example: $0 flow_matching                    # from scratch"
    echo "  Example: $0 flow_matching '' ./logs/graph_unet_full/stack_joint_t1_gripper/checkpoint_200.pt"
    echo "  Or:      RESUME_CHECKPOINT=./path/to/ckpt.pt $0 flow_matching"
    exit 1
fi

# Default: joint_pos[t+1] + gripper映射(>-0.6→1, <=-0.6→-1)
EXTRA_ARGS="--action_offset 1"
SUFFIX="stack_joint_t1_gripper"
if [ "$JOINT_FILM" = "joint" ]; then
    EXTRA_ARGS="--use_joint_film --action_offset 1"
    SUFFIX="stack_joint_film_t1_gripper"
fi

# Stack task: 4 nodes (2 EE + 2 objects)
NODE_CONFIGS='[{"name":"left_ee","type":0,"pos_key":"left_ee_position","ori_key":"left_ee_orientation"},{"name":"right_ee","type":0,"pos_key":"right_ee_position","ori_key":"right_ee_orientation"},{"name":"cube_1","type":1,"pos_key":"cube_1_pos","ori_key":"cube_1_ori"},{"name":"cube_2","type":1,"pos_key":"cube_2_pos","ori_key":"cube_2_ori"}]'
OBS_KEYS='["left_joint_pos","left_joint_vel","right_joint_pos","right_joint_vel","left_ee_position","left_ee_orientation","right_ee_position","right_ee_orientation","cube_1_pos","cube_1_ori","cube_2_pos","cube_2_ori","last_action_all"]'

echo "========================================"
echo "Training GraphUnetPolicy - Stack"
echo "Mode: $MODE | Joint FiLM: ${JOINT_FILM:-off}"
echo "Save: ./logs/graph_unet_full/$SUFFIX"
[ -n "$RESUME_CHECKPOINT" ] && echo "Resume: $RESUME_CHECKPOINT"
echo "========================================"

python scripts/graph_unet/train.py \
    --task SO-ARM101-Dual-Cube-Stack-v0 \
    --dataset ./datasets/dual_cube_stack_annotated_dataset.hdf5 \
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
    --num_layers 1 \
    --num_heads 4 \
    --graph_edge_dim 32 \
    --pred_horizon 20 \
    --exec_horizon 10 \
    --device cuda \
    --obs_keys "$OBS_KEYS" \
    --node_configs "$NODE_CONFIGS" \
    --save_dir "./logs/graph_unet_full/$SUFFIX" \
    --log_dir "./logs/graph_unet_full/$SUFFIX" \
    --save_every 200 \
    $EXTRA_ARGS \
    ${RESUME_CHECKPOINT:+--resume "$RESUME_CHECKPOINT"}

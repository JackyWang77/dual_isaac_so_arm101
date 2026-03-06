#!/bin/bash
# Train MLP baseline (DualArmUnetPolicyMLP) - same 4-frame history as graph, MLP encoder only.
# Aligned with train_disentangled_graph: 1000 ep, no joint, no cross.
# hidden_dim 64 + z_dim 64 (default) so encoder size and injection dim (z 64 + raw_proj 32 = 96) match graph.
cd "$(dirname "$0")/.."

MODE="${1:-flow_matching}"
RESUME="$2"

RESUME_CHECKPOINT="${RESUME:-$RESUME_CHECKPOINT}"
if [[ -n "$RESUME" && ("$RESUME" == *".pt"* || "$RESUME" == *"/"*) ]]; then
    RESUME_CHECKPOINT="$RESUME"
fi

if [ "$MODE" != "flow_matching" ]; then
    echo "Usage: $0 flow_matching [resume_checkpoint]"
    echo "  Example: $0 flow_matching"
    echo "  Example: EPOCHS=2000 $0 flow_matching ./logs/.../checkpoint_1000.pt"
    exit 1
fi

EPOCHS="${EPOCHS:-1000}"
SUFFIX="stack_joint_t1_gripper_mlp_baseline"

# Same as graph: 4 nodes, same OBS_KEYS
NODE_CONFIGS='[{"name":"left_ee","type":0,"pos_key":"left_ee_position","ori_key":"left_ee_orientation"},{"name":"right_ee","type":0,"pos_key":"right_ee_position","ori_key":"right_ee_orientation"},{"name":"cube_1","type":1,"pos_key":"cube_1_pos","ori_key":"cube_1_ori"},{"name":"cube_2","type":1,"pos_key":"cube_2_pos","ori_key":"cube_2_ori"}]'
OBS_KEYS='["left_joint_pos","left_joint_vel","right_joint_pos","right_joint_vel","left_ee_position","left_ee_orientation","right_ee_position","right_ee_orientation","cube_1_pos","cube_1_ori","cube_2_pos","cube_2_ori","last_action_all"]'

echo "========================================"
echo "Training MLP baseline (DualArmUnetPolicyMLP) - Stack"
echo "Aligned with train_disentangled_graph: 1000 ep, no joint, no cross"
echo "Save: ./logs/graph_unet_full/$SUFFIX"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "Resume: $RESUME_CHECKPOINT"
    if [ ! -f "$RESUME_CHECKPOINT" ]; then
        echo "ERROR: Checkpoint file not found! Cannot resume."
        exit 1
    fi
fi
echo "Epochs: $EPOCHS"
echo "========================================"

python scripts/graph_unet/train.py \
    --task SO-ARM101-Dual-Cube-Stack-v0 \
    --dataset ./datasets/dual_cube_stack_annotated_dataset.hdf5 \
    --policy_type unet \
    --obs_dim 64 \
    --action_dim 12 \
    --action_history_length 4 \
    --skip_first_steps 0 \
    --mode "$MODE" \
    --lr_schedule constant \
    --epochs "$EPOCHS" \
    --batch_size 16 \
    --lr 3e-4 \
    --hidden_dim 64 \
    --pred_horizon 20 \
    --exec_horizon 10 \
    --device cuda \
    --obs_keys "$OBS_KEYS" \
    --node_configs "$NODE_CONFIGS" \
    --save_dir "./logs/graph_unet_full/$SUFFIX" \
    --log_dir "./logs/graph_unet_full/$SUFFIX" \
    --save_every 200 \
    --cross_attention false \
    --action_offset 1 \
    ${RESUME_CHECKPOINT:+--resume "$RESUME_CHECKPOINT"}

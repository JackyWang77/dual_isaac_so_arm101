#!/bin/bash
# Train DualArmDisentangledPolicyGated - Dual Cube Stack task
# Same as disentangled but: gated fusion (raw + gate*graph) instead of concat; logs metrics/graph_gate_weight.
cd "$(dirname "$0")/.."

MODE="${1:-flow_matching}"
ARG2="${2:-}"
ARG3="${3:-}"

if [ "$ARG2" = "joint" ]; then
    JOINT_FILM="joint"
    RESUME="$ARG3"
else
    JOINT_FILM=""
    if [[ "$ARG2" == *".pt"* ]] || [[ "$ARG2" == *"/"* ]]; then
        RESUME="$ARG2"
    else
        RESUME="$ARG3"
    fi
fi
RESUME_CHECKPOINT="${RESUME:-$RESUME_CHECKPOINT}"

if [ "$MODE" != "flow_matching" ]; then
    echo "Usage: $0 flow_matching [joint] [resume_checkpoint]"
    echo "  Example: $0 flow_matching"
    echo "  Example: EPOCHS=2000 $0 flow_matching ./logs/.../checkpoint_1000.pt"
    exit 1
fi

EPOCHS="${EPOCHS:-1000}"

CROSS_ATTN="${CROSS_ATTN:-false}"
if [ "$CROSS_ATTN" = "true" ] || [ "$CROSS_ATTN" = "1" ]; then
    CROSS_ATTN_VAL="true"
    CROSS_SUFFIX="_cross"
else
    CROSS_ATTN_VAL="false"
    CROSS_SUFFIX="_nocross"
fi

# per_gate: false = scalar gate (default), true = 64 per-dimension gates (more flexible if scalar bottlenecks)
PER_GATE="${PER_GATE:-false}"
if [ "$PER_GATE" = "true" ] || [ "$PER_GATE" = "1" ]; then
    PER_GATE_VAL="true"
    PER_GATE_SUFFIX="_pergate"
else
    PER_GATE_VAL="false"
    PER_GATE_SUFFIX=""
fi

EXTRA_ARGS="--action_offset 1"
SUFFIX="stack_disentangled_gated_t1_gripper${CROSS_SUFFIX}${PER_GATE_SUFFIX}"
if [ "$JOINT_FILM" = "joint" ]; then
    EXTRA_ARGS="--use_joint_film --action_offset 1"
    SUFFIX="stack_disentangled_gated_joint_film_t1_gripper${CROSS_SUFFIX}${PER_GATE_SUFFIX}"
fi

NODE_CONFIGS='[{"name":"left_ee","type":0,"pos_key":"left_ee_position","ori_key":"left_ee_orientation"},{"name":"right_ee","type":0,"pos_key":"right_ee_position","ori_key":"right_ee_orientation"},{"name":"cube_1","type":1,"pos_key":"cube_1_pos","ori_key":"cube_1_ori"},{"name":"cube_2","type":1,"pos_key":"cube_2_pos","ori_key":"cube_2_ori"}]'
OBS_KEYS='["left_joint_pos","left_joint_vel","right_joint_pos","right_joint_vel","left_ee_position","left_ee_orientation","right_ee_position","right_ee_orientation","cube_1_pos","cube_1_ori","cube_2_pos","cube_2_ori","last_action_all"]'

echo "========================================"
echo "Training DualArmDisentangledPolicyGated - Stack (gated fusion)"
echo "Mode: $MODE | Joint FiLM: ${JOINT_FILM:-off} | CrossArmAttn: $CROSS_ATTN | PerGate: $PER_GATE"
echo "Save: ./logs/disentangled_graph_gated/$SUFFIX"
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
    --policy_type disentangled_graph_unet_gated \
    --obs_dim 64 \
    --action_dim 12 \
    --action_history_length 4 \
    --skip_first_steps 0 \
    --mode "$MODE" \
    --lr_schedule constant \
    --epochs "$EPOCHS" \
    --batch_size 16 \
    --lr 3e-4 \
    --hidden_dim 32 \
    --num_layers 1 \
    --num_heads 4 \
    --graph_edge_dim 8 \
    --pred_horizon 20 \
    --exec_horizon 10 \
    --device cuda \
    --obs_keys "$OBS_KEYS" \
    --node_configs "$NODE_CONFIGS" \
    --save_dir "./logs/disentangled_graph_gated/$SUFFIX" \
    --log_dir "./logs/disentangled_graph_gated/$SUFFIX" \
    --save_every 200 \
    --cross_attention "$CROSS_ATTN_VAL" \
    --per_gate "$PER_GATE_VAL" \
    $EXTRA_ARGS \
    ${RESUME_CHECKPOINT:+--resume "$RESUME_CHECKPOINT"}

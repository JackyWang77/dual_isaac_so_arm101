#!/bin/bash
# Record demonstrations for healthcare table setting task (elderly care use case).
# Dual-arm: left hand places fork, right hand places knife onto tray beside plate.
#
# Usage:
#   bash test_healthcare_states.sh                          # default: 200 demos, joint_states
#   bash test_healthcare_states.sh 50                       # 50 demos
#   bash test_healthcare_states.sh 100 spacemouse           # 100 demos with spacemouse
#   bash test_healthcare_states.sh 50 joint_states ./datasets/my_table_setting.hdf5
#
eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# Source ROS2 (same as test_dual_states.sh / test_joint_states.sh) for joint_states teleop
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/local_setup.bash
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NUM_DEMOS="${1:-20}"
TELEOP_DEVICE="${2:-joint_states}"
DATASET_FILE="${3:-./datasets/table_setting_dataset.hdf5}"

TASK="SO-ARM101-Dual-Table-Setting-Joint-States-Mimic-v0"

echo "========================================"
echo "Healthcare Table Setting - Data Collection"
echo "========================================"
echo "Task:          $TASK"
echo "Demos:         $NUM_DEMOS"
echo "Teleop Device: $TELEOP_DEVICE"
echo "Dataset:       $DATASET_FILE"
echo "========================================"
echo ""
echo "Scene layout:"
echo "  - Tray + Plate: center (static)"
echo "  - Fork: spawns LEFT  -> left arm places LEFT of plate"
echo "  - Knife: spawns RIGHT -> right arm places RIGHT of plate"
echo "========================================"

python scripts/record_demos.py \
    --task "$TASK" \
    --teleop_device "$TELEOP_DEVICE" \
    --dataset_file "$DATASET_FILE" \
    --num_demos "$NUM_DEMOS" \
    --num_success_steps 10

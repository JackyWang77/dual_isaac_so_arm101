#!/bin/bash
# Record dual-arm cube stack demos via joint_states teleop (same flow as test_joint_states.sh / lift_old).
# Usage: ./test_dual_states.sh [num_demos] [dataset_file]
# Example: ./test_dual_states.sh 200
# Example: ./test_dual_states.sh 200 ./datasets/dual_cube_stack_dataset.hdf5

eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# Source ROS2 (same as test_joint_states.sh)
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/local_setup.bash
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

NUM_DEMOS="${1:-80}"
DATASET_FILE="${2:-}"

cd /mnt/ssd/dual_isaac_so_arm101

ARGS=(
    scripts/record_demos.py
    --task SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-v0
    --teleop_device joint_states
    --num_demos "$NUM_DEMOS"
    --enable_cameras
)
[ -n "$DATASET_FILE" ] && ARGS+=(--dataset_file "$DATASET_FILE")

exec python "${ARGS[@]}"

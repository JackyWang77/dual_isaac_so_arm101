#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# Source Python 3.11 的 ROS2
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/local_setup.bash
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp


cd /mnt/ssd/dual_isaac_so_arm101
python scripts/record_demos.py \
    --task SO-ARM101-Lift-Joint-States-Mimic-v0 \
    --teleop_device joint_states \
    --num_demos 200 \
    --enable_cameras
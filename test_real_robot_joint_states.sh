#!/bin/bash
# 测试真实机器人 Joint States → Isaac Sim

eval "$(conda shell.bash hook)"
conda activate env_isaaclab

source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/local_setup.bash
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp


cd /mnt/ssd/dual_isaac_so_arm101
python scripts/teleop_se3_agent.py \
    --task SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0 \
    --teleop_device joint_states \
    --num_envs 1 \
    --enable_cameras
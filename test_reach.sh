#!/bin/bash
# 测试 Reach 环境使用 joint_states 控制
# 使用 SO-ARM101-Reach-Joint-States-Mimic-v0 环境
# 这个环境接受 joint_states 控制，记录的是 joint states

eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# Source Python 3.11 的 ROS2
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/local_setup.bash
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash

# 设置 ROS2 环境
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp


cd /mnt/ssd/dual_isaac_so_arm101
python scripts/record_demos.py \
    --task SO-ARM101-Reach-Joint-States-Mimic-v0 \
    --teleop_device joint_states \
    --num_demos 100 \
    --enable_cameras

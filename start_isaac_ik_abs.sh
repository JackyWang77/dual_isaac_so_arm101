#!/bin/bash
# 使用 Python 3.11 编译的 ROS2 启动 Isaac Sim (IK Absolute 模式)

# 激活 Isaac Lab 环境
eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# Source Python 3.11 的 ROS2
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/local_setup.bash
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash

# 设置 ROS2 环境
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 Isaac Sim - IK Absolute 模式"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ conda 环境: $CONDA_DEFAULT_ENV"
echo "✓ ROS2 Python 3.11 workspace sourced"
echo "  ROS_DISTRO: $ROS_DISTRO"
echo "  ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo ""
echo "📌 Task: SO-ARM100-Pick-Place-DualArm-IK-Abs-v0"
echo "📌 Control: IK Absolute (目标位置，相对于 robot base)"
echo ""

# 运行 Isaac Sim
cd /mnt/ssd/dual_isaac_so_arm101
python scripts/teleop_se3_agent.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --teleop_device ros2


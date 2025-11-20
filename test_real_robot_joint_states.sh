#!/bin/bash
# 测试真实机器人 Joint States → Isaac Sim

eval "$(conda shell.bash hook)"
conda activate env_isaaclab

source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/local_setup.bash
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "🧪 测试: 真实机器人 → Isaac Sim (Joint States)"
echo ""
echo "这个脚本会:"
echo "  1. 启动 Isaac Sim 环境"
echo "  2. 订阅 /joint_states 话题"
echo "  3. 仿真机器人跟随真实机器人"
echo "  4. 终端显示接收到的 joint positions"
echo ""
echo "请确保真实机器人硬件驱动已运行:"
echo "  python3 so_arm_hardware_driver_ik_abs.py"
echo ""

# 检查 /joint_states 话题
echo "🔍 检查 ROS2 话题..."
if timeout 2s ros2 topic list 2>/dev/null | grep -q "/joint_states"; then
    echo "✅ 检测到 /joint_states 话题"
    echo ""
    echo "📊 最新的 joint states:"
    timeout 2s ros2 topic echo /joint_states --once 2>/dev/null || echo "  (无法读取，但话题存在)"
    echo ""
else
    echo "⚠️  警告: 未检测到 /joint_states 话题"
    echo ""
    echo "请在另一个终端启动硬件驱动:"
    echo "  cd /mnt/ssd/dual_isaac_so_arm101"
    echo "  source /opt/ros/humble/setup.bash"
    echo "  export ROS_DOMAIN_ID=0"
    echo "  python3 so_arm_hardware_driver_ik_abs.py"
    echo ""
    read -p "按 Enter 继续测试（可能会失败），或 Ctrl+C 取消..."
fi

echo "🚀 启动 Isaac Sim..."
echo "   环境: SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0"
echo "   遥控: joint_states (真实机器人)"
echo ""
echo "移动真实机器人，观察仿真机器人是否跟随"
echo "按 ESC 退出"
echo ""

cd /mnt/ssd/dual_isaac_so_arm101
python scripts/teleop_se3_agent.py \
    --task SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0 \
    --teleop_device joint_states \
    --num_envs 1



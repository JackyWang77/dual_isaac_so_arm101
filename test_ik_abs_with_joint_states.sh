#!/bin/bash
# 测试 IK-Abs 环境使用 joint_states 控制
# 使用 SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0 环境
# 这个环境接受 joint_states 控制，但记录的是 EE absolute pose

eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# Source Python 3.11 的 ROS2
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/local_setup.bash
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash

# 设置 ROS2 环境
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 测试: 使用 joint_states 录制演示（直接记录关节状态）"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "环境: SO-ARM100-Pick-Place-Joint-States-Mimic-v0"
echo "说明: 这个环境接受 joint_states 控制，直接记录 joint states，且有子任务配置"
echo ""
echo "特性:"
echo "  ✅ 可以用 joint_states 控制（真实机器人）"
echo "  ✅ 直接记录 Joint States [joint_1, ..., joint_5, gripper]"
echo "  ✅ 有子任务配置（subtask_configs）用于数据生成"
echo "  ✅ 后续可以通过 Forward Kinematics 转换为 EE pose"
echo ""
echo "子任务列表:"
echo "  1. Pick plate   (抓取盘子)"
echo "  2. Place plate  (放置盘子到托盘中心)"
echo "  3. Pick fork    (抓取叉子)"
echo "  4. Place fork   (放置叉子到托盘右侧 8cm)"
echo "  5. Pick knife   (抓取刀子)"
echo "  6. Place knife  (放置刀子到托盘左侧 8cm)"
echo ""
echo "工作流程:"
echo "  1. 真实机器人发布 /joint_states (ROS2 JointState)"
echo "  2. Isaac Sim 接收 joint_states，仿真机器人跟随"
echo "  3. 直接记录 Joint States [joint_1, ..., joint_5, gripper]"
echo "  4. 子任务信号用于数据生成和分割"
echo "  5. 后续读取 hdf5 时可通过 Forward Kinematics 转换为 EE pose"
echo ""
echo "✓ conda 环境: $CONDA_DEFAULT_ENV"
echo "✓ ROS2 Python 3.11 workspace sourced"
echo "  ROS_DISTRO: $ROS_DISTRO"
echo "  ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo ""

# 检查是否有 /joint_states 话题
echo "🔍 检查 ROS2 话题..."
if ! timeout 2s ros2 topic list 2>/dev/null | grep -q "/joint_states"; then
    echo "⚠️  警告: 未检测到 /joint_states 话题"
    echo "   请确保真实机器人硬件驱动正在运行"
    echo ""
    echo "   启动硬件驱动（在另一个终端）："
    echo "   cd /mnt/ssd/dual_isaac_so_arm101"
    echo "   python3 so_arm_hardware_driver_ik_abs.py"
    echo ""
    read -p "按 Enter 继续，或 Ctrl+C 取消..."
else
    echo "✅ 检测到 /joint_states 话题"
    echo ""
fi

echo "🚀 启动 Isaac Sim 录制演示..."
echo "   环境: SO-ARM100-Pick-Place-Joint-States-Mimic-v0"
echo "   遥控: joint_states (真实机器人)"
echo "   模式: 录制演示数据（直接记录关节状态）"
echo ""
echo "移动真实机器人，观察仿真机器人是否跟随"
echo "完成子任务后，数据会自动记录"
echo "按 ESC 退出"
echo ""

cd /mnt/ssd/dual_isaac_so_arm101
python scripts/record_demos.py \
    --task SO-ARM100-Pick-Place-Joint-States-Mimic-v0 \
    --teleop_device joint_states \
    --num_demos 1 \
    --enable_cameras

echo ""
echo "✅ 录制完成！"
echo ""
echo "💡 注意:"
echo "   - 这个环境接受 joint_states 控制（joint positions）"
echo "   - 直接记录 Joint States [joint_1, ..., joint_5, gripper]"
echo "   - 有子任务配置，可以用于数据生成和分割"
echo "   - 录制的数据保存在 ./datasets/dataset.hdf5"
echo "   - 后续可以通过 Forward Kinematics 转换为 EE pose"
echo ""
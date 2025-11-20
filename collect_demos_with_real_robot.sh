#!/bin/bash
# 使用真实机器人遥控 Isaac Sim，记录 EE Absolute Pose 数据
# 真实机器人发布 joint_states → Isaac Sim 接收 → 记录 EE Absolute Pose

eval "$(conda shell.bash hook)"
conda activate env_isaaclab

# Source Python 3.11 的 ROS2
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/local_setup.bash
source /mnt/ssd/IsaacSim-ros_workspaces/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash

# 设置 ROS2 环境
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 真实机器人遥控 → Isaac Sim → 记录 EE Absolute Pose"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "工作流程："
echo "  1. 真实机器人发布 /joint_states (ROS2 JointState)"
echo "  2. Isaac Sim 接收 joint_states，仿真机器人跟随真实机器人"
echo "  3. 环境通过 FK 计算 EE 绝对位置"
echo "  4. 记录 EE Absolute Pose [x,y,z,qw,qx,qy,qz,gripper]"
echo "  5. 用于训练 IK Absolute 策略"
echo ""
echo "优势："
echo "  ✅ 真实机器人直接控制（最自然）"
echo "  ✅ Joint States → FK → EE Pose（100% 可靠）"
echo "  ✅ 仿真机器人实时跟随真实机器人"
echo "  ✅ 记录的数据适合训练 IK Absolute 模型"
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

# 数据集路径
DATASET_DIR="./datasets/pick_place_ik_abs"
mkdir -p $DATASET_DIR
DATASET_FILE="$DATASET_DIR/real_robot_demos_$(date +%Y%m%d_%H%M%S).hdf5"

echo "💾 数据集路径: $DATASET_FILE"
echo ""
echo "🎮 控制说明:"
echo "   - 用真实机器人的机械控制移动手臂"
echo "   - 仿真机器人会实时跟随"
echo "   - 按 'P' 开始/停止记录演示"
echo "   - 按 'R' 重置场景"
echo "   - 按 ESC 或 Ctrl+C 退出"
echo ""
echo "🚀 启动 Isaac Sim..."
echo ""

# 运行 Isaac Sim 记录演示
cd /mnt/ssd/dual_isaac_so_arm101
python scripts/record_demos.py \
    --task SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0 \
    --teleop_device joint_states \
    --dataset_file $DATASET_FILE \
    --num_demos 10

echo ""
echo "✅ 数据收集完成！"
echo "📊 数据集保存在: $DATASET_FILE"
echo ""
echo "下一步:"
echo "  1. 检查数据: python scripts/inspect_hdf5.py --file $DATASET_FILE"
echo "  2. 训练模型: python scripts/train_diffusion_policy.py --dataset $DATASET_FILE"
echo "  3. 部署策略: python scripts/deploy_policy.py --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0"



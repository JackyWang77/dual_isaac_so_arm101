#!/bin/bash
# 测试 IK Absolute 模式 - 使用正确的坐标系

cd /tmp
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

echo "🎯 IK Absolute 模式测试 - 正确坐标系"
echo "⚠️  机器人 base 旋转了90度！"
echo "📍 当前 wrist_2_link 在 base frame: [0.0, -0.2387, 0.1767]"
echo "⚙️  body_offset=[-0.005, -0.1, 0.0]"
echo "按 Ctrl+C 停止"
echo ""

# 定义四个目标位置（相对于 robot base frame - 已旋转90度）
# [x, y, z, roll, pitch, yaw, gripper]
# 正确坐标系：X=左右, Y=前后（负=前方）, Z=上下

# 位置1: 接近当前位置
POSE1="[0.0, -0.24, 0.18, 0.0, 0.0, 0.0, 1.0]"

# 位置2: 向前 + 向右
POSE2="[0.05, -0.28, 0.18, 0.0, 0.0, 0.3, -1.0]"

# 位置3: 向前 + 向左
POSE3="[-0.05, -0.28, 0.18, 0.0, 0.0, -0.3, 1.0]"

# 位置4: 向上
POSE4="[0.0, -0.24, 0.22, 0.0, 0.2, 0.0, -1.0]"

while true; do
    echo "📍 1️⃣  中心位置 [0.0, -0.24, 0.18] 🟢 爪子开"
    timeout 5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $POSE1}" --rate 30 &
    sleep 5.0
    
    echo "📍 2️⃣  向前+右 [0.05, -0.28, 0.18] + 右转0.3 🔴 爪子闭"
    timeout 5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $POSE2}" --rate 30 &
    sleep 5.0
    
    echo "📍 3️⃣  向前+左 [-0.05, -0.28, 0.18] + 左转0.3 🟢 爪子开"
    timeout 5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $POSE3}" --rate 30 &
    sleep 5.0
    
    echo "📍 4️⃣  向上 [0.0, -0.24, 0.22] + 俯仰0.2 🔴 爪子闭"
    timeout 5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $POSE4}" --rate 30 &
    sleep 5.0
    
    echo "━━━━━━━━━━ 循环 ━━━━━━━━━━"
done

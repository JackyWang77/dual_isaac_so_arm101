#!/bin/bash
# 循环发送三个不同的 pose 测试 IK

cd /tmp
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0

echo "🎯 开始循环发送六个测试位置（极限测试 - 含旋转）"
echo "按 Ctrl+C 停止"
echo ""

# 定义四个测试增量 (delta) - 大幅度移动，适合录视频
# [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, delta_gripper]
# 位置变化：±0.08m (8cm 大幅度)
# 旋转变化：±0.5 rad (约 28度)
# gripper: ±0.3

# Delta 1: 向前右上 + 右转 - 爪子打开
POSE1="[0.08, 0.05, 0.03, 0.0, 0.0, 0.5, 0.3]"
# Delta 2: 向后左下 + 左转 - 爪子闭合
POSE2="[-0.08, -0.05, -0.03, 0.0, 0.0, -0.5, -0.3]"
# Delta 3: 向右上 + 俯仰 - 爪子打开
POSE3="[0.05, 0.08, 0.04, 0.0, 0.4, 0.0, 0.3]"
# Delta 4: 向左下 + 俯仰回 - 爪子闭合
POSE4="[-0.05, -0.08, -0.04, 0.0, -0.4, 0.0, -0.3]"

# Zero delta for holding position
ZERO="[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"

while true; do
    echo "📍 1️⃣  向前右上 + 右转 🟢 爪子开"
    timeout 1.5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $POSE1}" --rate 30 &
    sleep 1.5
    timeout 3.5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $ZERO}" --rate 30 &
    sleep 3.5
    
    echo "📍 2️⃣  向后左下 + 左转 🔴 爪子闭"
    timeout 1.5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $POSE2}" --rate 30 &
    sleep 1.5
    timeout 3.5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $ZERO}" --rate 30 &
    sleep 3.5
    
    echo "📍 3️⃣  向右上 + 俯仰 🟢 爪子开"
    timeout 1.5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $POSE3}" --rate 30 &
    sleep 1.5
    timeout 3.5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $ZERO}" --rate 30 &
    sleep 3.5
    
    echo "📍 4️⃣  向左下 + 俯仰回 🔴 爪子闭"
    timeout 1.5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $POSE4}" --rate 30 &
    sleep 1.5
    timeout 3.5 ros2 topic pub /ee_pose std_msgs/Float64MultiArray "{data: $ZERO}" --rate 30 &
    sleep 3.5
    
    echo "━━━━━━━━━━ 循环 ━━━━━━━━━━"
done


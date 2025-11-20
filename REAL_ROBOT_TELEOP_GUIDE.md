# 真实机器人遥控 Isaac Sim 指南

## 🎯 核心理念

**最自然的数据收集方式**：直接用真实机器人控制仿真机器人！

```
真实机器人 (物理控制)
    ↓ 发布 /joint_states
Isaac Sim (接收 joint positions)
    ↓ 仿真机器人跟随
    ↓ FK 计算 EE pose
记录 EE Absolute Pose
    ↓ 保存到 HDF5
训练 IK Absolute 策略
```

---

## 🔧 不需要改环境！

你问得对：**只需要写个 joint 接收器就行**！

已创建：
- ✅ `SO_100/devices/joint_states_ros2.py` - Joint States ROS2 Device
- ✅ 更新 `record_demos.py` - 支持 `--teleop_device joint_states`
- ✅ 使用现有环境 `SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0`

---

## 🚀 使用方法

### 步骤 1: 启动真实机器人硬件驱动

```bash
# 终端 1: 启动硬件驱动（Python 3.10 系统 ROS2）
cd /mnt/ssd/dual_isaac_so_arm101
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# 运行硬件驱动，发布 /joint_states
python3 so_arm_hardware_driver_ik_abs.py
```

### 步骤 2: 启动 Isaac Sim 收集数据

```bash
# 终端 2: 启动 Isaac Sim（Python 3.11 Isaac Lab 环境）
chmod +x collect_demos_with_real_robot.sh
./collect_demos_with_real_robot.sh
```

### 步骤 3: 操作真实机器人收集数据

1. 用手移动真实机器人（或用机器人自己的控制界面）
2. 仿真机器人会实时跟随
3. 按 `P` 开始记录演示
4. 完成动作后再按 `P` 停止记录
5. 按 `R` 重置场景，开始下一个演示
6. 收集 10-20 个演示后按 `ESC` 退出

---

## 📊 数据流

### 真实机器人端 (系统 ROS2 Python 3.10)

```python
# so_arm_hardware_driver_ik_abs.py

1. 读取伺服电机位置 (STS3215 协议)
   joint_positions = [j1, j2, j3, j4, j5, gripper]

2. 发布到 ROS2
   JointState msg:
     names: ["shoulder_pan_joint", "shoulder_lift_joint", ...]
     positions: [j1, j2, j3, j4, j5, gripper]
   
   发布到: /joint_states
```

### Isaac Sim 端 (Isaac Lab Python 3.11)

```python
# JointStatesROS2 Device

1. 订阅 /joint_states
2. 提取 joint positions
3. 返回为 action: [j1, j2, j3, j4, j5, gripper]

# SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0 Environment

4. 接收 joint actions
5. 控制仿真机器人移动
6. 通过 FrameTransformer FK 计算 EE pose
7. 记录 EE Absolute Pose: [x,y,z,qw,qx,qy,qz,gripper]
```

---

## 💾 记录的数据格式

```python
demo_0.hdf5
├── observations/
│   ├── image_front: [T, H, W, 3]          # 前置相机
│   ├── robot_joint_pos: [T, 6]            # 关节角度
│   ├── plate_pos: [T, 3]                  # 物体位置
│   └── ...
└── actions/
    └── ee_absolute_pose: [T, 8]            # [x,y,z,qw,qx,qy,qz,gripper]
                                            # 相对于 robot base frame
```

**关键点**：
- 遥控输入：Joint Positions (6 DOF)
- 记录输出：EE Absolute Pose (8 DOF)
- 坐标系：Robot Base Frame

---

## 🎮 为什么这样做？

### 对比方案

| 方案 | 遥控方式 | 稳定性 | 自然度 |
|-----|---------|--------|--------|
| ❌ IK Absolute 遥控 | 发布目标 EE pose | 差（求解失败） | 低 |
| ⚠️ 键盘 Joint Control | 键盘按键 | 好 | 中 |
| ✅ **真实机器人 Joint States** | **物理控制真实机器人** | **最好** | **最高** |

### 优势

1. **最自然的遥控方式**
   - 直接移动真实机器人
   - 最符合人类的操作习惯
   - 动作流畅、准确

2. **数据质量最高**
   - 真实机器人的动力学特性
   - 真实的速度和加速度
   - 真实的碰撞反馈

3. **100% 可靠的转换**
   - Joint Positions → FK → EE Pose
   - 没有 IK 求解失败
   - 没有奇异点问题

4. **实时可视化**
   - 仿真机器人跟随真实机器人
   - 可以看到动作效果
   - 便于调试和验证

---

## 🔍 故障排查

### 问题 1: 仿真机器人不动

```bash
# 检查 /joint_states 话题
ros2 topic list | grep joint_states
ros2 topic echo /joint_states

# 如果没有输出，检查硬件驱动是否运行
ps aux | grep "so_arm_hardware_driver"
```

### 问题 2: ROS_DOMAIN_ID 不匹配

```bash
# 两个终端都设置相同的 DOMAIN_ID
export ROS_DOMAIN_ID=0

# 验证
echo $ROS_DOMAIN_ID
```

### 问题 3: Python 版本冲突

```bash
# 终端 1 (硬件驱动): 使用系统 Python 3.10
source /opt/ros/humble/setup.bash
python3 --version  # 应该是 3.10

# 终端 2 (Isaac Sim): 使用 Isaac Lab Python 3.11
conda activate env_isaaclab
python --version  # 应该是 3.11
```

### 问题 4: 仿真机器人延迟

```bash
# 降低硬件驱动的发布频率
# 在 so_arm_hardware_driver_ik_abs.py 中:
self.timer = self.create_timer(0.05, self.read_and_publish)  # 20Hz
# 改为:
self.timer = self.create_timer(0.033, self.read_and_publish)  # 30Hz
```

---

## 📋 完整工作流程

### 1. 准备

```bash
# 连接真实机器人到 /dev/ttyACM0
ls -l /dev/ttyACM0

# 赋予脚本执行权限
chmod +x collect_demos_with_real_robot.sh
```

### 2. 启动系统

```bash
# 终端 1: 硬件驱动
cd /mnt/ssd/dual_isaac_so_arm101
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
python3 so_arm_hardware_driver_ik_abs.py

# 终端 2: Isaac Sim
./collect_demos_with_real_robot.sh
```

### 3. 收集数据

```
1. 移动真实机器人到准备位置
2. 按 'P' 开始记录
3. 执行抓取/放置动作
4. 完成后按 'P' 停止
5. 按 'R' 重置场景
6. 重复步骤 1-5，收集 10-20 个演示
7. 按 ESC 退出
```

### 4. 训练和部署

```bash
# 检查数据
python scripts/inspect_hdf5.py --file ./datasets/pick_place_ik_abs/real_robot_demos_xxx.hdf5

# 训练
python scripts/train_diffusion_policy.py --dataset ./datasets/pick_place_ik_abs/real_robot_demos_xxx.hdf5

# 部署
python scripts/deploy_policy.py --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 --policy ./checkpoints/best.pth
```

---

## 🎉 总结

**不需要改环境，只需要一个 Joint States 接收器！**

核心组件：
- ✅ `JointStatesROS2` device - 接收真实机器人的 joint states
- ✅ `SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0` - 自动记录 EE Absolute Pose
- ✅ `collect_demos_with_real_robot.sh` - 一键启动脚本

工作流程：
```
真实机器人物理控制 → joint_states → Isaac Sim → FK → EE Absolute Pose → HDF5 → 训练
```

**最自然、最可靠的数据收集方式！** 🤖✨



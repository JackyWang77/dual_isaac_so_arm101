# SO-ARM100 真实机器人遥控 Isaac Sim - 最终配置

## 🎯 你的需求

> "我会用真实的机器人来 publish 一个 joint_states，Isaac Sim 里面的机械臂接收这个，记录的输入是 ee，用改变环境么？还是就写个 joint 的接收器就行？"

## ✅ 答案

**只需要写个 joint 接收器就行！不用改环境！**

已完成：
- ✅ `JointStatesROS2` device - 订阅 `/joint_states`
- ✅ 使用现有环境 `SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0`
- ✅ 环境自动记录 EE Absolute Pose（通过 FK）

---

## 📁 新增文件

### 1. 核心文件

```
source/SO_100/SO_100/devices/
├── joint_states_ros2.py          ← Joint States ROS2 Device (新)
├── se3_ros2.py                    (已有)
└── __init__.py                    (已更新)

scripts/
├── record_demos.py                (已更新，支持 joint_states)
└── teleop_se3_agent.py            (已更新，支持 joint_states)
```

### 2. 测试和收集脚本

```
collect_demos_with_real_robot.sh   ← 一键收集数据
test_real_robot_joint_states.sh    ← 快速测试
```

### 3. 文档

```
REAL_ROBOT_TELEOP_GUIDE.md         ← 详细使用指南
FINAL_SETUP.md                     ← 这个文件
```

---

## 🚀 使用方法（两步）

### 步骤 1: 启动真实机器人（终端 1）

```bash
cd /mnt/ssd/dual_isaac_so_arm101
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# 运行硬件驱动，发布 /joint_states
python3 so_arm_hardware_driver_ik_abs.py
```

### 步骤 2: 启动 Isaac Sim 收集数据（终端 2）

```bash
cd /mnt/ssd/dual_isaac_so_arm101
chmod +x collect_demos_with_real_robot.sh
./collect_demos_with_real_robot.sh
```

**就这么简单！**

---

## 🔄 完整数据流

```
┌─────────────────────┐
│  真实机器人           │
│  (物理手动控制)       │
└──────────┬──────────┘
           │ 读取伺服电机位置
           ↓
┌─────────────────────┐
│  硬件驱动             │
│  (Python 3.10)       │
└──────────┬──────────┘
           │ 发布 /joint_states
           │ [j1, j2, j3, j4, j5, gripper]
           ↓
┌─────────────────────┐
│  JointStatesROS2    │
│  Device             │
└──────────┬──────────┘
           │ 返回 joint positions
           ↓
┌─────────────────────┐
│  Isaac Sim 环境      │
│  (Joint-For-IK-Abs) │
└──────────┬──────────┘
           │ 1. 仿真机器人跟随
           │ 2. FK 计算 EE pose
           │ 3. 记录 EE Absolute Pose
           ↓
┌─────────────────────┐
│  HDF5 数据集         │
│  [x,y,z,qw,qx,qy,qz,gripper] │
└──────────┬──────────┘
           │ 训练
           ↓
┌─────────────────────┐
│  IK Absolute 策略   │
│  (Diffusion Policy) │
└─────────────────────┘
```

---

## 📊 记录的数据

```python
demo_0.hdf5
├── observations/
│   ├── image_front: [T, H, W, 3]
│   ├── robot_joint_pos: [T, 6]
│   ├── plate_pos: [T, 3]
│   └── ...
└── actions/
    └── [T, 8]  # [x, y, z, qw, qx, qy, qz, gripper]
                # EE Absolute Pose (相对于 robot base frame)
```

**关键点**：
- 遥控输入：Joint Positions (从真实机器人)
- 记录输出：EE Absolute Pose (自动转换)
- 训练目标：Observation → EE Absolute Pose

---

## 🧪 快速测试

```bash
# 终端 1: 启动硬件驱动
cd /mnt/ssd/dual_isaac_so_arm101
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
python3 so_arm_hardware_driver_ik_abs.py

# 终端 2: 测试连接
chmod +x test_real_robot_joint_states.sh
./test_real_robot_joint_states.sh
```

移动真实机器人，观察仿真机器人是否跟随！

---

## 💡 为什么这样设计？

### 对比方案

| 方案 | 需要改环境 | 复杂度 | 稳定性 |
|-----|----------|--------|--------|
| ❌ 重新设计环境 | 是 | 高 | 未知 |
| ✅ **Joint 接收器** | **否** | **低** | **高** |

### 设计原则

1. **最小侵入**
   - 不改环境，只加 device
   - 使用现有的 FK 功能
   - 复用现有的记录逻辑

2. **职责分离**
   - Device: 接收 ROS2 joint_states
   - Environment: FK + 记录 EE pose
   - 各司其职，清晰明了

3. **可靠性**
   - Joint Positions → FK 100% 可靠
   - 没有 IK 求解失败
   - 没有奇异点问题

---

## 🎮 操作流程

### 收集演示

```
1. 启动硬件驱动（终端 1）
2. 启动 Isaac Sim（终端 2）
3. 移动真实机器人到准备位置
4. 在 Isaac Sim 中按 'P' 开始记录
5. 执行抓取/放置动作（移动真实机器人）
6. 完成后按 'P' 停止记录
7. 按 'R' 重置场景
8. 重复步骤 3-7，收集 10-20 个演示
9. 按 ESC 退出
```

### 查看数据

```bash
python scripts/inspect_hdf5.py \
    --file ./datasets/pick_place_ik_abs/real_robot_demos_xxx.hdf5
```

### 训练模型

```bash
python scripts/train_diffusion_policy.py \
    --dataset ./datasets/pick_place_ik_abs/real_robot_demos_xxx.hdf5 \
    --action_dim 8 \
    --epochs 100
```

### 部署策略

```bash
python scripts/deploy_policy.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --policy_path ./checkpoints/policy_best.pth
```

---

## 🔍 故障排查

### 问题: 仿真机器人不动

```bash
# 1. 检查 /joint_states 话题
ros2 topic list | grep joint_states
ros2 topic echo /joint_states

# 2. 检查 ROS_DOMAIN_ID
echo $ROS_DOMAIN_ID  # 两个终端应该都是 0

# 3. 检查硬件驱动是否运行
ps aux | grep "so_arm_hardware_driver"
```

### 问题: 关节名称不匹配

如果真实机器人的关节名称不同，修改 `JointStatesROS2Cfg`:

```python
joint_names=[
    "your_joint_1",  # 改成真实机器人的关节名
    "your_joint_2",
    "your_joint_3",
    "your_joint_4",
    "your_joint_5",
    "your_gripper"
]
```

---

## 📋 文件清单

### 必需文件（已创建）

- ✅ `source/SO_100/SO_100/devices/joint_states_ros2.py`
- ✅ `source/SO_100/SO_100/devices/__init__.py`
- ✅ `scripts/record_demos.py` (已更新)
- ✅ `scripts/teleop_se3_agent.py` (已更新)
- ✅ `so_arm_hardware_driver_ik_abs.py`
- ✅ `collect_demos_with_real_robot.sh`
- ✅ `test_real_robot_joint_states.sh`

### 使用的环境（已存在）

- ✅ `SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0`
  - 接收 Joint Position actions
  - 记录 EE Absolute Pose
  - 无需修改

---

## 🎉 总结

**你的理解完全正确！**

- ✅ 真实机器人发布 `joint_states`
- ✅ Isaac Sim 接收并跟随
- ✅ 自动记录 EE Absolute Pose
- ✅ **只需要写个 joint 接收器，不用改环境！**

**一切就绪！开始收集数据吧！** 🚀

---

## 📚 相关文档

- `REAL_ROBOT_TELEOP_GUIDE.md` - 详细使用指南
- `JOINT_FOR_IK_ABS_GUIDE.md` - Joint Control 数据收集指南
- `SETUP_SUMMARY.md` - 配置总结



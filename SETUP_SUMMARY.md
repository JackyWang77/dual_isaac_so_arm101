# SO-ARM100 IK Absolute 数据收集 - 完整配置总结

## 🎯 核心理念

**你说得完全对！** Joint Control 遥控更稳定，IK 求解确实是"一坨"。

所以我们采用：
- ✅ **遥控用 Joint Control**（稳定、直观）
- ✅ **记录用 EE Absolute Pose**（适合训练）
- ✅ **训练学习 Observation → EE Pose**
- ✅ **部署用 IK Absolute Controller**（单步求解，比实时遥控稳定）

---

## 📁 新增文件

### 1. 环境定义
- `source/SO_100/SO_100/tasks/pick_place/pick_place_joint_for_ik_abs_env.py`
  - 用 Joint Control 遥控
  - 记录时自动转换为 EE Absolute Pose

### 2. 硬件驱动
- `so_arm_hardware_driver_ik_abs.py`
  - 读取真实机器人关节角度
  - 通过 FK 计算 EE 绝对位置
  - 发布到 `/ee_pose` 话题

### 3. 测试和收集脚本
- `collect_demos_joint_for_ik_abs.sh` - 收集数据（键盘或真实机器人）
- `test_joint_for_ik_abs.sh` - 快速测试
- `test_real_robot_ik_abs.sh` - 测试真实机器人

### 4. 文档
- `JOINT_FOR_IK_ABS_GUIDE.md` - 详细使用指南

---

## 🚀 快速开始

### 测试 1: 键盘遥控测试

```bash
chmod +x test_joint_for_ik_abs.sh
./test_joint_for_ik_abs.sh
```

用键盘控制机器人，观察是否稳定。

### 测试 2: 收集少量数据

```bash
chmod +x collect_demos_joint_for_ik_abs.sh
./collect_demos_joint_for_ik_abs.sh keyboard
```

尝试收集 1-2 个演示，检查数据格式是否正确。

### 测试 3: 真实机器人遥控（如果有）

```bash
# 终端 1
chmod +x test_real_robot_ik_abs.sh
./test_real_robot_ik_abs.sh

# 终端 2
./collect_demos_joint_for_ik_abs.sh ros2
```

---

## 🔄 数据流

```
┌─────────────────┐
│  键盘 / 真实机器人 │
│  (Joint Control)│
└────────┬────────┘
         │ Joint Positions
         ↓
┌─────────────────┐
│  Isaac Sim Env  │
│  + FrameTransformer │
└────────┬────────┘
         │ Forward Kinematics (FK)
         ↓
┌─────────────────┐
│  EE Absolute Pose│
│  [x,y,z,qw,qx,qy,qz,gripper]│
└────────┬────────┘
         │ 记录到 HDF5
         ↓
┌─────────────────┐
│  Training Data  │
│  obs → EE Pose  │
└────────┬────────┘
         │ Diffusion Policy 训练
         ↓
┌─────────────────┐
│  Trained Policy │
│  obs → EE Pose  │
└────────┬────────┘
         │ 部署
         ↓
┌─────────────────┐
│  IK Absolute    │
│  Controller     │
└────────┬────────┘
         │ Inverse Kinematics (IK)
         ↓
┌─────────────────┐
│  Joint Commands │
│  执行到真实机器人 │
└─────────────────┘
```

---

## 🎮 控制说明

### Joint Control 键盘映射

遥控时使用的是 **Joint Position Action**，对应键盘：

- `W/S`: Joint 1 - Base Rotation (shoulder_pan_joint)
- `A/D`: Joint 2 - Shoulder Lift (shoulder_lift_joint)
- `Q/E`: Joint 3 - Elbow (elbow_joint)
- `Z/X`: Joint 4 - Wrist Pitch (wrist_pitch_joint)
- `C/V`: Joint 5 - Wrist Roll (wrist_roll_joint)
- `Space`: Gripper (开/关)
- `P`: 开始/停止记录
- `ESC`: 退出

### 记录的 Action 格式

虽然你遥控时输入的是关节角度，但 **记录的是 EE Absolute Pose**：

```python
action = [
    x,      # 末端执行器 X 位置（相对于 robot base）
    y,      # 末端执行器 Y 位置
    z,      # 末端执行器 Z 位置
    qw,     # 四元数 W
    qx,     # 四元数 X
    qy,     # 四元数 Y
    qz,     # 四元数 Z
    gripper # 夹爪 (-1.0 闭合, 1.0 打开)
]
```

---

## 📊 环境对比

| 环境 ID | 遥控模式 | 记录格式 | 用途 |
|--------|---------|---------|-----|
| `SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0` | ✅ Joint Control | EE Absolute | **数据收集** |
| `SO-ARM100-Pick-Place-DualArm-IK-Abs-v0` | IK Absolute | EE Absolute | 策略部署 |
| `SO-ARM100-Pick-Place-DualArm-IK-Rel-Mimic-v0` | IK Relative | EE Delta | (旧方案) |

---

## 🛠️ 下一步

1. **测试环境**
   ```bash
   ./test_joint_for_ik_abs.sh
   ```

2. **收集演示数据**
   ```bash
   ./collect_demos_joint_for_ik_abs.sh keyboard
   ```

3. **检查数据格式**
   ```bash
   python scripts/inspect_hdf5.py --file ./datasets/pick_place_ik_abs/demos_xxx.hdf5
   ```

4. **训练策略**
   ```bash
   python scripts/train_diffusion_policy.py \
       --dataset ./datasets/pick_place_ik_abs/demos_xxx.hdf5
   ```

5. **部署到 IK Absolute 环境**
   ```bash
   python scripts/deploy_policy.py \
       --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
       --policy_path ./checkpoints/best.pth
   ```

---

## 💡 关键点

1. **遥控和记录是分离的**
   - 你的感受是对的：IK 遥控体验差
   - 用 Joint Control 遥控（稳定）
   - 自动记录为 EE Absolute Pose（训练需要）

2. **训练学习的是 Observation → EE Pose**
   - 不关心遥控时用的是什么
   - 只关心 observation 和对应的 EE 目标位置

3. **部署时才用 IK**
   - IK 单步求解（action → joint command）比实时遥控稳定得多
   - 每一步都是独立的 IK 求解，没有累积误差

---

## 🎉 总结

你的直觉完全正确！

- ❌ **不要**用 IK 遥控（体验差，数据质量差）
- ✅ **用** Joint Control 遥控（稳定，容易控制）
- ✅ **记录** EE Absolute Pose（通过 FK 自动转换）
- ✅ **训练** 学习 obs → EE Pose 的映射
- ✅ **部署** 用 IK Absolute Controller（单步求解，稳定）

这就是为什么我们创建了 `SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0` 环境！



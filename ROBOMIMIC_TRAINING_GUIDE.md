# Isaac Lab Robomimic 模仿学习训练指南

## 📚 概述

Isaac Lab 使用 **Robomimic** 框架进行模仿学习（Imitation Learning）训练。这个指南基于 Isaac Lab 官方实现，展示如何为你的任务设置 BC (Behavioral Cloning) 训练。

---

## 🏗️ 核心架构

### 1. 训练脚本

**位置**: `/mnt/ssd/IsaacLab/scripts/imitation_learning/robomimic/train.py`

**主要功能**:
- 加载 HDF5 数据集
- 使用 Robomimic 算法（如 BC）训练 Policy
- 支持观测归一化和动作归一化
- 自动保存模型检查点

**关键流程**:
```
1. 加载数据集 (HDF5)
2. 创建环境
3. 初始化 Robomimic 模型
4. 训练循环
5. 保存检查点
```

### 2. 配置文件格式 (JSON)

**位置**: `source/SO_100/SO_100/tasks/pick_place/agents/robomimic/bc.json`

**结构**:
```json
{
    "algo_name": "bc",              // 算法名称
    "experiment": {                 // 实验配置
        "name": "bc_experiment",
        "validate": true,
        "save": { ... }
    },
    "train": {                      // 训练配置
        "data": null,               // 数据集路径（可通过命令行覆盖）
        "batch_size": 100,
        "num_epochs": 2000,
        "seq_length": 10
    },
    "algo": {                       // 算法特定配置
        "optim_params": { ... },
        "loss": { ... },
        "actor_layer_dims": [512, 512],  // 网络结构
        "rnn": { ... }              // RNN 配置（可选）
    },
    "observation": {                // 观测配置
        "modalities": {
            "obs": {
                "low_dim": ["joint_pos", "joint_vel", "object", ...]
            }
        }
    }
}
```

---

## 📝 步骤 1: 创建 Robomimic 配置文件

### 创建配置文件目录

```bash
mkdir -p source/SO_100/SO_100/tasks/pick_place/agents/robomimic
```

### 创建 BC 配置文件

创建文件: `source/SO_100/SO_100/tasks/pick_place/agents/robomimic/bc_rnn_low_dim.json`

```json
{
    "algo_name": "bc",
    "experiment": {
        "name": "bc_rnn_pick_place",
        "validate": false,
        "logging": {
            "terminal_output_to_txt": true,
            "log_tb": true
        },
        "save": {
            "enabled": true,
            "every_n_epochs": 100,
            "on_best_rollout_success_rate": true
        },
        "epoch_every_n_steps": 100,
        "env": null,
        "render": false,
        "render_video": false
    },
    "train": {
        "data": null,
        "num_data_workers": 4,
        "hdf5_cache_mode": "all",
        "hdf5_normalize_obs": false,
        "seq_length": 10,
        "dataset_keys": ["actions"],
        "cuda": true,
        "batch_size": 100,
        "num_epochs": 2000,
        "seed": 101
    },
    "algo": {
        "optim_params": {
            "policy": {
                "optimizer_type": "adam",
                "learning_rate": {
                    "initial": 0.001,
                    "decay_factor": 0.1,
                    "epoch_schedule": [],
                    "scheduler_type": "multistep"
                },
                "regularization": {
                    "L2": 0.0
                }
            }
        },
        "loss": {
            "l2_weight": 1.0,
            "l1_weight": 0.0,
            "cos_weight": 0.0
        },
        "actor_layer_dims": [512, 512],
        "rnn": {
            "enabled": true,
            "horizon": 10,
            "hidden_dim": 400,
            "rnn_type": "LSTM",
            "num_layers": 2,
            "open_loop": false
        }
    },
    "observation": {
        "modalities": {
            "obs": {
                "low_dim": [
                    "actions",
                    "joint_pos",
                    "joint_vel",
                    "object",
                    "object_positions",
                    "object_orientations",
                    "eef_pos",
                    "eef_quat",
                    "gripper_pos"
                ],
                "rgb": [],
                "depth": [],
                "scan": []
            }
        }
    }
}
```

---

## 📝 步骤 2: 注册 Robomimic 配置到环境

在环境注册时添加 `robomimic_bc_cfg_entry_point`:

**文件**: `source/SO_100/SO_100/tasks/pick_place/__init__.py`

```python
from . import agents

gym.register(
    id="SO-ARM100-Pick-Place-DualArm-IK-Abs-v0",
    entry_point=_ENTRY_POINT_ABS,
    kwargs={
        "env_cfg_entry_point": _ENV_CFG_ABS,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",  # 添加这行
    },
    disable_env_checker=True,
)
```

**注意**: 
- `robomimic_bc_cfg_entry_point` 格式: `{agents.__name__}:robomimic/{config_file}`
- 路径相对于 `agents` 模块

---

## 📝 步骤 3: 数据集准备

### HDF5 数据集格式

你的数据集应该包含以下结构:

```
/data/
  /demo_0/
    /observations/
      /actions: (N, 8)  # IK Absolute actions [x, y, z, qw, qx, qy, qz, gripper]
      /joint_pos: (N, 5)
      /joint_vel: (N, 5)
      /object: (N, 27)
      ...
    /actions: (N, 8)  # 主要 actions
    /rewards: (N,)
    /dones: (N,)
  /demo_1/
    ...
```

### 观测键名映射

配置文件中的 `observation.modalities.obs.low_dim` 需要与数据集中的键名匹配。

**常用映射**:
- `joint_pos` → 对应数据集的 `joint_pos`
- `joint_vel` → 对应数据集的 `joint_vel`
- `object` → 对应数据集的 `object`
- `eef_pos` → 对应数据集的 `eef_pos`
- `eef_quat` → 对应数据集的 `eef_quat`
- `gripper_pos` → 对应数据集的 `gripper_pos`

---

## 🚀 步骤 4: 开始训练

### 训练命令

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --algo bc \
    --normalize_training_actions \
    --dataset ./datasets/generated_dataset_pick_place.hdf5 \
    --log_dir robomimic \
    --epochs 2000
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--task` | 环境任务名称 | `SO-ARM100-Pick-Place-DualArm-IK-Abs-v0` |
| `--algo` | 算法名称 | `bc` (Behavioral Cloning) |
| `--dataset` | HDF5 数据集路径 | `./datasets/my_dataset.hdf5` |
| `--normalize_training_actions` | 归一化 actions 到 [-1, 1] | 标志位 |
| `--log_dir` | 日志目录 | `robomimic` |
| `--epochs` | 训练轮数 | `2000` |

---

## 📊 配置文件关键参数说明

### 网络结构 (`algo.actor_layer_dims`)

```json
"actor_layer_dims": [512, 512]  // 全连接层维度
```

### RNN 配置 (`algo.rnn`)

```json
"rnn": {
    "enabled": true,           // 启用 RNN
    "horizon": 10,             // 序列长度
    "hidden_dim": 400,         // 隐藏层维度
    "rnn_type": "LSTM",        // 类型: "LSTM" 或 "GRU"
    "num_layers": 2            // RNN 层数
}
```

### 训练参数 (`train`)

```json
"train": {
    "batch_size": 100,         // 批次大小
    "num_epochs": 2000,        // 训练轮数
    "seq_length": 10,          // 序列长度（与 RNN horizon 一致）
    "learning_rate": {         // 学习率
        "initial": 0.001
    }
}
```

---

## 🔍 观测键名配置

### 关键点

配置文件中的观测键名必须与：
1. **HDF5 数据集中的键名**匹配
2. **环境的观测空间**匹配

### 检查观测键名

你的环境观测键名在 `pick_place_env_cfg.py` 中定义:

```python
class PolicyCfg(ObsGroup):
    actions = ObsTerm(func=mdp.last_action)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    object = ObsTerm(func=mdp.object_obs)
    object_positions = ObsTerm(func=mdp.object_positions_in_world_frame)
    object_orientations = ObsTerm(func=mdp.object_orientations_in_world_frame)
    eef_pos = ObsTerm(func=mdp.ee_frame_pos)
    eef_quat = ObsTerm(func=mdp.ee_frame_quat)
    gripper_pos = ObsTerm(func=mdp.gripper_pos)
```

这些键名需要在 JSON 配置文件的 `observation.modalities.obs.low_dim` 中列出。

---

## 📁 文件结构示例

```
source/SO_100/SO_100/tasks/pick_place/
├── __init__.py                    # 注册环境（添加 robomimic_bc_cfg_entry_point）
├── agents/
│   ├── __init__.py
│   ├── rsl_rl_ppo_cfg.py         # RL 配置
│   └── robomimic/                 # Robomimic 配置目录
│       ├── bc_rnn_low_dim.json    # BC with RNN 配置
│       └── bc.json                 # 简单 BC 配置
└── ...
```

---

## 🎯 训练流程总结

```
1. 收集演示数据 → record_demos.py
   ↓
2. 生成 HDF5 数据集 → datasets/pick_place.hdf5
   ↓
3. 创建 Robomimic 配置文件 → agents/robomimic/bc.json
   ↓
4. 注册配置到环境 → __init__.py
   ↓
5. 开始训练 → scripts/imitation_learning/robomimic/train.py
   ↓
6. 模型保存在 → logs/robomimic/SO-ARM100-Pick-Place-DualArm-IK-Abs-v0/.../models/
```

---

## 🔗 参考资源

1. **Isaac Lab Robomimic 文档**:
   - `/mnt/ssd/IsaacLab/scripts/imitation_learning/robomimic/train.py`
   - `/mnt/ssd/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/`

2. **Robomimic 官方文档**:
   - https://robomimic.github.io/

3. **示例配置文件**:
   - `/mnt/ssd/IsaacLab/source/isaaclab_tasks/.../agents/robomimic/bc_rnn_low_dim.json`

---

## ⚠️ 注意事项

1. **观测键名匹配**: 确保配置文件中的观测键名与数据集和环境匹配
2. **Action 归一化**: 使用 `--normalize_training_actions` 时，actions 会被归一化到 [-1, 1]
3. **序列长度**: `seq_length` 和 `rnn.horizon` 应该一致（如果使用 RNN）
4. **数据集路径**: 可以使用相对路径或绝对路径

---

祝你训练顺利！🚀



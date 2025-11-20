# Isaac Lab 自定义 Policy 训练指南

## 📚 概述

在 Isaac Lab 中自定义 Policy 有两种方式：
1. **配置方式**：通过修改 `RslRlPpoActorCriticCfg` 参数（推荐，简单）
2. **实现方式**：完全自定义网络结构（高级，需要深入了解）

本文档主要介绍**配置方式**，这是最常用的方法。

---

## 🏗️ Policy 配置结构

### 1. 基础配置类

```python
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class MyCustomPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 训练参数
    num_steps_per_env = 24        # 每个环境的步数
    max_iterations = 1500         # 最大训练迭代次数
    save_interval = 100           # 保存间隔
    experiment_name = "my_task"   # 实验名称
    
    # Policy 网络结构配置
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                    # 初始探索噪声标准差
        actor_obs_normalization=True,          # Actor 观测归一化
        critic_obs_normalization=True,         # Critic 观测归一化
        actor_hidden_dims=[256, 128, 64],      # Actor 网络隐藏层维度
        critic_hidden_dims=[256, 128, 64],     # Critic 网络隐藏层维度
        activation="elu",                      # 激活函数: "elu", "relu", "tanh"
    )
    
    # PPO 算法参数
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,           # 价值损失系数
        use_clipped_value_loss=True,    # 使用裁剪的价值损失
        clip_param=0.2,                 # PPO 裁剪参数
        entropy_coef=0.006,             # 熵系数（探索奖励）
        num_learning_epochs=5,          # 每次更新的学习轮数
        num_mini_batches=4,             # 小批量数量
        learning_rate=1.0e-4,           # 学习率
        schedule="adaptive",            # 学习率调度: "adaptive", "constant"
        gamma=0.98,                     # 折扣因子
        lam=0.95,                       # GAE lambda
        desired_kl=0.01,                # 期望的 KL 散度
        max_grad_norm=1.0,              # 梯度裁剪
    )
```

---

## 📝 主要配置参数说明

### Policy 网络结构 (`RslRlPpoActorCriticCfg`)

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `actor_hidden_dims` | Actor 网络隐藏层维度列表 | `[256, 128, 64]` (大任务)<br/>`[128, 64]` (小任务) |
| `critic_hidden_dims` | Critic 网络隐藏层维度列表 | 通常与 Actor 相同 |
| `activation` | 激活函数 | `"elu"` (推荐)<br/>`"relu"`, `"tanh"` |
| `init_noise_std` | 初始探索噪声 | `0.5-2.0` (根据任务调整) |
| `actor_obs_normalization` | Actor 观测归一化 | `True` (推荐) |
| `critic_obs_normalization` | Critic 观测归一化 | `True` (推荐) |

### 训练参数 (`RslRlPpoAlgorithmCfg`)

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `learning_rate` | 学习率 | `1e-4` (默认)<br/>`1e-3` (简单任务)<br/>`1e-5` (复杂任务) |
| `num_steps_per_env` | 每个环境的步数 | `16-32` |
| `num_learning_epochs` | 每次更新的学习轮数 | `5-10` |
| `num_mini_batches` | 小批量数量 | `4-8` |
| `gamma` | 折扣因子 | `0.98-0.99` |
| `clip_param` | PPO 裁剪参数 | `0.1-0.3` |
| `entropy_coef` | 熵系数（探索） | `0.001-0.01` |

---

## 🚀 快速开始

### 步骤 1: 创建自定义 Policy 配置文件

在 `source/SO_100/SO_100/tasks/pick_place/agents/` 目录下创建或修改配置文件：

```python
# source/SO_100/SO_100/tasks/pick_place/agents/my_custom_policy_cfg.py

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class MyCustomPickPlacePPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "pick_place_custom"
    empirical_normalization = False
    
    # 自定义 Policy 网络结构
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],      # 更大的网络
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    # 自定义算法参数
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,                      # 更多探索
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-5,                   # 较小的学习率
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

### 步骤 2: 注册 Policy 配置

在环境注册时添加 `rsl_rl_cfg_entry_point`：

```python
# source/SO_100/SO_100/tasks/pick_place/__init__.py

from . import agents

gym.register(
    id="SO-ARM100-Pick-Place-DualArm-IK-Abs-v0",
    entry_point=_ENTRY_POINT_ABS,
    kwargs={
        "env_cfg_entry_point": _ENV_CFG_ABS,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.my_custom_policy_cfg:MyCustomPickPlacePPOCfg",
    },
    disable_env_checker=True,
)
```

### 步骤 3: 开始训练

```bash
python scripts/rsl_rl/train.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --headless
```

---

## 🎨 常见自定义场景

### 场景 1: 更大的网络（复杂任务）

```python
policy = RslRlPpoActorCriticCfg(
    actor_hidden_dims=[512, 256, 128, 64],
    critic_hidden_dims=[512, 256, 128, 64],
    activation="elu",
)
```

### 场景 2: 更小的网络（简单任务，快速训练）

```python
policy = RslRlPpoActorCriticCfg(
    actor_hidden_dims=[128, 64],
    critic_hidden_dims=[128, 64],
    activation="relu",
)
```

### 场景 3: 更多探索

```python
algorithm = RslRlPpoAlgorithmCfg(
    entropy_coef=0.02,        # 更高的熵系数
    init_noise_std=1.5,       # 更大的初始噪声
    # ... 其他参数
)
```

### 场景 4: 更稳定的训练

```python
algorithm = RslRlPpoAlgorithmCfg(
    learning_rate=1.0e-5,     # 较小的学习率
    clip_param=0.15,          # 更小的裁剪
    num_learning_epochs=10,   # 更多的学习轮数
    # ... 其他参数
)
```

---

## 📊 查看现有配置

参考现有配置：
- `source/SO_100/SO_100/tasks/pick_place/agents/rsl_rl_ppo_cfg.py` - Pick-Place 任务
- `source/SO_100/SO_100/tasks/lift/agents/rsl_rl_ppo_cfg.py` - Lift 任务

---

## 🔗 参考资源

1. **Isaac Lab 官方文档**:
   - [训练指南](https://docs.robotsfan.com/isaaclab/source/overview/reinforcement-learning/training_guide.html)
   - [环境设计指南](https://docs.robotsfan.com/isaaclab/source/setup/walkthrough/technical_env_design.html)

2. **RSL-RL 文档**:
   - PPO 算法详解
   - Actor-Critic 网络结构

3. **实际示例**:
   - 查看 `scripts/rsl_rl/train.py` 了解训练流程
   - 查看 `scripts/rsl_rl/play.py` 了解推理流程

---

## ⚠️ 注意事项

1. **网络大小**: 更大的网络需要更多训练时间和内存
2. **学习率**: 根据任务复杂度调整，复杂任务用更小的学习率
3. **探索**: 使用 `entropy_coef` 和 `init_noise_std` 控制探索程度
4. **观测维度**: 确保网络输入维度与观测空间匹配（自动处理）

---

## 🎯 下一步

1. 创建你的自定义 Policy 配置文件
2. 注册到环境 entry point
3. 开始训练并监控性能
4. 根据训练结果调整参数

祝你训练顺利！🚀



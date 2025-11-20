# 自定义策略（Policy）实现完整指南

## 📚 三种方式对比

| 方式 | 难度 | 灵活性 | 适用场景 |
|------|------|--------|----------|
| **1. 配置自定义** | ⭐ 简单 | 中 | 修改网络结构、超参数 |
| **2. Robomimic** | ⭐⭐ 中等 | 中高 | 模仿学习、序列建模 |
| **3. 完全自定义** | ⭐⭐⭐ 复杂 | 最高 | 完全自定义网络和训练流程 |

---

## 🎯 方式 1: 配置自定义（推荐初学者）

### 适用场景
- 修改网络层数和维度
- 调整激活函数
- 修改训练超参数
- 使用 RSL-RL 框架

### 实现步骤

#### 1. 创建配置文件

**文件**: `source/SO_100/SO_100/tasks/pick_place/agents/my_custom_policy_cfg.py`

```python
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
    experiment_name = "my_custom_policy"
    empirical_normalization = False
    
    # 自定义 Policy 网络结构
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],      # 自定义 Actor 网络
        critic_hidden_dims=[512, 256, 128],     # 自定义 Critic 网络
        activation="elu",                       # 激活函数: "elu", "relu", "tanh"
    )
    
    # 自定义算法参数
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

#### 2. 注册配置

**文件**: `source/SO_100/SO_100/tasks/pick_place/__init__.py`

```python
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

#### 3. 开始训练

```bash
python scripts/rsl_rl/train.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --headless
```

---

## 🎯 方式 2: Robomimic（模仿学习）

### 适用场景
- 使用演示数据训练（HDF5 数据集）
- 需要序列建模（RNN/Transformer）
- 模仿学习任务

### 实现步骤

#### 1. 创建 Robomimic 配置文件

**文件**: `source/SO_100/SO_100/tasks/pick_place/agents/robomimic/my_bc_policy.json`

```json
{
    "algo_name": "bc",
    "experiment": {
        "name": "my_bc_policy",
        "validate": false,
        "save": {
            "enabled": true,
            "every_n_epochs": 100,
            "on_best_rollout_success_rate": true
        }
    },
    "train": {
        "data": null,
        "batch_size": 100,
        "num_epochs": 2000,
        "seq_length": 10,
        "cuda": true
    },
    "algo": {
        "optim_params": {
            "policy": {
                "optimizer_type": "adam",
                "learning_rate": {
                    "initial": 0.001
                }
            }
        },
        "loss": {
            "l2_weight": 1.0
        },
        "actor_layer_dims": [512, 512, 256],
        "rnn": {
            "enabled": true,
            "horizon": 10,
            "hidden_dim": 400,
            "rnn_type": "LSTM",
            "num_layers": 2
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
                    "eef_pos",
                    "eef_quat",
                    "gripper_pos"
                ]
            }
        }
    }
}
```

#### 2. 注册配置

**文件**: `source/SO_100/SO_100/tasks/pick_place/__init__.py`

```python
from . import agents

gym.register(
    id="SO-ARM100-Pick-Place-DualArm-IK-Abs-v0",
    entry_point=_ENTRY_POINT_ABS,
    kwargs={
        "env_cfg_entry_point": _ENV_CFG_ABS,
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/my_bc_policy.json",
    },
    disable_env_checker=True,
)
```

#### 3. 训练

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --algo bc \
    --dataset ./datasets/pick_place.hdf5 \
    --normalize_training_actions
```

---

## 🎯 方式 3: 完全自定义 PyTorch 训练脚本（最高灵活性）

### 适用场景
- 完全自定义网络架构（Transformer、Diffusion 等）
- 自定义训练流程
- 不依赖现有框架

### 实现步骤

#### 1. 创建自定义 Policy 网络

**文件**: `source/SO_100/SO_100/policies/my_custom_policy.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomPolicy(nn.Module):
    """完全自定义的 Policy 网络。
    
    输入: 观测 (obs_dim,)
    输出: 动作 (action_dim,)
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list = [256, 256, 128]):
        super().__init__()
        
        # 构建网络层
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # LayerNorm
            layers.append(nn.GELU())  # GELU 激活
            layers.append(nn.Dropout(0.1))  # Dropout
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # 输出归一化到 [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            obs: 观测张量 [batch_size, obs_dim] 或 [obs_dim]
            
        Returns:
            actions: 动作张量 [batch_size, action_dim] 或 [action_dim]
        """
        return self.network(obs)
```

#### 2. 创建训练脚本

**文件**: `scripts/custom_policy/train_my_policy.py`

```python
"""完全自定义的 Policy 训练脚本。

这个脚本展示了如何从零开始训练一个自定义 Policy。
"""

from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm

import gymnasium as gym
import SO_100.tasks  # noqa: F401  # 注册环境
from SO_100.policies.my_custom_policy import MyCustomPolicy


class HDF5Dataset(Dataset):
    """HDF5 数据集加载器"""
    
    def __init__(self, hdf5_path: str, obs_keys: list):
        self.hdf5_path = hdf5_path
        self.obs_keys = obs_keys
        self.episodes = []
        
        # 加载所有 episode
        with h5py.File(hdf5_path, 'r') as f:
            for demo_key in f['data'].keys():
                demo = f[f'data/{demo_key}']
                obs_dict = {key: np.array(demo['observations'][key]) for key in obs_keys}
                actions = np.array(demo['actions'])
                
                # 存储每个 (obs, action) 对
                for i in range(len(actions)):
                    obs = np.concatenate([obs_dict[key][i] for key in obs_keys])
                    self.episodes.append({
                        'obs': obs.astype(np.float32),
                        'action': actions[i].astype(np.float32)
                    })
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        return torch.from_numpy(episode['obs']), torch.from_numpy(episode['action'])


def train_behavioral_cloning(
    task_name: str,
    dataset_path: str,
    obs_keys: list,
    obs_dim: int,
    action_dim: int,
    hidden_dims: list = [512, 256, 128],
    batch_size: int = 256,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    save_dir: str = "./logs/custom_policy",
):
    """训练 Behavioral Cloning Policy
    
    Args:
        task_name: 环境任务名称
        dataset_path: HDF5 数据集路径
        obs_keys: 观测键名列表
        obs_dim: 观测维度（所有观测拼接后的总维度）
        action_dim: 动作维度
        hidden_dims: 隐藏层维度列表
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备 ("cuda" 或 "cpu")
        save_dir: 模型保存目录
    """
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建环境（用于获取观测和动作维度）
    env_cfg = None  # 从任务中自动获取
    env = gym.make(task_name, cfg=env_cfg)
    
    # 获取实际的观测和动作维度
    obs_space = env.observation_space
    action_space = env.action_space
    
    if hasattr(obs_space, 'shape'):
        actual_obs_dim = sum(obs_space.shape) if isinstance(obs_space.shape, tuple) else obs_space.shape[0]
    else:
        # 字典空间：需要手动计算
        actual_obs_dim = obs_dim  # 使用提供的维度
    
    actual_action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else action_dim
    
    print(f"[INFO] 观测维度: {actual_obs_dim}")
    print(f"[INFO] 动作维度: {actual_action_dim}")
    
    # 创建 Policy 网络
    policy = MyCustomPolicy(
        obs_dim=actual_obs_dim,
        action_dim=actual_action_dim,
        hidden_dims=hidden_dims
    ).to(device)
    
    print(f"[INFO] Policy 网络结构:")
    print(policy)
    
    # 加载数据集
    print(f"[INFO] 加载数据集: {dataset_path}")
    dataset = HDF5Dataset(dataset_path, obs_keys)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    print(f"[INFO] 数据集大小: {len(dataset)}")
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 训练循环
    policy.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for obs, actions in pbar:
            obs = obs.to(device)
            actions = actions.to(device)
            
            # 前向传播
            pred_actions = policy(obs)
            
            # 计算损失
            loss = criterion(pred_actions, actions)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        print(f"[Epoch {epoch+1}/{num_epochs}] 平均损失: {avg_loss:.6f}")
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"policy_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"[INFO] 保存检查点: {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join(save_dir, "policy_final.pt")
    torch.save(policy.state_dict(), final_path)
    print(f"[INFO] 保存最终模型: {final_path}")
    
    env.close()
    return policy


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练自定义 Policy")
    parser.add_argument("--task", type=str, required=True, help="任务名称")
    parser.add_argument("--dataset", type=str, required=True, help="HDF5 数据集路径")
    parser.add_argument("--obs_dim", type=int, default=72, help="观测维度")
    parser.add_argument("--action_dim", type=int, default=8, help="动作维度")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--save_dir", type=str, default="./logs/custom_policy", help="保存目录")
    
    args = parser.parse_args()
    
    # 观测键名（需要与数据集匹配）
    obs_keys = [
        "actions",
        "joint_pos",
        "joint_vel",
        "object",
        "object_positions",
        "object_orientations",
        "eef_pos",
        "eef_quat",
        "gripper_pos"
    ]
    
    # 开始训练
    train_behavioral_cloning(
        task_name=args.task,
        dataset_path=args.dataset,
        obs_keys=obs_keys,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dims=[512, 256, 128],
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
```

#### 3. 创建推理脚本

**文件**: `scripts/custom_policy/play_my_policy.py`

```python
"""使用训练好的自定义 Policy 进行推理"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym
import SO_100.tasks  # noqa: F401
from SO_100.policies.my_custom_policy import MyCustomPolicy


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="使用自定义 Policy 推理")
    parser.add_argument("--task", type=str, required=True, help="任务名称")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--obs_dim", type=int, default=72, help="观测维度")
    parser.add_argument("--action_dim", type=int, default=8, help="动作维度")
    
    args = parser.parse_args()
    
    # 创建环境
    env = gym.make(args.task)
    obs, _ = env.reset()
    
    # 加载 Policy
    policy = MyCustomPolicy(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dims=[512, 256, 128]
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(checkpoint, dict):
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    policy = policy.to(env.unwrapped.device)
    
    print(f"[INFO] 加载模型: {args.checkpoint}")
    
    # 推理循环
    with torch.inference_mode():
        while simulation_app.is_running():
            # 处理观测（如果是字典，需要拼接）
            if isinstance(obs, dict):
                obs_tensor = torch.cat([torch.from_numpy(obs[key]).flatten() for key in obs.keys()], dim=0)
            else:
                obs_tensor = torch.from_numpy(obs).flatten()
            
            obs_tensor = obs_tensor.unsqueeze(0).to(env.unwrapped.device)
            
            # 获取动作
            action = policy(obs_tensor)
            action = action.cpu().numpy()[0]
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

#### 4. 训练和使用

```bash
# 训练
python scripts/custom_policy/train_my_policy.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --dataset ./datasets/pick_place.hdf5 \
    --obs_dim 72 \
    --action_dim 8 \
    --epochs 200 \
    --batch_size 256

# 推理
python scripts/custom_policy/play_my_policy.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --checkpoint ./logs/custom_policy/policy_final.pt \
    --obs_dim 72 \
    --action_dim 8
```

---

## 📊 三种方式对比总结

### 方式 1: 配置自定义
- ✅ 最简单
- ✅ 使用现有框架（RSL-RL）
- ✅ 快速开始
- ❌ 灵活性有限

### 方式 2: Robomimic
- ✅ 适合模仿学习
- ✅ 支持序列建模
- ✅ 配置灵活
- ❌ 需要 HDF5 数据集

### 方式 3: 完全自定义
- ✅ 完全控制
- ✅ 可以实现任何网络结构
- ✅ 自定义训练流程
- ❌ 需要自己实现训练逻辑
- ❌ 代码量较大

---

## 🎯 推荐选择

- **初学者/快速原型**: 使用**方式 1**（配置自定义）
- **模仿学习任务**: 使用**方式 2**（Robomimic）
- **高级自定义需求**: 使用**方式 3**（完全自定义）

---

## 📝 下一步

1. 选择适合你的方式
2. 创建相应的配置文件/脚本
3. 准备数据集（如果需要）
4. 开始训练

祝你训练顺利！🚀



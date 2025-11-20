# Isaac Lab 添加自定义学习库完整指南

基于 [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/main/source/how-to/add_own_library.html)

---

## 📚 概述

Isaac Lab 支持集成你自己的学习库（如自定义的 RL 框架、模仿学习库等）。有两种方式：

1. **使用不同版本的现有库**（如修改后的 rsl-rl）
2. **集成全新的库**（需要创建 wrapper）

---

## 🎯 方式 1: 使用不同版本的现有库

### 场景
- 使用自己修改过的 rsl-rl 版本
- 使用不同版本的 SKRL、RL-Games 等
- 测试新版本的库

### 步骤

#### 1. 克隆或获取你的库

```bash
# 例如：克隆修改过的 rsl-rl
git clone git@github.com:yourusername/rsl_rl.git
cd rsl_rl
```

#### 2. 安装到 Isaac Lab 环境

```bash
# 在 Isaac Lab 根目录下
cd /mnt/ssd/IsaacLab

# 安装你的库（使用 -e 表示可编辑模式）
./isaaclab.sh -p -m pip install -e /path/to/your/rsl_rl

# 或者直接安装到当前环境
./isaaclab.sh -p -m pip install -e ~/git/rsl_rl
```

#### 3. 验证安装

```bash
# 检查库的位置和版本
./isaaclab.sh -p -m pip show rsl-rl-lib

# 输出应该显示你的库位置
# Location: /path/to/your/rsl_rl
```

#### 4. 使用你的库

```bash
# 正常使用，Isaac Lab 会自动使用你安装的版本
python scripts/rsl_rl/train.py --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0
```

---

## 🎯 方式 2: 集成全新的学习库

### 场景
- 集成全新的 RL 框架（如 Diffusion Policy、π0 等）
- 创建自定义的训练流程
- 添加新的算法库

### 完整步骤

根据官方文档，需要完成以下步骤：

### 步骤 1: 在 setup.py 中添加依赖

**文件**: `source/SO_100/setup.py` 或 `source/isaaclab_rl/setup.py`

```python
# 在 EXTRAS_REQUIRE 中添加你的库
EXTRAS_REQUIRE = {
    "sb3": ["stable-baselines3>=2.6"],
    "skrl": ["skrl>=1.4.3"],
    "rsl-rl": ["rsl-rl-lib==3.1.2"],
    "your_library": ["your-library>=1.0.0"],  # 添加你的库
}

# 或者添加依赖链接
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
```

### 步骤 2: 安装你的库

```bash
# 安装到 Isaac Lab 环境
./isaaclab.sh -p -m pip install your-library

# 或从源码安装
./isaaclab.sh -p -m pip install -e /path/to/your/library
```

### 步骤 3: 创建环境 Wrapper

**文件**: `source/SO_100/SO_100/wrappers/your_library_wrapper.py`

参考 `RslRlVecEnvWrapper` 的实现：

```python
"""Wrapper to configure an environment instance for your custom library."""

import torch
from typing import Any
from gymnasium import Env

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class YourLibraryVecEnvWrapper:
    """Wraps around Isaac Lab environment for your custom library.
    
    This wrapper adapts Isaac Lab's environment interface to your library's interface.
    
    Reference:
        See RslRlVecEnvWrapper for example implementation.
    """
    
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, **kwargs):
        """Initialize the wrapper.
        
        Args:
            env: The Isaac Lab environment to wrap.
            **kwargs: Additional arguments for your library.
        """
        # 验证环境类型
        if not isinstance(env.unwrapped, (ManagerBasedRLEnv, DirectRLEnv)):
            raise ValueError(
                f"The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. "
                f"Got: {type(env)}"
            )
        
        self.env = env
        self.unwrapped = env.unwrapped
        
        # 存储环境信息
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        
        # 获取观测和动作维度
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        # 初始化环境
        self.env.reset()
    
    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset the environment.
        
        Returns:
            tuple: (observations, info_dict)
        """
        obs_dict, info = self.env.reset()
        # 转换为你的库需要的格式
        obs_tensor = self._process_observations(obs_dict)
        return obs_tensor, info
    
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment.
        
        Args:
            actions: Actions from your policy [num_envs, action_dim]
            
        Returns:
            tuple: (observations, rewards, dones, info_dict)
        """
        # 执行动作
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)
        
        # 转换为你的库需要的格式
        obs_tensor = self._process_observations(obs_dict)
        dones = (terminated | truncated).to(dtype=torch.long)
        
        return obs_tensor, rewards, dones, info
    
    def _process_observations(self, obs_dict: dict) -> torch.Tensor:
        """Process observations from dict to tensor.
        
        Args:
            obs_dict: Dictionary of observations
            
        Returns:
            torch.Tensor: Flattened observations [num_envs, obs_dim]
        """
        # 如果是字典观测，需要拼接
        if isinstance(obs_dict, dict):
            obs_list = []
            for key in sorted(obs_dict.keys()):
                obs = obs_dict[key]
                if isinstance(obs, torch.Tensor):
                    obs_list.append(obs.flatten(start_dim=1))
                else:
                    obs_list.append(torch.from_numpy(obs).flatten(start_dim=1))
            return torch.cat(obs_list, dim=1)
        else:
            # 已经是 tensor
            if isinstance(obs_dict, torch.Tensor):
                return obs_dict
            else:
                return torch.from_numpy(obs_dict)
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    def seed(self, seed: int = -1) -> int:
        """Set the random seed."""
        return self.unwrapped.seed(seed)
    
    @property
    def cfg(self):
        """Returns the environment configuration."""
        return self.unwrapped.cfg
```

### 步骤 4: 创建训练脚本

**文件**: `scripts/your_library/train.py`

```python
"""Training script using your custom library."""

from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse
import gymnasium as gym
import torch

import SO_100.tasks  # noqa: F401  # Register environments
from SO_100.wrappers.your_library_wrapper import YourLibraryVecEnvWrapper
from your_library import YourTrainer  # 你的库的 Trainer


def main():
    parser = argparse.ArgumentParser(description="Train with your custom library")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
    # ... 其他参数
    
    args = parser.parse_args()
    
    # 创建环境
    env_cfg = None  # 从任务自动获取
    env = gym.make(args.task, cfg=env_cfg, num_envs=args.num_envs)
    
    # 包装环境
    env = YourLibraryVecEnvWrapper(env)
    
    # 创建 Trainer（你的库的接口）
    trainer = YourTrainer(
        env=env,
        # ... 你的库的参数
    )
    
    # 开始训练
    trainer.train(num_iterations=1000)
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

### 步骤 5: 创建配置文件（可选）

如果需要配置系统，创建配置文件：

**文件**: `source/SO_100/SO_100/agents/your_library_cfg.py`

```python
from isaaclab.utils import configclass


@configclass
class YourLibraryTrainerCfg:
    """Configuration for your custom library trainer."""
    
    # 训练参数
    num_iterations: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 256
    
    # 网络结构
    hidden_dims: list[int] = [256, 128, 64]
    activation: str = "elu"
```

### 步骤 6: 注册配置到环境（可选）

如果需要通过环境注册使用配置：

**文件**: `source/SO_100/SO_100/tasks/pick_place/__init__.py`

```python
from . import agents

gym.register(
    id="SO-ARM100-Pick-Place-DualArm-IK-Abs-v0",
    entry_point=_ENTRY_POINT_ABS,
    kwargs={
        "env_cfg_entry_point": _ENV_CFG_ABS,
        "your_library_cfg_entry_point": f"{agents.__name__}.your_library_cfg:YourLibraryTrainerCfg",
    },
    disable_env_checker=True,
)
```

---

## 📊 参考实现

### RSL-RL Wrapper 示例

参考 `/mnt/ssd/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`:

关键特性：
- 继承自 `VecEnv`（RSL-RL 的接口）
- 实现 `reset()`, `step()` 方法
- 转换观测格式（dict → TensorDict）
- 处理动作裁剪
- 管理 episode 长度缓冲区

### SKRL Wrapper 示例

参考 `/mnt/ssd/IsaacLab/source/isaaclab_rl/isaaclab_rl/skrl.py`:

关键特性：
- 调用 SKRL 库的 `wrap_env` 函数
- 支持多框架（torch/jax）
- 自动检测单/多智能体环境

---

## 🔧 关键要点

### 1. Wrapper 需要实现的接口

你的 Wrapper 需要实现以下方法（根据你的库要求）：

```python
class YourLibraryWrapper:
    def reset(self) -> tuple:      # 重置环境
        pass
    
    def step(self, actions) -> tuple:  # 执行动作
        pass
    
    def close(self):               # 关闭环境
        pass
    
    @property
    def observation_space(self):   # 观测空间
        pass
    
    @property
    def action_space(self):        # 动作空间
        pass
```

### 2. 观测格式转换

Isaac Lab 环境返回字典格式的观测，你的库可能需要：
- **Tensor 格式**: 需要拼接字典中的值
- **字典格式**: 直接使用
- **其他格式**: 需要转换

### 3. 批处理处理

Isaac Lab 是向量化环境（多个并行环境），你的库需要：
- 支持批处理观测 `[num_envs, obs_dim]`
- 支持批处理动作 `[num_envs, action_dim]`
- 支持批处理奖励和 done 信号

---

## 🎯 实际示例：为你的项目添加自定义库

假设你想添加一个简单的自定义训练库，可以这样做：

### 1. 创建 Wrapper

**文件**: `source/SO_100/SO_100/wrappers/__init__.py`

```python
from .your_library_wrapper import YourLibraryVecEnvWrapper

__all__ = ["YourLibraryVecEnvWrapper"]
```

### 2. 创建训练脚本

**文件**: `scripts/my_library/train.py`

（参考上面的示例）

### 3. 使用

```bash
python scripts/my_library/train.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --num_envs 4096
```

---

## 📝 测试（可选但推荐）

根据官方文档建议，添加测试：

**文件**: `source/SO_100/test/test_your_library_wrapper.py`

```python
"""Tests for your library wrapper."""

import torch
import gymnasium as gym
import SO_100.tasks  # noqa: F401
from SO_100.wrappers.your_library_wrapper import YourLibraryVecEnvWrapper


def test_wrapper_basic():
    """Test basic wrapper functionality."""
    # 创建环境
    env = gym.make("SO-ARM100-Pick-Place-DualArm-IK-Abs-v0")
    
    # 包装环境
    wrapped_env = YourLibraryVecEnvWrapper(env)
    
    # 测试 reset
    obs, info = wrapped_env.reset()
    assert obs.shape[0] == wrapped_env.num_envs
    
    # 测试 step
    actions = torch.zeros((wrapped_env.num_envs, wrapped_env.action_space.shape[0]))
    obs, rewards, dones, info = wrapped_env.step(actions)
    
    assert obs.shape[0] == wrapped_env.num_envs
    assert rewards.shape[0] == wrapped_env.num_envs
    
    wrapped_env.close()
```

---

## 🔗 参考资源

1. **官方文档**: [Adding your own learning library](https://isaac-sim.github.io/IsaacLab/main/source/how-to/add_own_library.html)
2. **RSL-RL Wrapper**: `/mnt/ssd/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`
3. **SKRL Wrapper**: `/mnt/ssd/IsaacLab/source/isaaclab_rl/isaaclab_rl/skrl.py`
4. **Setup.py 示例**: `/mnt/ssd/IsaacLab/source/isaaclab_rl/setup.py`

---

## ⚠️ 注意事项

1. **Python 版本兼容性**: 确保你的库支持 Isaac Sim 使用的 Python 版本（3.10 或 3.11）
2. **GPU 支持**: 确保你的库支持 CUDA（Isaac Lab 主要在 GPU 上运行）
3. **批处理**: 你的库必须支持批处理（向量化环境）
4. **接口兼容性**: Wrapper 需要正确转换 Isaac Lab 的接口到你的库的接口

---

## 🎯 下一步

1. 决定使用哪种方式（方式 1 或方式 2）
2. 创建 Wrapper（如果需要）
3. 创建训练脚本
4. 测试集成
5. 开始训练

祝你集成顺利！🚀



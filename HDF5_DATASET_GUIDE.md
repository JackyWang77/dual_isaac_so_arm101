# HDF5 数据集读取指南

## 📚 概述

Isaac Lab 使用 HDF5 格式存储演示数据。本指南展示如何读取和分析这些数据集。

---

## 📁 HDF5 数据集结构

Isaac Lab 生成的 HDF5 数据集结构如下：

```
dataset.hdf5
└── data/
    ├── demo_0/
    │   ├── observations/
    │   │   ├── actions: (N, action_dim)
    │   │   ├── joint_pos: (N, num_joints)
    │   │   ├── joint_vel: (N, num_joints)
    │   │   ├── object: (N, object_dim)
    │   │   ├── eef_pos: (N, 3)
    │   │   ├── eef_quat: (N, 4)
    │   │   └── ...
    │   ├── actions: (N, action_dim)
    │   ├── rewards: (N,)
    │   └── dones: (N,)
    ├── demo_1/
    │   └── ...
    └── ...
```

### 关键结构

- **`data/`**: 顶层组，包含所有演示
- **`demo_{i}/`**: 第 i 个演示的数据
- **`observations/`**: 观测数据字典（每个键对应一个观测类型）
- **`actions`**: 动作数组 `[num_steps, action_dim]`
- **`rewards`**: 奖励数组 `[num_steps,]`
- **`dones`**: 结束标志数组 `[num_steps,]`

---

## 🔍 方法 1: 使用 inspect_hdf5_dataset.py（推荐）

### 检查数据集结构

```bash
python scripts/inspect_hdf5_dataset.py --dataset ./datasets/pick_place.hdf5
```

### 查看特定演示

```bash
python scripts/inspect_hdf5_dataset.py \
    --dataset ./datasets/pick_place.hdf5 \
    --demo_idx 0 \
    --show_samples 10
```

### 加载特定样本

```bash
python scripts/inspect_hdf5_dataset.py \
    --dataset ./datasets/pick_place.hdf5 \
    --demo_idx 0 \
    --step_idx 5
```

---

## 📖 方法 2: 直接使用 h5py 读取

### 基本读取

```python
import h5py
import numpy as np

# 打开 HDF5 文件
with h5py.File('dataset.hdf5', 'r') as f:
    data_group = f['data']
    
    # 获取所有演示的键
    demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
    print(f"Found {len(demo_keys)} demonstrations")
    
    # 读取第一个演示
    demo_key = demo_keys[0]
    demo = data_group[demo_key]
    
    # 读取观测
    obs_dict = {}
    for key in demo['observations'].keys():
        obs_dict[key] = np.array(demo['observations'][key])
    
    # 读取动作
    actions = np.array(demo['actions'])
    
    # 读取奖励和结束标志
    rewards = np.array(demo['rewards']) if 'rewards' in demo else None
    dones = np.array(demo['dones']) if 'dones' in demo else None
    
    print(f"Actions shape: {actions.shape}")
    print(f"Observations keys: {list(obs_dict.keys())}")
```

---

## 📖 方法 3: 迭代读取（内存高效）

```python
import h5py
import numpy as np

def read_hdf5_iterative(dataset_path, obs_keys):
    """逐演示读取数据集（内存高效）"""
    
    all_obs = []
    all_actions = []
    
    with h5py.File(dataset_path, 'r') as f:
        data_group = f['data']
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
        
        for demo_key in demo_keys:
            demo = data_group[demo_key]
            
            # 提取观测
            obs_list = []
            for key in obs_keys:
                if key in demo['observations']:
                    obs_val = np.array(demo['observations'][key])
                    # 展平（如果需要）
                    if len(obs_val.shape) > 2:
                        obs_val = obs_val.reshape(obs_val.shape[0], -1)
                    obs_list.append(obs_val)
            
            # 拼接观测
            obs_concat = np.concatenate(obs_list, axis=1)
            actions = np.array(demo['actions'])
            
            all_obs.append(obs_concat)
            all_actions.append(actions)
    
    # 拼接所有演示
    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    return observations, actions

# 使用
obs_keys = ['joint_pos', 'joint_vel', 'eef_pos', 'eef_quat', ...]
observations, actions = read_hdf5_iterative('dataset.hdf5', obs_keys)
print(f"Observations: {observations.shape}, Actions: {actions.shape}")
```

---

## 📖 方法 4: PyTorch DataLoader（用于训练）

```python
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class HDF5Dataset(Dataset):
    """PyTorch Dataset for HDF5 files."""
    
    def __init__(self, dataset_path, obs_keys, normalize=True):
        self.dataset_path = dataset_path
        self.obs_keys = obs_keys
        self.normalize = normalize
        
        # 预加载数据（小数据集）或实现延迟加载（大数据集）
        self._load_data()
    
    def _load_data(self):
        """加载所有数据"""
        all_obs = []
        all_actions = []
        
        with h5py.File(self.dataset_path, 'r') as f:
            data_group = f['data']
            demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
            
            for demo_key in demo_keys:
                demo = data_group[demo_key]
                
                # 提取观测
                obs_list = []
                for key in self.obs_keys:
                    if key in demo['observations']:
                        obs_val = np.array(demo['observations'][key])
                        if len(obs_val.shape) > 2:
                            obs_val = obs_val.reshape(obs_val.shape[0], -1)
                        obs_list.append(obs_val)
                
                obs_concat = np.concatenate(obs_list, axis=1)
                actions = np.array(demo['actions'])
                
                all_obs.append(obs_concat)
                all_actions.append(actions)
        
        self.observations = np.concatenate(all_obs, axis=0).astype(np.float32)
        self.actions = np.concatenate(all_actions, axis=0).astype(np.float32)
        
        # 归一化
        if self.normalize:
            self.obs_mean = np.mean(self.observations, axis=0, keepdims=True)
            self.obs_std = np.std(self.observations, axis=0, keepdims=True) + 1e-8
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        obs = self.observations[idx]
        action = self.actions[idx]
        
        # 归一化
        if self.normalize:
            obs = (obs - self.obs_mean.squeeze()) / self.obs_std.squeeze()
        
        return torch.from_numpy(obs), torch.from_numpy(action)

# 创建 DataLoader
obs_keys = ['joint_pos', 'joint_vel', 'eef_pos', ...]
dataset = HDF5Dataset('dataset.hdf5', obs_keys)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 使用
for obs_batch, action_batch in dataloader:
    # obs_batch: [batch_size, obs_dim]
    # action_batch: [batch_size, action_dim]
    pass
```

---

## 📖 方法 5: 使用 Isaac Lab 的 robomimic 工具（推荐用于训练）

Isaac Lab 的 `train.py` 脚本使用 robomimic 的工具函数：

```python
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.train_utils as TrainUtils

# 1. 获取数据集元数据
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path='dataset.hdf5')
shape_meta = FileUtils.get_shape_metadata_from_dataset(
    dataset_path='dataset.hdf5',
    all_obs_keys=['joint_pos', 'joint_vel', ...],
    verbose=True
)

# 2. 创建配置（简化示例）
from robomimic.config import Config
config = Config(...)
config.train.data = 'dataset.hdf5'
config.all_obs_keys = ['joint_pos', 'joint_vel', ...]

# 3. 加载训练数据
trainset, validset = TrainUtils.load_data_for_training(
    config,
    obs_keys=shape_meta["all_obs_keys"]
)

# 4. 创建 DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(
    dataset=trainset,
    batch_size=256,
    shuffle=True,
    num_workers=4,
)
```

---

## 🔧 完整示例脚本

### 脚本 1: inspect_hdf5_dataset.py

检查数据集结构和内容：

```bash
python scripts/inspect_hdf5_dataset.py --dataset ./datasets/pick_place.hdf5
```

### 脚本 2: read_hdf5_example.py

展示不同的读取方法：

```bash
python scripts/read_hdf5_example.py --dataset ./datasets/pick_place.hdf5 --method 4
```

---

## 🎯 常见观测键名

根据你的环境配置，常见的观测键名包括：

- **`actions`**: 上一步动作（用于历史）
- **`joint_pos`**: 关节位置
- **`joint_vel`**: 关节速度
- **`object`**: 物体状态（位置、方向等）
- **`object_positions`**: 物体位置（世界坐标系）
- **`object_orientations`**: 物体方向（世界坐标系）
- **`eef_pos`**: 末端执行器位置
- **`eef_quat`**: 末端执行器四元数
- **`gripper_pos`**: 夹爪位置

---

## 💡 提示

### 1. 内存管理

- **小数据集**: 直接加载到内存
- **大数据集**: 使用迭代读取或延迟加载
- **训练**: 使用 PyTorch DataLoader 的多进程加载

### 2. 数据归一化

在训练前归一化观测和动作：

```python
# 观测归一化
obs_mean = np.mean(observations, axis=0)
obs_std = np.std(observations, axis=0) + 1e-8
normalized_obs = (observations - obs_mean) / obs_std

# 动作归一化（如果需要）
action_mean = np.mean(actions, axis=0)
action_std = np.std(actions, axis=0) + 1e-8
normalized_actions = (actions - action_mean) / action_std
```

### 3. 数据验证

读取数据后验证形状和范围：

```python
print(f"Observations shape: {observations.shape}")
print(f"Actions shape: {actions.shape}")
print(f"Action range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
print(f"Observations range: [{np.min(observations):.3f}, {np.max(observations):.3f}]")
```

---

## 🔗 参考

1. **Isaac Lab 训练脚本**: `/mnt/ssd/IsaacLab/scripts/imitation_learning/robomimic/train.py`
2. **HDF5 官方文档**: https://www.h5py.org/
3. **robomimic 文档**: https://robomimic.github.io/

---

祝你使用顺利！🚀



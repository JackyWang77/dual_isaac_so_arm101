# Isaac Lab 图像录制和存储指南

## 📚 概述

Isaac Lab 支持录制相机图像观测并存储到 HDF5 数据集中。本指南展示如何：
1. 配置相机传感器
2. 添加图像观测到环境
3. 录制和存储图像数据

---

## 🎯 步骤 1: 在环境中添加相机传感器

### 1.1 在环境配置中添加相机

**文件**: `source/SO_100/SO_100/tasks/pick_place/pick_place_env_cfg.py`

```python
from isaaclab.sensors import CameraCfg
from omni.isaac.lab.sim import PinholeCameraCfg
from omni.isaac.lab.utils import configclass

@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    # ... 其他配置 ...
    
    # 添加相机传感器
    camera_front = CameraCfg(
        data_types=["rgb"],  # 或 ["rgb", "distance_to_image_plane"]
        spawn=PinholeCameraCfg(
            focal_length=24.0,  # mm
            focus_distance=400.0,  # mm
            horizontal_aperture=20.955,  # mm
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 0.5),  # 相机位置 (x, y, z)
            rot=(0.5, -0.5, 0.5, -0.5),  # 四元数 (w, x, y, z)
            convention="ros",
        ),
        prim_path="{ENV_REGEX_NS}/World/origin/front_camera",
        debug_vis=True,
    )
    
    # 可选：添加腕部相机
    camera_wrist = CameraCfg(
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
        prim_path="{ENV_REGEX_NS}/Robot/wrist_2_link/camera_wrist",
        debug_vis=False,
    )
```

---

## 🎯 步骤 2: 添加图像观测到观测配置

### 2.1 创建图像观测组

**文件**: `source/SO_100/SO_100/tasks/pick_place/pick_place_env_cfg.py`

```python
from isaaclab.envs.mdp import ObsTerm

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # ... 低维观测 ...
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # ...
    
    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """包含 RGB 图像的观测组"""
        
        # 前置相机图像
        image_front = ObsTerm(
            func=mdp.generated_commands,  # 占位符，实际使用相机数据
            params={
                "asset_cfg": SceneEntityCfg("camera_front"),
                "command_name": "rgb",
            },
        )
        
        # 腕部相机图像（可选）
        image_wrist = ObsTerm(
            func=mdp.generated_commands,
            params={
                "asset_cfg": SceneEntityCfg("camera_wrist"),
                "command_name": "rgb",
            },
        )
        
        # 仍然包含低维观测
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)

@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    # ... 其他配置 ...
    
    # 使用包含图像的观测配置
    observations = ObservationsCfg()
    observations.policy = ObservationsCfg.RGBCameraPolicyCfg()
```

### 2.2 实际获取相机数据（推荐方法）

更好的方法是直接从相机传感器读取：

```python
@configclass
class ObservationsCfg:
    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """包含 RGB 图像的观测组"""
        
        # 直接从相机传感器读取 RGB 图像
        image_front = ObsTerm(
            func=lambda env: env.scene["camera_front"].data.rgb,  # 直接访问相机数据
        )
        
        # 或使用环境辅助函数（如果可用）
        # image_front = ObsTerm(
        #     func=mdp.image_from_camera,
        #     params={"camera_name": "camera_front"},
        # )
        
        # 低维观测
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # ...
```

---

## 🎯 步骤 3: HDF5 数据集中的图像存储格式

### 3.1 HDF5 数据集结构（包含图像）

```
dataset.hdf5
└── data/
    ├── demo_0/
    │   ├── observations/
    │   │   ├── image_front: (N, H, W, 3)      # RGB 图像 [0-255]
    │   │   ├── image_wrist: (N, H, W, 3)      # RGB 图像 [0-255]
    │   │   ├── joint_pos: (N, num_joints)
    │   │   ├── joint_vel: (N, num_joints)
    │   │   └── ...
    │   ├── actions: (N, action_dim)
    │   ├── rewards: (N,)
    │   └── dones: (N,)
    └── ...
```

### 3.2 图像数据格式

- **数据类型**: `uint8` (0-255)
- **形状**: `[num_steps, height, width, channels]`
- **通道顺序**: RGB (最后一个维度)
- **存储格式**: HDF5 数组

---

## 📝 完整示例：添加相机到 Pick-Place 环境

### 示例 1: 添加前置相机

**文件**: `source/SO_100/SO_100/tasks/pick_place/pick_place_ik_abs_env_cfg.py`

```python
from isaaclab.sensors import CameraCfg
from omni.isaac.lab.sim import PinholeCameraCfg
from omni.isaac.lab.utils import configclass

@configclass
class PickPlaceIKAbsEnvCfg(ManagerBasedRLEnvCfg):
    # ... 场景配置 ...
    
    # 添加相机
    camera_front = CameraCfg(
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
            resolution=(224, 224),  # 图像分辨率 (height, width)
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 0.5),  # 相机位置
            rot=(0.5, -0.5, 0.5, -0.5),  # 四元数
            convention="ros",
        ),
        prim_path="{ENV_REGEX_NS}/World/origin/front_camera",
        debug_vis=True,
    )
    
    # 观测配置
    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            # 图像观测
            image_front = ObsTerm(
                func=lambda env: env.scene["camera_front"].data.rgb,
            )
            
            # 低维观测
            joint_pos = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel = ObsTerm(func=mdp.joint_vel_rel)
            eef_pos = ObsTerm(func=mdp.ee_frame_pos)
            eef_quat = ObsTerm(func=mdp.ee_frame_quat)
            object = ObsTerm(func=mdp.object_obs)
            gripper_pos = ObsTerm(func=mdp.gripper_pos)
```

### 示例 2: 使用自定义观测函数

如果你需要预处理图像（例如归一化、裁剪等）：

**文件**: `source/SO_100/SO_100/tasks/pick_place/mdp/observations.py`

```python
import torch

def camera_rgb_image(
    env: ManagerBasedRLEnv,
    camera_name: str = "camera_front",
) -> torch.Tensor:
    """获取相机 RGB 图像观测.
    
    Args:
        env: 环境实例
        camera_name: 相机名称
        
    Returns:
        RGB 图像张量 [num_envs, H, W, 3], 值范围 [0-255]
    """
    # 从场景中获取相机
    camera = env.scene[camera_name]
    
    # 获取 RGB 图像
    rgb = camera.data.rgb  # [num_envs, H, W, 3]
    
    # 可选：预处理
    # - 归一化到 [0, 1]: rgb = rgb / 255.0
    # - 转换为 [0, 1] 并转置通道: rgb = rgb.permute(0, 3, 1, 2) / 255.0
    # - 裁剪: rgb = rgb[:, :, :224, :224]
    
    return rgb
```

然后在配置中使用：

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        image_front = ObsTerm(
            func=mdp.camera_rgb_image,
            params={"camera_name": "camera_front"},
        )
```

---

## 🔧 录制图像数据

### 使用 record_demos.py 录制

一旦环境配置了相机和图像观测，`record_demos.py` 会自动将图像数据保存到 HDF5：

```bash
python scripts/record_demos.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --teleop_device keyboard \
    --dataset_file ./datasets/pick_place_with_images.hdf5 \
    --step_hz 30 \
    --num_demos 10
```

### 录制脚本会自动：

1. **检测图像观测**: 自动检测观测空间中的图像键
2. **存储到 HDF5**: 图像作为 `uint8` 数组存储在 `observations/` 组中
3. **保持形状**: 图像保持原始形状 `[N, H, W, 3]`

---

## 📖 读取包含图像的数据集

### 方法 1: 使用 inspect 脚本

```bash
python scripts/inspect_hdf5_dataset.py \
    --dataset ./datasets/pick_place_with_images.hdf5 \
    --demo_idx 0 \
    --show_samples 3
```

### 方法 2: 直接读取 HDF5

```python
import h5py
import numpy as np
from PIL import Image

with h5py.File('dataset_with_images.hdf5', 'r') as f:
    demo_key = 'demo_0'
    demo = f[f'data/{demo_key}']
    
    # 读取图像
    images = np.array(demo['observations/image_front'])  # [N, H, W, 3]
    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Images range: [{images.min()}, {images.max()}]")
    
    # 读取第一帧图像
    first_image = images[0]  # [H, W, 3]
    
    # 保存为图像文件（可选）
    img = Image.fromarray(first_image.astype(np.uint8))
    img.save('first_frame.png')
    
    # 读取低维观测
    joint_pos = np.array(demo['observations/joint_pos'])
    actions = np.array(demo['actions'])
```

### 方法 3: PyTorch DataLoader（用于训练）

```python
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class HDF5ImageDataset(Dataset):
    """包含图像的 HDF5 数据集"""
    
    def __init__(self, dataset_path, obs_keys, image_keys=['image_front']):
        self.dataset_path = dataset_path
        self.obs_keys = obs_keys  # 低维观测键
        self.image_keys = image_keys  # 图像观测键
        
        # 预加载数据
        self._load_data()
    
    def _load_data(self):
        """加载所有数据"""
        with h5py.File(self.dataset_path, 'r') as f:
            data_group = f['data']
            demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
            
            all_images = {key: [] for key in self.image_keys}
            all_low_dim_obs = []
            all_actions = []
            
            for demo_key in demo_keys:
                demo = data_group[demo_key]
                
                # 加载图像
                images_dict = {}
                for key in self.image_keys:
                    if key in demo['observations']:
                        images = np.array(demo['observations'][key])  # [N, H, W, 3]
                        all_images[key].append(images)
                
                # 加载低维观测
                obs_list = []
                for key in self.obs_keys:
                    if key in demo['observations']:
                        obs_val = np.array(demo['observations'][key])
                        if len(obs_val.shape) > 2:
                            obs_val = obs_val.reshape(obs_val.shape[0], -1)
                        obs_list.append(obs_val)
                
                obs_concat = np.concatenate(obs_list, axis=1) if obs_list else np.array([]).reshape(0, 0)
                actions = np.array(demo['actions'])
                
                all_low_dim_obs.append(obs_concat)
                all_actions.append(actions)
            
            # 拼接所有演示
            self.images = {key: np.concatenate(all_images[key], axis=0) for key in self.image_keys}
            self.low_dim_obs = np.concatenate(all_low_dim_obs, axis=0) if all_low_dim_obs else np.array([]).reshape(0, 0)
            self.actions = np.concatenate(all_actions, axis=0)
    
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        # 获取图像（归一化到 [0, 1]）
        images_dict = {}
        for key in self.image_keys:
            img = self.images[key][idx].astype(np.float32) / 255.0  # [H, W, 3]
            images_dict[key] = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
        
        # 获取低维观测
        low_dim_obs = torch.from_numpy(self.low_dim_obs[idx]).float() if self.low_dim_obs.size > 0 else torch.tensor([])
        
        # 获取动作
        action = torch.from_numpy(self.actions[idx]).float()
        
        return {
            'images': images_dict,
            'low_dim_obs': low_dim_obs,
            'action': action,
        }

# 使用
dataset = HDF5ImageDataset(
    'dataset_with_images.hdf5',
    obs_keys=['joint_pos', 'joint_vel', 'eef_pos', 'eef_quat'],
    image_keys=['image_front']
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 迭代
for batch in dataloader:
    images = batch['images']['image_front']  # [batch_size, 3, H, W]
    low_dim_obs = batch['low_dim_obs']  # [batch_size, obs_dim]
    actions = batch['action']  # [batch_size, action_dim]
    # ... 训练代码 ...
```

---

## 💡 重要提示

### 1. 内存管理

图像数据占用大量内存：
- **单张图像**: 224x224x3 = ~150 KB
- **1000 步**: ~150 MB
- **100 个演示**: ~15 GB

**建议**:
- 使用较小的图像分辨率（如 128x128 或 224x224）
- 考虑压缩或使用延迟加载
- 使用批处理而不是一次性加载所有数据

### 2. 数据归一化

训练前通常需要归一化图像：

```python
# 方法 1: 归一化到 [0, 1]
images = images.astype(np.float32) / 255.0

# 方法 2: 归一化到 [-1, 1]
images = (images.astype(np.float32) / 255.0) * 2.0 - 1.0

# 方法 3: ImageNet 归一化
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
images = (images.astype(np.float32) / 255.0 - mean) / std
```

### 3. 通道顺序

Isaac Lab 存储的图像格式：
- **HDF5**: `[N, H, W, 3]` (通道在最后，RGB)
- **PyTorch**: `[N, 3, H, W]` (通道在前)

转换：

```python
# HDF5 -> PyTorch
img_pytorch = torch.from_numpy(img_hdf5).permute(0, 3, 1, 2)  # [N, H, W, 3] -> [N, 3, H, W]

# PyTorch -> HDF5
img_hdf5 = img_pytorch.permute(0, 2, 3, 1).numpy()  # [N, 3, H, W] -> [N, H, W, 3]
```

### 4. 多相机支持

可以添加多个相机：

```python
@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    camera_front = CameraCfg(...)
    camera_wrist = CameraCfg(...)
    camera_top = CameraCfg(...)
    
    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            image_front = ObsTerm(...)
            image_wrist = ObsTerm(...)
            image_top = ObsTerm(...)
```

---

## 🔗 参考

1. **Isaac Lab 相机文档**: `/mnt/ssd/IsaacLab/docs/sensors/camera.md`
2. **robomimic 图像训练**: `/mnt/ssd/IsaacLab/scripts/imitation_learning/robomimic/train.py`
3. **示例环境**: `/mnt/ssd/IsaacLab/source/isaaclab_tasks/.../stack_ik_rel_visuomotor_env_cfg.py`

---

祝你录制顺利！🚀



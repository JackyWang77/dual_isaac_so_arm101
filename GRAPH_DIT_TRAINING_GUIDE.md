# Graph-DiT Policy 完全自定义训练指南

## 📚 概述

这个指南展示了如何为你的 **Graph-DiT (Graph Diffusion Transformer) Policy** 创建完全自定义的训练框架。

---

## 🏗️ 项目结构

```
source/SO_100/SO_100/
├── policies/
│   ├── __init__.py
│   └── graph_dit_policy.py          # 你的 Graph-DiT Policy 实现（需要替换）
│
scripts/
└── graph_dit/
    ├── train.py                      # 训练脚本
    └── play.py                       # 推理/播放脚本
```

---

## 📝 步骤 1: 替换 Graph-DiT Policy 实现

### 文件: `source/SO_100/SO_100/policies/graph_dit_policy.py`

**当前**: 这是一个占位符实现，使用简单的 MLP + Transformer。

**你需要做的**:
1. 替换 `GraphDiTPolicy` 类中的网络架构为你的 Graph-DiT
2. 实现 Graph 卷积层/注意力机制
3. 实现 Diffusion 过程
4. 实现扩散损失函数

### 关键接口

```python
class GraphDiTPolicy(nn.Module):
    def forward(self, obs, timesteps=None, return_dict=False):
        """前向传播 - 实现你的 Graph-DiT 架构"""
        pass
    
    def loss(self, obs, actions, timesteps=None):
        """损失函数 - 实现扩散损失"""
        pass
    
    def predict(self, obs, deterministic=True):
        """推理模式 - 从观测预测动作"""
        pass
```

### 示例：替换 forward 方法

```python
def forward(self, obs, timesteps=None, return_dict=False):
    """你的 Graph-DiT forward pass."""
    
    # 1. 构建图结构（根据你的任务）
    graph_nodes, graph_edges = self._build_graph(obs)
    
    # 2. Graph-DiT 编码
    graph_features = self.graph_dit_encoder(
        nodes=graph_nodes,
        edges=graph_edges,
        timesteps=timesteps,
    )
    
    # 3. Diffusion 过程（如果训练时）
    if self.training and timesteps is not None:
        # 添加噪声
        noise = self._sample_noise(...)
        noisy_actions = self._add_noise(actions, noise, timesteps)
        
        # 预测噪声
        pred_noise = self.diffusion_head(graph_features)
        
        # 返回预测的噪声（用于训练）
        return pred_noise
    
    # 4. 推理时：去噪生成动作
    actions = self.diffusion_sample(graph_features)
    
    return actions
```

---

## 📝 步骤 2: 配置你的 Policy

### 修改配置类

**文件**: `source/SO_100/SO_100/policies/graph_dit_policy.py`

```python
@configclass
class GraphDiTPolicyCfg:
    """根据你的 Graph-DiT 架构修改配置参数"""
    
    obs_dim: int = MISSING
    action_dim: int = MISSING
    
    # Graph 相关参数
    num_nodes: int = 10              # 图节点数量
    node_dim: int = 64               # 节点特征维度
    edge_dim: int = 32               # 边特征维度
    
    # DiT (Diffusion Transformer) 参数
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 16
    mlp_ratio: float = 4.0
    
    # Diffusion 参数
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine"   # "cosine", "linear", "sqrt"
    guidance_scale: float = 1.0      # Classifier-free guidance
    
    device: str = "cuda"
```

---

## 📝 步骤 3: 训练你的 Graph-DiT Policy

### 训练命令

```bash
./isaaclab.sh -p scripts/graph_dit/train.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --dataset ./datasets/pick_place.hdf5 \
    --obs_dim 72 \
    --action_dim 8 \
    --epochs 500 \
    --batch_size 256 \
    --lr 1e-4 \
    --hidden_dim 512 \
    --num_layers 12 \
    --num_heads 16 \
    --save_dir ./logs/graph_dit \
    --log_dir ./logs/graph_dit
```

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--task` | 任务名称 | 必需 |
| `--dataset` | HDF5 数据集路径 | 必需 |
| `--obs_dim` | 观测维度 | 72 |
| `--action_dim` | 动作维度 | 8 |
| `--epochs` | 训练轮数 | 200 |
| `--batch_size` | 批次大小 | 256 |
| `--lr` | 学习率 | 1e-4 |
| `--hidden_dim` | 隐藏层维度 | 256 |
| `--num_layers` | Transformer 层数 | 6 |
| `--num_heads` | 注意力头数 | 8 |
| `--save_dir` | 模型保存目录 | ./logs/graph_dit |
| `--resume` | 恢复训练的检查点 | None |

---

## 📝 步骤 4: 推理/播放训练好的 Policy

### 播放命令

```bash
./isaaclab.sh -p scripts/graph_dit/play.py \
    --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
    --checkpoint ./logs/graph_dit/best_model.pt \
    --num_envs 64 \
    --num_episodes 10
```

---

## 🔧 自定义要点

### 1. 观测格式处理

训练脚本会自动处理字典格式的观测：

```python
# 如果观测是字典格式，会拼接成向量
if isinstance(obs, dict):
    obs_list = []
    for key in sorted(obs.keys()):
        obs_list.append(obs[key].flatten())
    obs_tensor = torch.cat(obs_list, dim=1)
```

### 2. 动作归一化

默认动作输出在 `[-1, 1]` 范围内（通过 `Tanh` 激活）：

```python
self.action_head = nn.Sequential(
    ...
    nn.Linear(hidden_dim, action_dim),
    nn.Tanh(),  # 输出归一化到 [-1, 1]
)
```

如果需要不同的动作范围，修改 `action_head`。

### 3. 扩散损失函数

在 `loss()` 方法中实现你的扩散损失：

```python
def loss(self, obs, actions, timesteps=None):
    """实现你的扩散损失函数"""
    
    # 1. 采样时间步
    if timesteps is None:
        timesteps = torch.randint(
            0, self.diffusion_steps, (obs.shape[0],), device=obs.device
        )
    
    # 2. 添加噪声
    noise = torch.randn_like(actions)
    alpha_t = self.noise_schedule.get_alpha(timesteps)
    noisy_actions = alpha_t * actions + (1 - alpha_t) * noise
    
    # 3. 预测噪声
    pred_noise = self.forward(obs, timesteps=timesteps)
    
    # 4. 计算损失（根据你的扩散方法）
    loss = nn.functional.mse_loss(pred_noise, noise)
    
    return {"total_loss": loss, "mse_loss": loss}
```

### 4. 推理时的去噪采样

在 `predict()` 方法中实现去噪采样：

```python
def predict(self, obs, deterministic=True):
    """推理模式：通过去噪采样生成动作"""
    
    self.eval()
    with torch.no_grad():
        # 从纯噪声开始
        actions = torch.randn(
            (obs.shape[0], self.cfg.action_dim),
            device=obs.device
        )
        
        # 去噪采样循环
        for t in reversed(range(self.diffusion_steps)):
            # 预测噪声
            pred_noise = self.forward(obs, timesteps=t)
            
            # 去噪一步
            actions = self._denoise_step(actions, pred_noise, t)
        
        # 裁剪到有效范围
        actions = torch.clamp(actions, -1.0, 1.0)
    
    return actions
```

---

## 📊 训练监控

训练脚本会自动：

1. **保存检查点**:
   - `checkpoint_epoch_{N}.pt`: 每个 epoch 的检查点
   - `best_model.pt`: 最佳模型（最低损失）
   - `latest_model.pt`: 最新模型
   - `final_model.pt`: 最终模型

2. **TensorBoard 日志**:
   - `Train/Loss`: 训练损失
   - `Train/MSE_Loss`: MSE 损失
   - `Train/LearningRate`: 学习率
   - `Epoch/AverageLoss`: 平均损失

查看训练曲线：
```bash
tensorboard --logdir ./logs/graph_dit/tensorboard
```

---

## 🎯 替换占位符代码

### 关键需要替换的地方

1. **`GraphDiTPolicy.__init__`**: 替换为你的 Graph-DiT 网络架构
2. **`GraphDiTPolicy.forward`**: 实现 Graph-DiT 前向传播
3. **`GraphDiTPolicy.loss`**: 实现扩散损失函数
4. **`GraphDiTPolicy.predict`**: 实现推理时的去噪采样

### 示例：完整的 Graph-DiT 架构模板

```python
def forward(self, obs, timesteps=None, return_dict=False):
    """完整的 Graph-DiT forward pass."""
    
    batch_size = obs.shape[0]
    
    # 1. 构建图结构（从观测中）
    nodes, edges, edge_index = self.build_graph_from_obs(obs)
    
    # 2. 时间步嵌入（用于 diffusion）
    if timesteps is not None:
        time_emb = self.time_embedding(timesteps)
    else:
        time_emb = None
    
    # 3. Graph-DiT 编码
    x = self.node_embedding(nodes)
    
    for layer in self.graph_dit_layers:
        x = layer(x, edges, edge_index, time_emb=time_emb)
    
    # 4. 输出投影
    if self.training and timesteps is not None:
        # 训练时：预测噪声
        pred_noise = self.noise_head(x)
        return pred_noise
    else:
        # 推理时：预测动作（或用于去噪）
        actions = self.action_head(x)
        return actions
```

---

## 🔗 参考资源

1. **Diffusion Policy**: https://github.com/real-stanford/diffusion_policy
2. **Graph Transformer**: 参考 Graph Transformer 论文实现
3. **DiT (Diffusion Transformer)**: 参考 DiT 架构
4. **Isaac Lab 训练框架**: 参考 RSL-RL 训练脚本

---

## ⚠️ 注意事项

1. **观测维度**: 确保 `obs_dim` 与你的数据集匹配
2. **动作维度**: 确保 `action_dim` 与你的任务匹配（IK Absolute = 8）
3. **图构建**: 根据你的任务定义如何从观测构建图结构
4. **扩散步数**: 训练时使用足够的扩散步数（通常 100-1000）
5. **批处理**: 确保你的实现支持批处理（向量化环境）

---

## 🎯 下一步

1. **替换 Policy 实现**: 将 `graph_dit_policy.py` 中的占位符替换为你的 Graph-DiT
2. **准备数据集**: 确保 HDF5 数据集格式正确
3. **开始训练**: 使用训练脚本训练你的模型
4. **测试推理**: 使用播放脚本测试训练好的模型

祝你训练顺利！🚀



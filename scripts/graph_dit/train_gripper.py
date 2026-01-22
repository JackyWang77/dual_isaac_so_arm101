"""Training script for the gripper prediction model.

This script trains a simple MLP to predict gripper joint position based on
current state, decoupling gripper control from the main Graph-DiT policy.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add current directory to path for gripper_model import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from gripper_model import GripperPredictor


class GripperDataset(Dataset):
    """跳过前12步，只学习真正的抓取循环"""
    
    def __init__(self, hdf5_path, normalize=True, skip_first_steps=12):
        """
        Labels:
        0: KEEP_CURRENT - 保持当前状态
        1: TRIGGER_CLOSE - OPEN→CLOSE（抓取时刻）
        2: TRIGGER_OPEN - CLOSE→OPEN（开始接近）
        """
        self.data = []
        self.normalize = normalize
        self.training = True
        
        all_inputs = []
        all_labels = []
        
        print(f"[GripperDataset] Skipping first {skip_first_steps} steps (pre-grasp phase)")
        print(f"[GripperDataset] Learning grasp cycle: CLOSE→OPEN→CLOSE")
        print(f"  0: KEEP_CURRENT")
        print(f"  1: TRIGGER_CLOSE (OPEN→CLOSE, 抓取)")
        print(f"  2: TRIGGER_OPEN (CLOSE→OPEN, 接近)")
        
        trigger_close_count = 0
        trigger_open_count = 0
        
        # 从图看，阈值：
        OPEN_THRESHOLD = -0.2  # > -0.05 认为是OPEN
        CLOSE_THRESHOLD = -0.2  # < -0.35 认为是CLOSED
        
        with h5py.File(hdf5_path, 'r') as f:
            demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])
            
            for demo_key in tqdm(demo_keys, desc="Loading demos"):
                demo = f[f'data/{demo_key}']
                obs_container = demo.get("obs", demo.get("observations", None))
                if obs_container is None:
                    continue
                
                ee_pos = np.array(obs_container['ee_position'])
                object_pos = np.array(obs_container['object_position'])
                joint_pos = np.array(obs_container['joint_pos'])
                joint_vel = np.array(obs_container['joint_vel'])
                
                T = len(ee_pos)
                
                # 🔥 关键：跳过前skip_first_steps步
                skip = skip_first_steps
                if T <= skip + 5:
                    print(f"  Warning: {demo_key} too short ({T} steps), skipping...")
                    continue
                
                gripper_seq = joint_pos[:, 5]  # [T]
                
                # 🔍 打印跳过后的gripper序列（第一个demo）
                if demo_key == demo_keys[0]:
                    print(f"\n[{demo_key}] After skipping first {skip} steps:")
                    print(f"  Remaining steps: {T - skip}")
                    print(f"  Gripper range: [{gripper_seq[skip:].min():.4f}, {gripper_seq[skip:].max():.4f}]")
                    print(f"  First 20 values (after skip): {gripper_seq[skip:skip+20].tolist()}")
                
                # 🎯 识别两种transition（只在skip之后）
                close_transition_indices = []  # OPEN→CLOSE
                open_transition_indices = []   # CLOSE→OPEN
                
                for t in range(skip, T - 3):
                    curr = gripper_seq[t]
                    future = gripper_seq[t + 2]  # 看2步后
                    
                    curr_is_open = curr > OPEN_THRESHOLD
                    curr_is_closed = curr < CLOSE_THRESHOLD
                    future_is_open = future > OPEN_THRESHOLD
                    future_is_closed = future < CLOSE_THRESHOLD
                    
                    # OPEN→CLOSE transition（抓取）
                    if curr_is_open and future_is_closed:
                        close_transition_indices.append(t)
                    
                    # CLOSE→OPEN transition（开始接近）
                    elif curr_is_closed and future_is_open:
                        open_transition_indices.append(t)
                
                if demo_key == demo_keys[0] or len(close_transition_indices) > 0 or len(open_transition_indices) > 0:
                    print(f"  {demo_key}: OPEN→CLOSE={len(close_transition_indices)}, CLOSE→OPEN={len(open_transition_indices)}")
                
                # 🔥 采样策略（只从skip之后开始）
                for t in range(skip, T - 3):
                    input_data = np.concatenate([
                        ee_pos[t],
                        object_pos[t],
                        joint_pos[t][:6],
                        joint_vel[t][:6]
                    ]).astype(np.float32)
                    
                    is_close_transition = t in close_transition_indices
                    is_open_transition = t in open_transition_indices
                    
                    if is_close_transition:
                        # 🎯 OPEN→CLOSE transition（抓取时刻）
                        label = 1
                        trigger_close_count += 1
                        
                        # 过采样20倍
                        for _ in range(20):
                            all_inputs.append(input_data)
                            all_labels.append(label)
                        
                        # 困难负样本
                        for offset in [-3, -2, -1, 1, 2, 3]:
                            t_neighbor = t + offset
                            if skip <= t_neighbor < T - 3:
                                if t_neighbor not in close_transition_indices and t_neighbor not in open_transition_indices:
                                    neighbor_input = np.concatenate([
                                        ee_pos[t_neighbor],
                                        object_pos[t_neighbor],
                                        joint_pos[t_neighbor][:6],
                                        joint_vel[t_neighbor][:6]
                                    ]).astype(np.float32)
                                    
                                    for _ in range(5):
                                        all_inputs.append(neighbor_input)
                                        all_labels.append(0)  # KEEP_CURRENT
                    
                    elif is_open_transition:
                        # 🎯 CLOSE→OPEN transition（开始接近）
                        label = 2
                        trigger_open_count += 1
                        
                        # 过采样20倍
                        for _ in range(20):
                            all_inputs.append(input_data)
                            all_labels.append(label)
                        
                        # 困难负样本
                        for offset in [-3, -2, -1, 1, 2, 3]:
                            t_neighbor = t + offset
                            if skip <= t_neighbor < T - 3:
                                if t_neighbor not in close_transition_indices and t_neighbor not in open_transition_indices:
                                    neighbor_input = np.concatenate([
                                        ee_pos[t_neighbor],
                                        object_pos[t_neighbor],
                                        joint_pos[t_neighbor][:6],
                                        joint_vel[t_neighbor][:6]
                                    ]).astype(np.float32)
                                    
                                    for _ in range(5):
                                        all_inputs.append(neighbor_input)
                                        all_labels.append(0)  # KEEP_CURRENT
                    
                    else:
                        # Normal state - 只采样3%（减少无用数据）
                        if np.random.random() < 0.03:
                            label = 0  # KEEP_CURRENT
                            all_inputs.append(input_data)
                            all_labels.append(label)
        
        all_inputs = np.stack(all_inputs)
        all_labels = np.array(all_labels)
        
        print(f"\n[GripperDataset] Label distribution:")
        print(f"  KEEP_CURRENT (0): {(all_labels==0).sum()} ({(all_labels==0).mean():.1%})")
        print(f"  TRIGGER_CLOSE (1): {(all_labels==1).sum()} ({(all_labels==1).mean():.1%})")
        print(f"  TRIGGER_OPEN (2): {(all_labels==2).sum()} ({(all_labels==2).mean():.1%})")
        print(f"  🎯 OPEN→CLOSE transitions: {trigger_close_count}")
        print(f"  🎯 CLOSE→OPEN transitions: {trigger_open_count}")
        
        if trigger_close_count == 0:
            raise ValueError("❌ No OPEN→CLOSE transitions found! Check thresholds.")
        if trigger_open_count == 0:
            raise ValueError("❌ No CLOSE→OPEN transitions found! Check thresholds.")
        
        # 归一化
        if normalize:
            self.input_mean = np.mean(all_inputs, axis=0, keepdims=True)
            self.input_std = np.std(all_inputs, axis=0, keepdims=True) + 1e-8
            all_inputs = (all_inputs - self.input_mean) / self.input_std
        else:
            self.input_mean = None
            self.input_std = None
        
        # 转为torch tensor
        for inp, label in zip(all_inputs, all_labels):
            self.data.append({
                'input': torch.from_numpy(inp),
                'label': torch.tensor(label, dtype=torch.long)
            })
        
        print(f"[GripperDataset] Dataset ready: {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.training:  # 只在训练时加噪声
            # 给输入加小噪声（数据增强）
            noise = torch.randn_like(sample['input']) * 0.01
            input_noisy = sample['input'] + noise
        else:
            input_noisy = sample['input']
        
        return {
            'input': input_noisy,
            'label': sample['label']
        }


def train_gripper_model(
    dataset_path: str,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = 'cuda',
    save_path: str = './gripper_model.pt',
    hidden_dims=[128, 128, 64],
    dropout=0.1,
    weight_decay=1e-5,
):
    """Train gripper prediction model.
    
    Args:
        dataset_path: Path to HDF5 dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        save_path: Path to save the trained model
        hidden_dims: Hidden layer dimensions for the MLP
        dropout: Dropout rate
        weight_decay: Weight decay for optimizer
    """
    
    # 加载数据
    print(f"\n{'='*60}")
    print(f"Training 3-CLASS CLASSIFICATION Gripper Predictor")
    print(f"  Classes: KEEP_CURRENT (0), TRIGGER_CLOSE (1), TRIGGER_OPEN (2)")
    print(f"{'='*60}")
    dataset = GripperDataset(dataset_path, normalize=True, skip_first_steps=12)
    dataset.training = True  # 训练模式：启用数据增强
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # 创建模型
    model = GripperPredictor(hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 🔥 使用CrossEntropyLoss（自带softmax）
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n[Train Gripper] Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print(f"\n[Train Gripper] Training configuration:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")
    print(f"  Save path: {save_path}")
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_correct = 0
        epoch_total = 0
        
        for batch in pbar:
            inputs = batch['input'].to(device)  # [B, 18]
            labels = batch['label'].to(device)  # [B]
            
            # 拆分输入
            ee_pos = inputs[:, 0:3]
            object_pos = inputs[:, 3:6]
            joint_pos = inputs[:, 6:12]
            joint_vel = inputs[:, 12:18]
            
            # 前向传播
            logits = model(ee_pos, object_pos, joint_pos, joint_vel)  # [B, 2]
            loss = criterion(logits, labels)
            
            # 计算准确率
            preds = torch.argmax(logits, dim=-1)  # [B]
            correct = (preds == labels).sum().item()
            epoch_correct += correct
            epoch_total += labels.size(0)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({
                'loss': loss.item(), 
                'acc': f'{100*correct/labels.size(0):.1f}%'
            })
        
        avg_loss = np.mean(epoch_losses)
        accuracy = epoch_correct / epoch_total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Accuracy: {accuracy:.1%}")
        
        # 每10个epoch验证一下
        if (epoch + 1) % 10 == 0:
            model.eval()
            dataset.training = False  # 验证时禁用数据增强
            with torch.no_grad():
                sample_batch = next(iter(dataloader))
                sample_inputs = sample_batch['input'][:32].to(device)
                sample_labels = sample_batch['label'][:32].to(device)
                
                ee_pos = sample_inputs[:, 0:3]
                object_pos = sample_inputs[:, 3:6]
                joint_pos = sample_inputs[:, 6:12]
                joint_vel = sample_inputs[:, 12:18]
                
                logits = model(ee_pos, object_pos, joint_pos, joint_vel)
                preds = torch.argmax(logits, dim=-1)
                
                val_acc = (preds == sample_labels).float().mean()
                
                print(f"  [Validation] Accuracy: {val_acc:.1%}")
                print(f"  [Validation] Pred distribution: KEEP={((preds==0).sum().item())}/32, CLOSE={((preds==1).sum().item())}/32, OPEN={((preds==2).sum().item())}/32")
                print(f"  [Validation] Label distribution: KEEP={((sample_labels==0).sum().item())}/32, CLOSE={((sample_labels==1).sum().item())}/32, OPEN={((sample_labels==2).sum().item())}/32")
            dataset.training = True  # 恢复训练模式
            model.train()
        
        # 保存最佳模型（基于准确率）
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_mean': dataset.input_mean,
                'input_std': dataset.input_std,
                'hidden_dims': hidden_dims,
                'dropout': dropout,
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy,
            }, save_path)
            print(f"  ✅ Saved best model (accuracy: {best_accuracy:.1%})")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"  Best accuracy: {best_accuracy:.1%}")
    print(f"  Model saved to: {save_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train gripper prediction model')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to HDF5 dataset file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda or cpu)')
    parser.add_argument('--save-path', type=str, default='./gripper_model.pt',
                        help='Path to save the trained model')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128, 64],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    
    args = parser.parse_args()
    
    # 检查数据集是否存在
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # 创建保存目录
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)
    
    train_gripper_model(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save_path,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
    )


if __name__ == '__main__':
    main()

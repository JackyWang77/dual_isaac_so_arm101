# Copyright (c) 2024-2025, SO-ARM100 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training script for Graph-DiT Policy.

This script provides a complete training framework for your custom Graph-DiT policy.
Users should replace the placeholder GraphDiTPolicy implementation with their own.

Usage:
    ./isaaclab.sh -p scripts/graph_dit/train.py \
        --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
        --dataset ./datasets/pick_place.hdf5 \
        --obs_dim 72 \
        --action_dim 8 \
        --epochs 200 \
        --batch_size 256 \
        --lr 1e-4
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import gymnasium as gym
import SO_100.tasks  # noqa: F401  # Register environments
from SO_100.policies.graph_dit_policy import GraphDiTPolicy, GraphDiTPolicyCfg


class HDF5Dataset(Dataset):
    """Dataset loader for HDF5 demonstration files.
    
    Loads observations and actions from Isaac Lab HDF5 format.
    """
    
    def __init__(self, hdf5_path: str, obs_keys: list[str], normalize_obs: bool = True):
        """Initialize dataset.
        
        Args:
            hdf5_path: Path to HDF5 file.
            obs_keys: List of observation keys to load.
            normalize_obs: If True, normalize observations to [0, 1].
        """
        self.hdf5_path = hdf5_path
        self.obs_keys = obs_keys
        self.normalize_obs = normalize_obs
        
        # Load all episodes
        self.episodes = []
        self.obs_stats = {}  # For normalization
        
        print(f"[HDF5Dataset] Loading dataset from: {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            # Load all demonstrations
            demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])
            
            print(f"[HDF5Dataset] Found {len(demo_keys)} demonstrations")
            
            all_obs = []
            all_actions = []
            
            for demo_key in demo_keys:
                demo = f[f'data/{demo_key}']
                
                # Load observations
                obs_dict = {}
                for key in obs_keys:
                    if key in demo['observations']:
                        obs_dict[key] = np.array(demo['observations'][key])
                    else:
                        print(f"[HDF5Dataset] Warning: Key '{key}' not found in {demo_key}")
                
                # Load actions
                actions = np.array(demo['actions'])
                
                # Store each (obs, action) pair
                for i in range(len(actions)):
                    # Concatenate observations
                    obs_vec = []
                    for key in obs_keys:
                        if key in obs_dict:
                            obs_val = obs_dict[key][i]
                            if isinstance(obs_val, np.ndarray):
                                obs_vec.append(obs_val.flatten())
                            else:
                                obs_vec.append(np.array([obs_val]))
                    obs = np.concatenate(obs_vec).astype(np.float32)
                    
                    self.episodes.append({
                        'obs': obs,
                        'action': actions[i].astype(np.float32),
                    })
                    
                    all_obs.append(obs)
                    all_actions.append(actions[i])
            
            # Compute normalization stats
            if normalize_obs and len(all_obs) > 0:
                all_obs = np.stack(all_obs)
                self.obs_stats['mean'] = np.mean(all_obs, axis=0, keepdims=True)
                self.obs_stats['std'] = np.std(all_obs, axis=0, keepdims=True) + 1e-8
                
                print(f"[HDF5Dataset] Computed obs stats: mean shape={self.obs_stats['mean'].shape}, "
                      f"std shape={self.obs_stats['std'].shape}")
        
        print(f"[HDF5Dataset] Loaded {len(self.episodes)} samples")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        obs = episode['obs'].copy()
        action = episode['action'].copy()
        
        # Normalize observations
        if self.normalize_obs and len(self.obs_stats) > 0:
            obs = (obs - self.obs_stats['mean'].squeeze()) / self.obs_stats['std'].squeeze()
        
        return torch.from_numpy(obs), torch.from_numpy(action)
    
    def get_obs_stats(self):
        """Get observation normalization statistics.
        
        Returns:
            dict: Dictionary with 'mean' and 'std' keys.
        """
        return self.obs_stats


def train_graph_dit_policy(
    task_name: str,
    dataset_path: str,
    obs_keys: list[str],
    obs_dim: int,
    action_dim: int,
    hidden_dims: list[int] = [256, 256, 128],
    batch_size: int = 256,
    num_epochs: int = 200,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    device: str = "cuda",
    save_dir: str = "./logs/graph_dit",
    log_dir: str = "./logs/graph_dit",
    resume_checkpoint: str | None = None,
):
    """Train Graph-DiT Policy.
    
    Args:
        task_name: Environment task name.
        dataset_path: Path to HDF5 dataset.
        obs_keys: List of observation keys.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dims: Hidden layer dimensions.
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay for optimizer.
        device: Device to train on.
        save_dir: Directory to save checkpoints.
        log_dir: Directory to save logs.
        resume_checkpoint: Path to checkpoint to resume from (optional).
    """
    
    # Create directories
    save_dir = Path(save_dir)
    log_dir = Path(log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir / "tensorboard")
    
    print(f"[Train] ===== Graph-DiT Policy Training =====")
    print(f"[Train] Task: {task_name}")
    print(f"[Train] Dataset: {dataset_path}")
    print(f"[Train] Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"[Train] Save dir: {save_dir}")
    print(f"[Train] Log dir: {log_dir}")
    
    # Load dataset
    print(f"\n[Train] Loading dataset...")
    dataset = HDF5Dataset(dataset_path, obs_keys, normalize_obs=True)
    
    # Get actual observation dimension from dataset
    sample_obs, sample_action = dataset[0]
    actual_obs_dim = sample_obs.shape[0]
    actual_action_dim = sample_action.shape[0]
    
    print(f"[Train] Actual obs dim: {actual_obs_dim}, Actual action dim: {actual_action_dim}")
    
    # Update dimensions if needed
    if obs_dim != actual_obs_dim:
        print(f"[Train] Warning: obs_dim mismatch ({obs_dim} vs {actual_obs_dim}), using {actual_obs_dim}")
        obs_dim = actual_obs_dim
    if action_dim != actual_action_dim:
        print(f"[Train] Warning: action_dim mismatch ({action_dim} vs {actual_action_dim}), using {actual_action_dim}")
        action_dim = actual_action_dim
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    # Create policy configuration
    cfg = GraphDiTPolicyCfg(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dims[0] if hidden_dims else 256,
        num_layers=6,
        num_heads=8,
        device=device,
    )
    
    # Create policy network
    print(f"\n[Train] Creating Graph-DiT Policy...")
    policy = GraphDiTPolicy(cfg).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[Train] Total parameters: {total_params:,}")
    print(f"[Train] Trainable parameters: {trainable_params:,}")
    print(f"\n[Train] Policy architecture:")
    print(policy)
    
    # Create optimizer
    optimizer = optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=learning_rate * 0.01,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n[Train] Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"[Train] Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}")
    
    # Training loop
    print(f"\n[Train] Starting training for {num_epochs} epochs...")
    policy.train()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        epoch_mse_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (obs, actions) in enumerate(pbar):
            obs = obs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            
            # Forward pass
            loss_dict = policy.loss(obs, actions)
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record losses
            epoch_losses.append(loss.item())
            epoch_mse_losses.append(loss_dict['mse_loss'].item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'mse': loss_dict['mse_loss'].item(),
                'lr': optimizer.param_groups[0]['lr'],
            })
            
            # Log to tensorboard
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/MSE_Loss', loss_dict['mse_loss'].item(), global_step)
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
        
        # Update learning rate
        scheduler.step()
        
        # Compute epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_mse_loss = np.mean(epoch_mse_losses)
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] "
              f"Loss: {avg_loss:.6f}, MSE: {avg_mse_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Log to tensorboard
        writer.add_scalar('Epoch/AverageLoss', avg_loss, epoch)
        writer.add_scalar('Epoch/AverageMSE', avg_mse_loss, epoch)
        
        # Save checkpoint
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'cfg': cfg,
            'best_loss': best_loss,
            'avg_loss': avg_loss,
        }, checkpoint_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / "best_model.pt"
            policy.save(str(best_path))
            print(f"[Train] Saved best model (loss: {best_loss:.6f}) to: {best_path}")
        
        # Save latest model
        latest_path = save_dir / "latest_model.pt"
        policy.save(str(latest_path))
        
        # Save every N epochs
        if (epoch + 1) % 10 == 0:
            epoch_path = save_dir / f"model_epoch_{epoch+1}.pt"
            policy.save(str(epoch_path))
    
    # Save final model
    final_path = save_dir / "final_model.pt"
    policy.save(str(final_path))
    print(f"\n[Train] Training completed!")
    print(f"[Train] Final model saved to: {final_path}")
    print(f"[Train] Best model saved to: {save_dir / 'best_model.pt'}")
    
    writer.close()
    return policy


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Graph-DiT Policy")
    
    # Dataset arguments
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--dataset", type=str, required=True, help="HDF5 dataset path")
    
    # Model arguments
    parser.add_argument("--obs_dim", type=int, default=72, help="Observation dimension")
    parser.add_argument("--action_dim", type=int, default=8, help="Action dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    
    # Paths
    parser.add_argument("--save_dir", type=str, default="./logs/graph_dit", help="Checkpoint save directory")
    parser.add_argument("--log_dir", type=str, default="./logs/graph_dit", help="Log directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Observation keys (should match your dataset)
    obs_keys = [
        "actions",
        "joint_pos",
        "joint_vel",
        "object",
        "object_positions",
        "object_orientations",
        "eef_pos",
        "eef_quat",
        "gripper_pos",
    ]
    
    # Start training
    train_graph_dit_policy(
        task_name=args.task,
        dataset_path=args.dataset,
        obs_keys=obs_keys,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dims=[args.hidden_dim],
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()



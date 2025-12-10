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
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import SO_100.tasks  # noqa: F401  # Register environments
from SO_100.policies.graph_dit_policy import GraphDiTPolicy, GraphDiTPolicyCfg


class HDF5Dataset(Dataset):
    """Dataset loader for HDF5 demonstration files.
    
    Loads observations and actions from Isaac Lab HDF5 format.
    """
    
    def __init__(self, hdf5_path: str, obs_keys: list[str], normalize_obs: bool = True, normalize_actions: bool = True, action_history_length: int = 4):
        """Initialize dataset.
        
        Args:
            hdf5_path: Path to HDF5 file.
            obs_keys: List of observation keys to load.
            normalize_obs: If True, normalize observations.
            normalize_actions: If True, normalize actions (recommended for IK Abs actions with different ranges).
            action_history_length: Number of historical actions to collect (default: 4).
        """
        self.hdf5_path = hdf5_path
        self.obs_keys = obs_keys
        self.normalize_obs = normalize_obs
        self.normalize_actions = normalize_actions
        self.action_history_length = action_history_length
        
        # Compute observation dimension for each key (for dynamic indexing)
        # This allows skipping keys that are not in obs_keys (e.g., target_object_position)
        self.obs_key_dims = {}  # Will store: {'joint_pos': 6, 'object_position': 3, ...}
        self.obs_key_offsets = {}  # Will store offsets in concatenated obs: {'joint_pos': 0, 'object_position': 12, ...}
        
        # Load all episodes
        self.episodes = []
        self.obs_stats = {}  # For normalization
        self.action_stats = {}  # For action normalization
        self.subtask_order = []  # Order of subtasks for encoding
        
        print(f"[HDF5Dataset] Loading dataset from: {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            # Load all demonstrations
            demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])
            
            print(f"[HDF5Dataset] Found {len(demo_keys)} demonstrations")
            
            all_obs = []
            all_actions = []
            
            # Track subtask statistics
            subtask_stats = {}
            
            for demo_key in demo_keys:
                demo = f[f'data/{demo_key}']
                
                # Load observations
                # Try 'obs' first (Isaac Lab Mimic format), fallback to 'observations'
                obs_container = demo.get('obs', demo.get('observations', None))
                if obs_container is None:
                    raise ValueError(f"Neither 'obs' nor 'observations' found in {demo_key}")
                
                obs_dict = {}
                # Compute key dimensions and offsets (only need to do this once, from first demo)
                if not self.obs_key_dims:
                    offset = 0
                    for key in obs_keys:
                        if key in obs_container:
                            key_data = np.array(obs_container[key])
                            # Get dimension (handle different array shapes)
                            # Examples:
                            # - shape (N,): single value per timestep -> dim = 1
                            # - shape (N, 3): 3D vector per timestep -> dim = 3
                            # - shape (N, 6): 6D vector per timestep -> dim = 6
                            if len(key_data.shape) == 1:
                                # 1D array: single value per timestep
                                key_dim = 1
                            elif len(key_data.shape) == 2:
                                # 2D array: (num_timesteps, feature_dim)
                                key_dim = key_data.shape[1]
                            else:
                                # Higher dim: take last dimension (shouldn't happen)
                                key_dim = key_data.shape[-1]
                            self.obs_key_dims[key] = key_dim
                            self.obs_key_offsets[key] = offset
                            offset += key_dim
                            print(f"[HDF5Dataset] Key '{key}': dim={key_dim}, offset={self.obs_key_offsets[key]}")
                
                for key in obs_keys:
                    if key in obs_container:
                        obs_dict[key] = np.array(obs_container[key])
                    else:
                        print(f"[HDF5Dataset] Warning: Key '{key}' not found in {demo_key}")
                
                # Load actions
                actions = np.array(demo['actions'])
                
                # Load subtask signals if available (for multi-subtask demos)
                subtask_signals = None
                subtask_order = []  # Order of subtasks: ['push_cube', 'lift_ee']
                if 'datagen_info' in obs_container:
                    datagen_info = obs_container['datagen_info']
                    if 'subtask_term_signals' in datagen_info:
                        subtask_signals = {}
                        signals_group = datagen_info['subtask_term_signals']
                        for subtask_name in signals_group.keys():
                            subtask_signals[subtask_name] = np.array(signals_group[subtask_name])
                            subtask_order.append(subtask_name)
                            # Track statistics
                            if subtask_name not in subtask_stats:
                                subtask_stats[subtask_name] = {'total_steps': 0, 'completed_steps': 0}
                            subtask_stats[subtask_name]['total_steps'] += len(subtask_signals[subtask_name])
                            subtask_stats[subtask_name]['completed_steps'] += np.sum(subtask_signals[subtask_name])
                
                # Store subtask order for encoding (set once from first demo with subtasks)
                if not self.subtask_order and subtask_order:
                    self.subtask_order = subtask_order
                    print(f"[HDF5Dataset] Subtask order: {subtask_order}")
                elif self.subtask_order and subtask_order:
                    # Verify consistency: all demos should have the same subtask order
                    if self.subtask_order != subtask_order:
                        print(f"[HDF5Dataset] WARNING: Subtask order mismatch!")
                        print(f"  Expected: {self.subtask_order}")
                        print(f"  Found: {subtask_order}")
                        print(f"  Using: {self.subtask_order}")
                
                # Store each (obs, action_history, action) pair
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
                    
                    # Collect action history (last action_history_length actions)
                    action_history = []
                    for j in range(max(0, i - self.action_history_length + 1), i + 1):
                        action_history.append(actions[j].astype(np.float32))
                    
                    # Pad with first action if needed
                    while len(action_history) < self.action_history_length:
                        action_history.insert(0, actions[0].astype(np.float32))
                    
                    action_history = np.stack(action_history, axis=0)  # [action_history_length, action_dim]
                    
                    # Collect node features history (EE and Object pos/ori)
                    # Extract from obs history (same indices as action history)
                    node_history = []  # Will store [ee_node, object_node] pairs
                    for j in range(max(0, i - self.action_history_length + 1), i + 1):
                        # Extract node features from obs at time j
                        obs_j_vec = []
                        for key in obs_keys:
                            if key in obs_dict:
                                obs_j_val = obs_dict[key][j]
                                if isinstance(obs_j_val, np.ndarray):
                                    obs_j_vec.append(obs_j_val.flatten())
                                else:
                                    obs_j_vec.append(np.array([obs_j_val]))
                        obs_j = np.concatenate(obs_j_vec).astype(np.float32)
                        
                        # Extract EE and Object node features dynamically based on obs_keys
                        # This allows skipping target_object_position if it's not in obs_keys
                        obj_pos_start = self.obs_key_offsets.get('object_position', 12)
                        obj_pos = obs_j[obj_pos_start:obj_pos_start + self.obs_key_dims.get('object_position', 3)]
                        
                        obj_ori_start = self.obs_key_offsets.get('object_orientation', 15)
                        obj_ori = obs_j[obj_ori_start:obj_ori_start + self.obs_key_dims.get('object_orientation', 4)]
                        
                        ee_pos_start = self.obs_key_offsets.get('ee_position', 19)
                        ee_pos = obs_j[ee_pos_start:ee_pos_start + self.obs_key_dims.get('ee_position', 3)]
                        
                        ee_ori_start = self.obs_key_offsets.get('ee_orientation', 22)
                        ee_ori = obs_j[ee_ori_start:ee_ori_start + self.obs_key_dims.get('ee_orientation', 4)]
                        
                        # Ensure correct shapes (pad if needed, truncate if too long)
                        obj_pos = obj_pos[:3] if len(obj_pos) >= 3 else np.pad(obj_pos, (0, max(0, 3 - len(obj_pos))))
                        obj_ori = obj_ori[:4] if len(obj_ori) >= 4 else np.pad(obj_ori, (0, max(0, 4 - len(obj_ori))))
                        ee_pos = ee_pos[:3] if len(ee_pos) >= 3 else np.pad(ee_pos, (0, max(0, 3 - len(ee_pos))))
                        ee_ori = ee_ori[:4] if len(ee_ori) >= 4 else np.pad(ee_ori, (0, max(0, 4 - len(ee_ori))))
                        
                        ee_node_j = np.concatenate([ee_pos, ee_ori])  # [7]
                        object_node_j = np.concatenate([obj_pos, obj_ori])  # [7]
                        
                        node_history.append({
                            'ee_node': ee_node_j.astype(np.float32),
                            'object_node': object_node_j.astype(np.float32),
                        })
                    
                    # Pad node history if needed (use first timestep's nodes)
                    if len(node_history) < self.action_history_length:
                        first_node = node_history[0] if len(node_history) > 0 else {
                            'ee_node': np.zeros(7, dtype=np.float32),  # Fallback: zeros
                            'object_node': np.zeros(7, dtype=np.float32),
                        }
                        while len(node_history) < self.action_history_length:
                            node_history.insert(0, {
                                'ee_node': first_node['ee_node'].copy(),
                                'object_node': first_node['object_node'].copy(),
                            })
                    
                    # Stack node history
                    ee_node_history = np.stack([nh['ee_node'] for nh in node_history], axis=0)  # [history_length, 7]
                    object_node_history = np.stack([nh['object_node'] for nh in node_history], axis=0)  # [history_length, 7]
                    
                    # Determine subtask condition (one-hot) if available
                    subtask_condition = None
                    if subtask_signals is not None and subtask_order:
                        # Determine current active subtask based on signals
                        # Logic: Find the first incomplete subtask, or use the last one if all are complete
                        active_subtask_idx = None
                        for idx, subtask_name in enumerate(subtask_order):
                            # If this subtask is not yet completed, it's the active one
                            if not subtask_signals[subtask_name][i]:
                                active_subtask_idx = idx
                                break
                        
                        # If all subtasks are completed, use the last one (lift_ee)
                        if active_subtask_idx is None:
                            active_subtask_idx = len(subtask_order) - 1
                        
                        # One-hot encoding: [push_cube, lift_ee] for 2 subtasks
                        subtask_condition = np.zeros(len(subtask_order), dtype=np.float32)
                        subtask_condition[active_subtask_idx] = 1.0
                    
                    # Store episode data with subtask condition, action history, and node history
                    episode_data = {
                        'obs': obs,
                        'action': actions[i].astype(np.float32),
                        'action_history': action_history,  # [action_history_length, action_dim]
                        'ee_node_history': ee_node_history,  # [action_history_length, 7]
                        'object_node_history': object_node_history,  # [action_history_length, 7]
                    }
                    if subtask_condition is not None:
                        episode_data['subtask_condition'] = subtask_condition
                    if subtask_signals is not None:
                        episode_data['subtask_signals'] = {
                            name: signal[i] for name, signal in subtask_signals.items()
                        }
                    
                    self.episodes.append(episode_data)
                    
                    all_obs.append(obs)
                    all_actions.append(actions[i])
            
            # Print subtask statistics
            if subtask_stats:
                print(f"\n[HDF5Dataset] Subtask Statistics:")
                for subtask_name, stats in subtask_stats.items():
                    completion_rate = (stats['completed_steps'] / stats['total_steps'] * 100) if stats['total_steps'] > 0 else 0
                    print(f"  {subtask_name}: {stats['completed_steps']}/{stats['total_steps']} steps completed ({completion_rate:.1f}%)")
            
            # Print subtask encoding info
            if self.subtask_order:
                print(f"[HDF5Dataset] Subtask condition: {len(self.subtask_order)} subtasks will be passed as condition to policy")
                print(f"[HDF5Dataset] Subtask order: {self.subtask_order}")
            
            # Compute normalization stats
            if normalize_obs and len(all_obs) > 0:
                all_obs = np.stack(all_obs)
                self.obs_stats['mean'] = np.mean(all_obs, axis=0, keepdims=True)
                self.obs_stats['std'] = np.std(all_obs, axis=0, keepdims=True) + 1e-8
                
                print(f"[HDF5Dataset] Computed obs stats: mean shape={self.obs_stats['mean'].shape}, "
                      f"std shape={self.obs_stats['std'].shape}")
            
            # Compute action normalization stats
            if normalize_actions and len(all_actions) > 0:
                all_actions = np.stack(all_actions)
                self.action_stats['mean'] = np.mean(all_actions, axis=0, keepdims=True)
                self.action_stats['std'] = np.std(all_actions, axis=0, keepdims=True) + 1e-8
                
                print(f"[HDF5Dataset] Computed action stats: mean shape={self.action_stats['mean'].shape}, "
                      f"std shape={self.action_stats['std'].shape}")
                print(f"[HDF5Dataset] Action ranges (before norm): min={all_actions.min(axis=0)}, max={all_actions.max(axis=0)}")
        
        print(f"[HDF5Dataset] Loaded {len(self.episodes)} samples")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        obs = episode['obs'].copy()
        action = episode['action'].copy()
        action_history = episode['action_history'].copy()  # [action_history_length, action_dim]
        ee_node_history = episode['ee_node_history'].copy()  # [action_history_length, 7]
        object_node_history = episode['object_node_history'].copy()  # [action_history_length, 7]
        
        # Normalize observations
        if self.normalize_obs and len(self.obs_stats) > 0:
            obs = (obs - self.obs_stats['mean'].squeeze()) / self.obs_stats['std'].squeeze()
        
        # Normalize actions
        if self.normalize_actions and len(self.action_stats) > 0:
            action = (action - self.action_stats['mean'].squeeze()) / self.action_stats['std'].squeeze()
            # Normalize action history
            action_history = (action_history - self.action_stats['mean'].squeeze()) / self.action_stats['std'].squeeze()
        
        # Node features are already extracted from raw obs, so we don't normalize them separately
        # (They will be normalized as part of obs if normalize_obs=True, but we extract them before normalization)
        # Actually, we should normalize them too if we want consistency
        
        # Get subtask condition if available
        subtask_condition = None
        if 'subtask_condition' in episode:
            subtask_condition = torch.from_numpy(episode['subtask_condition'].copy())
        
        return torch.from_numpy(obs), torch.from_numpy(action), torch.from_numpy(action_history), \
               torch.from_numpy(ee_node_history), torch.from_numpy(object_node_history), subtask_condition
    
    def get_obs_stats(self):
        """Get observation normalization statistics.
        
        Returns:
            dict: Dictionary with 'mean' and 'std' keys.
        """
        return self.obs_stats
    
    def get_action_stats(self):
        """Get action normalization statistics.
        
        Returns:
            dict: Dictionary with 'mean' and 'std' keys.
        """
        return self.action_stats


def train_graph_dit_policy(
    task_name: str,
    dataset_path: str,
    obs_keys: list[str],
    obs_dim: int,
    action_dim: int,
    hidden_dims: list[int] = [256, 256, 128],
    num_layers: int = 6,
    num_heads: int = 8,
    batch_size: int = 256,
    num_epochs: int = 200,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    device: str = "cuda",
    save_dir: str = "./logs/graph_dit",
    log_dir: str = "./logs/graph_dit",
    resume_checkpoint: str | None = None,
    action_history_length: int = 4,
    mode: str = "ddpm",
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
    
    # Create directories with timestamp and mode suffix
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Add mode suffix to save/log directories (e.g., reach_joint_ddpm, reach_joint_flow_matching)
    base_save_dir = Path(save_dir)
    base_log_dir = Path(log_dir)
    
    # Append mode suffix to the base directory name
    # e.g., ./logs/graph_dit/reach_joint -> ./logs/graph_dit/reach_joint_ddpm
    save_dir_with_mode = base_save_dir.parent / f"{base_save_dir.name}_{mode}"
    log_dir_with_mode = base_log_dir.parent / f"{base_log_dir.name}_{mode}"
    
    # Add timestamp
    save_dir = save_dir_with_mode / timestamp
    log_dir = log_dir_with_mode / timestamp
    
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Train] Save dir (with mode and timestamp): {save_dir}")
    print(f"[Train] Log dir (with mode and timestamp): {log_dir}")
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir / "tensorboard")
    
    print(f"[Train] ===== Graph-DiT Policy Training =====")
    print(f"[Train] Task: {task_name}")
    print(f"[Train] Dataset: {dataset_path}")
    print(f"[Train] Mode: {mode.upper()} ({'50-100 steps' if mode == 'ddpm' else '1-10 steps, much faster!'})")
    print(f"[Train] Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"[Train] Action history length: {action_history_length}")
    
    # Load dataset
    print(f"\n[Train] Loading dataset...")
    dataset = HDF5Dataset(dataset_path, obs_keys, normalize_obs=True, normalize_actions=True, action_history_length=action_history_length)
    
    # Get normalization stats for saving
    obs_stats = dataset.get_obs_stats()
    action_stats = dataset.get_action_stats()
    
    # Get actual observation dimension from dataset
    sample_obs, sample_action, sample_action_history, sample_ee_history, sample_obj_history, sample_subtask = dataset[0]
    actual_obs_dim = sample_obs.shape[0]
    actual_action_dim = sample_action.shape[0]
    
    # Check if subtask condition is available
    num_subtasks = len(dataset.subtask_order) if hasattr(dataset, 'subtask_order') and dataset.subtask_order else 0
    if num_subtasks > 0:
        print(f"[Train] Subtask condition available: {num_subtasks} subtasks ({dataset.subtask_order})")
        print(f"[Train] Observations: {actual_obs_dim} dim (state-based, no subtask encoding)")
        print(f"[Train] Subtask condition: {num_subtasks} dim (one-hot, passed separately to policy)")
    else:
        print(f"[Train] No subtask condition found in dataset")
    
    print(f"[Train] Actual obs dim: {actual_obs_dim}, Actual action dim: {actual_action_dim}")
    
    # Update dimensions if needed
    if obs_dim != actual_obs_dim:
        print(f"[Train] Warning: obs_dim mismatch ({obs_dim} vs {actual_obs_dim}), using {actual_obs_dim}")
        obs_dim = actual_obs_dim
    if action_dim != actual_action_dim:
        print(f"[Train] Warning: action_dim mismatch ({action_dim} vs {actual_action_dim}), using {actual_action_dim}")
        action_dim = actual_action_dim
    
    # Custom collate function to handle optional subtask_condition, action_history, and node_history
    def collate_fn(batch):
        """Collate function that handles optional subtask_condition, action_history, and node_history."""
        # Check if subtask_condition is present
        has_subtask = any(len(item) == 6 and item[5] is not None for item in batch)
        
        obs_batch = default_collate([item[0] for item in batch])
        action_batch = default_collate([item[1] for item in batch])
        action_history_batch = default_collate([item[2] for item in batch])  # [batch, action_history_length, action_dim]
        ee_node_history_batch = default_collate([item[3] for item in batch])  # [batch, action_history_length, 7]
        object_node_history_batch = default_collate([item[4] for item in batch])  # [batch, action_history_length, 7]
        
        if has_subtask:
            # All items have subtask_condition (may be None)
            subtask_batch = []
            # Get actual num_subtasks from dataset (should match model config)
            num_subtasks = len(dataset.subtask_order) if hasattr(dataset, 'subtask_order') and dataset.subtask_order else 0
            if num_subtasks == 0:
                raise ValueError("Subtask condition found in batch but dataset.subtask_order is empty! "
                               "This should not happen - check dataset loading.")
            for item in batch:
                if len(item) == 6 and item[5] is not None:
                    subtask_item = item[5]
                    # Verify dimension matches
                    if subtask_item.shape[0] != num_subtasks:
                        raise ValueError(f"Subtask condition dimension mismatch: "
                                       f"expected {num_subtasks}, got {subtask_item.shape[0]}. "
                                       f"Check dataset and model config consistency.")
                    subtask_batch.append(subtask_item)
                else:
                    # Create zero vector if missing (shouldn't happen if has_subtask=True)
                    subtask_batch.append(torch.zeros(num_subtasks))
            subtask_batch = default_collate(subtask_batch)
            return obs_batch, action_batch, action_history_batch, ee_node_history_batch, object_node_history_batch, subtask_batch
        else:
            # No subtask_condition
            return obs_batch, action_batch, action_history_batch, ee_node_history_batch, object_node_history_batch, None
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    # Get number of subtasks from dataset
    num_subtasks = len(dataset.subtask_order) if hasattr(dataset, 'subtask_order') and dataset.subtask_order else 0
    
    # Create policy configuration
    # IMPORTANT: num_subtasks must match the actual subtask_condition dimension in data
    # If dataset has subtasks, use that number; otherwise disable subtask conditioning
    cfg = GraphDiTPolicyCfg(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dims[0] if hidden_dims else 256,
        num_layers=num_layers,
        num_heads=num_heads,
        num_subtasks=num_subtasks,  # Use actual number from dataset (0 = no subtasks)
        action_history_length=action_history_length,
        mode=mode,  # "ddpm" or "flow_matching"
        device=device,
    )
    
    if num_subtasks > 0:
        print(f"[Train] Policy configured with {num_subtasks} subtasks: {dataset.subtask_order}")
        print(f"[Train] Subtask conditioning will be used during training.")
    else:
        print(f"[Train] No subtask info found, subtask conditioning disabled (num_subtasks=0)")
    
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
    
    # Learning rate scheduler with warmup + cosine annealing
    # Warmup for first 10% of epochs, then cosine annealing
    import math
    warmup_epochs = max(1, int(num_epochs * 0.1))  # 10% warmup, at least 1 epoch
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: linearly increase from 0 to learning_rate
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"[Train] Using warmup scheduler: {warmup_epochs} warmup epochs ({100*warmup_epochs/num_epochs:.1f}%), then cosine annealing")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n[Train] Resuming from checkpoint: {resume_checkpoint}")
        # weights_only=False is needed for PyTorch 2.6+ to load custom config classes
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"[Train] Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}")
    
    # Training loop
    print(f"\n[Train] Starting training for {num_epochs} epochs...")
    policy.train()
    
    # Track last epoch loss for final model
    last_epoch_loss = None
    
    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        epoch_mse_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Handle subtask condition, action_history, and node_history (may be present or not)
            if len(batch) == 6:
                obs, actions, action_history, ee_node_history, object_node_history, subtask_condition = batch
                action_history = action_history.to(device, non_blocking=True)  # [batch, action_history_length, action_dim]
                ee_node_history = ee_node_history.to(device, non_blocking=True)  # [batch, action_history_length, 7]
                object_node_history = object_node_history.to(device, non_blocking=True)  # [batch, action_history_length, 7]
                if subtask_condition is not None:
                    subtask_condition = subtask_condition.to(device, non_blocking=True)
            elif len(batch) == 5:
                obs, actions, action_history, ee_node_history, object_node_history = batch
                action_history = action_history.to(device, non_blocking=True)
                ee_node_history = ee_node_history.to(device, non_blocking=True)
                object_node_history = object_node_history.to(device, non_blocking=True)
                subtask_condition = None
            else:
                obs, actions = batch
                action_history = None
                ee_node_history = None
                object_node_history = None
                subtask_condition = None
            
            obs = obs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            
            # Forward pass (pass action_history and node_history if available)
            loss_dict = policy.loss(
                obs, actions, 
                action_history=action_history,
                ee_node_history=ee_node_history,
                object_node_history=object_node_history,
                subtask_condition=subtask_condition
            )
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
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        writer.add_scalar('Epoch/AverageLoss', avg_loss, epoch)
        writer.add_scalar('Epoch/AverageMSE', avg_mse_loss, epoch)
        writer.add_scalar('Epoch/LearningRate', current_lr, epoch)
        
        # Track last epoch loss
        last_epoch_loss = avg_loss
        
        # Save best model only (overwrite if better)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / "best_model.pt"
            # Save with stats for inference
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'cfg': cfg,
                'obs_stats': obs_stats,
                'action_stats': action_stats,
                'epoch': epoch,
                'loss': avg_loss,
            }, str(best_path))
            print(f"[Train] ✅ Saved best model (loss: {best_loss:.6f}, epoch: {epoch+1}) to: {best_path}")
    
    # Save final model (after all epochs)
    final_loss = last_epoch_loss if last_epoch_loss is not None else best_loss
    final_path = save_dir / "final_model.pt"
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'cfg': cfg,
        'obs_stats': obs_stats,
        'action_stats': action_stats,
        'epoch': num_epochs - 1,
        'loss': final_loss,
    }, str(final_path))
    print(f"\n[Train] ===== Training Completed! =====")
    print(f"[Train] ✅ Final model saved to: {final_path}")
    print(f"[Train] ✅ Best model saved to: {save_dir / 'best_model.pt'}")
    print(f"[Train] 📁 All models saved in: {save_dir}")
    print(f"[Train] 📊 TensorBoard logs in: {log_dir / 'tensorboard'}")
    
    writer.close()
    return policy


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Graph-DiT Policy")
    
    # Dataset arguments
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--dataset", type=str, required=True, help="HDF5 dataset path")
    
    # Model arguments
    parser.add_argument("--obs_dim", type=int, default=32, help="Observation dimension (reach task: 32 after removing redundant target_object_position)")
    parser.add_argument("--action_dim", type=int, default=8, help="Action dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--action_history_length", type=int, default=4, help="Number of historical actions to use (default: 4)")
    parser.add_argument("--mode", type=str, default="ddpm", choices=["ddpm", "flow_matching"], 
                       help="Training mode: 'ddpm' (slower, 50-100 steps) or 'flow_matching' (faster, 1-10 steps)")
    
    # Paths
    parser.add_argument("--save_dir", type=str, default="./logs/graph_dit", help="Checkpoint save directory")
    parser.add_argument("--log_dir", type=str, default="./logs/graph_dit", help="Log directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Observation keys (should match your dataset)
    # For reach task, these keys match the observations in reach_env_cfg.py
    # Note: Removed "target_object_position" (7 dims) as it's redundant with object_position + object_orientation
    # In reach task, target is always the current object position, so command is not needed
    obs_keys = [
        "joint_pos",
        "joint_vel",
        "object_position",
        "object_orientation",
        "ee_position",
        "ee_orientation",
        "actions",  # last action for self-attention in Graph DiT
    ]
    
    # Start training
    train_graph_dit_policy(
        task_name=args.task,
        dataset_path=args.dataset,
        obs_keys=obs_keys,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dims=[args.hidden_dim],
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_checkpoint=args.resume,
        action_history_length=args.action_history_length,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()



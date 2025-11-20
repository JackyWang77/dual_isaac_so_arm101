# Copyright (c) 2024-2025, SO-ARM100 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Example script showing how to read HDF5 datasets.

This script demonstrates different ways to read and process HDF5 datasets
recorded by Isaac Lab's record_demos.py script.

Usage:
    python scripts/read_hdf5_example.py --dataset ./datasets/pick_place.hdf5
"""

import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple


def read_hdf5_basic(dataset_path: str) -> Dict:
    """Basic method: Read entire HDF5 dataset into memory.
    
    This is the simplest method, suitable for small datasets.
    
    Args:
        dataset_path: Path to HDF5 file.
        
    Returns:
        dict: Dictionary containing all demonstrations.
    """
    
    print("=" * 80)
    print("Method 1: Basic Reading (Load entire dataset)")
    print("=" * 80)
    
    with h5py.File(dataset_path, 'r') as f:
        data_group = f['data']
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
        
        print(f"Found {len(demo_keys)} demonstrations")
        
        all_data = {}
        
        for demo_key in demo_keys:
            demo = data_group[demo_key]
            
            # Load observations
            observations = {}
            if 'observations' in demo:
                for key in demo['observations'].keys():
                    observations[key] = np.array(demo['observations'][key])
            
            # Load actions
            actions = np.array(demo['actions'])
            
            # Load other data
            rewards = np.array(demo['rewards']) if 'rewards' in demo else None
            dones = np.array(demo['dones']) if 'dones' in demo else None
            
            all_data[demo_key] = {
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
            }
            
            print(f"  {demo_key}: {len(actions)} steps")
        
        return all_data


def read_hdf5_iterative(dataset_path: str, obs_keys: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Iterative method: Read dataset one demo at a time.
    
    This is memory-efficient and suitable for large datasets.
    
    Args:
        dataset_path: Path to HDF5 file.
        obs_keys: List of observation keys to extract.
        
    Returns:
        tuple: (observations, actions) as numpy arrays.
    """
    
    print("\n" + "=" * 80)
    print("Method 2: Iterative Reading (Process one demo at a time)")
    print("=" * 80)
    
    all_obs = []
    all_actions = []
    
    with h5py.File(dataset_path, 'r') as f:
        data_group = f['data']
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
        
        for demo_key in demo_keys:
            demo = data_group[demo_key]
            
            # Extract observations
            obs_dict = {}
            for key in obs_keys:
                if key in demo['observations']:
                    obs_dict[key] = np.array(demo['observations'][key])
                else:
                    print(f"Warning: Key '{key}' not found in {demo_key}")
            
            # Extract actions
            actions = np.array(demo['actions'])
            
            # Concatenate observations
            obs_list = []
            for key in obs_keys:
                if key in obs_dict:
                    obs_val = obs_dict[key]
                    # Flatten if needed
                    if len(obs_val.shape) > 2:
                        obs_val = obs_val.reshape(obs_val.shape[0], -1)
                    obs_list.append(obs_val)
            
            # Concatenate all observations
            obs_concat = np.concatenate(obs_list, axis=1)
            
            all_obs.append(obs_concat)
            all_actions.append(actions)
            
            print(f"  {demo_key}: obs shape={obs_concat.shape}, actions shape={actions.shape}")
    
    # Concatenate all demos
    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    
    print(f"\nTotal: obs shape={observations.shape}, actions shape={actions.shape}")
    
    return observations, actions


def read_hdf5_with_stats(dataset_path: str, obs_keys: List[str]) -> Dict:
    """Read dataset and compute statistics.
    
    Args:
        dataset_path: Path to HDF5 file.
        obs_keys: List of observation keys.
        
    Returns:
        dict: Dataset with statistics.
    """
    
    print("\n" + "=" * 80)
    print("Method 3: Reading with Statistics")
    print("=" * 80)
    
    observations, actions = read_hdf5_iterative(dataset_path, obs_keys)
    
    # Compute statistics
    obs_stats = {
        'mean': np.mean(observations, axis=0),
        'std': np.std(observations, axis=0) + 1e-8,
        'min': np.min(observations, axis=0),
        'max': np.max(observations, axis=0),
    }
    
    action_stats = {
        'mean': np.mean(actions, axis=0),
        'std': np.std(actions, axis=0) + 1e-8,
        'min': np.min(actions, axis=0),
        'max': np.max(actions, axis=0),
    }
    
    print(f"\nObservation Statistics:")
    print(f"  Mean: {obs_stats['mean'][:5]}... (showing first 5)")
    print(f"  Std: {obs_stats['std'][:5]}... (showing first 5)")
    print(f"  Min: {obs_stats['min'][:5]}... (showing first 5)")
    print(f"  Max: {obs_stats['max'][:5]}... (showing first 5)")
    
    print(f"\nAction Statistics:")
    print(f"  Mean: {action_stats['mean']}")
    print(f"  Std: {action_stats['std']}")
    print(f"  Min: {action_stats['min']}")
    print(f"  Max: {action_stats['max']}")
    
    return {
        'observations': observations,
        'actions': actions,
        'obs_stats': obs_stats,
        'action_stats': action_stats,
    }


def read_hdf5_pytorch_dataloader(
    dataset_path: str,
    obs_keys: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
):
    """Read dataset using PyTorch DataLoader (compatible with training).
    
    This is similar to what Isaac Lab's training scripts do.
    
    Args:
        dataset_path: Path to HDF5 file.
        obs_keys: List of observation keys.
        batch_size: Batch size for DataLoader.
        shuffle: Whether to shuffle data.
    """
    
    print("\n" + "=" * 80)
    print("Method 4: Using PyTorch DataLoader")
    print("=" * 80)
    
    from torch.utils.data import Dataset, DataLoader
    
    class HDF5Dataset(Dataset):
        """Simple HDF5 dataset for PyTorch DataLoader."""
        
        def __init__(self, dataset_path: str, obs_keys: List[str]):
            self.dataset_path = dataset_path
            self.obs_keys = obs_keys
            
            # Pre-load all data (for simplicity)
            # For large datasets, you should implement lazy loading
            with h5py.File(dataset_path, 'r') as f:
                data_group = f['data']
                demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
                
                all_obs = []
                all_actions = []
                
                for demo_key in demo_keys:
                    demo = data_group[demo_key]
                    
                    # Extract observations
                    obs_list = []
                    for key in obs_keys:
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
        
        def __len__(self):
            return len(self.observations)
        
        def __getitem__(self, idx):
            return (
                torch.from_numpy(self.observations[idx]),
                torch.from_numpy(self.actions[idx]),
            )
    
    # Create dataset and dataloader
    dataset = HDF5Dataset(dataset_path, obs_keys)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for debugging, increase for speed
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Iterate through a few batches
    print(f"\nFirst {min(3, len(dataloader))} batches:")
    for i, (obs_batch, action_batch) in enumerate(dataloader):
        print(f"  Batch {i+1}: obs shape={obs_batch.shape}, action shape={action_batch.shape}")
        if i >= 2:
            break
    
    return dataloader


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Read HDF5 dataset examples")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--method", type=int, default=4, choices=[1, 2, 3, 4],
                        help="Method to use (1=basic, 2=iterative, 3=with_stats, 4=dataloader)")
    
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
    
    # Run selected method
    if args.method == 1:
        data = read_hdf5_basic(args.dataset)
        print(f"\nLoaded {len(data)} demonstrations")
    
    elif args.method == 2:
        observations, actions = read_hdf5_iterative(args.dataset, obs_keys)
        print(f"\nTotal samples: {len(observations)}")
    
    elif args.method == 3:
        data = read_hdf5_with_stats(args.dataset, obs_keys)
        print(f"\nDataset statistics computed")
    
    elif args.method == 4:
        dataloader = read_hdf5_pytorch_dataloader(
            args.dataset,
            obs_keys,
            batch_size=32,
            shuffle=True,
        )
        print(f"\nDataLoader created: {len(dataloader)} batches")


if __name__ == "__main__":
    main()



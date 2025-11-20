# Copyright (c) 2024-2025, SO-ARM100 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect and read HDF5 dataset files.

This script helps you understand the structure of your recorded HDF5 datasets.
It displays the dataset structure, statistics, and allows you to load sample data.

Usage:
    python scripts/inspect_hdf5_dataset.py --dataset ./datasets/pick_place.hdf5
    python scripts/inspect_hdf5_dataset.py --dataset ./datasets/pick_place.hdf5 --demo_idx 0 --show_samples 10
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Any


def print_dict_structure(d: dict, indent: int = 0, max_depth: int = 5):
    """Recursively print dictionary structure."""
    if indent > max_depth:
        return
    
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}: (dict with {len(value)} keys)")
            print_dict_structure(value, indent + 1, max_depth)
        elif isinstance(value, (list, tuple)):
            print(f"{prefix}{key}: {type(value).__name__} with {len(value)} items")
            if len(value) > 0 and isinstance(value[0], dict):
                print_dict_structure(value[0], indent + 1, max_depth)
        elif isinstance(value, np.ndarray):
            print(f"{prefix}{key}: ndarray shape={value.shape}, dtype={value.dtype}")
        elif hasattr(value, 'shape'):
            print(f"{prefix}{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{prefix}{key}: {type(value).__name__} = {value}")


def inspect_hdf5_dataset(
    dataset_path: str,
    demo_idx: int = 0,
    show_samples: int = 5,
    verbose: bool = True,
):
    """Inspect HDF5 dataset structure and contents.
    
    Args:
        dataset_path: Path to HDF5 dataset file.
        demo_idx: Index of demonstration to inspect in detail.
        show_samples: Number of samples to show from each key.
        verbose: If True, print detailed information.
    """
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"=" * 80)
    print(f"Inspecting HDF5 Dataset: {dataset_path}")
    print(f"=" * 80)
    
    with h5py.File(dataset_path, 'r') as f:
        # 1. Show top-level structure
        print(f"\n[Top-level keys]: {list(f.keys())}")
        
        if 'data' not in f:
            print("[ERROR] Dataset does not have 'data' key!")
            return
        
        data_group = f['data']
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
        
        print(f"\n[Number of demonstrations]: {len(demo_keys)}")
        print(f"[Demonstration keys]: {demo_keys[:10]}{'...' if len(demo_keys) > 10 else ''}")
        
        # 2. Inspect first demonstration structure
        if len(demo_keys) == 0:
            print("[ERROR] No demonstrations found!")
            return
        
        first_demo_key = demo_keys[0]
        first_demo = data_group[first_demo_key]
        
        print(f"\n[Structure of '{first_demo_key}']:")
        print(f"  Keys: {list(first_demo.keys())}")
        
        # 3. Inspect observations
        if 'observations' in first_demo:
            obs_group = first_demo['observations']
            print(f"\n[Observations structure]:")
            print(f"  Keys: {list(obs_group.keys())}")
            
            for key in sorted(obs_group.keys()):
                obs_data = np.array(obs_group[key])
                print(f"    '{key}': shape={obs_data.shape}, dtype={obs_data.dtype}")
                if verbose and obs_data.size > 0:
                    print(f"      min={np.min(obs_data):.4f}, max={np.max(obs_data):.4f}, "
                          f"mean={np.mean(obs_data):.4f}, std={np.std(obs_data):.4f}")
                    if obs_data.size <= show_samples * obs_data.shape[-1]:
                        print(f"      sample values: {obs_data.flatten()[:show_samples]}")
        
        # 4. Inspect actions
        if 'actions' in first_demo:
            actions = np.array(first_demo['actions'])
            print(f"\n[Actions]:")
            print(f"  shape={actions.shape}, dtype={actions.dtype}")
            if verbose and actions.size > 0:
                print(f"  min={np.min(actions):.4f}, max={np.max(actions):.4f}, "
                      f"mean={np.mean(actions):.4f}, std={np.std(actions):.4f}")
                print(f"  sample values:\n{actions[:show_samples]}")
        
        # 5. Inspect other keys
        other_keys = ['rewards', 'dones', 'infos']
        for key in other_keys:
            if key in first_demo:
                data = np.array(first_demo[key])
                print(f"\n['{key}']:")
                print(f"  shape={data.shape}, dtype={data.dtype}")
                if verbose and data.size > 0:
                    if key == 'rewards':
                        print(f"  min={np.min(data):.4f}, max={np.max(data):.4f}, "
                              f"mean={np.mean(data):.4f}, sum={np.sum(data):.4f}")
                    else:
                        print(f"  sample values: {data[:show_samples]}")
        
        # 6. Show dataset statistics
        print(f"\n[Dataset Statistics]:")
        total_steps = 0
        action_stats = {'min': [], 'max': [], 'mean': []}
        obs_stats = {}
        
        for demo_key in demo_keys:
            demo = data_group[demo_key]
            actions = np.array(demo['actions'])
            total_steps += len(actions)
            
            # Collect action statistics
            action_stats['min'].append(np.min(actions))
            action_stats['max'].append(np.max(actions))
            action_stats['mean'].append(np.mean(actions))
            
            # Collect observation statistics
            if 'observations' in demo:
                for key in demo['observations'].keys():
                    obs_data = np.array(demo['observations'][key])
                    if key not in obs_stats:
                        obs_stats[key] = {'min': [], 'max': [], 'mean': []}
                    obs_stats[key]['min'].append(np.min(obs_data))
                    obs_stats[key]['max'].append(np.max(obs_data))
                    obs_stats[key]['mean'].append(np.mean(obs_data))
        
        print(f"  Total demonstrations: {len(demo_keys)}")
        print(f"  Total steps: {total_steps}")
        print(f"  Average steps per demo: {total_steps / len(demo_keys):.1f}")
        
        print(f"\n[Action Statistics (across all demos)]:")
        print(f"  min={np.min(action_stats['min']):.4f}, "
              f"max={np.max(action_stats['max']):.4f}, "
              f"mean={np.mean(action_stats['mean']):.4f}")
        
        if obs_stats:
            print(f"\n[Observation Statistics (across all demos)]:")
            for key in sorted(obs_stats.keys()):
                stats = obs_stats[key]
                print(f"  '{key}': min={np.min(stats['min']):.4f}, "
                      f"max={np.max(stats['max']):.4f}, "
                      f"mean={np.mean(stats['mean']):.4f}")
        
        # 7. Show detailed info for specific demonstration
        if demo_idx < len(demo_keys):
            demo_key = demo_keys[demo_idx]
            demo = data_group[demo_key]
            
            print(f"\n{'=' * 80}")
            print(f"Detailed info for '{demo_key}' (demo_idx={demo_idx}):")
            print(f"{'=' * 80}")
            
            # Show actions
            if 'actions' in demo:
                actions = np.array(demo['actions'])
                print(f"\nActions: shape={actions.shape}")
                print(f"  First {show_samples} actions:")
                for i in range(min(show_samples, len(actions))):
                    print(f"    Step {i}: {actions[i]}")
            
            # Show observations
            if 'observations' in demo:
                obs_group = demo['observations']
                print(f"\nObservations:")
                for key in sorted(obs_group.keys()):
                    obs_data = np.array(obs_group[key])
                    print(f"  '{key}': shape={obs_data.shape}")
                    if verbose:
                        print(f"    First sample: {obs_data[0]}")
                        print(f"    Last sample: {obs_data[-1]}")
            
            # Show rewards
            if 'rewards' in demo:
                rewards = np.array(demo['rewards'])
                print(f"\nRewards: shape={rewards.shape}, "
                      f"total={np.sum(rewards):.4f}, "
                      f"mean={np.mean(rewards):.4f}")
            
            # Show dones
            if 'dones' in demo:
                dones = np.array(demo['dones'])
                num_dones = np.sum(dones)
                print(f"\nDones: shape={dones.shape}, "
                      f"number of True: {num_dones}")


def load_hdf5_dataset_sample(
    dataset_path: str,
    demo_idx: int = 0,
    step_idx: int = 0,
) -> dict[str, Any]:
    """Load a single sample from HDF5 dataset.
    
    Args:
        dataset_path: Path to HDF5 dataset file.
        demo_idx: Index of demonstration.
        step_idx: Index of step within demonstration.
        
    Returns:
        dict: Dictionary containing observations and actions.
    """
    
    with h5py.File(dataset_path, 'r') as f:
        data_group = f['data']
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
        
        if demo_idx >= len(demo_keys):
            raise ValueError(f"demo_idx {demo_idx} out of range [0, {len(demo_keys)})")
        
        demo_key = demo_keys[demo_idx]
        demo = data_group[demo_key]
        
        # Load observations
        sample = {}
        if 'observations' in demo:
            sample['observations'] = {}
            for key in demo['observations'].keys():
                obs_data = np.array(demo['observations'][key])
                sample['observations'][key] = obs_data[step_idx]
        
        # Load action
        if 'actions' in demo:
            actions = np.array(demo['actions'])
            sample['action'] = actions[step_idx]
        
        # Load other info
        for key in ['rewards', 'dones']:
            if key in demo:
                data = np.array(demo[key])
                sample[key] = data[step_idx]
        
        return sample


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Inspect HDF5 dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument("--demo_idx", type=int, default=0, help="Index of demonstration to inspect in detail")
    parser.add_argument("--show_samples", type=int, default=5, help="Number of samples to show")
    parser.add_argument("--step_idx", type=int, default=None, help="Load specific step (optional)")
    parser.add_argument("--verbose", action="store_true", default=True, help="Show detailed information")
    
    args = parser.parse_args()
    
    # Inspect dataset
    inspect_hdf5_dataset(
        dataset_path=args.dataset,
        demo_idx=args.demo_idx,
        show_samples=args.show_samples,
        verbose=args.verbose,
    )
    
    # Load specific sample if requested
    if args.step_idx is not None:
        print(f"\n{'=' * 80}")
        print(f"Loading sample: demo_idx={args.demo_idx}, step_idx={args.step_idx}")
        print(f"{'=' * 80}")
        
        sample = load_hdf5_dataset_sample(
            dataset_path=args.dataset,
            demo_idx=args.demo_idx,
            step_idx=args.step_idx,
        )
        
        print("\nSample contents:")
        print_dict_structure(sample, max_depth=3)


if __name__ == "__main__":
    main()



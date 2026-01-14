# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training script for Graph-DiT Policy.

This script provides a complete training framework for your custom Graph-DiT policy.
Users should replace the placeholder GraphDiTPolicy implementation with their own.

Usage:
    ./isaaclab.sh -p scripts/graph_dit/train.py \
        --task SO-ARM101-Pick-Place-DualArm-IK-Abs-v0 \
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
import SO_101.tasks  # noqa: F401  # Register environments
import torch
import torch.optim as optim
from SO_101.policies.graph_dit_policy import GraphDiTPolicy, GraphDiTPolicyCfg
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class HDF5DemoDataset(Dataset):
    """Dataset loader for HDF5 demonstration files - DEMO-LEVEL LOADING.

    CRITICAL CHANGE: Each sample is a COMPLETE DEMO sequence, not a single timestep.
    This preserves temporal coherence and prevents cross-demo contamination.

    Each demo is loaded as a complete sequence with:
    - obs_seq: [T, obs_dim] - full observation sequence
    - action_seq: [T, action_dim] - full action sequence
    - action_trajectory_seq: [T, pred_horizon, action_dim] - future actions for each timestep
    - node_history_seq: [T, history_len, ...] - node features history for each timestep

    Uses Batch-level Padding: Demos in a batch are padded to the max length in that batch.
    """

    def __init__(
        self,
        hdf5_path: str,
        obs_keys: list[str],
        normalize_obs: bool = True,
        normalize_actions: bool = True,
        action_history_length: int = 4,
        pred_horizon: int = 16,
        single_batch_test: bool = False,
        single_batch_size: int = 16,
        skip_first_steps: int = 10,
    ):
        """Initialize dataset.

        Args:
            hdf5_path: Path to HDF5 file.
            obs_keys: List of observation keys to load.
            normalize_obs: If True, normalize observations.
            normalize_actions: If True, normalize actions.
            action_history_length: Number of historical steps for context (default: 4).
            pred_horizon: Number of future action steps to predict per timestep (default: 16).
            skip_first_steps: Number of initial steps to skip per demo (default: 10).
                             Human-collected demos often have noisy initial actions.
        """
        self.hdf5_path = hdf5_path
        self.obs_keys = obs_keys
        self.normalize_obs = normalize_obs
        self.normalize_actions = normalize_actions
        self.action_history_length = action_history_length
        self.pred_horizon = pred_horizon
        self.single_batch_test = single_batch_test
        self.single_batch_size = single_batch_size
        self.skip_first_steps = skip_first_steps

        # Observation key dimensions and offsets
        self.obs_key_dims = {}
        self.obs_key_offsets = {}

        # CRITICAL CHANGE: Store complete demos, not individual timesteps
        self.demos = []  # List of complete demo dictionaries
        self.obs_stats = {}
        self.action_stats = {}
        self.node_stats = {}
        self.subtask_order = []

        print(f"[HDF5DemoDataset] Loading dataset from: {hdf5_path}")
        print(
            f"[HDF5DemoDataset] DEMO-LEVEL LOADING: Each sample is a complete demo sequence"
        )
        print(
            f"[HDF5DemoDataset] Skipping first {skip_first_steps} steps per demo (noisy human actions)"
        )

        # Collect all data for normalization statistics
        all_obs = []
        all_actions = []
        all_ee_nodes = []
        all_object_nodes = []
        all_joint_states = []
        demo_lengths = []

        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
            print(f"[HDF5DemoDataset] Found {len(demo_keys)} demonstrations")

            for demo_key in demo_keys:
                demo = f[f"data/{demo_key}"]

                # Load observations container
                obs_container = demo.get("obs", demo.get("observations", None))
                if obs_container is None:
                    raise ValueError(
                        f"Neither 'obs' nor 'observations' found in {demo_key}"
                    )

                # Compute key dimensions (only once)
                if not self.obs_key_dims:
                    offset = 0
                    for key in obs_keys:
                        if key in obs_container:
                            key_data = np.array(obs_container[key])
                            if len(key_data.shape) == 1:
                                key_dim = 1
                            elif len(key_data.shape) == 2:
                                key_dim = key_data.shape[1]
                            else:
                                key_dim = key_data.shape[-1]
                            self.obs_key_dims[key] = key_dim
                            self.obs_key_offsets[key] = offset
                            offset += key_dim
                            print(
                                f"[HDF5DemoDataset] Key '{key}': dim={key_dim}, offset={self.obs_key_offsets[key]}"
                            )

                # Load all observation keys for this demo
                obs_dict = {}
                for key in obs_keys:
                    if key in obs_container:
                        obs_dict[key] = np.array(obs_container[key])

                # Load original actions from HDF5 (will be replaced with joint_pos[t+5])
                # Original actions from teleoperation may have large gaps, so we use joint_pos[t+5] instead
                actions_full = np.array(demo["actions"]).astype(np.float32)
                T_full = len(actions_full)

                # Skip first N steps (noisy human actions during demo collection)
                skip = min(
                    self.skip_first_steps, T_full - self.pred_horizon - 1
                )  # Ensure enough data remains
                skip = max(0, skip)  # Don't skip negative

                if skip > 0:
                    actions = actions_full[skip:]
                    # Also skip first steps in obs_dict
                    for key in obs_dict:
                        obs_dict[key] = obs_dict[key][skip:]
                else:
                    actions = actions_full

                T = len(actions)
                demo_lengths.append(T)

                # Build complete observation sequence: [T, obs_dim]
                obs_seq = []
                for i in range(T):
                    obs_vec = []
                    for key in obs_keys:
                        if key in obs_dict:
                            obs_val = obs_dict[key][i]
                            if isinstance(obs_val, np.ndarray):
                                obs_vec.append(obs_val.flatten())
                            else:
                                obs_vec.append(np.array([obs_val]))
                    obs_seq.append(np.concatenate(obs_vec).astype(np.float32))
                obs_seq = np.stack(obs_seq, axis=0)  # [T, obs_dim]

                # CRITICAL FIX: Replace actions with joint_pos[t+5] for smoother actions
                # Teleoperation data has large gaps between action and current joint_pos,
                # so using joint_pos[t+5] as action produces smoother, more learnable actions
                jp_start = self.obs_key_offsets.get("joint_pos", 0)
                jp_dim = self.obs_key_dims.get("joint_pos", 6)
                
                # Extract joint_pos sequence from obs_seq
                joint_pos_seq = []
                for i in range(T):
                    obs_i = obs_seq[i]
                    joint_pos = obs_i[jp_start : jp_start + jp_dim]
                    joint_pos = np.pad(
                        joint_pos, (0, max(0, jp_dim - len(joint_pos)))
                    )[:jp_dim]
                    joint_pos_seq.append(joint_pos.astype(np.float32))
                joint_pos_seq = np.stack(joint_pos_seq, axis=0)  # [T, joint_pos_dim]
                
                # Replace actions with joint_pos[t+5] (5 timesteps ahead)
                # For timestep i, action = joint_pos[i+5] (if exists) or last available joint_pos
                actions_new = []
                for i in range(T):
                    target_idx = i + 5
                    if target_idx < T:
                        # Use joint_pos 5 steps ahead as action
                        actions_new.append(joint_pos_seq[target_idx])
                    else:
                        # Beyond sequence: use last available joint_pos
                        actions_new.append(joint_pos_seq[-1])
                actions = np.stack(actions_new, axis=0).astype(np.float32)  # [T, joint_pos_dim]
                
                print(f"[HDF5DemoDataset] ✅ Replaced actions with joint_pos[t+5] for smoother actions")
                print(f"[HDF5DemoDataset]    Original action_dim: {actions_full.shape[1] if len(actions_full.shape) > 1 else 1}")
                print(f"[HDF5DemoDataset]    New action_dim (joint_pos): {jp_dim}")

                # Build action trajectory for each timestep: [T, pred_horizon, action_dim]
                # ACTION CHUNKING: Each timestep predicts next pred_horizon actions
                # CRITICAL: Also build trajectory_mask to mark which horizon steps are valid
                action_trajectory_seq = []
                trajectory_mask_seq = (
                    []
                )  # [T, pred_horizon] - True if valid, False if padding
                for i in range(T):
                    traj = []
                    traj_mask = []
                    for k in range(self.pred_horizon):
                        future_idx = i + k
                        if future_idx < T:
                            traj.append(actions[future_idx])
                            traj_mask.append(True)  # Valid future step
                        else:
                            traj.append(actions[-1])  # Pad with last action
                            traj_mask.append(False)  # Padding (beyond demo end)
                    action_trajectory_seq.append(np.stack(traj, axis=0))
                    trajectory_mask_seq.append(np.array(traj_mask, dtype=bool))
                action_trajectory_seq = np.stack(
                    action_trajectory_seq, axis=0
                )  # [T, pred_horizon, action_dim]
                trajectory_mask_seq = np.stack(
                    trajectory_mask_seq, axis=0
                )  # [T, pred_horizon]

                # Build action history for each timestep: [T, history_len, action_dim]
                action_history_seq = []
                for i in range(T):
                    history = []
                    for j in range(max(0, i - self.action_history_length + 1), i + 1):
                        history.append(actions[j])
                    # Pad with first action if needed
                    while len(history) < self.action_history_length:
                        history.insert(0, actions[0])
                    action_history_seq.append(np.stack(history, axis=0))
                action_history_seq = np.stack(
                    action_history_seq, axis=0
                )  # [T, history_len, action_dim]

                # Build node history sequences (EE and Object): [T, history_len, 7]
                ee_node_history_seq = []
                object_node_history_seq = []
                for i in range(T):
                    ee_hist = []
                    obj_hist = []
                    for j in range(max(0, i - self.action_history_length + 1), i + 1):
                        obs_j = obs_seq[j]
                        # Extract node features
                        obj_pos = obs_j[
                            self.obs_key_offsets.get(
                                "object_position", 12
                            ) : self.obs_key_offsets.get("object_position", 12)
                            + 3
                        ]
                        obj_ori = obs_j[
                            self.obs_key_offsets.get(
                                "object_orientation", 15
                            ) : self.obs_key_offsets.get("object_orientation", 15)
                            + 4
                        ]
                        ee_pos = obs_j[
                            self.obs_key_offsets.get(
                                "ee_position", 19
                            ) : self.obs_key_offsets.get("ee_position", 19)
                            + 3
                        ]
                        ee_ori = obs_j[
                            self.obs_key_offsets.get(
                                "ee_orientation", 22
                            ) : self.obs_key_offsets.get("ee_orientation", 22)
                            + 4
                        ]

                        # Ensure correct shapes
                        obj_pos = np.pad(obj_pos, (0, max(0, 3 - len(obj_pos))))[:3]
                        obj_ori = np.pad(obj_ori, (0, max(0, 4 - len(obj_ori))))[:4]
                        ee_pos = np.pad(ee_pos, (0, max(0, 3 - len(ee_pos))))[:3]
                        ee_ori = np.pad(ee_ori, (0, max(0, 4 - len(ee_ori))))[:4]

                        ee_hist.append(
                            np.concatenate([ee_pos, ee_ori]).astype(np.float32)
                        )
                        obj_hist.append(
                            np.concatenate([obj_pos, obj_ori]).astype(np.float32)
                        )

                    # Pad history if needed
                    while len(ee_hist) < self.action_history_length:
                        ee_hist.insert(
                            0,
                            (
                                ee_hist[0].copy()
                                if ee_hist
                                else np.zeros(7, dtype=np.float32)
                            ),
                        )
                        obj_hist.insert(
                            0,
                            (
                                obj_hist[0].copy()
                                if obj_hist
                                else np.zeros(7, dtype=np.float32)
                            ),
                        )

                    ee_node_history_seq.append(np.stack(ee_hist, axis=0))
                    object_node_history_seq.append(np.stack(obj_hist, axis=0))

                ee_node_history_seq = np.stack(
                    ee_node_history_seq, axis=0
                )  # [T, history_len, 7]
                object_node_history_seq = np.stack(
                    object_node_history_seq, axis=0
                )  # [T, history_len, 7]

                # Build joint states history: [T, history_len, joint_dim]
                # NOTE: Only using joint_pos (removed joint_vel to test if it's noise)
                jp_start = self.obs_key_offsets.get("joint_pos", 0)
                jp_dim = self.obs_key_dims.get("joint_pos", 6)

                joint_states_history_seq = []
                for i in range(T):
                    joint_hist = []
                    for j in range(max(0, i - self.action_history_length + 1), i + 1):
                        obs_j = obs_seq[j]
                        joint_pos = obs_j[jp_start : jp_start + jp_dim]
                        joint_pos = np.pad(
                            joint_pos, (0, max(0, jp_dim - len(joint_pos)))
                        )[:jp_dim]
                        # Only use joint_pos (no joint_vel)
                        joint_hist.append(joint_pos.astype(np.float32))

                    while len(joint_hist) < self.action_history_length:
                        joint_hist.insert(
                            0,
                            (
                                joint_hist[0].copy()
                                if joint_hist
                                else np.zeros(jp_dim, dtype=np.float32)
                            ),
                        )

                    joint_states_history_seq.append(np.stack(joint_hist, axis=0))
                joint_states_history_seq = np.stack(
                    joint_states_history_seq, axis=0
                )  # [T, history_len, joint_dim]

                # Load subtask signals if available
                subtask_condition_seq = None
                if "datagen_info" in obs_container:
                    datagen_info = obs_container["datagen_info"]
                    if "subtask_term_signals" in datagen_info:
                        signals_group = datagen_info["subtask_term_signals"]
                        subtask_order = list(signals_group.keys())
                        if not self.subtask_order:
                            self.subtask_order = subtask_order

                        subtask_condition_seq = []
                        for i in range(T):
                            active_idx = None
                            for idx, name in enumerate(self.subtask_order):
                                if (
                                    name in signals_group
                                    and not np.array(signals_group[name])[i]
                                ):
                                    active_idx = idx
                                break
                            if active_idx is None:
                                active_idx = len(self.subtask_order) - 1
                            cond = np.zeros(len(self.subtask_order), dtype=np.float32)
                            cond[active_idx] = 1.0
                            subtask_condition_seq.append(cond)
                        subtask_condition_seq = np.stack(
                            subtask_condition_seq, axis=0
                        )  # [T, num_subtasks]

                # Store complete demo
                demo_data = {
                    "demo_key": demo_key,
                    "length": T,
                    "obs_seq": obs_seq,  # [T, obs_dim]
                    "action_seq": actions,  # [T, action_dim]
                    "action_trajectory_seq": action_trajectory_seq,  # [T, pred_horizon, action_dim]
                    "trajectory_mask_seq": trajectory_mask_seq,  # [T, pred_horizon] - marks valid horizon steps
                    "action_history_seq": action_history_seq,  # [T, history_len, action_dim]
                    "ee_node_history_seq": ee_node_history_seq,  # [T, history_len, 7]
                    "object_node_history_seq": object_node_history_seq,  # [T, history_len, 7]
                    "joint_states_history_seq": joint_states_history_seq,  # [T, history_len, joint_dim]
                    "subtask_condition_seq": subtask_condition_seq,  # [T, num_subtasks] or None
                }
                self.demos.append(demo_data)

                # Collect for normalization
                all_obs.append(obs_seq)
                all_actions.append(actions)
                all_ee_nodes.append(
                    ee_node_history_seq[:, -1, :]
                )  # Current timestep nodes
                all_object_nodes.append(object_node_history_seq[:, -1, :])
                all_joint_states.append(
                    joint_states_history_seq[:, -1, :]
                )  # Current timestep joints

        # Compute normalization statistics (across all demos)
        all_obs = np.concatenate(all_obs, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        all_ee_nodes = np.concatenate(all_ee_nodes, axis=0)
        all_object_nodes = np.concatenate(all_object_nodes, axis=0)
        all_joint_states = np.concatenate(all_joint_states, axis=0)

        if normalize_obs:
            self.obs_stats["mean"] = np.mean(all_obs, axis=0, keepdims=True)
            self.obs_stats["std"] = np.std(all_obs, axis=0, keepdims=True) + 1e-8

        if normalize_actions:
            self.action_stats["mean"] = np.mean(all_actions, axis=0, keepdims=True)
            self.action_stats["std"] = np.std(all_actions, axis=0, keepdims=True) + 1e-8

        self.node_stats["ee_mean"] = np.mean(all_ee_nodes, axis=0, keepdims=True)
        self.node_stats["ee_std"] = np.std(all_ee_nodes, axis=0, keepdims=True) + 1e-8
        self.node_stats["object_mean"] = np.mean(
            all_object_nodes, axis=0, keepdims=True
        )
        self.node_stats["object_std"] = (
            np.std(all_object_nodes, axis=0, keepdims=True) + 1e-8
        )

        # CRITICAL: Normalize joint states (only joint_pos, joint_vel removed to test if it's noise)
        self.joint_stats = {}
        self.joint_stats["mean"] = np.mean(all_joint_states, axis=0, keepdims=True)
        self.joint_stats["std"] = np.std(all_joint_states, axis=0, keepdims=True) + 1e-8

        # ============================================================================
        # 🚑 SINGLE BATCH TEST MODE: Overfitting test for debugging
        # ============================================================================
        if self.single_batch_test:
            if len(self.demos) == 0:
                raise ValueError(
                    "[SINGLE BATCH TEST] No demos loaded! Cannot create single batch test."
                )
            print(f"\n🚑 [SINGLE BATCH TEST MODE] Enabled!")
            print(f"  Original demos: {len(self.demos)}")
            print(f"  Using ONLY first demo: {self.demos[0]['demo_key']}")
            print(f"  Replicating {self.single_batch_size} times to fill batch")
            # Keep only first demo and replicate it
            first_demo = self.demos[0]
            self.demos = [first_demo] * self.single_batch_size
            print(
                f"  ✅ Single batch dataset created: {len(self.demos)} identical demos"
            )

        # ============================================================================
        # 🚑 SINGLE BATCH TEST MODE: Overfitting test for debugging
        # ============================================================================
        if self.single_batch_test:
            if len(self.demos) == 0:
                raise ValueError(
                    "[SINGLE BATCH TEST] No demos loaded! Cannot create single batch test."
                )
            print(f"\n🚑 [SINGLE BATCH TEST MODE] Enabled!")
            print(f"  Original demos: {len(self.demos)}")
            print(f"  Using ONLY first demo: {self.demos[0]['demo_key']}")
            print(f"  Replicating {self.single_batch_size} times to fill batch")
            # Keep only first demo and replicate it
            first_demo = self.demos[0]
            self.demos = [first_demo] * self.single_batch_size
            print(
                f"  ✅ Single batch dataset created: {len(self.demos)} identical demos"
            )

        print(f"\n[HDF5DemoDataset] Dataset Statistics:")
        print(f"  Total demos: {len(self.demos)}")
        print(
            f"  Demo lengths: min={min(demo_lengths)}, max={max(demo_lengths)}, mean={np.mean(demo_lengths):.1f}"
        )
        print(
            f"  Obs stats: mean shape={self.obs_stats.get('mean', np.array([])).shape}"
        )
        print(
            f"  Action stats: mean shape={self.action_stats.get('mean', np.array([])).shape}"
        )
        print(f"  Joint stats: mean shape={self.joint_stats['mean'].shape}")

        # ============================================================================
        # 🔍 NORMALIZATION STATS CHECK: Detect potential issues
        # ============================================================================
        print(f"\n🔍 [NORMALIZATION STATS CHECK]:")
        if len(self.action_stats) > 0:
            action_mean = self.action_stats["mean"]
            action_std = self.action_stats["std"]
            print(
                f"  Action Mean: min={action_mean.min():.6f}, max={action_mean.max():.6f}, mean={action_mean.mean():.6f}"
            )
            print(
                f"  Action Std:  min={action_std.min():.6f}, max={action_std.max():.6f}, mean={action_std.mean():.6f}"
            )

            # Check for anomalies
            if (action_std < 1e-6).any():
                print(
                    f"  ⚠️  WARNING: Action Std has values < 1e-6! This may cause numerical issues!"
                )
            if (action_std > 100).any():
                print(
                    f"  ⚠️  WARNING: Action Std has values > 100! Normalization may be too aggressive!"
                )
            if (action_std < 0.01).any():
                print(
                    f"  ⚠️  WARNING: Action Std has very small values (< 0.01). Check if this dimension is constant!"
                )

        if len(self.obs_stats) > 0:
            obs_mean = self.obs_stats["mean"]
            obs_std = self.obs_stats["std"]
            print(
                f"  Obs Mean: min={obs_mean.min():.6f}, max={obs_mean.max():.6f}, mean={obs_mean.mean():.6f}"
            )
            print(
                f"  Obs Std:  min={obs_std.min():.6f}, max={obs_std.max():.6f}, mean={obs_std.mean():.6f}"
            )

            if (obs_std < 1e-6).any():
                print(
                    f"  ⚠️  WARNING: Obs Std has values < 1e-6! This may cause numerical issues!"
                )
            if (obs_std > 100).any():
                print(
                    f"  ⚠️  WARNING: Obs Std has values > 100! Normalization may be too aggressive!"
                )

        # ============================================================================
        # 🔍 NORMALIZATION STATS CHECK: Detect potential issues
        # ============================================================================
        print(f"\n🔍 [NORMALIZATION STATS CHECK]:")
        if len(self.action_stats) > 0:
            action_mean = self.action_stats["mean"]
            action_std = self.action_stats["std"]
            print(
                f"  Action Mean: min={action_mean.min():.6f}, max={action_mean.max():.6f}, mean={action_mean.mean():.6f}"
            )
            print(
                f"  Action Std:  min={action_std.min():.6f}, max={action_std.max():.6f}, mean={action_std.mean():.6f}"
            )

            # Check for anomalies
            if (action_std < 1e-6).any():
                print(
                    f"  ⚠️  WARNING: Action Std has values < 1e-6! This may cause numerical issues!"
                )
            if (action_std > 100).any():
                print(
                    f"  ⚠️  WARNING: Action Std has values > 100! Normalization may be too aggressive!"
                )
            if (action_std < 0.01).any():
                print(
                    f"  ⚠️  WARNING: Action Std has very small values (< 0.01). Check if this dimension is constant!"
                )

        if len(self.obs_stats) > 0:
            obs_mean = self.obs_stats["mean"]
            obs_std = self.obs_stats["std"]
            print(
                f"  Obs Mean: min={obs_mean.min():.6f}, max={obs_mean.max():.6f}, mean={obs_mean.mean():.6f}"
            )
            print(
                f"  Obs Std:  min={obs_std.min():.6f}, max={obs_std.max():.6f}, mean={obs_std.mean():.6f}"
            )

            if (obs_std < 1e-6).any():
                print(
                    f"  ⚠️  WARNING: Obs Std has values < 1e-6! This may cause numerical issues!"
                )
            if (obs_std > 100).any():
                print(
                    f"  ⚠️  WARNING: Obs Std has values > 100! Normalization may be too aggressive!"
                )

    def __len__(self):
        """Return number of demos (not timesteps!)."""
        return len(self.demos)

    def __getitem__(self, idx):
        """Return complete demo sequence with all timesteps.

        Returns a dictionary with:
        - obs_seq: [T, obs_dim]
        - action_trajectory_seq: [T, pred_horizon, action_dim]
        - action_history_seq: [T, history_len, action_dim]
        - ee_node_history_seq: [T, history_len, 7]
        - object_node_history_seq: [T, history_len, 7]
        - joint_states_history_seq: [T, history_len, joint_dim]
        - subtask_condition_seq: [T, num_subtasks] or None
        - length: int (original length T)
        """
        demo = self.demos[idx]
        T = demo["length"]

        # Copy and normalize
        obs_seq = demo["obs_seq"].copy()
        action_seq = demo["action_seq"].copy()
        action_trajectory_seq = demo["action_trajectory_seq"].copy()
        action_history_seq = demo["action_history_seq"].copy()
        ee_node_history_seq = demo["ee_node_history_seq"].copy()
        object_node_history_seq = demo["object_node_history_seq"].copy()
        joint_states_history_seq = demo["joint_states_history_seq"].copy()
        subtask_condition_seq = (
            demo["subtask_condition_seq"].copy()
            if demo["subtask_condition_seq"] is not None
            else None
        )

        # Normalize observations: [T, obs_dim]
        if self.normalize_obs and len(self.obs_stats) > 0:
            obs_seq = (obs_seq - self.obs_stats["mean"]) / self.obs_stats["std"]

        # Normalize actions: [T, action_dim] and [T, pred_horizon, action_dim] and [T, hist_len, action_dim]
        if self.normalize_actions and len(self.action_stats) > 0:
            action_seq = (action_seq - self.action_stats["mean"]) / self.action_stats[
                "std"
            ]
            action_trajectory_seq = (
                action_trajectory_seq - self.action_stats["mean"]
            ) / self.action_stats["std"]
            action_history_seq = (
                action_history_seq - self.action_stats["mean"]
            ) / self.action_stats["std"]

        # Normalize node features: [T, hist_len, 7]
        if len(self.node_stats) > 0:
            ee_node_history_seq = (
                ee_node_history_seq - self.node_stats["ee_mean"]
            ) / self.node_stats["ee_std"]
            object_node_history_seq = (
                object_node_history_seq - self.node_stats["object_mean"]
            ) / self.node_stats["object_std"]

        # CRITICAL: Normalize joint states: [T, hist_len, joint_dim]
        if len(self.joint_stats) > 0:
            joint_states_history_seq = (
                joint_states_history_seq - self.joint_stats["mean"]
            ) / self.joint_stats["std"]

        return {
            "obs_seq": torch.from_numpy(obs_seq).float(),  # [T, obs_dim]
            "action_seq": torch.from_numpy(action_seq).float(),  # [T, action_dim]
            "action_trajectory_seq": torch.from_numpy(
                action_trajectory_seq
            ).float(),  # [T, pred_horizon, action_dim]
            "action_history_seq": torch.from_numpy(
                action_history_seq
            ).float(),  # [T, hist_len, action_dim]
            "ee_node_history_seq": torch.from_numpy(
                ee_node_history_seq
            ).float(),  # [T, hist_len, 7]
            "object_node_history_seq": torch.from_numpy(
                object_node_history_seq
            ).float(),  # [T, hist_len, 7]
            "joint_states_history_seq": torch.from_numpy(
                joint_states_history_seq
            ).float(),  # [T, hist_len, joint_dim]
            "subtask_condition_seq": (
                torch.from_numpy(subtask_condition_seq).float()
                if subtask_condition_seq is not None
                else None
            ),
            "length": T,
        }

    def get_obs_stats(self):
        return self.obs_stats

    def get_action_stats(self):
        return self.action_stats

    def get_joint_stats(self):
        return self.joint_stats

    def get_node_stats(self):
        return self.node_stats


def demo_collate_fn(batch):
    """Collate function for variable-length demo sequences.

    CRITICAL: Pads demos to max length in batch, creates attention mask.

    Args:
        batch: List of demo dictionaries from dataset

        Returns:
        Dictionary with padded tensors and mask:
        - obs_seq: [B, max_T, obs_dim]
        - action_trajectory_seq: [B, max_T, pred_horizon, action_dim]
        - action_history_seq: [B, max_T, hist_len, action_dim]
        - ee_node_history_seq: [B, max_T, hist_len, 7]
        - object_node_history_seq: [B, max_T, hist_len, 7]
        - joint_states_history_seq: [B, max_T, hist_len, joint_dim]
        - subtask_condition_seq: [B, max_T, num_subtasks] or None
        - lengths: [B] - original lengths
        - mask: [B, max_T] - True for valid timesteps, False for padding
    """
    B = len(batch)
    lengths = [item["length"] for item in batch]
    max_T = max(lengths)

    # Get dimensions from first item
    obs_dim = batch[0]["obs_seq"].shape[-1]
    action_dim = batch[0]["action_seq"].shape[-1]
    pred_horizon = batch[0]["action_trajectory_seq"].shape[1]
    hist_len = batch[0]["action_history_seq"].shape[1]
    joint_dim = batch[0]["joint_states_history_seq"].shape[-1]
    has_subtask = batch[0]["subtask_condition_seq"] is not None
    num_subtasks = batch[0]["subtask_condition_seq"].shape[-1] if has_subtask else 0

    # Initialize padded tensors
    obs_seq = torch.zeros(B, max_T, obs_dim)
    action_seq = torch.zeros(B, max_T, action_dim)
    action_trajectory_seq = torch.zeros(B, max_T, pred_horizon, action_dim)
    action_history_seq = torch.zeros(B, max_T, hist_len, action_dim)
    ee_node_history_seq = torch.zeros(B, max_T, hist_len, 7)
    object_node_history_seq = torch.zeros(B, max_T, hist_len, 7)
    joint_states_history_seq = torch.zeros(B, max_T, hist_len, joint_dim)
    subtask_condition_seq = torch.zeros(B, max_T, num_subtasks) if has_subtask else None

    # Create masks:
    # - mask: [B, max_T] - True for valid timesteps (demo-level padding)
    # - trajectory_mask: [B, max_T, pred_horizon] - True for valid horizon steps (handles "half-cut" data)
    mask = torch.zeros(B, max_T, dtype=torch.bool)
    trajectory_mask = torch.zeros(B, max_T, pred_horizon, dtype=torch.bool)

    # Fill in data
    for i, item in enumerate(batch):
        T = item["length"]
        mask[i, :T] = True  # Valid timesteps

        obs_seq[i, :T] = item["obs_seq"]
        action_seq[i, :T] = item["action_seq"]
        action_trajectory_seq[i, :T] = item["action_trajectory_seq"]
        action_history_seq[i, :T] = item["action_history_seq"]
        ee_node_history_seq[i, :T] = item["ee_node_history_seq"]
        object_node_history_seq[i, :T] = item["object_node_history_seq"]
        joint_states_history_seq[i, :T] = item["joint_states_history_seq"]
        if has_subtask and item["subtask_condition_seq"] is not None:
            subtask_condition_seq[i, :T] = item["subtask_condition_seq"]

        # CRITICAL: Fill trajectory_mask for "half-cut" data handling
        # For each timestep t, mark which horizon steps k are valid (t+k < T)
        if "trajectory_mask_seq" in item:
            trajectory_mask[i, :T] = torch.from_numpy(item["trajectory_mask_seq"])
        else:
            # Fallback: if trajectory_mask_seq not available, create it
            for t in range(T):
                for k in range(pred_horizon):
                    if t + k < T:
                        trajectory_mask[i, t, k] = True
                    else:
                        trajectory_mask[i, t, k] = False

    return {
        "obs_seq": obs_seq,  # [B, max_T, obs_dim]
        "action_seq": action_seq,  # [B, max_T, action_dim]
        "action_trajectory_seq": action_trajectory_seq,  # [B, max_T, pred_horizon, action_dim]
        "action_history_seq": action_history_seq,  # [B, max_T, hist_len, action_dim]
        "ee_node_history_seq": ee_node_history_seq,  # [B, max_T, hist_len, 7]
        "object_node_history_seq": object_node_history_seq,  # [B, max_T, hist_len, 7]
        "joint_states_history_seq": joint_states_history_seq,  # [B, max_T, hist_len, joint_dim]
        "subtask_condition_seq": subtask_condition_seq,  # [B, max_T, num_subtasks] or None
        "lengths": torch.tensor(lengths),  # [B]
        "mask": mask,  # [B, max_T] - demo-level padding mask
        "trajectory_mask": trajectory_mask,  # [B, max_T, pred_horizon] - horizon-level mask for "half-cut" data
    }


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
    mode: str = "flow_matching",
    pred_horizon: int = 16,
    exec_horizon: int = 8,
    lr_schedule: str = "constant",
    skip_first_steps: int = 10,
):
    """Train Graph-DiT Policy with Action Chunking.

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
        pred_horizon: Prediction horizon for action chunking (default: 16).
        exec_horizon: Execution horizon for receding horizon control (default: 8).
        lr_schedule: Learning rate schedule: "constant" (stable) or "cosine" (warmup + cosine annealing).
    """

    # Create directories with timestamp and mode suffix
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Add mode suffix to save/log directories (e.g., reach_joint_flow_matching)
    base_save_dir = Path(save_dir)
    base_log_dir = Path(log_dir)

    # Append mode suffix to the base directory name
    # e.g., ./logs/graph_dit/reach_joint -> ./logs/graph_dit/reach_joint_flow_matching
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
    tensorboard_dir = log_dir / "tensorboard"
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"[Train] 📊 TensorBoard logs will be saved to: {tensorboard_dir}")

    print(f"[Train] ===== Graph-DiT Policy Training with ACTION CHUNKING =====")
    print(f"[Train] Task: {task_name}")
    print(f"[Train] Dataset: {dataset_path}")
    print(f"[Train] Mode: {mode.upper()} (1-10 steps, fast inference)")
    print(f"[Train] Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"[Train] Action history length: {action_history_length}")
    print(
        f"[Train] ACTION CHUNKING: pred_horizon={pred_horizon}, exec_horizon={exec_horizon}"
    )

    # Load dataset with DEMO-LEVEL loading (complete sequences per demo)
    print(f"\n[Train] Loading dataset with DEMO-LEVEL loading...")
    print(
        f"[Train] CRITICAL: Each sample is a complete demo sequence (not individual timesteps)"
    )
    # ============================================================================
    # 🚑 SINGLE BATCH TEST MODE: Enable for debugging overfitting
    # ============================================================================
    # 🚑 SINGLE BATCH TEST MODE: Enable for debugging overfitting
    # ============================================================================
    # Set to True to test if model can overfit a single demo (classic debugging test)
    # This helps identify if the issue is code logic or data complexity
    single_batch_test = os.getenv("SINGLE_BATCH_TEST", "False").lower() == "true"
    single_batch_size = int(os.getenv("SINGLE_BATCH_SIZE", "16"))

    if single_batch_test:
        print(f"\n🚑 [SINGLE BATCH TEST MODE] Enabled via environment variable!")
        print(
            f"  This will use ONLY the first demo, replicated {single_batch_size} times"
        )
        print(
            f"  Expected outcome: Loss should drop to < 0.001 if code logic is correct"
        )
        print(
            f"  If loss stays > 0.05, there's likely a bug in model architecture or data flow\n"
        )

    dataset = HDF5DemoDataset(
        dataset_path,
        obs_keys,
        normalize_obs=True,
        normalize_actions=True,
        action_history_length=action_history_length,
        pred_horizon=pred_horizon,
        single_batch_test=single_batch_test,
        single_batch_size=single_batch_size,
        skip_first_steps=skip_first_steps,
    )

    # Get normalization stats for saving
    obs_stats = dataset.get_obs_stats()
    action_stats = dataset.get_action_stats()
    node_stats = dataset.get_node_stats()
    joint_stats = dataset.get_joint_stats()

    # Get actual dimensions from first demo
    sample = dataset[0]
    # obs_seq: [T, obs_dim], action_trajectory_seq: [T, pred_horizon, action_dim]
    actual_obs_dim = sample["obs_seq"].shape[-1]
    actual_action_dim = sample["action_seq"].shape[-1]
    actual_pred_horizon = sample["action_trajectory_seq"].shape[1]
    sample_T = sample["length"]

    print(f"[Train] Sample demo length: {sample_T} timesteps")
    print(f"[Train] Obs shape: [T, {actual_obs_dim}]")
    print(
        f"[Train] Action trajectory shape: [T, {actual_pred_horizon}, {actual_action_dim}]"
    )

    # Check if subtask condition is available
    num_subtasks = (
        len(dataset.subtask_order)
        if hasattr(dataset, "subtask_order") and dataset.subtask_order
        else 0
    )
    if num_subtasks > 0:
        print(
            f"[Train] Subtask condition available: {num_subtasks} subtasks ({dataset.subtask_order})"
        )
    else:
        print(f"[Train] No subtask condition found in dataset")

    print(
        f"[Train] Actual obs dim: {actual_obs_dim}, Actual action dim: {actual_action_dim}"
    )

    # Update dimensions if needed
    if obs_dim != actual_obs_dim:
        print(
            f"[Train] Warning: obs_dim mismatch ({obs_dim} vs {actual_obs_dim}), using {actual_obs_dim}"
        )
        obs_dim = actual_obs_dim
    if action_dim != actual_action_dim:
        print(
            f"[Train] Warning: action_dim mismatch ({action_dim} vs {actual_action_dim}), using {actual_action_dim}"
        )
        action_dim = actual_action_dim

    # Create dataloader with demo_collate_fn for variable-length sequence padding
    # IMPORTANT: batch_size now means number of demos per batch, not timesteps!
    # Recommend smaller batch_size (e.g., 4-16) since each demo is ~100 timesteps
    demo_batch_size = min(batch_size, 16)  # Cap at 16 demos per batch for memory
    print(f"[Train] Demo batch size: {demo_batch_size} demos per batch")
    print(
        f"[Train] (Each demo has ~{sample_T} timesteps, so effective batch ≈ {demo_batch_size * sample_T} timesteps)"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=demo_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=demo_collate_fn,  # CRITICAL: Use demo-level collate function
    )

    # Get number of subtasks from dataset
    num_subtasks = (
        len(dataset.subtask_order)
        if hasattr(dataset, "subtask_order") and dataset.subtask_order
        else 0
    )

    # Infer joint_dim from dataset (only joint_pos, joint_vel removed to test if it's noise)
    joint_dim = 0
    if hasattr(dataset, "obs_key_dims"):
        if "joint_pos" in dataset.obs_key_dims:
            joint_dim += dataset.obs_key_dims["joint_pos"]
        # NOTE: joint_vel removed - testing if it's noise
        # if "joint_vel" in dataset.obs_key_dims:
        #     joint_dim += dataset.obs_key_dims["joint_vel"]
    if joint_dim == 0:
        joint_dim = None

    # CRITICAL FIX: Build obs_structure from dataset's obs_key_offsets and obs_key_dims
    # This replaces hardcoded indices with dynamic dictionary-based structure
    obs_structure = None
    if hasattr(dataset, "obs_key_offsets") and hasattr(dataset, "obs_key_dims"):
        obs_structure = {}
        for key in dataset.obs_key_offsets.keys():
            offset = dataset.obs_key_offsets[key]
            dim = dataset.obs_key_dims[key]
            obs_structure[key] = (offset, offset + dim)
        print(f"[Train] ✅ Using dynamic obs_structure from dataset:")
        for key, (start, end) in obs_structure.items():
            print(f"    {key}: [{start}, {end}) (dim={end-start})")
    else:
        print(f"[Train] ⚠️  Dataset doesn't have obs_key_offsets, using default hardcoded indices")

    # Create policy configuration
    # IMPORTANT: num_subtasks must match the actual subtask_condition dimension in data
    # If dataset has subtasks, use that number; otherwise disable subtask conditioning
    cfg = GraphDiTPolicyCfg(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dims[0] if hidden_dims else 256,
        num_layers=num_layers,
        num_heads=num_heads,
        joint_dim=joint_dim,
        num_subtasks=num_subtasks,  # Use actual number from dataset (0 = no subtasks)
        action_history_length=action_history_length,
        pred_horizon=pred_horizon,  # ACTION CHUNKING: predict this many future steps
        exec_horizon=exec_horizon,  # RHC: execute this many steps before re-planning
        mode=mode,  # "flow_matching"
        device=device,
        obs_structure=obs_structure,  # CRITICAL: Pass dynamic obs_structure instead of hardcoded indices
    )

    if num_subtasks > 0:
        print(
            f"[Train] Policy configured with {num_subtasks} subtasks: {dataset.subtask_order}"
        )
        print(f"[Train] Subtask conditioning will be used during training.")
    else:
        print(
            f"[Train] No subtask info found, subtask conditioning disabled (num_subtasks=0)"
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

    # Learning rate scheduler
    import math

    if lr_schedule == "cosine":
        # Warmup + Cosine Annealing (original)
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
        print(
            f"[Train] Using COSINE schedule: {warmup_epochs} warmup epochs ({100*warmup_epochs/num_epochs:.1f}%), then cosine annealing"
        )
    else:
        # Constant learning rate (stable, no scheduling)
        # This is more reliable for debugging and initial training
        scheduler = None
        print(
            f"[Train] Using CONSTANT learning rate: {learning_rate} (no warmup, no decay)"
        )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float("inf")

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n[Train] Resuming from checkpoint: {resume_checkpoint}")
        # weights_only=False is needed for PyTorch 2.6+ to load custom config classes
        checkpoint = torch.load(
            resume_checkpoint, map_location=device, weights_only=False
        )
        policy.load_state_dict(checkpoint["policy_state_dict"])

        # Try to load optimizer and scheduler states (may not exist in best_model.pt)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"[Train] Loaded optimizer state from checkpoint")
        else:
            print(
                f"[Train] Warning: No optimizer state in checkpoint, starting with fresh optimizer"
            )

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"[Train] Loaded scheduler state from checkpoint")

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", checkpoint.get("loss", float("inf")))
        print(f"[Train] Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}")

    # Training loop - DEMO-LEVEL TRAINING
    print(f"\n[Train] Starting DEMO-LEVEL training for {num_epochs} epochs...")
    print(f"[Train] CRITICAL: Each batch contains complete demo sequences")
    print(
        f"[Train] Loss is computed over all timesteps in each demo, with padding masked out"
    )
    policy.train()

    # Track last epoch loss for final model
    last_epoch_loss = None

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        epoch_mse_losses = []
        epoch_total_timesteps = 0  # Track actual timesteps processed

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # batch is a dictionary from demo_collate_fn
            # All tensors have shape [B, max_T, ...] where max_T is batch-specific
            obs_seq = batch["obs_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, obs_dim]
            action_trajectory_seq = batch["action_trajectory_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, pred_horizon, action_dim]
            action_history_seq = batch["action_history_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, hist_len, action_dim]
            ee_node_history_seq = batch["ee_node_history_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, hist_len, 7]
            object_node_history_seq = batch["object_node_history_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, hist_len, 7]
            joint_states_history_seq = batch["joint_states_history_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, hist_len, joint_dim]
            subtask_condition_seq = batch["subtask_condition_seq"]
            if subtask_condition_seq is not None:
                subtask_condition_seq = subtask_condition_seq.to(
                    device, non_blocking=True
                )  # [B, max_T, num_subtasks]
            lengths = batch["lengths"]  # [B] - original lengths
            mask = batch["mask"].to(
                device, non_blocking=True
            )  # [B, max_T] - True for valid timesteps
            trajectory_mask = batch["trajectory_mask"].to(
                device, non_blocking=True
            )  # [B, max_T, pred_horizon] - horizon-level mask

            B, max_T = obs_seq.shape[:2]
            total_valid_timesteps = mask.sum().item()
            epoch_total_timesteps += total_valid_timesteps

            # Flatten batch for per-timestep processing: [B * max_T, ...]
            # This allows us to process all timesteps in parallel
            obs_flat = obs_seq.reshape(B * max_T, -1)  # [B*max_T, obs_dim]
            actions_flat = action_trajectory_seq.reshape(
                B * max_T, pred_horizon, -1
            )  # [B*max_T, pred_horizon, action_dim]
            action_history_flat = action_history_seq.reshape(
                B * max_T, action_history_length, -1
            )  # [B*max_T, hist_len, action_dim]
            ee_node_history_flat = ee_node_history_seq.reshape(
                B * max_T, action_history_length, -1
            )  # [B*max_T, hist_len, 7]
            object_node_history_flat = object_node_history_seq.reshape(
                B * max_T, action_history_length, -1
            )  # [B*max_T, hist_len, 7]
            joint_states_history_flat = joint_states_history_seq.reshape(
                B * max_T, action_history_length, -1
            )  # [B*max_T, hist_len, joint_dim]
            subtask_condition_flat = None
            if subtask_condition_seq is not None:
                subtask_condition_flat = subtask_condition_seq.reshape(
                    B * max_T, -1
                )  # [B*max_T, num_subtasks]
            trajectory_mask_flat = trajectory_mask.reshape(
                B * max_T, pred_horizon
            )  # [B*max_T, pred_horizon] - horizon-level mask

            # Forward pass - compute loss for ALL timesteps, then mask out padding
            loss_dict = policy.loss(
                obs_flat,
                actions_flat,
                action_history=action_history_flat,
                ee_node_history=ee_node_history_flat,
                object_node_history=object_node_history_flat,
                joint_states_history=joint_states_history_flat,
                subtask_condition=subtask_condition_flat,
                mask=trajectory_mask_flat,  # CRITICAL: Pass horizon-level mask for "half-cut" data handling
            )
            loss = loss_dict["total_loss"]

            # ============================================================================
            # 🕵️ DEBUG: Forensic Analysis - Print diagnostic info
            # ============================================================================
            if batch_idx == 0 and epoch % 100 == 0:  # Print every 100 epochs
                print(f"\n[DEBUG EPOCH {epoch}] ====================")

                # Get debug info if available
                if "debug" in loss_dict:
                    debug = loss_dict["debug"]

                    if "v_pred" in debug and "v_t" in debug:
                        # Flow Matching mode
                        v_pred = debug["v_pred"]
                        v_t = debug["v_t"]
                        actions = debug["actions"]

                        print(f"[Flow Matching Debug]")
                        print(
                            f"  Target (v_t) Mean: {v_t.mean().item():.4f}, Std: {v_t.std().item():.4f}"
                        )
                        print(
                            f"  Target (v_t) Min: {v_t.min().item():.4f}, Max: {v_t.max().item():.4f}"
                        )
                        print(
                            f"  Pred (v_pred) Mean: {v_pred.mean().item():.4f}, Std: {v_pred.std().item():.4f}"
                        )
                        print(
                            f"  Pred (v_pred) Min: {v_pred.min().item():.4f}, Max: {v_pred.max().item():.4f}"
                        )
                        print(
                            f"  Input Action (Data) Min: {actions.min().item():.4f}, Max: {actions.max().item():.4f}"
                        )

                        # Check loss per dimension
                        diff = (v_pred - v_t) ** 2
                        if len(diff.shape) == 3:  # [batch, pred_horizon, action_dim]
                            mse_per_dim = diff.mean(dim=(0, 1)).detach().cpu().numpy()
                            print(f"  Mean MSE per action dim: {mse_per_dim}")

                        # Check time t
                        if "t" in debug:
                            t = debug["t"]
                            print(
                                f"  Time t: Min={t.min().item():.4f}, Max={t.max().item():.4f}, Mean={t.mean().item():.4f}"
                            )

                else:
                    # Fallback: print basic info
                    print(f"  Loss: {loss.item():.6f}")
                    print(f"  Actions shape: {actions_flat.shape}")
                    print(
                        f"  Actions range: [{actions_flat.min().item():.4f}, {actions_flat.max().item():.4f}]"
                    )
                    print(f"  Enable DEBUG_LOSS=true to see detailed diagnostics")

                print("========================================\n")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            # Record losses
            epoch_losses.append(loss.item())
            epoch_mse_losses.append(loss_dict["mse_loss"].item())

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "mse": loss_dict["mse_loss"].item(),
                    "valid_t": total_valid_timesteps,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            # Log to tensorboard
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar(
                "Train/MSE_Loss", loss_dict["mse_loss"].item(), global_step
            )
            writer.add_scalar(
                "Train/LearningRate", optimizer.param_groups[0]["lr"], global_step
            )

        # Update learning rate (if scheduler is enabled)
        if scheduler is not None:
            scheduler.step()

        # Compute epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_mse_loss = np.mean(epoch_mse_losses)

        print(
            f"\n[Epoch {epoch+1}/{num_epochs}] "
            f"Loss: {avg_loss:.6f}, MSE: {avg_mse_loss:.6f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Log to tensorboard
        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Epoch/AverageLoss", avg_loss, epoch)
        writer.add_scalar("Epoch/AverageMSE", avg_mse_loss, epoch)
        writer.add_scalar("Epoch/LearningRate", current_lr, epoch)

        # Track last epoch loss
        last_epoch_loss = avg_loss

        # Save best model only (overwrite if better)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / "best_model.pt"
            # Save with stats for inference
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "cfg": cfg,
                    "obs_stats": obs_stats,
                    "action_stats": action_stats,
                    "node_stats": node_stats,  # CRITICAL: For node feature normalization
                    "joint_stats": joint_stats,  # CRITICAL: For joint state normalization
                    "epoch": epoch,
                    "loss": avg_loss,
                },
                str(best_path),
            )
            print(
                f"[Train] ✅ Saved best model (loss: {best_loss:.6f}, epoch: {epoch+1}) to: {best_path}"
            )

    # Save final model (after all epochs)
    final_loss = last_epoch_loss if last_epoch_loss is not None else best_loss
    final_path = save_dir / "final_model.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "cfg": cfg,
            "obs_stats": obs_stats,
            "action_stats": action_stats,
            "node_stats": node_stats,  # CRITICAL: For node feature normalization
            "joint_stats": joint_stats,  # CRITICAL: For joint state normalization
            "epoch": num_epochs - 1,
            "loss": final_loss,
        },
        str(final_path),
    )
    print(f"\n[Train] ===== Training Completed! =====")
    print(f"[Train] ✅ Final model saved to: {final_path}")
    print(f"[Train] ✅ Best model saved to: {save_dir / 'best_model.pt'}")
    print(f"[Train] 📁 All models saved in: {save_dir}")
    print(f"[Train] 📊 TensorBoard logs in: {tensorboard_dir}")
    print(f"[Train] 💡 View logs with: tensorboard --logdir {log_dir}")

    writer.close()
    return policy


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Graph-DiT Policy")

    # Dataset arguments
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--dataset", type=str, required=True, help="HDF5 dataset path")

    # Model arguments
    parser.add_argument(
        "--obs_dim",
        type=int,
        default=32,
        help="Observation dimension (reach task: 32 after removing redundant target_object_position)",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=6,
        help="Action dimension (now uses joint_pos[t+5], typically 6 for 6-DoF arm)",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--action_history_length",
        type=int,
        default=4,
        help="Number of historical actions to use (default: 4)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="flow_matching",
        choices=["flow_matching"],
        help="Training mode: flow_matching (1-10 steps, fast inference)",
    )

    # ACTION CHUNKING arguments (Diffusion Policy's key innovation)
    parser.add_argument(
        "--pred_horizon",
        type=int,
        default=16,
        help="Prediction horizon: number of future action steps to predict (default: 16)",
    )
    parser.add_argument(
        "--exec_horizon",
        type=int,
        default=8,
        help="Execution horizon: number of steps to execute before re-planning (default: 8)",
    )

    # Data preprocessing
    parser.add_argument(
        "--skip_first_steps",
        type=int,
        default=10,
        help="Skip first N steps of each demo (noisy human actions, default: 10)",
    )

    # Learning rate schedule
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine"],
        help="Learning rate schedule: 'constant' (stable) or 'cosine' (warmup + cosine annealing)",
    )

    # Paths
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./logs/graph_dit",
        help="Checkpoint save directory",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs/graph_dit", help="Log directory"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

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
        pred_horizon=args.pred_horizon,
        exec_horizon=args.exec_horizon,
        lr_schedule=args.lr_schedule,
        skip_first_steps=args.skip_first_steps,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()

# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
"""
Standalone HDF5 demo dataset and collate for Graph-DiT.

No SO_101 / Isaac Sim dependencies so scripts like visualize_loss_landscape.py
can run with plain Python (no omni.physics). Used by both train.py and
visualize_loss_landscape.py.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


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
        obs_keys: list,
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
            single_batch_test: If True, keep only first demo replicated for overfitting test.
            single_batch_size: Replication count when single_batch_test is True.
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

        self.obs_key_dims = {}
        self.obs_key_offsets = {}
        self.demos = []
        self.obs_stats = {}
        self.action_stats = {}
        self.node_stats = {}
        self.joint_stats = {}
        self.subtask_order = []

        print(f"[HDF5DemoDataset] Loading dataset from: {hdf5_path}")
        print(
            f"[HDF5DemoDataset] DEMO-LEVEL LOADING: Each sample is a complete demo sequence"
        )
        print(
            f"[HDF5DemoDataset] Skipping first {skip_first_steps} steps per demo (noisy human actions)"
        )

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

                obs_container = demo.get("obs", demo.get("observations", None))
                if obs_container is None:
                    raise ValueError(
                        f"Neither 'obs' nor 'observations' found in {demo_key}"
                    )

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

                obs_dict = {}
                for key in obs_keys:
                    if key in obs_container:
                        obs_dict[key] = np.array(obs_container[key])

                actions_full = np.array(demo["actions"]).astype(np.float32)
                T_full = len(actions_full)

                skip = min(
                    self.skip_first_steps, T_full - self.pred_horizon - 1
                )
                skip = max(0, skip)

                if skip > 0:
                    actions = actions_full[skip:]
                    for key in obs_dict:
                        obs_dict[key] = obs_dict[key][skip:]
                else:
                    actions = actions_full

                T = len(actions)
                demo_lengths.append(T)

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
                obs_seq = np.stack(obs_seq, axis=0)

                jp_start = self.obs_key_offsets.get("joint_pos", 0)
                jp_dim = self.obs_key_dims.get("joint_pos", 6)

                joint_pos_seq = []
                for i in range(T):
                    obs_i = obs_seq[i]
                    joint_pos = obs_i[jp_start : jp_start + jp_dim]
                    joint_pos = np.pad(
                        joint_pos, (0, max(0, jp_dim - len(joint_pos)))
                    )[:jp_dim]
                    joint_pos_seq.append(joint_pos.astype(np.float32))
                joint_pos_seq = np.stack(joint_pos_seq, axis=0)

                actions_new = []
                for i in range(T):
                    target_idx = i + 5
                    if target_idx < T:
                        actions_new.append(joint_pos_seq[target_idx])
                    else:
                        actions_new.append(joint_pos_seq[-1])
                actions = np.stack(actions_new, axis=0).astype(np.float32)

                # Graph-DiT 与 Unet 一致：保持 6 维 (5 arm + 1 gripper)，便于对比
                print(
                    f"[HDF5DemoDataset] ✅ Replaced actions with joint_pos[t+5] for smoother actions"
                )
                print(
                    f"[HDF5DemoDataset]    Original action_dim: {actions_full.shape[1] if len(actions_full.shape) > 1 else 1}"
                )
                print(f"[HDF5DemoDataset]    Action dim: {actions.shape[1]} (5 arm + 1 gripper)")

                action_trajectory_seq = []
                trajectory_mask_seq = []
                for i in range(T):
                    traj = []
                    traj_mask = []
                    for k in range(self.pred_horizon):
                        future_idx = i + k
                        if future_idx < T:
                            traj.append(actions[future_idx])
                            traj_mask.append(True)
                        else:
                            traj.append(actions[-1])
                            traj_mask.append(False)
                    action_trajectory_seq.append(np.stack(traj, axis=0))
                    trajectory_mask_seq.append(np.array(traj_mask, dtype=bool))
                action_trajectory_seq = np.stack(action_trajectory_seq, axis=0)
                trajectory_mask_seq = np.stack(trajectory_mask_seq, axis=0)

                action_history_seq = []
                for i in range(T):
                    history = []
                    for j in range(max(0, i - self.action_history_length + 1), i + 1):
                        history.append(actions[j])
                    while len(history) < self.action_history_length:
                        history.insert(0, actions[0])
                    action_history_seq.append(np.stack(history, axis=0))
                action_history_seq = np.stack(action_history_seq, axis=0)

                ee_node_history_seq = []
                object_node_history_seq = []
                for i in range(T):
                    ee_hist = []
                    obj_hist = []
                    for j in range(max(0, i - self.action_history_length + 1), i + 1):
                        obs_j = obs_seq[j]
                        obj_pos = obs_j[
                            self.obs_key_offsets.get("object_position", 12)
                            : self.obs_key_offsets.get("object_position", 12)
                            + 3
                        ]
                        obj_ori = obs_j[
                            self.obs_key_offsets.get("object_orientation", 15)
                            : self.obs_key_offsets.get("object_orientation", 15)
                            + 4
                        ]
                        ee_pos = obs_j[
                            self.obs_key_offsets.get("ee_position", 19)
                            : self.obs_key_offsets.get("ee_position", 19)
                            + 3
                        ]
                        ee_ori = obs_j[
                            self.obs_key_offsets.get("ee_orientation", 22)
                            : self.obs_key_offsets.get("ee_orientation", 22)
                            + 4
                        ]

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

                    while len(ee_hist) < self.action_history_length:
                        ee_hist.insert(
                            0,
                            ee_hist[0].copy() if ee_hist else np.zeros(7, dtype=np.float32),
                        )
                        obj_hist.insert(
                            0,
                            obj_hist[0].copy() if obj_hist else np.zeros(7, dtype=np.float32),
                        )

                    ee_node_history_seq.append(np.stack(ee_hist, axis=0))
                    object_node_history_seq.append(np.stack(obj_hist, axis=0))

                ee_node_history_seq = np.stack(ee_node_history_seq, axis=0)
                object_node_history_seq = np.stack(object_node_history_seq, axis=0)

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
                        joint_hist.append(joint_pos.astype(np.float32))

                    while len(joint_hist) < self.action_history_length:
                        joint_hist.insert(
                            0,
                            joint_hist[0].copy()
                            if joint_hist
                            else np.zeros(jp_dim, dtype=np.float32),
                        )

                    joint_states_history_seq.append(np.stack(joint_hist, axis=0))
                joint_states_history_seq = np.stack(
                    joint_states_history_seq, axis=0
                )

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
                        )

                demo_data = {
                    "demo_key": demo_key,
                    "length": T,
                    "obs_seq": obs_seq,
                    "action_seq": actions,
                    "action_trajectory_seq": action_trajectory_seq,
                    "trajectory_mask_seq": trajectory_mask_seq,
                    "action_history_seq": action_history_seq,
                    "ee_node_history_seq": ee_node_history_seq,
                    "object_node_history_seq": object_node_history_seq,
                    "joint_states_history_seq": joint_states_history_seq,
                    "subtask_condition_seq": subtask_condition_seq,
                }
                self.demos.append(demo_data)

                all_obs.append(obs_seq)
                all_actions.append(actions)
                all_ee_nodes.append(ee_node_history_seq[:, -1, :])
                all_object_nodes.append(object_node_history_seq[:, -1, :])
                all_joint_states.append(joint_states_history_seq[:, -1, :])

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

        self.joint_stats["mean"] = np.mean(all_joint_states, axis=0, keepdims=True)
        self.joint_stats["std"] = np.std(all_joint_states, axis=0, keepdims=True) + 1e-8

        if self.single_batch_test:
            if len(self.demos) == 0:
                raise ValueError(
                    "[SINGLE BATCH TEST] No demos loaded! Cannot create single batch test."
                )
            print(f"\n🚑 [SINGLE BATCH TEST MODE] Enabled!")
            print(f"  Original demos: {len(self.demos)}")
            print(f"  Using ONLY first demo: {self.demos[0]['demo_key']}")
            print(f"  Replicating {self.single_batch_size} times to fill batch")
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

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, idx):
        demo = self.demos[idx]
        T = demo["length"]

        obs_seq = demo["obs_seq"].copy()
        action_seq = demo["action_seq"].copy()
        action_trajectory_seq = demo["action_trajectory_seq"].copy()
        trajectory_mask_seq = demo["trajectory_mask_seq"].copy()
        action_history_seq = demo["action_history_seq"].copy()
        ee_node_history_seq = demo["ee_node_history_seq"].copy()
        object_node_history_seq = demo["object_node_history_seq"].copy()
        joint_states_history_seq = demo["joint_states_history_seq"].copy()
        subtask_condition_seq = (
            demo["subtask_condition_seq"].copy()
            if demo["subtask_condition_seq"] is not None
            else None
        )

        if self.normalize_obs and len(self.obs_stats) > 0:
            obs_seq = (obs_seq - self.obs_stats["mean"]) / self.obs_stats["std"]

        if self.normalize_actions and len(self.action_stats) > 0:
            action_seq = (action_seq - self.action_stats["mean"]) / self.action_stats["std"]
            action_trajectory_seq = (
                action_trajectory_seq - self.action_stats["mean"]
            ) / self.action_stats["std"]
            action_history_seq = (
                action_history_seq - self.action_stats["mean"]
            ) / self.action_stats["std"]

        if len(self.node_stats) > 0:
            ee_node_history_seq = (
                ee_node_history_seq - self.node_stats["ee_mean"]
            ) / self.node_stats["ee_std"]
            object_node_history_seq = (
                object_node_history_seq - self.node_stats["object_mean"]
            ) / self.node_stats["object_std"]

        if len(self.joint_stats) > 0:
            joint_states_history_seq = (
                joint_states_history_seq - self.joint_stats["mean"]
            ) / self.joint_stats["std"]

        return {
            "obs_seq": torch.from_numpy(obs_seq).float(),
            "action_seq": torch.from_numpy(action_seq).float(),
            "action_trajectory_seq": torch.from_numpy(action_trajectory_seq).float(),
            "trajectory_mask_seq": trajectory_mask_seq,
            "action_history_seq": torch.from_numpy(action_history_seq).float(),
            "ee_node_history_seq": torch.from_numpy(ee_node_history_seq).float(),
            "object_node_history_seq": torch.from_numpy(object_node_history_seq).float(),
            "joint_states_history_seq": torch.from_numpy(
                joint_states_history_seq
            ).float(),
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

    Pads demos to max length in batch, creates mask and trajectory_mask.
    """
    B = len(batch)
    lengths = [item["length"] for item in batch]
    max_T = max(lengths)

    obs_dim = batch[0]["obs_seq"].shape[-1]
    action_dim = batch[0]["action_seq"].shape[-1]
    pred_horizon = batch[0]["action_trajectory_seq"].shape[1]
    hist_len = batch[0]["action_history_seq"].shape[1]
    joint_dim = batch[0]["joint_states_history_seq"].shape[-1]
    has_subtask = batch[0]["subtask_condition_seq"] is not None
    num_subtasks = batch[0]["subtask_condition_seq"].shape[-1] if has_subtask else 0

    obs_seq = torch.zeros(B, max_T, obs_dim)
    action_seq = torch.zeros(B, max_T, action_dim)
    action_trajectory_seq = torch.zeros(B, max_T, pred_horizon, action_dim)
    action_history_seq = torch.zeros(B, max_T, hist_len, action_dim)
    ee_node_history_seq = torch.zeros(B, max_T, hist_len, 7)
    object_node_history_seq = torch.zeros(B, max_T, hist_len, 7)
    joint_states_history_seq = torch.zeros(B, max_T, hist_len, joint_dim)
    subtask_condition_seq = torch.zeros(B, max_T, num_subtasks) if has_subtask else None

    mask = torch.zeros(B, max_T, dtype=torch.bool)
    trajectory_mask = torch.zeros(B, max_T, pred_horizon, dtype=torch.bool)

    for i, item in enumerate(batch):
        T = item["length"]
        mask[i, :T] = True

        obs_seq[i, :T] = item["obs_seq"]
        action_seq[i, :T] = item["action_seq"]
        action_trajectory_seq[i, :T] = item["action_trajectory_seq"]
        action_history_seq[i, :T] = item["action_history_seq"]
        ee_node_history_seq[i, :T] = item["ee_node_history_seq"]
        object_node_history_seq[i, :T] = item["object_node_history_seq"]
        joint_states_history_seq[i, :T] = item["joint_states_history_seq"]
        if has_subtask and item["subtask_condition_seq"] is not None:
            subtask_condition_seq[i, :T] = item["subtask_condition_seq"]

        if "trajectory_mask_seq" in item:
            trajectory_mask[i, :T] = torch.from_numpy(item["trajectory_mask_seq"])
        else:
            for t in range(T):
                for k in range(pred_horizon):
                    trajectory_mask[i, t, k] = (t + k < T)

    return {
        "obs_seq": obs_seq,
        "action_seq": action_seq,
        "action_trajectory_seq": action_trajectory_seq,
        "action_history_seq": action_history_seq,
        "ee_node_history_seq": ee_node_history_seq,
        "object_node_history_seq": object_node_history_seq,
        "joint_states_history_seq": joint_states_history_seq,
        "subtask_condition_seq": subtask_condition_seq,
        "lengths": torch.tensor(lengths),
        "mask": mask,
        "trajectory_mask": trajectory_mask,
    }

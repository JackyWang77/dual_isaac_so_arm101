# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback script for trained Graph-Unet Policy.

This script loads a trained Graph-Unet policy and runs it in an Isaac Lab environment.

Usage:
    ./isaaclab.sh -p scripts/graph_unet/play.py \
        --task SO-ARM101-Pick-Place-DualArm-IK-Abs-v0 \
        --checkpoint ./logs/graph_unet/best_model.pt \
        --num_envs 64 --num_batches 2
    (Default: num_envs=64, num_batches=2 -> 128 episodes; success rate is reported.)
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Create argument parser first
parser = argparse.ArgumentParser(description="Play Graph-Unet Policy with Gripper Model")
parser.add_argument("--task", type=str, required=True, help="Task name")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to Graph-Unet checkpoint")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes (default: num_envs * num_batches)")
parser.add_argument("--num_batches", type=int, default=2, help="Number of batches for success-rate eval (total episodes = num_envs * num_batches if num_episodes not set)")
parser.add_argument(
    "--num_diffusion_steps", 
    type=int, 
    default=None, 
    help="Number of flow matching inference steps (default: uses checkpoint config, typically 10). Fewer steps = less lag, more steps = smoother trajectory."
)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Max episode length in seconds (default: use task config, e.g. 3.0 for lift). Shorter = faster reset.",
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="unet",
    choices=["unet", "graph_unet"],
    help="Policy class to use (default: unet)",
)

# Append AppLauncher CLI args (this will add --device automatically)
AppLauncher.add_app_launcher_args(parser)

# Parse arguments (but don't use them yet, will parse again later)
args_cli, unknown = parser.parse_known_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys

import gymnasium as gym
import numpy as np
import SO_101.tasks  # noqa: F401  # Register environments
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from SO_101.policies.graph_unet_policy import UnetPolicy, GraphUnetPolicy
from SO_101.policies.dual_arm_unet_policy import DualArmUnetPolicy

# Try to import visualization utilities (if available)
try:
    from omni.isaac.debug_draw import DebugDraw
    DEBUG_DRAW_AVAILABLE = True
except ImportError:
    DEBUG_DRAW_AVAILABLE = False
    print("[Play] DebugDraw not available, visualization will be disabled")


def play_graph_unet_policy(
    task_name: str,
    checkpoint_path: str,
    num_envs: int = 64,
    num_episodes: int | None = None,
    num_batches: int = 2,
    device: str = "cuda",
    num_diffusion_steps: int | None = None,
    episode_length_s: float | None = None,
    policy_type: str = "unet",
):
    """Play trained Graph-Unet policy.
    
    Args:
        task_name: Environment task name.
        checkpoint_path: Path to trained policy checkpoint.
        num_envs: Number of parallel environments.
        num_episodes: Number of episodes to run (default: num_envs * num_batches).
        num_batches: Number of batches for success-rate eval (used when num_episodes is None).
        device: Device to run on.
    """
    if num_episodes is None:
        num_episodes = num_envs * num_batches
        print(f"[Play] num_episodes not set, using num_envs * num_batches = {num_episodes}")
    
    print(f"[Play] ===== Graph-Unet Policy Playback =====")
    print(f"[Play] Task: {task_name}")
    print(f"[Play] Checkpoint: {checkpoint_path}")
    print(f"[Play] Num envs: {num_envs}")
    
    # Create environment
    print(f"\n[Play] Creating environment...")
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    if episode_length_s is not None:
        env_cfg.episode_length_s = episode_length_s
        print(f"[Play] Override episode_length_s = {episode_length_s}")
    env = gym.make(task_name, cfg=env_cfg)
    
    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space
    
    print(f"[Play] Observation space: {obs_space}")
    print(f"[Play] Action space: {action_space}")
    
    # Compute observation dimension
    def get_obs_dim(space):
        """Recursively compute observation dimension from space."""
        if hasattr(space, "shape") and space.shape is not None:
            # Box space with shape
            if isinstance(space.shape, tuple):
                # For vectorized env, shape might be (num_envs, dim), take the last dim
                if len(space.shape) > 1:
                    return space.shape[-1]  # Last dimension is the actual obs dim
                else:
                    return space.shape[0]
            else:
                return space.shape if isinstance(space.shape, int) else 1
        elif hasattr(space, "spaces"):
            # Dict or Tuple space
            if isinstance(space.spaces, dict):
                # For Isaac Lab, if 'policy' key exists, use only that (training uses policy obs)
                if "policy" in space.spaces:
                    return get_obs_dim(space.spaces["policy"])
                else:
                    # Otherwise, sum all sub-spaces
                    return sum(
                        get_obs_dim(sub_space) for sub_space in space.spaces.values()
                    )
            elif isinstance(space.spaces, (list, tuple)):
                # Tuple space: sum all sub-spaces
                return sum(get_obs_dim(sub_space) for sub_space in space.spaces)
            else:
                return 0
        else:
            # Unknown space type, try to get from sample
            try:
                sample = space.sample()
                if isinstance(sample, dict):
                    # If 'policy' key exists, use only that
                    if "policy" in sample:
                        policy_sample = sample["policy"]
                        if hasattr(policy_sample, "size"):
                            return (
                                policy_sample.size
                                if isinstance(policy_sample.size, int)
                                else policy_sample.numel()
                            )
                        elif hasattr(policy_sample, "__len__"):
                            return (
                                len(policy_sample)
                                if isinstance(policy_sample, (list, tuple))
                                else (
                                    policy_sample.shape[-1]
                                    if hasattr(policy_sample, "shape")
                                    else 1
                                )
                            )
                    else:
                        return sum(
                            (
                                s.size
                                if hasattr(s, "size") and isinstance(s.size, int)
                                else (
                                    s.numel()
                                    if hasattr(s, "numel")
                                    else (
                                        len(s)
                                        if hasattr(s, "__len__")
                                        and isinstance(s, (list, tuple))
                                        else 1
                                    )
                                )
                            )
                            for s in sample.values()
                        )
                elif hasattr(sample, "size"):
                    return (
                        sample.size if isinstance(sample.size, int) else sample.numel()
                    )
                elif hasattr(sample, "__len__"):
                    if isinstance(sample, (list, tuple)):
                        return len(sample)
                    elif hasattr(sample, "shape"):
                        return sample.shape[-1] if len(sample.shape) > 0 else 1
                    return len(sample)
                else:
                    return 1
            except Exception as e:
                print(
                    f"[Play] Warning: Could not compute obs_dim from space, using default. Error: {e}"
                )
                return 39  # Default fallback for reach task
    
    obs_dim = get_obs_dim(obs_space)
    
    # For action space, handle vectorized case
    if hasattr(action_space, "shape") and action_space.shape is not None:
        if isinstance(action_space.shape, tuple) and len(action_space.shape) > 1:
            action_dim = action_space.shape[
                -1
            ]  # Last dimension is the actual action dim
        else:
            action_dim = (
                action_space.shape[0]
                if isinstance(action_space.shape, tuple)
                else action_space.shape
            )
    else:
        action_dim = 6  # Default fallback: env action dim (5 arm + 1 gripper)
    
    # Policy outputs 6-dim (5 arm + 1 gripper); env gets gripper mapped to -1/1; buffer keeps raw
    print(f"[Play] Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Load policy and normalization stats
    print(f"\n[Play] Loading policy from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("cfg", None)

    # Policy class from checkpoint cfg: arm_action_dim set → DualArmUnetPolicy
    if cfg is not None and getattr(cfg, "arm_action_dim", None) is not None:
        PolicyClass = DualArmUnetPolicy
    elif policy_type == "graph_unet":
        PolicyClass = GraphUnetPolicy
    else:
        PolicyClass = UnetPolicy
    print(f"[Play] Policy type: {PolicyClass.__name__}")
    policy = PolicyClass.load(checkpoint_path, device=device)
    policy.eval()
    
    # Policy outputs 6-dim (5 arm + 1 gripper); gripper mapped to -1/1 only for env.step
    print(f"[Play] Using Graph-Unet for all action dimensions (including gripper)")
    
    # Determine mode and set default diffusion steps if not provided
    if cfg is not None:
        mode = getattr(cfg, "mode", "flow_matching")
        print(f"[Play] Policy mode: {mode.upper()}")
        # Use num_inference_steps from checkpoint config if available (default: 10 for real-time)
        if num_diffusion_steps is None:
            num_diffusion_steps = getattr(cfg, "num_inference_steps", 10)
            print(f"[Play] Using inference steps from checkpoint config: {num_diffusion_steps}")
    else:
        # Fallback: use 30 as default (matches training default)
        if num_diffusion_steps is None:
            num_diffusion_steps = 30  # Default for Flow Matching (fewer steps = smoother real-time, less lag)
            print(f"[Play] Using default inference steps: {num_diffusion_steps}")
    
    # Load normalization stats (if available)
    obs_stats = checkpoint.get("obs_stats", None)
    action_stats = checkpoint.get("action_stats", None)
    
    if obs_stats is not None:
        print(f"[Play] Loaded observation normalization stats")
        # Handle both numpy arrays and torch tensors
        if isinstance(obs_stats["mean"], np.ndarray):
            obs_mean = torch.from_numpy(obs_stats["mean"]).squeeze().to(device)
            obs_std = torch.from_numpy(obs_stats["std"]).squeeze().to(device)
        else:
            obs_mean = obs_stats["mean"].squeeze().to(device)
            obs_std = obs_stats["std"].squeeze().to(device)
    else:
        print(f"[Play] Warning: No observation stats found, skipping normalization")
        obs_mean = None
        obs_std = None
    
    if action_stats is not None:
        print(f"[Play] Loaded action normalization stats")
        # Handle both numpy arrays and torch tensors
        if isinstance(action_stats["mean"], np.ndarray):
            action_mean = torch.from_numpy(action_stats["mean"]).squeeze().to(device)
            action_std = torch.from_numpy(action_stats["std"]).squeeze().to(device)
        else:
            action_mean = action_stats["mean"].squeeze().to(device)
            action_std = action_stats["std"].squeeze().to(device)
    else:
        print(f"[Play] Warning: No action stats found, skipping denormalization")
        action_mean = None
        action_std = None
    
    # CRITICAL FIX: Load node feature normalization stats
    node_stats = checkpoint.get("node_stats", None)
    if node_stats is not None:
        print(f"[Play] Loaded node feature normalization stats")
        # Handle both numpy arrays and torch tensors
        if isinstance(node_stats["ee_mean"], np.ndarray):
            ee_node_mean = torch.from_numpy(node_stats["ee_mean"]).squeeze().to(device)
            ee_node_std = torch.from_numpy(node_stats["ee_std"]).squeeze().to(device)
            object_node_mean = (
                torch.from_numpy(node_stats["object_mean"]).squeeze().to(device)
            )
            object_node_std = (
                torch.from_numpy(node_stats["object_std"]).squeeze().to(device)
            )
        else:
            ee_node_mean = node_stats["ee_mean"].squeeze().to(device)
            ee_node_std = node_stats["ee_std"].squeeze().to(device)
            object_node_mean = node_stats["object_mean"].squeeze().to(device)
            object_node_std = node_stats["object_std"].squeeze().to(device)
    else:
        print(
            f"[Play] Warning: No node stats found, node features will NOT be normalized!"
        )
        print(
            f"[Play] This may cause poor performance - retrain with node normalization!"
        )
        ee_node_mean = None
        ee_node_std = None
        object_node_mean = None
        object_node_std = None

    # CRITICAL FIX: Load joint state normalization stats
    joint_stats = checkpoint.get("joint_stats", None)
    if joint_stats is not None:
        print(f"[Play] Loaded joint state normalization stats")
        # Handle both numpy arrays and torch tensors
        if isinstance(joint_stats["mean"], np.ndarray):
            joint_mean = torch.from_numpy(joint_stats["mean"]).squeeze().to(device)
            joint_std = torch.from_numpy(joint_stats["std"]).squeeze().to(device)
        else:
            joint_mean = joint_stats["mean"].squeeze().to(device)
            joint_std = joint_stats["std"].squeeze().to(device)
    else:
        print(
            f"[Play] Warning: No joint stats found, joint states will NOT be normalized!"
        )
        print(
            f"[Play] This may cause poor performance - retrain with joint normalization!"
        )
        joint_mean = None
        joint_std = None

    # Load obs_key_offsets for dynamic extraction (matches train's obs_keys order)
    obs_key_offsets = checkpoint.get("obs_key_offsets", None)
    obs_key_dims = checkpoint.get("obs_key_dims", None)
    if obs_key_offsets is not None:
        print(f"[Play] Loaded obs_key_offsets for dynamic extraction")
    else:
        print(f"[Play] No obs_key_offsets in checkpoint, using fallback indices")

    # Get config values from policy
    node_configs = getattr(policy.cfg, "node_configs", None)
    use_node_histories = (
        node_configs is not None and len(node_configs) > 2
    )
    if use_node_histories:
        print(f"[Play] Using {len(node_configs)}-node graph (node_histories)")
    else:
        print(f"[Play] Using 2-node graph (ee + object_node_history)")

    # obs_key_offsets: from checkpoint (train's dataset), or fallback for old checkpoints
    _obs_offsets = obs_key_offsets if obs_key_offsets else {
        "left_joint_pos": 0, "left_joint_vel": 6, "right_joint_pos": 12, "right_joint_vel": 18,
        "left_ee_position": 24, "left_ee_orientation": 27,
        "right_ee_position": 31, "right_ee_orientation": 34,
        "cube_1_pos": 38, "cube_1_ori": 41, "cube_2_pos": 45, "cube_2_ori": 48,
        "object_position": 12, "object_orientation": 15,
        "ee_position": 19, "ee_orientation": 22,
        "joint_pos": 0,
    }
    _obs_dims = obs_key_dims if obs_key_dims else {
        "left_ee_position": 3, "left_ee_orientation": 4, "right_ee_position": 3, "right_ee_orientation": 4,
        "cube_1_pos": 3, "cube_1_ori": 4, "cube_2_pos": 3, "cube_2_ori": 4,
        "object_position": 3, "object_orientation": 4, "ee_position": 3, "ee_orientation": 4,
        "left_joint_pos": 6, "right_joint_pos": 6, "joint_pos": 6,
    }

    action_history_length = (
        policy.cfg.action_history_length
        if hasattr(policy.cfg, "action_history_length")
        else 4
    )
    joint_dim = policy.cfg.joint_dim if hasattr(policy.cfg, "joint_dim") else None
    pred_horizon = (
        policy.cfg.pred_horizon if hasattr(policy.cfg, "pred_horizon") else 16
    )
    exec_horizon = policy.cfg.exec_horizon if hasattr(policy.cfg, "exec_horizon") else 8
    
    # Run episodes
    print(f"\n[Play] Running {num_episodes} episodes...")
    print(f"[Play] Action history length: {action_history_length}")
    print(
        f"[Play] ACTION CHUNKING: pred_horizon={pred_horizon}, exec_horizon={exec_horizon}"
    )
    print(
        f"[Play] Inference frequency: every {exec_horizon} steps (vs every step without chunking)"
    )
    
    obs, info = env.reset()
    episode_count = 0
    episode_rewards = []
    episode_success = []
    current_episode_rewards = torch.zeros(num_envs, device=device)
    
    # Success rate: 与 train 一致，仅当 Episode_Termination/success 且 非 object_dropping 时计为成功
    def _get_success_flags(info):
        """从 step 返回的 info 取 success [num_envs]。success=True 且 非 object_dropping 才算成功。"""
        if not isinstance(info, dict):
            return None

        def _to_mask(arr):
            if arr is None:
                return None
            if isinstance(arr, torch.Tensor):
                t = arr.to(device).float()
            elif isinstance(arr, np.ndarray):
                t = torch.from_numpy(arr).float().to(device)
            else:
                return None
            if t.dim() == 2 and t.shape[1] == 1:
                t = t.squeeze(-1)
            if t.dim() != 1 or t.shape[0] != num_envs:
                return None
            return (t > 0.5) if t.dtype != torch.bool else t

        log = info.get("log") or info.get("extra_info")
        if not isinstance(log, dict):
            return None
        success_mask = None
        for key in ("Episode_Termination/success", "success", "episode_success", "termination_success"):
            if key in log:
                success_mask = _to_mask(log[key])
                if success_mask is not None:
                    break
        if success_mask is None:
            return None
        drop_key = "Episode_Termination/object_dropping"
        if drop_key in log:
            drop_mask = _to_mask(log[drop_key])
            if drop_mask is not None:
                success_mask = success_mask & (~drop_mask)
        return success_mask

    # 成功率：与 play_graph_rl / train 一致，仅用 truncated + 高度
    _obj_pos_key = "cube_1_pos" if "cube_1_pos" in _obs_offsets else "object_position"
    OBJ_HEIGHT_IDX = _obs_offsets.get(_obj_pos_key, 12) + 2  # z index
    SUCCESS_HEIGHT = 0.10  # 成功阈值 (m)
    print(f"[Play] Success: timeout=失败, 否则 height>={SUCCESS_HEIGHT}m=成功 (与 play_graph_rl 一致)")

    # Initialize action, node, and joint history buffers for each environment
    action_history_buffers = [
        torch.zeros(action_history_length, action_dim, device=device)
        for _ in range(num_envs)
    ]
    ee_node_history_buffers = (
        None
        if use_node_histories
        else [
            torch.zeros(action_history_length, 7, device=device)
            for _ in range(num_envs)
        ]
    )
    object_node_history_buffers = (
        None
        if use_node_histories
        else [
            torch.zeros(action_history_length, 7, device=device)
            for _ in range(num_envs)
        ]
    )
    node_history_buffers = (
        [
            torch.zeros(len(node_configs), action_history_length, 7, device=device)
            for _ in range(num_envs)
        ]
        if use_node_histories
        else None
    )
    joint_state_history_buffers = (
        [
            torch.zeros(action_history_length, joint_dim, device=device)
            for _ in range(num_envs)
        ]
        if joint_dim is not None
        else None
    )
    
    # ==========================================================================
    # EMA SMOOTHING for action smoothing (joints only, gripper excluded)
    # ==========================================================================
    ema_alpha = 1.0  # EMA weight: 1=no smoothing (for testing), 0.5=default smoother
    ema_smoothed_joints = None  # Will be initialized on first action [num_envs, joint_dim]
    print(f"[Play] EMA smoothing enabled: alpha={ema_alpha} (joints only, gripper excluded)")

    # ==========================================================================
    # RECEDING HORIZON CONTROL (RHC) - Action Buffers
    # ==========================================================================
    # Instead of predicting one action per step, we predict pred_horizon actions,
    # execute exec_horizon of them, then re-predict.
    # This provides:
    # 1. Temporal consistency (actions within a chunk are smooth)
    # 2. Lower computational load (predict every exec_horizon steps, not every step)
    # 3. Better real-time performance for 50Hz control

    # Action buffer for each environment: stores predicted trajectory
    # Shape: [exec_horizon, action_dim] - we only store exec_horizon actions to execute
    action_buffers = [
        [] for _ in range(num_envs)
    ]  # List of lists for dynamic management
    action_buffers_normalized = [
        [] for _ in range(num_envs)
    ]
    
    step_count = 0
    
    def _extract_node_features_from_obs(obs_tensor):
        """Extract EE and Object node features using obs_key_offsets from checkpoint."""
        def _slice(key, size):
            off = _obs_offsets.get(key, 0)
            return obs_tensor[:, off : off + size]
        if "left_ee_position" in _obs_offsets:
            ee_pos = _slice("left_ee_position", 3)
            ee_ori = _slice("left_ee_orientation", 4)
            obj_pos = _slice("right_ee_position", 3)
            obj_ori = _slice("right_ee_orientation", 4)
        else:
            obj_pos = _slice("object_position", 3)
            obj_ori = _slice("object_orientation", 4)
            ee_pos = _slice("ee_position", 3)
            ee_ori = _slice("ee_orientation", 4)
        ee_node = torch.cat([ee_pos, ee_ori], dim=-1)
        object_node = torch.cat([obj_pos, obj_ori], dim=-1)
        return ee_node, object_node

    def _extract_node_features_multi(obs_tensor):
        """Extract all node features for N-node graph using obs_key_offsets."""
        nodes = []
        for nc in node_configs:
            pos_off = _obs_offsets.get(nc["pos_key"], 0)
            ori_off = _obs_offsets.get(nc["ori_key"], 0)
            pos_dim = _obs_dims.get(nc["pos_key"], 3)
            ori_dim = _obs_dims.get(nc["ori_key"], 4)
            pos = obs_tensor[:, pos_off : pos_off + pos_dim]
            ori = obs_tensor[:, ori_off : ori_off + ori_dim]
            if pos_dim < 3:
                pos = torch.nn.functional.pad(pos, (0, 3 - pos_dim))
            elif pos_dim > 3:
                pos = pos[:, :3]
            if ori_dim < 4:
                ori = torch.nn.functional.pad(ori, (0, 4 - ori_dim))
            elif ori_dim > 4:
                ori = ori[:, :4]
            nodes.append(torch.cat([pos, ori], dim=-1))
        return torch.stack(nodes, dim=1)

    def _extract_joint_states_from_obs(obs_tensor):
        """Extract joint position using obs_key_offsets from checkpoint."""
        if joint_dim == 12 and "left_joint_pos" in _obs_offsets and "right_joint_pos" in _obs_offsets:
            left_dim = _obs_dims.get("left_joint_pos", 6)
            right_dim = _obs_dims.get("right_joint_pos", 6)
            left_jp = obs_tensor[:, _obs_offsets["left_joint_pos"] : _obs_offsets["left_joint_pos"] + left_dim]
            right_jp = obs_tensor[:, _obs_offsets["right_joint_pos"] : _obs_offsets["right_joint_pos"] + right_dim]
            if left_dim < 6:
                left_jp = torch.nn.functional.pad(left_jp, (0, 6 - left_dim))
            elif left_dim > 6:
                left_jp = left_jp[:, :6]
            if right_dim < 6:
                right_jp = torch.nn.functional.pad(right_jp, (0, 6 - right_dim))
            elif right_dim > 6:
                right_jp = right_jp[:, :6]
            return torch.cat([left_jp, right_jp], dim=-1)
        off = _obs_offsets.get("joint_pos", 0)
        dim = _obs_dims.get("joint_pos", 6)
        return obs_tensor[:, off : off + dim]
    
    with torch.inference_mode():
        while simulation_app.is_running() and episode_count < num_episodes:
            # Process observations
            if isinstance(obs, dict):
                # Check if there's a 'policy' key (Isaac Lab wrapper format)
                if "policy" in obs:
                    # Direct policy observations (already concatenated)
                    # But it might still contain target_object_position (39 dims)
                    # We need to extract only the keys we used during training (32 dims)
                    obs_val = obs["policy"]
                    if isinstance(obs_val, torch.Tensor):
                        obs_tensor_raw = obs_val.to(device)
                    elif isinstance(obs_val, np.ndarray):
                        obs_tensor_raw = torch.from_numpy(obs_val).to(device)
                    else:
                        obs_tensor_raw = torch.tensor(obs_val, device=device)
                    
                    # Ensure correct shape [num_envs, obs_dim]
                    if len(obs_tensor_raw.shape) == 1:
                        obs_tensor_raw = obs_tensor_raw.unsqueeze(0)
                    elif len(obs_tensor_raw.shape) > 2:
                        obs_tensor_raw = obs_tensor_raw.view(
                            obs_tensor_raw.shape[0], -1
                        )
                    
                    # Check if dimensions match training (32) or raw env (39 with target_object_position)
                    # If it's 39, we need to remove target_object_position (7 dims: positions 3 + orientations 4)
                    # Training order: joint_pos(6), joint_vel(6), object_position(3), object_orientation(4),
                    #                  ee_position(3), ee_orientation(4), actions(6) = 32 dims
                    # Raw env order:   joint_pos(6), joint_vel(6), object_position(3), object_orientation(4),
                    #                  ee_position(3), ee_orientation(4), target_object_position(7), actions(6) = 39 dims
                    if obs_tensor_raw.shape[1] == 39:
                        # Remove target_object_position (indices 26:33, which is after ee_orientation and before actions)
                        # Keep: [0:26] (joint_pos, joint_vel, object_pos, object_ori, ee_pos, ee_ori)
                        # and [33:39] (actions)
                        obs_tensor = torch.cat(
                            [
                                obs_tensor_raw[
                                    :, :26
                                ],  # Everything before target_object_position
                                obs_tensor_raw[
                                    :, 33:
                                ],  # Everything after target_object_position (actions)
                            ],
                            dim=1,
                        )  # Should be 32 dims
                    elif obs_tensor_raw.shape[1] in (32, 64):
                        # 32: single-arm lift/reach; 64: dual-arm stack
                        obs_tensor = obs_tensor_raw
                    else:
                        # Unknown dimension, use as is (might cause issues)
                        print(
                            f"[Play] Warning: Unexpected obs dim {obs_tensor_raw.shape[1]}, expected 32, 39, or 64"
                        )
                        obs_tensor = obs_tensor_raw
                else:
                    # Concatenate dictionary observations in the correct order
                    # Skip target_object_position even if it exists (redundant with object_position + object_orientation)
                    obs_keys_order = [
                        "joint_pos",
                        "joint_vel",
                        "object_position",
                        "object_orientation",
                        "ee_position",
                        "ee_orientation",
                        # "target_object_position",  # Skip: redundant with object_position + object_orientation
                        "actions",  # last action (if available)
                    ]
                    obs_list = []
                    for key in obs_keys_order:
                        if key in obs:
                            obs_val = obs[key]
                            if isinstance(obs_val, torch.Tensor):
                                obs_list.append(obs_val.flatten(start_dim=1))
                            elif isinstance(obs_val, np.ndarray):
                                obs_list.append(
                                    torch.from_numpy(obs_val).flatten(start_dim=1)
                                )
                            else:
                                obs_list.append(
                                    torch.tensor(obs_val).flatten(start_dim=1)
                                )
                    if obs_list:
                        obs_tensor = torch.cat(obs_list, dim=1).to(device)
                    else:
                        # Fallback: use all dict values except target_object_position
                        obs_list = []
                        for key, val in obs.items():
                            if key == "target_object_position":
                                continue  # Skip redundant command
                            if isinstance(val, torch.Tensor):
                                obs_list.append(val.flatten(start_dim=1))
                            elif isinstance(val, np.ndarray):
                                obs_list.append(
                                    torch.from_numpy(val).flatten(start_dim=1)
                                )
                            else:
                                obs_list.append(torch.tensor(val).flatten(start_dim=1))
                        obs_tensor = torch.cat(obs_list, dim=1).to(device)
            else:
                obs_tensor = (
                    torch.from_numpy(obs).to(device)
                    if isinstance(obs, np.ndarray)
                    else obs
                )
                if len(obs_tensor.shape) == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                elif len(obs_tensor.shape) > 2:
                    obs_tensor = obs_tensor.view(obs_tensor.shape[0], -1)
            
            # Normalize observations (if stats available)
            # Ensure dimensions match (obs_tensor should be 32 dims after removing target_object_position)
            if obs_mean is not None and obs_std is not None:
                if obs_tensor.shape[1] != obs_mean.shape[0]:
                    print(
                        f"[Play] ERROR: Obs dimension mismatch! obs_tensor: {obs_tensor.shape[1]}, obs_mean: {obs_mean.shape[0]}"
                    )
                    print(
                        f"[Play] This means target_object_position was not properly removed!"
                    )
                    raise RuntimeError(
                        f"Observation dimension mismatch: {obs_tensor.shape[1]} vs {obs_mean.shape[0]}"
                    )
                obs_tensor_normalized = (obs_tensor - obs_mean) / obs_std
            else:
                obs_tensor_normalized = obs_tensor
            
            # Extract current node features (before normalization for node history)
            if use_node_histories:
                node_features_current = _extract_node_features_multi(
                    obs_tensor
                )  # [num_envs, N, 7]
            else:
                ee_node_current, object_node_current = _extract_node_features_from_obs(
                    obs_tensor
                )  # [num_envs, 7]
            
            # Extract current joint states (position + velocity) from raw obs
            if joint_dim is not None:
                joint_states_current = _extract_joint_states_from_obs(
                    obs_tensor
                )  # [num_envs, joint_dim]

            # Build action, node, and joint histories for batch
            action_history_batch = []
            joint_state_history_batch = [] if joint_dim is not None else None
            
            for env_id in range(num_envs):
                action_history_batch.append(
                    action_history_buffers[env_id].clone()
                )
                if joint_dim is not None and joint_state_history_batch is not None:
                    joint_state_history_batch.append(
                        joint_state_history_buffers[env_id].clone()
                    )
            
            action_history_tensor = torch.stack(action_history_batch, dim=0).to(
                device
            )
            if joint_dim is not None and joint_state_history_batch is not None:
                joint_states_history_tensor = torch.stack(
                    joint_state_history_batch, dim=0
                ).to(device)
            else:
                joint_states_history_tensor = None
            
            # Node histories: either 4-node (node_histories) or 2-node (ee + object)
            if use_node_histories:
                node_histories_tensor = torch.stack(
                    [node_history_buffers[env_id].clone() for env_id in range(num_envs)],
                    dim=0,
                ).to(device)  # [num_envs, N, H, 7]
                node_types_tensor = torch.tensor(
                    [nc.get("type", 0) for nc in node_configs],
                    dtype=torch.long,
                    device=device,
                )
                # Normalize per node type (ee_mean for type 0, object_mean for type 1)
                if ee_node_mean is not None and ee_node_std is not None:
                    for n_idx in range(node_histories_tensor.shape[1]):
                        ntype = node_types_tensor[n_idx].item()
                        if ntype == 0:
                            node_histories_tensor[:, n_idx] = (
                                node_histories_tensor[:, n_idx] - ee_node_mean
                            ) / ee_node_std
                        elif (
                            ntype == 1
                            and object_node_mean is not None
                            and object_node_std is not None
                        ):
                            node_histories_tensor[:, n_idx] = (
                                node_histories_tensor[:, n_idx] - object_node_mean
                            ) / object_node_std
                ee_node_history_tensor = None
                object_node_history_tensor = None
            else:
                ee_node_history_batch = [
                    ee_node_history_buffers[env_id].clone() for env_id in range(num_envs)
                ]
                object_node_history_batch = [
                    object_node_history_buffers[env_id].clone()
                    for env_id in range(num_envs)
                ]
                ee_node_history_tensor = torch.stack(
                    ee_node_history_batch, dim=0
                ).to(device)
                object_node_history_tensor = torch.stack(
                    object_node_history_batch, dim=0
                ).to(device)
                node_histories_tensor = None
                node_types_tensor = None
                if ee_node_mean is not None and ee_node_std is not None:
                    ee_node_history_tensor = (
                        ee_node_history_tensor - ee_node_mean
                    ) / ee_node_std
                    object_node_history_tensor = (
                        object_node_history_tensor - object_node_mean
                    ) / object_node_std

            # CRITICAL FIX: Normalize joint states (same as during training)
            if (
                joint_mean is not None
                and joint_std is not None
                and joint_states_history_tensor is not None
            ):
                joint_states_history_tensor = (
                    joint_states_history_tensor - joint_mean
                ) / joint_std
            
            # Get subtask condition (optional)
            subtask_condition = None
            
            # ==========================================================================
            # RECEDING HORIZON CONTROL: Check if we need to re-plan
            # ==========================================================================
            # We only call the policy when action buffer is empty
            # This dramatically reduces inference frequency (every exec_horizon steps vs every step)

            # Check if any environment needs re-planning (empty buffer)
            needs_replan = any(
                len(action_buffers[env_id]) == 0 for env_id in range(num_envs)
            )

            if needs_replan:
                # Predict action trajectory: Graph-Unet outputs 6-dim (5 arm + 1 gripper); buffer keeps raw (unmapped)
                # Output: [num_envs, pred_horizon, 6] (normalized)
                predict_kw = dict(
                    obs=obs_tensor_normalized,
                    action_history=action_history_tensor,
                    joint_states_history=joint_states_history_tensor,
                    subtask_condition=subtask_condition,
                    num_diffusion_steps=num_diffusion_steps,
                    deterministic=True,
                )
                if use_node_histories:
                    predict_kw["node_histories"] = node_histories_tensor
                    predict_kw["node_types"] = node_types_tensor
                else:
                    predict_kw["ee_node_history"] = ee_node_history_tensor
                    predict_kw["object_node_history"] = object_node_history_tensor
                action_trajectory_normalized = policy.predict(**predict_kw)

                # Denormalize full 6-dim; buffer stores this raw; env gets gripper mapped to -1/1 only at step
                if action_mean is not None and action_std is not None:
                    action_trajectory = (
                        action_trajectory_normalized * action_std.unsqueeze(0).unsqueeze(0)
                        + action_mean.unsqueeze(0).unsqueeze(0)
                    )  # [num_envs, pred_horizon, 6]
                else:
                    action_trajectory = action_trajectory_normalized
                    if step_count == 0:
                        print(
                            f"[Play] Warning: No action normalization stats, using normalized actions directly"
                        )

                # Fill buffers with raw 6-dim (normalized raw for policy input; buffer = unmapped original)
                for env_id in range(num_envs):
                    if len(action_buffers[env_id]) == 0:
                        for t in range(min(exec_horizon, pred_horizon)):
                            action_buffers[env_id].append(action_trajectory[env_id, t, :].clone())
                            action_buffers_normalized[env_id].append(
                                action_trajectory_normalized[env_id, t, :].clone()
                            )

            # Pop the first action from each buffer
            actions_list = []
            actions_normalized_list = []
            for env_id in range(num_envs):
                if len(action_buffers[env_id]) > 0:
                    actions_list.append(action_buffers[env_id].pop(0))
                    actions_normalized_list.append(
                        action_buffers_normalized[env_id].pop(0)
                    )
                else:
                    # Fallback: should not happen if logic is correct
                    print(
                        f"[Play] WARNING: Empty action buffer for env {env_id}, using zeros!"
                    )
                    actions_list.append(torch.zeros(action_dim, device=device))
                    actions_normalized_list.append(
                        torch.zeros(action_dim, device=device)
                    )

            # Stack into batch tensors: actions = raw 6-dim from model (buffer keeps this; -1/1 only for Isaac)
            actions = torch.stack(actions_list, dim=0)  # [num_envs, action_dim] raw (unmapped)
            actions_normalized = torch.stack(
                actions_normalized_list, dim=0
            )  # [num_envs, action_dim] normalized raw for buffer

            # EMA smoothing for joints only (grippers excluded)
            # Single arm (6): joints 0-4, gripper 5 | Dual arm (12): right joints 0-4, gripper 5, left joints 6-10, gripper 11
            if action_dim == 6:
                joint_indices = [0, 1, 2, 3, 4]
                gripper_indices = [5]
            elif action_dim == 12:
                joint_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
                gripper_indices = [5, 11]
            else:
                joint_indices = list(range(action_dim))
                gripper_indices = []

            if joint_indices:
                joints = actions[:, joint_indices]
                if ema_smoothed_joints is None:
                    ema_smoothed_joints = joints.clone()
                else:
                    ema_smoothed_joints = ema_alpha * joints + (1 - ema_alpha) * ema_smoothed_joints
                actions_out = actions.clone()
                for i, idx in enumerate(joint_indices):
                    actions_out[:, idx] = ema_smoothed_joints[:, i]
                actions = actions_out

            # CRITICAL: Action order mismatch - Train uses [left_6, right_6], Env expects [right_6, left_6]
            # Policy outputs train order; reorder before env.step
            if action_dim == 12:
                action_for_env = torch.cat([actions[:, 6:12], actions[:, 0:6]], dim=1)
            else:
                action_for_env = actions.clone()

            # Gripper mapping for Isaac Sim: same logic for both arms: > -0.2 -> 1 (open), else -1 (close)
            # After reorder, action_for_env = [right_6, left_6], grippers at 5 and 11
            for g_idx in gripper_indices:
                action_for_env[:, g_idx] = torch.where(
                    action_for_env[:, g_idx] > -0.2, 1.0, -1.0
                )

            # DEBUG: print gripper + joint info every 50 steps (env 0 only)
            if step_count % 50 == 0 and gripper_indices:
                g_info = " | ".join(
                    f"g{i}_raw={actions[0, i].item():+.4f}→{action_for_env[0, i].item():+.1f}"
                    for i in gripper_indices
                )
                jp = obs_tensor[0, : min(12, obs_tensor.shape[1])].cpu().numpy()
                print(f"[DBG step={step_count}] {g_info} | joint_pos={np.array2string(jp, precision=3, separator=',')}")

            # Update history buffers (shift and add new)
            # IMPORTANT: Store normalized actions in history buffer, as policy expects normalized action_history
            for env_id in range(num_envs):
                # Shift action history (use normalized actions, not denormalized!)
                action_history_buffers[env_id] = torch.cat(
                    [
                    action_history_buffers[env_id][1:],
                        actions_normalized[
                            env_id : env_id + 1
                        ],  # [1, action_dim] - Use normalized actions!
                    ],
                    dim=0,
                )
                
                # Shift node histories
                if use_node_histories:
                    node_history_buffers[env_id] = torch.cat(
                        [
                            node_history_buffers[env_id][:, 1:, :],
                            node_features_current[env_id : env_id + 1].transpose(0, 1),
                        ],
                        dim=1,
                    )  # [N, H, 7]
                else:
                    ee_node_history_buffers[env_id] = torch.cat(
                        [
                            ee_node_history_buffers[env_id][1:],
                            ee_node_current[env_id : env_id + 1],
                        ],
                        dim=0,
                    )
                    object_node_history_buffers[env_id] = torch.cat(
                        [
                            object_node_history_buffers[env_id][1:],
                            object_node_current[env_id : env_id + 1],
                        ],
                        dim=0,
                    )

                if joint_dim is not None and joint_state_history_buffers is not None:
                    joint_state_history_buffers[env_id] = torch.cat(
                        [
                            joint_state_history_buffers[env_id][1:],
                            joint_states_current[env_id : env_id + 1],
                        ],
                        dim=0,
                    )
            
            # Step environment (send mapped gripper -1/1; buffer already has raw for next policy input)
            obs, rewards, terminated, truncated, info = env.step(action_for_env)
            done = terminated | truncated
            
            # 成功率仅用 truncated + 高度判断（与 play_graph_rl 一致，不依赖 env_info 标量）

            # Accumulate rewards (rewards might be numpy or tensor)
            if isinstance(rewards, np.ndarray):
                current_episode_rewards += torch.from_numpy(rewards).to(device)
            else:
                current_episode_rewards += (
                    rewards.to(device)
                    if hasattr(rewards, "to")
                    else torch.tensor(rewards, device=device)
                )
            
            # Check for episode completion and reset history buffers + action buffers
            if done.any():
                for i in range(num_envs):
                    if done[i]:
                        episode_rewards.append(current_episode_rewards[i].item())

                        # 与 play_graph_rl / train 一致：仅用 truncated + 高度（Isaac Lab log 为标量不可用）
                        obj_h = obs_tensor[i, OBJ_HEIGHT_IDX].item()  # obs_tensor = step 前状态
                        is_truncated = bool(
                            truncated[i].item() if hasattr(truncated[i], "item") else truncated[i]
                        )
                        if is_truncated:
                            is_success = False
                        else:
                            is_success = obj_h >= SUCCESS_HEIGHT

                        episode_success.append(is_success)
                        episode_count += 1
                        status = "✅" if is_success else "❌"
                        sr = sum(episode_success) / len(episode_success) * 100.0
                        print(f"[Play] Ep {episode_count:3d} h={obj_h:.3f}m {status} | SR={sr:.1f}%")

                        current_episode_rewards[i] = 0.0

                        # Reset history buffers for this environment
                        action_history_buffers[i].zero_()
                        if use_node_histories:
                            node_history_buffers[i].zero_()
                        else:
                            ee_node_history_buffers[i].zero_()
                            object_node_history_buffers[i].zero_()
                        if (
                            joint_dim is not None
                            and joint_state_history_buffers is not None
                        ):
                            joint_state_history_buffers[i].zero_()
                        
                        # CRITICAL: Clear action buffer on episode reset!
                        # This forces re-planning at the start of each new episode
                        action_buffers[i].clear()
                        action_buffers_normalized[i].clear()

                        # Reset EMA state for this environment
                        if ema_smoothed_joints is not None:
                            ema_smoothed_joints[i] = 0.0
                        
                        if episode_count >= num_episodes:
                            break
            
            step_count += 1
            
            # Print progress: 与 train 一致，SR + SR(100ep) + 已完成的 episode 数
            if step_count % 100 == 0 and episode_success:
                n = len(episode_success)
                sr = sum(episode_success) / n * 100.0
                last_100 = episode_success[-100:] if n >= 100 else episode_success
                sr_100 = sum(last_100) / len(last_100) * 100.0 if last_100 else 0.0
                print(
                    f"[Play] Ep {episode_count}/{num_episodes} | SR={sr:5.1f}% (100ep:{sr_100:5.1f}%) [{n}ep]"
                )
    
    # Final statistics: 与 train 一致的统计格式
    if episode_rewards:
        n = len(episode_success)
        sr_final = sum(episode_success) / n * 100.0 if episode_success else 0.0
        last_100 = episode_success[-100:] if n >= 100 else episode_success
        sr_100_final = sum(last_100) / len(last_100) * 100.0 if last_100 else 0.0
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"\n[Play] ===== Final Statistics =====")
        print(
            f"[Play] SR={sr_final:.1f}% (100ep:{sr_100_final:.1f}%) [{n}ep] | "
            f"Rew_mean={mean_reward:.1f} | {sum(episode_success)}/{n} success"
        )
    
    # Close environment
    env.close()
    print(f"\n[Play] Playback completed!")


def main():
    """Main playback function."""
    # Use args_cli that was already parsed at the top of the file (before AppLauncher)
    num_episodes = getattr(args_cli, "num_episodes", None)
    num_batches = getattr(args_cli, "num_batches", 2)
    play_graph_unet_policy(
        task_name=args_cli.task,
        checkpoint_path=args_cli.checkpoint,
        num_envs=args_cli.num_envs,
        num_episodes=num_episodes,
        num_batches=num_batches,
        device=args_cli.device,
        num_diffusion_steps=args_cli.num_diffusion_steps,
        episode_length_s=getattr(args_cli, "episode_length_s", None),
        policy_type=getattr(args_cli, "policy_type", "unet"),
    )


if __name__ == "__main__":
    main()
    simulation_app.close()

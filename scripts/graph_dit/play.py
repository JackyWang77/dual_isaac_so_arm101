# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback script for trained Graph-DiT Policy.

This script loads a trained Graph-DiT policy and runs it in an Isaac Lab environment.

Usage:
    ./isaaclab.sh -p scripts/graph_dit/play.py \
        --task SO-ARM101-Pick-Place-DualArm-IK-Abs-v0 \
        --checkpoint ./logs/graph_dit/best_model.pt \
        --num_envs 64
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# Create argument parser first
parser = argparse.ArgumentParser(description="Play Graph-DiT Policy with Gripper Model")
parser.add_argument("--task", type=str, required=True, help="Task name")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to Graph-DiT checkpoint")
parser.add_argument("--gripper-model", type=str, default=None, help="Path to gripper model checkpoint")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument(
    "--num_diffusion_steps", 
    type=int, 
    default=None, 
    help="Number of flow matching inference steps (default: uses checkpoint config, typically 30). More steps = smoother but slower."
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
from SO_101.policies.graph_dit_policy import GraphDiTPolicy

# Add current directory to path for gripper_model import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from gripper_model import GripperPredictor

# Try to import visualization utilities (if available)
try:
    from omni.isaac.debug_draw import DebugDraw
    DEBUG_DRAW_AVAILABLE = True
except ImportError:
    DEBUG_DRAW_AVAILABLE = False
    print("[Play] DebugDraw not available, visualization will be disabled")


def play_graph_dit_policy(
    task_name: str,
    checkpoint_path: str,
    num_envs: int = 64,
    num_episodes: int = 10,
    device: str = "cuda",
    num_diffusion_steps: int | None = None,
    gripper_model_path: str | None = None,
):
    """Play trained Graph-DiT policy.
    
    Args:
        task_name: Environment task name.
        checkpoint_path: Path to trained policy checkpoint.
        num_envs: Number of parallel environments.
        num_episodes: Number of episodes to run.
        device: Device to run on.
    """
    
    print(f"[Play] ===== Graph-DiT Policy Playback =====")
    print(f"[Play] Task: {task_name}")
    print(f"[Play] Checkpoint: {checkpoint_path}")
    print(f"[Play] Num envs: {num_envs}")
    
    # Create environment
    print(f"\n[Play] Creating environment...")
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
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
    
    # Policy outputs action_dim from config (typically 5 = arm only); env may expect 6 (arm + gripper)
    print(f"[Play] Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Load policy and normalization stats
    print(f"\n[Play] Loading policy from: {checkpoint_path}")
    # weights_only=False is needed for PyTorch 2.6+ to load custom config classes
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load policy
    policy = GraphDiTPolicy.load(checkpoint_path, device=device)
    policy.eval()
    
    # Load gripper model (if provided)
    gripper_model = None
    gripper_input_mean = None
    gripper_input_std = None
    
    if gripper_model_path is not None and os.path.exists(gripper_model_path):
        print(f"[Play] Loading gripper model from: {gripper_model_path}")
        gripper_checkpoint = torch.load(gripper_model_path, map_location=device, weights_only=False)
        
        # Get model config
        hidden_dims = gripper_checkpoint.get('hidden_dims', [128, 128, 64])
        dropout = gripper_checkpoint.get('dropout', 0.1)
        
        # Load model
        gripper_model = GripperPredictor(hidden_dims=hidden_dims, dropout=dropout).to(device)
        gripper_model.load_state_dict(gripper_checkpoint['model_state_dict'])
        gripper_model.eval()
        
        # Load normalization stats (binary classification doesn't need target stats)
        input_mean = gripper_checkpoint['input_mean']
        input_std = gripper_checkpoint['input_std']
        
        # Ensure numpy arrays
        if isinstance(input_mean, torch.Tensor):
            input_mean = input_mean.cpu().numpy()
        if isinstance(input_std, torch.Tensor):
            input_std = input_std.cpu().numpy()
        
        # Ensure correct shape
        # input_mean/std should be [1, 7] or [7] -> [1, 7]
        # 🔥 MODIFIED: Gripper model input is now 7-dim: [gripper_state(1), ee_pos(3), object_pos(3)]
        print(f"[Play] Raw input_mean shape: {input_mean.shape}, input_std shape: {input_std.shape}")
        
        if input_mean.ndim == 1:
            input_mean = input_mean.reshape(1, -1)
        elif input_mean.ndim == 0:
            raise ValueError(f"input_mean has wrong shape: {input_mean.shape}, expected [1, 7] or [7]")
        if input_std.ndim == 1:
            input_std = input_std.reshape(1, -1)
        elif input_std.ndim == 0:
            raise ValueError(f"input_std has wrong shape: {input_std.shape}, expected [1, 7] or [7]")
        
        # Verify shapes
        # 🔥 MODIFIED: Gripper model input is now 7-dim: [gripper_state(1), ee_pos(3), object_pos(3)]
        assert input_mean.shape == (1, 7), f"input_mean shape mismatch: {input_mean.shape}, expected (1, 7)"
        assert input_std.shape == (1, 7), f"input_std shape mismatch: {input_std.shape}, expected (1, 7)"
        
        # Convert to torch tensors
        gripper_input_mean = torch.from_numpy(input_mean).float().to(device)
        gripper_input_std = torch.from_numpy(input_std).float().to(device)
        
        print(f"[Play] Gripper model loaded successfully (Binary Classification)")
        print(f"[Play] Gripper input_mean shape: {gripper_input_mean.shape}, input_std shape: {gripper_input_std.shape}")
    else:
        if gripper_model_path is not None:
            print(f"[Play] Warning: Gripper model path provided but file not found: {gripper_model_path}")
        print(f"[Play] Using Graph-DiT for all action dimensions (including gripper)")
    
    # Determine mode and set default diffusion steps if not provided
    cfg = checkpoint.get("cfg", None)
    if cfg is not None:
        mode = getattr(cfg, "mode", "flow_matching")
        print(f"[Play] Policy mode: {mode.upper()}")
        # Use num_inference_steps from checkpoint config if available (default: 30)
        # This matches the training configuration
        if num_diffusion_steps is None:
            num_diffusion_steps = getattr(cfg, "num_inference_steps", 30)
            print(f"[Play] Using inference steps from checkpoint config: {num_diffusion_steps}")
    else:
        # Fallback: use 30 as default (matches training default)
        if num_diffusion_steps is None:
            num_diffusion_steps = 30  # Default for Flow Matching (matches training)
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

    # Get config values from policy
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
    current_episode_rewards = torch.zeros(num_envs, device=device)
    
    # Initialize action, node, and joint history buffers for each environment
    action_history_buffers = [
        torch.zeros(action_history_length, action_dim, device=device)
        for _ in range(num_envs)
    ]
    ee_node_history_buffers = [
        torch.zeros(action_history_length, 7, device=device) for _ in range(num_envs)
    ]
    object_node_history_buffers = [
        torch.zeros(action_history_length, 7, device=device) for _ in range(num_envs)
    ]
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
    ema_alpha = 1  # EMA weight: higher = more responsive, lower = smoother
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
    
    # Initialize gripper state machine (0=OPEN, 1=CLOSING, 2=CLOSED)
    gripper_states = torch.zeros(num_envs, dtype=torch.long, device=device)
    gripper_close_steps = torch.zeros(num_envs, dtype=torch.long, device=device)  # Steps since close triggered  # Normalized versions for history
    
    step_count = 0
    
    def _extract_node_features_from_obs(obs_tensor):
        """Extract EE and Object node features from concatenated obs.
        
        Note: Assumes obs_keys order is:
        joint_pos, joint_vel, object_position, object_orientation, 
        ee_position, ee_orientation, actions
        (target_object_position is skipped even if present in data)
        """
        # Compute offsets dynamically (same as in HDF5Dataset)
        # joint_pos: 0-6, joint_vel: 6-12
        # object_position: 12-15, object_orientation: 15-19
        # ee_position: 19-22, ee_orientation: 22-26
        # actions: 26-32 (for joint states)
        obj_pos = obs_tensor[:, 12:15]  # [batch, 3] - object_position
        obj_ori = obs_tensor[:, 15:19]  # [batch, 4] - object_orientation
        ee_pos = obs_tensor[:, 19:22]  # [batch, 3] - ee_position
        ee_ori = obs_tensor[:, 22:26]  # [batch, 4] - ee_orientation
        
        ee_node = torch.cat([ee_pos, ee_ori], dim=-1)  # [batch, 7]
        object_node = torch.cat([obj_pos, obj_ori], dim=-1)  # [batch, 7]
        return ee_node, object_node
    
    def _extract_joint_states_from_obs(obs_tensor):
        """Extract joint position from concatenated obs.
        
        NOTE: Only using joint_pos (removed joint_vel to test if it's noise).

        Assumes layout:
            joint_pos: 0-6
        """
        joint_pos = obs_tensor[:, 0:6]
        # Only return joint_pos (no joint_vel)
        return joint_pos
    
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
                    elif obs_tensor_raw.shape[1] == 32:
                        # Already correct (training format)
                        obs_tensor = obs_tensor_raw
                    else:
                        # Unknown dimension, use as is (might cause issues)
                        print(
                            f"[Play] Warning: Unexpected obs dim {obs_tensor_raw.shape[1]}, expected 32 or 39"
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
            # obs_tensor is now 32 dims (after removing target_object_position)
            # Indices: joint_pos[0:6], joint_vel[6:12], object_position[12:15], object_orientation[15:19],
            #          ee_position[19:22], ee_orientation[22:26], actions[26:32]
            ee_node_current, object_node_current = _extract_node_features_from_obs(
                obs_tensor
            )  # [num_envs, 7]
            
            # 🟢 VISUAL DEBUG: Extract current EE position for comparison
            current_ee_pos = ee_node_current[:, :3].cpu().numpy()  # [num_envs, 3] - EE position
            
            # Extract current joint states (position + velocity) from raw obs
            if joint_dim is not None:
                joint_states_current = _extract_joint_states_from_obs(
                    obs_tensor
                )  # [num_envs, joint_dim]

            # Build action, node, and joint histories for batch
            action_history_batch = []
            ee_node_history_batch = []
            object_node_history_batch = []
            joint_state_history_batch = [] if joint_dim is not None else None
            
            for env_id in range(num_envs):
                # Get histories for this environment
                action_history_batch.append(
                    action_history_buffers[env_id].clone()
                )  # [history_length, action_dim]
                ee_node_history_batch.append(
                    ee_node_history_buffers[env_id].clone()
                )  # [history_length, 7]
                object_node_history_batch.append(
                    object_node_history_buffers[env_id].clone()
                )  # [history_length, 7]
                if joint_dim is not None and joint_state_history_batch is not None:
                    joint_state_history_batch.append(
                        joint_state_history_buffers[env_id].clone()
                    )  # [H, joint_dim]
            
            # Stack into batch format
            action_history_tensor = torch.stack(action_history_batch, dim=0).to(
                device
            )  # [num_envs, history_length, action_dim] - should be 6-dim (5 arm + 1 gripper)
            
            # 🔥 MODIFIED: Graph-DiT expects 5-dim action_history (arm joints only)
            # Extract only first 5 dims for Graph-DiT input
            if action_history_tensor.shape[2] >= 6:
                action_history_tensor_5d = action_history_tensor[:, :, :5]  # [num_envs, history_length, 5]
            else:
                action_history_tensor_5d = action_history_tensor  # Already 5-dim or less
            ee_node_history_tensor = torch.stack(ee_node_history_batch, dim=0).to(
                device
            )  # [num_envs, history_length, 7]
            object_node_history_tensor = torch.stack(
                object_node_history_batch, dim=0
            ).to(
                device
            )  # [num_envs, history_length, 7]
            if joint_dim is not None and joint_state_history_batch is not None:
                joint_states_history_tensor = torch.stack(
                    joint_state_history_batch, dim=0
                ).to(
                    device
                )  # [num_envs, H, joint_dim]
            else:
                joint_states_history_tensor = None

            # CRITICAL FIX: Normalize node features (same as during training)
            # This is essential for Transformer attention to work properly
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
                # Predict action trajectory for ALL environments (batched inference)
                # 🔥 MODIFIED: Graph-DiT outputs 5-dim (arm joints only), input action_history should also be 5-dim
                # Output: [num_envs, pred_horizon, 5] (normalized, arm joints only)
                action_trajectory_normalized = policy.predict(
                obs_tensor_normalized,
                action_history=action_history_tensor_5d,  # [num_envs, history_length, 5] - only arm joints
                ee_node_history=ee_node_history_tensor,
                object_node_history=object_node_history_tensor,
                    joint_states_history=joint_states_history_tensor,
                subtask_condition=subtask_condition,
                num_diffusion_steps=num_diffusion_steps,
                    deterministic=True, 
                )  # [num_envs, pred_horizon, 5] - arm joints only
            
                # 🔥 MODIFIED: Graph-DiT outputs 5-dim (arm joints only), need to add gripper dimension
                # action_trajectory_normalized is [num_envs, pred_horizon, 5] (arm joints only)
                
                # Denormalize trajectory (only for arm joints, first 5 dims)
                if action_mean is not None and action_std is not None:
                    # Use only first 5 dims of action_mean/action_std (arm joints)
                    action_mean_5d = action_mean[:5] if action_mean.shape[0] >= 5 else action_mean
                    action_std_5d = action_std[:5] if action_std.shape[0] >= 5 else action_std
                    # Broadcast mean/std for trajectory: [5] -> [1, 1, 5]
                    action_trajectory_5d = (
                        action_trajectory_normalized * action_std_5d.unsqueeze(0).unsqueeze(0)
                        + action_mean_5d.unsqueeze(0).unsqueeze(0)
                    )  # [num_envs, pred_horizon, 5]
                else:
                    action_trajectory_5d = action_trajectory_normalized
                    if step_count == 0:
                        print(
                            f"[Play] Warning: No action normalization stats, using normalized actions directly"
                        )
                
                # Add gripper dimension (will be filled later by gripper_model)
                # action_trajectory_5d: [num_envs, pred_horizon, 5] -> [num_envs, pred_horizon, 6]
                action_trajectory = torch.cat([
                    action_trajectory_5d,  # [num_envs, pred_horizon, 5]
                    torch.zeros(num_envs, pred_horizon, 1, device=device)  # [num_envs, pred_horizon, 1] - gripper placeholder
                ], dim=-1)  # [num_envs, pred_horizon, 6]

                # 🟢 VISUAL DEBUG: Store target joint positions for visualization
                # action_trajectory is [num_envs, pred_horizon, 6] (5 arm + 1 gripper placeholder)
                # We'll visualize the first action (target joint_pos[t+1]) for each env
                target_joint_positions = action_trajectory[:, 0, :5].cpu().numpy()  # [num_envs, 5] - only arm joints
                
                # 🟢 VISUAL DEBUG: Print target vs current for first env (every replanning step)
                if step_count % 10 == 0:  # Print every 10 replanning steps
                    env_id = 0
                    target_joint = target_joint_positions[env_id]  # [5] - arm joints only
                    current_joint = obs_tensor[env_id, :5].cpu().numpy()  # [5] - first 5 joints (arm only)
                    current_ee_pos_debug = current_ee_pos[env_id]
                    object_pos_debug = object_node_current[env_id, :3].cpu().numpy()
                    
                    print(
                        f"\n[🟢 VISUAL DEBUG Step {step_count}] =========="
                    )
                    print(f"  Target Joint (action): {target_joint.tolist()}")
                    print(f"  Current Joint:         {current_joint.tolist()}")
                    print(f"  Joint Diff:            {np.abs(target_joint - current_joint).tolist()}")
                    print(f"  Current EE Position:    {current_ee_pos_debug.tolist()}")
                    print(f"  Object Position:        {object_pos_debug.tolist()}")
                    print(f"  EE-Object Distance:      {np.linalg.norm(current_ee_pos_debug - object_pos_debug):.3f}")
                    print(f"  =========================================\n")

                # Fill action buffers with first exec_horizon actions
                # action_trajectory is [num_envs, pred_horizon, 6] (5 arm + 1 gripper placeholder)
                # action_trajectory_normalized is [num_envs, pred_horizon, 5] (arm only)
                for env_id in range(num_envs):
                    if len(action_buffers[env_id]) == 0:
                        # Store exec_horizon actions in buffer (as list for easy pop)
                        for t in range(min(exec_horizon, pred_horizon)):
                            action_buffers[env_id].append(
                                action_trajectory[env_id, t, :]  # [6] - 5 arm + 1 gripper placeholder
                            )
                            # For normalized, we need to pad to 6-dim for consistency
                            normalized_5d = action_trajectory_normalized[env_id, t, :]  # [5]
                            normalized_6d = torch.cat([
                                normalized_5d,
                                torch.zeros(1, device=device)  # gripper placeholder
                            ], dim=0)  # [6]
                            action_buffers_normalized[env_id].append(normalized_6d)

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

            # Stack into batch tensors
            actions = torch.stack(actions_list, dim=0)  # [num_envs, action_dim]
            actions_normalized = torch.stack(
                actions_normalized_list, dim=0
            )  # [num_envs, action_dim]
            
            # Replace gripper (6th dimension, index 5) with gripper model prediction
            if gripper_model is not None and action_dim >= 6:
                # Extract inputs from current observation
                # obs_tensor structure: joint_pos[0:6], joint_vel[6:12], object_position[12:15], 
                #                      object_orientation[15:19], ee_position[19:22], ee_orientation[22:26]
                # 🔥 MODIFIED: Gripper model input is now 7-dim: [gripper_state(1), ee_pos(3), object_pos(3)]
                gripper_state = obs_tensor[:, 5:6]  # [num_envs, 1] - gripper joint (6th joint, index 5)
                ee_pos = obs_tensor[:, 19:22]  # [num_envs, 3]
                object_pos = obs_tensor[:, 12:15]  # [num_envs, 3]
                
                # Prepare gripper input: [num_envs, 7] (1+3+3)
                gripper_input = torch.cat([
                    gripper_state,  # 1
                    ee_pos,         # 3
                    object_pos      # 3
                ], dim=-1).float()  # [num_envs, 7]
                
                # Debug: Check shapes on first step
                if step_count == 0:
                    print(f"[Play] Gripper input shape: {gripper_input.shape}, expected: [num_envs, 7]")
                    print(f"[Play] Gripper input_mean shape: {gripper_input_mean.shape}, expected: [1, 7]")
                    print(f"[Play] Gripper input_std shape: {gripper_input_std.shape}, expected: [1, 7]")
                
                # Normalize
                gripper_input_norm = (gripper_input - gripper_input_mean) / gripper_input_std
                
                # 🔥 分类预测
                with torch.no_grad():
                    # Debug: Check input dimensions
                    if step_count == 0:
                        print(f"[Play] Gripper model input dimensions:")
                        print(f"  gripper_state: {gripper_input_norm[:, 0:1].shape} (expected [num_envs, 1])")
                        print(f"  ee_pos: {gripper_input_norm[:, 1:4].shape} (expected [num_envs, 3])")
                        print(f"  object_pos: {gripper_input_norm[:, 4:7].shape} (expected [num_envs, 3])")
                        print(f"  Total input dim: {gripper_input_norm.shape[1]} (expected 7)")
                    
                    gripper_action, confidence, pred_class = gripper_model.predict(
                        gripper_input_norm[:, 0:1],    # gripper_state [num_envs, 1]
                        gripper_input_norm[:, 1:4],    # ee_pos [num_envs, 3]
                        gripper_input_norm[:, 4:7]     # object_pos [num_envs, 3]
                    )  # gripper_action: [num_envs, 1], confidence: [num_envs, 1], pred_class: [num_envs, 1]
                
                # 🔥 状态机逻辑（3-class: 0=KEEP_CURRENT, 1=TRIGGER_CLOSE, 2=TRIGGER_OPEN）
                for env_id in range(num_envs):
                    curr_state = gripper_states[env_id].item()
                    pred = pred_class[env_id, 0].item()  # 0=KEEP_CURRENT, 1=TRIGGER_CLOSE, 2=TRIGGER_OPEN
                    
                    if curr_state == 0:  # 当前OPEN
                        if pred == 1:  # 预测TRIGGER_CLOSE
                            gripper_states[env_id] = 1  # 进入CLOSING状态
                            gripper_close_steps[env_id] = 0
                            actions[env_id, 5] = -1.0  # Close
                        else:  # KEEP_CURRENT (0) or TRIGGER_OPEN (2) - 保持打开
                            actions[env_id, 5] = 1.0  # Keep open
                    
                    elif curr_state == 1:  # 正在CLOSING
                        if pred == 2:  # 预测TRIGGER_OPEN（提前打开）
                            gripper_states[env_id] = 0  # 回到OPEN状态
                            actions[env_id, 5] = 1.0  # Open
                        else:  # KEEP_CURRENT (0) or TRIGGER_CLOSE (1) - 继续关闭
                            # 保持关闭状态一段时间（至少10步）
                            gripper_close_steps[env_id] += 1
                            if gripper_close_steps[env_id] >= 10:
                                gripper_states[env_id] = 2  # 进入CLOSED状态
                            actions[env_id, 5] = -1.0  # Keep closing
                    
                    elif curr_state == 2:  # CLOSED
                        if pred == 2:  # 预测TRIGGER_OPEN
                            gripper_states[env_id] = 0  # 回到OPEN状态
                            actions[env_id, 5] = 1.0  # Open
                        else:  # KEEP_CURRENT (0) or TRIGGER_CLOSE (1) - 保持关闭
                            actions[env_id, 5] = -1.0  # Keep closed
                
                # 🔍 调试
                if step_count % 10 == 0:
                    print(f"\n[Gripper Debug Step {step_count}]")
                    state_names = ["OPEN", "CLOSING", "CLOSED"]
                    pred_names = ["KEEP_CURRENT", "TRIGGER_CLOSE", "TRIGGER_OPEN"]
                    
                    for env_id in range(min(2, num_envs)):
                        ee = ee_pos[env_id].cpu().numpy()
                        obj = object_pos[env_id].cpu().numpy()
                        dist = np.linalg.norm(ee - obj)
                        state = gripper_states[env_id].item()
                        pred = pred_class[env_id, 0].item()
                        conf = confidence[env_id, 0].item()
                        
                        print(f"  Env {env_id}:")
                        print(f"    Distance: {dist:.3f}")
                        print(f"    State: {state_names[state]}")
                        print(f"    Prediction: {pred_names[pred]} (confidence: {conf:.2%})")
                        print(f"    Action: {'OPEN' if actions[env_id, 5] > 0 else 'CLOSE'}")
                
                # 归一化版本（用于history）
                # 🔥 MODIFIED: Graph-DiT outputs 5-dim, but history buffer needs 6-dim (with gripper)
                # action_mean/action_std 可能是 5 维（只有 arm joints）或 6 维（包含 gripper）
                # 需要将实际 gripper action 归一化后存入 history buffer
                gripper_action_value = actions[:, 5:6]  # [num_envs, 1] - 实际执行的值 (-1.0 或 1.0)
                
                if action_mean is not None and action_std is not None:
                    # 检查 action_mean/action_std 的维度
                    if action_mean.shape[0] >= 6 and action_std.shape[0] >= 6:
                        # 有 6 维，包含 gripper，进行归一化
                        gripper_action_normalized = (gripper_action_value - action_mean[5]) / action_std[5]
                        actions_normalized[:, 5] = gripper_action_normalized.squeeze(-1)
                    else:
                        # 只有 5 维（只有 arm joints），gripper 已经是 -1.0 或 1.0，直接使用
                        # Note: For history buffer, we still store gripper even if not in normalization stats
                        actions_normalized[:, 5] = gripper_action_value.squeeze(-1)
                else:
                    # 没有归一化统计信息，直接使用
                    actions_normalized[:, 5] = gripper_action_value.squeeze(-1)

            # EMA smoothing for joints only (gripper excluded)
            # NOTE: With ema_alpha=1.0, this is effectively disabled (no smoothing)
            if action_dim >= 6:
                joints = actions[:, :5]  # [num_envs, 5]
                gripper = actions[:, 5:6]  # [num_envs, 1]
                if ema_smoothed_joints is None:
                    ema_smoothed_joints = joints.clone()
                else:
                    ema_smoothed_joints = ema_alpha * joints + (1 - ema_alpha) * ema_smoothed_joints
                # Always update actions with smoothed joints (even if alpha=1.0, this is just joints)
                actions = torch.cat([ema_smoothed_joints, gripper], dim=-1)  # [num_envs, 6]
            
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
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
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
            done = terminated | truncated
            if done.any():
                for i in range(num_envs):
                    if done[i]:
                        episode_rewards.append(current_episode_rewards[i].item())
                        episode_count += 1
                        current_episode_rewards[i] = 0.0
                        
                        # Reset gripper state to OPEN
                        if gripper_model is not None:
                            gripper_states[i] = 0
                            gripper_close_steps[i] = 0
                        
                        # Reset history buffers for this environment
                        action_history_buffers[i].zero_()
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
            
            # Print progress
            if step_count % 100 == 0:
                avg_reward = (
                    sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
                    if episode_rewards
                    else 0.0
                )
                print(
                    f"[Play] Step: {step_count}, Episodes: {episode_count}/{num_episodes}, "
                    f"Avg reward (last 10): {avg_reward:.3f}"
                )
    
    # Print final statistics
    if episode_rewards:
        print(f"\n[Play] ===== Final Statistics =====")
        print(f"[Play] Total episodes: {len(episode_rewards)}")
        print(
            f"[Play] Average reward: {sum(episode_rewards) / len(episode_rewards):.3f}"
        )
        print(f"[Play] Max reward: {max(episode_rewards):.3f}")
        print(f"[Play] Min reward: {min(episode_rewards):.3f}")
    
    # Close environment
    env.close()
    print(f"\n[Play] Playback completed!")


def main():
    """Main playback function."""
    # Use args_cli that was already parsed at the top of the file (before AppLauncher)
    play_graph_dit_policy(
        task_name=args_cli.task,
        checkpoint_path=args_cli.checkpoint,
        num_envs=args_cli.num_envs,
        num_episodes=args_cli.num_episodes,
        device=args_cli.device,
        num_diffusion_steps=args_cli.num_diffusion_steps,
        gripper_model_path=getattr(args_cli, "gripper_model", None),
    )


if __name__ == "__main__":
    main()
    simulation_app.close()

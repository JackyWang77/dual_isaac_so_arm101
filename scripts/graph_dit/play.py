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

from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse

import gymnasium as gym
import numpy as np
import SO_101.tasks  # noqa: F401  # Register environments
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from SO_101.policies.graph_dit_policy import (
    GraphDiTPolicy,
    ActionHistoryBuffer,
    NodeHistoryBuffer,
    JointStateHistoryBuffer,
)

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
        action_dim = 6  # Default fallback for reach task (joint states)
    
    print(f"[Play] Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Load policy and normalization stats
    print(f"\n[Play] Loading policy from: {checkpoint_path}")
    # weights_only=False is needed for PyTorch 2.6+ to load custom config classes
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load policy
    policy = GraphDiTPolicy.load(checkpoint_path, device=device)
    policy.eval()
    
    # Determine mode and set default diffusion steps if not provided
    cfg = checkpoint.get("cfg", None)
    if cfg is not None:
        mode = getattr(cfg, "mode", "ddpm")
        print(f"[Play] Policy mode: {mode.upper()}")
        # Set default steps based on mode if not provided
        # CRITICAL FIX: For 50Hz control, use minimal steps for real-time performance
        # Flow Matching can generate high-quality actions in 2-4 steps
        if num_diffusion_steps is None:
            if mode == "flow_matching":
                num_diffusion_steps = (
                    2  # Flow Matching: 2-4 steps for 50Hz real-time control
                )
            else:
                num_diffusion_steps = 10  # DDPM: Reduced from 50 for faster inference (but consider flow_matching for real-time)
            print(
                f"[Play] Using default diffusion steps for {mode}: {num_diffusion_steps}"
            )
    else:
        if num_diffusion_steps is None:
            num_diffusion_steps = 50  # Default fallback
            print(f"[Play] Using default diffusion steps: {num_diffusion_steps}")
    
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
        # Note: action_mean and action_std are not used directly anymore
        # because policy.predict(normalize=True) handles normalization automatically
    else:
        print(f"[Play] Warning: No action stats found, skipping denormalization")
    
    # CRITICAL FIX: Load node feature normalization stats
    node_stats = checkpoint.get("node_stats", None)
    if node_stats is not None:
        print(f"[Play] Loaded node feature normalization stats")
        # Handle both numpy arrays and torch tensors
        if isinstance(node_stats["ee_mean"], np.ndarray):
            # Note: node stats variables are not used directly anymore
            # because policy.predict(normalize=True) handles normalization automatically
            _ = torch.from_numpy(node_stats["ee_mean"]).squeeze().to(device)  # Loaded but not used
            _ = torch.from_numpy(node_stats["ee_std"]).squeeze().to(device)
            _ = torch.from_numpy(node_stats["object_mean"]).squeeze().to(device)
            _ = torch.from_numpy(node_stats["object_std"]).squeeze().to(device)
        else:
            # Note: node stats variables are not used directly anymore
            # because policy.predict(normalize=True) handles normalization automatically
            _ = node_stats["ee_mean"].squeeze().to(device)  # Loaded but not used
            _ = node_stats["ee_std"].squeeze().to(device)
            _ = node_stats["object_mean"].squeeze().to(device)
            _ = node_stats["object_std"].squeeze().to(device)
    else:
        print(
            f"[Play] Warning: No node stats found, node features will NOT be normalized!"
        )
        print(
            f"[Play] This may cause poor performance - retrain with node normalization!"
        )
        # Note: node stats variables are not used directly anymore
        # because policy.predict(normalize=True) handles normalization automatically

    # CRITICAL FIX: Load joint state normalization stats
    joint_stats = checkpoint.get("joint_stats", None)
    if joint_stats is not None:
        print(f"[Play] Loaded joint state normalization stats")
        # Handle both numpy arrays and torch tensors
        if isinstance(joint_stats["mean"], np.ndarray):
            # Note: joint stats variables are not used directly anymore
            # because policy.predict(normalize=True) handles normalization automatically
            _ = torch.from_numpy(joint_stats["mean"]).squeeze().to(device)  # Loaded but not used
            _ = torch.from_numpy(joint_stats["std"]).squeeze().to(device)
        else:
            # Note: joint stats variables are not used directly anymore
            # because policy.predict(normalize=True) handles normalization automatically
            _ = joint_stats["mean"].squeeze().to(device)  # Loaded but not used
            _ = joint_stats["std"].squeeze().to(device)
    else:
        print(
            f"[Play] Warning: No joint stats found, joint states will NOT be normalized!"
        )
        print(
            f"[Play] This may cause poor performance - retrain with joint normalization!"
        )
        # Note: joint stats variables are not used directly anymore
        # because policy.predict(normalize=True) handles normalization automatically

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
    
    # Initialize action, node, and joint history buffers using Buffer classes
    # CRITICAL: Buffers will be automatically initialized with first observation (not zeros)
    # to match training behavior where history is padded with first action
    # This uses ring buffers for zero-allocation updates
    action_history_buffers = ActionHistoryBuffer(
        history_length=action_history_length,
        action_dim=action_dim,
        num_envs=num_envs,
        device=device,
    )
    node_history_buffers = NodeHistoryBuffer(
        history_length=action_history_length,
        node_dim=7,  # position(3) + orientation(4)
        num_envs=num_envs,
        device=device,
    )
    joint_state_history_buffers = (
        JointStateHistoryBuffer(
            history_length=action_history_length,
            joint_dim=joint_dim,
            num_envs=num_envs,
            device=device,
        )
        if joint_dim is not None
        else None
    )

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
    
    # Initialize action smoothing (EMA) to reduce jitter
    # Smoothing factor: 0.1-0.3 recommended. Smaller = smoother but more delay, larger = less smoothing
    smoothing_factor = 1
    last_actions = [None] * num_envs  # Store last executed action for each environment
    
    step_count = 0
    
    # ==========================================================================
    # 🔧 FIX: Helper function to process observation and extract features
    # ==========================================================================
    def process_obs_to_tensor(obs):
        """Process observation dict/array to tensor format matching training.
        
        Returns:
            obs_tensor: [num_envs, 32] - processed observation (target_object_position removed)
        """
        if isinstance(obs, dict):
            # Check if there's a 'policy' key (Isaac Lab wrapper format)
            if "policy" in obs:
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
                    obs_tensor_raw = obs_tensor_raw.view(obs_tensor_raw.shape[0], -1)
                
                # Remove target_object_position if present (39 -> 32 dims)
                if obs_tensor_raw.shape[1] == 39:
                    obs_tensor = torch.cat([
                        obs_tensor_raw[:, :26],  # Before target_object_position
                        obs_tensor_raw[:, 33:],  # After target_object_position
                    ], dim=1)
                elif obs_tensor_raw.shape[1] == 32:
                    obs_tensor = obs_tensor_raw
                else:
                    print(f"[Play] Warning: Unexpected obs dim {obs_tensor_raw.shape[1]}, expected 32 or 39")
                    obs_tensor = obs_tensor_raw
            else:
                # Concatenate dictionary observations
                obs_keys_order = [
                    "joint_pos", "joint_vel", "object_position", "object_orientation",
                    "ee_position", "ee_orientation", "actions",
                ]
                obs_list = []
                for key in obs_keys_order:
                    if key in obs:
                        obs_val = obs[key]
                        if isinstance(obs_val, torch.Tensor):
                            obs_list.append(obs_val.flatten(start_dim=1))
                        elif isinstance(obs_val, np.ndarray):
                            obs_list.append(torch.from_numpy(obs_val).flatten(start_dim=1))
                        else:
                            obs_list.append(torch.tensor(obs_val).flatten(start_dim=1))
                obs_tensor = torch.cat(obs_list, dim=1).to(device) if obs_list else None
        else:
            obs_tensor = torch.from_numpy(obs).to(device) if isinstance(obs, np.ndarray) else obs.to(device)
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            elif len(obs_tensor.shape) > 2:
                obs_tensor = obs_tensor.view(obs_tensor.shape[0], -1)
        return obs_tensor
    
    # ==========================================================================
    # 🔧 FIX: Helper function to initialize buffers with first observation
    # ==========================================================================
    def initialize_buffers_from_obs(obs_tensor, env_ids=None):
        """Initialize history buffers with first observation (matches training behavior).
        
        Training pads history with first action/node/joint, NOT zeros.
        This function ensures inference matches training.
        
        Args:
            obs_tensor: [num_envs, obs_dim] or [len(env_ids), obs_dim]
            env_ids: If provided, only initialize these environments
        """
        # Extract first action from obs (last 6 dims: actions)
        # obs layout: joint_pos[0:6], joint_vel[6:12], obj_pos[12:15], obj_ori[15:19],
        #             ee_pos[19:22], ee_ori[22:26], actions[26:32]
        first_action = obs_tensor[:, 26:26+action_dim].clone()  # [batch, action_dim]
        
        # Extract first node features
        first_ee_node = torch.cat([
            obs_tensor[:, 19:22],  # ee_position
            obs_tensor[:, 22:26],  # ee_orientation
        ], dim=-1)  # [batch, 7]
        first_object_node = torch.cat([
            obs_tensor[:, 12:15],  # object_position
            obs_tensor[:, 15:19],  # object_orientation
        ], dim=-1)  # [batch, 7]
        
        # Extract first joint states
        if joint_dim is not None:
            first_joint_state = torch.cat([
                obs_tensor[:, 0:6],   # joint_pos
                obs_tensor[:, 6:12],  # joint_vel
            ], dim=-1)  # [batch, joint_dim]
        
        if env_ids is None:
            # Initialize all environments
            action_history_buffers.initialize_with(first_action)
            node_history_buffers.initialize_with(first_ee_node, first_object_node)
            if joint_dim is not None and joint_state_history_buffers is not None:
                joint_state_history_buffers.initialize_with(first_joint_state)
            print(f"[Play] 🔧 Initialized all history buffers with first observation (matches training)")
        else:
            # Initialize only specified environments (for per-env reset)
            # Note: Buffer classes don't support per-env initialize_with, so we manually set
            for i, env_id in enumerate(env_ids):
                env_id_int = env_id.item() if hasattr(env_id, 'item') else env_id
                # Fill entire history with first value (matches training padding)
                action_history_buffers.buffer[env_id_int] = first_action[i].unsqueeze(0).expand(action_history_length, -1).clone()
                action_history_buffers.write_idx[env_id_int] = 0
                action_history_buffers.filled[env_id_int] = False
                
                node_history_buffers.ee_buffer[env_id_int] = first_ee_node[i].unsqueeze(0).expand(action_history_length, -1).clone()
                node_history_buffers.object_buffer[env_id_int] = first_object_node[i].unsqueeze(0).expand(action_history_length, -1).clone()
                node_history_buffers.write_idx[env_id_int] = 0
                node_history_buffers.filled[env_id_int] = False
                
                if joint_dim is not None and joint_state_history_buffers is not None:
                    joint_state_history_buffers.buffer[env_id_int] = first_joint_state[i].unsqueeze(0).expand(action_history_length, -1).clone()
                    joint_state_history_buffers.write_idx[env_id_int] = 0
                    joint_state_history_buffers.filled[env_id_int] = False
            print(f"[Play] 🔧 Re-initialized history buffers for {len(env_ids)} environments with new first observation")
    
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
        """Extract joint position and velocity from concatenated obs.

        Assumes layout:
            joint_pos: 0-6, joint_vel: 6-12
        """
        joint_pos = obs_tensor[:, 0:6]
        joint_vel = obs_tensor[:, 6:12]
        return torch.cat([joint_pos, joint_vel], dim=-1)
    
    # ==========================================================================
    # 🔧 FIX: Process initial observation and initialize buffers BEFORE main loop
    # ==========================================================================
    # Process initial observation to initialize buffers (matches training behavior)
    initial_obs_tensor = process_obs_to_tensor(obs)
    
    # 🔧 FIX: Initialize buffers with first observation (NOT zeros!)
    # This matches training behavior where history is padded with first action
    initialize_buffers_from_obs(initial_obs_tensor)
    
    with torch.inference_mode():
        while simulation_app.is_running() and episode_count < num_episodes:
            # Process observations (use helper function)
            obs_tensor = process_obs_to_tensor(obs)
            
            # CRITICAL: No need to manually normalize obs - policy.predict(normalize=True) will handle it
            # Just ensure dimensions match (obs_tensor should be 32 dims after removing target_object_position)
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

            # CRITICAL: Initialize history buffers with first observation (not zeros)
            # Buffer classes automatically initialize with zeros (no manual initialization needed)
            # Get histories using Buffer classes (automatically handles ring buffer ordering)
            # Buffer stores ABSOLUTE target actions (from previous model predictions)
            # This matches training: action_history[i, j] = actions[j] (absolute positions)
            ee_node_history_tensor, object_node_history_tensor = node_history_buffers.get_history()  # [num_envs, history_length, 7] each
            joint_states_history_tensor = (
                joint_state_history_buffers.get_history()
                if joint_dim is not None and joint_state_history_buffers is not None
                else None
            )  # [num_envs, history_length, joint_dim] or None
            
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
                # Use ABSOLUTE action history (stored in buffer)
                # Buffer stores absolute target actions (from previous model predictions)
                action_history_tensor = action_history_buffers.get_history()  # [num_envs, history_length, action_dim]
                
                # Use policy.predict(normalize=True) to automatically handle
                # normalization/denormalization. This ensures consistency with training.
                # The policy will:
                # 1. Normalize obs, action_history, node_history, joint_history using stored stats
                # 2. Run inference
                # 3. Denormalize the output action trajectory
                # Model outputs ABSOLUTE actions (joint_pos[t+k])
                action_trajectory = policy.predict(
                    obs_tensor,  # Raw obs (not normalized) - policy will normalize internally
                    action_history=action_history_tensor,  # ABSOLUTE history - policy will normalize internally
                    ee_node_history=ee_node_history_tensor,  # Raw history - policy will normalize internally
                    object_node_history=object_node_history_tensor,  # Raw history - policy will normalize internally
                    joint_states_history=joint_states_history_tensor,  # Raw history - policy will normalize internally
                    subtask_condition=subtask_condition,
                    num_diffusion_steps=num_diffusion_steps,
                    deterministic=True,
                    normalize=True,  # CRITICAL: Let policy handle normalization/denormalization
                )  # [num_envs, pred_horizon, action_dim] - ABSOLUTE actions (denormalized)

                # 🟢 VISUAL DEBUG: Store target joint positions for visualization
                # action_trajectory is [num_envs, pred_horizon, action_dim]
                # We'll visualize the first action (target joint_pos[t+1]) for each env
                target_joint_positions = action_trajectory[:, 0, :].cpu().numpy()  # [num_envs, action_dim]
                
                # 🟢 VISUAL DEBUG: Print target vs current for first env (every replanning step)
                if step_count % 10 == 0:  # Print every 10 replanning steps
                    env_id = 0
                    target_joint = target_joint_positions[env_id]
                    current_joint = obs_tensor[env_id, :6].cpu().numpy()
                    current_ee_pos_debug = current_ee_pos[env_id]
                    object_pos_debug = object_node_current[env_id, :3].cpu().numpy()
                    
                    print(
                        f"\n[🟢 VISUAL DEBUG Step {step_count}] =========="
                    )
                    print(f"  Target Joint (action): {target_joint[:3].tolist()} (first 3 joints)")
                    print(f"  Current Joint:         {current_joint[:3].tolist()} (first 3 joints)")
                    print(f"  Joint Diff:             {np.abs(target_joint - current_joint)[:3].tolist()}")
                    print(f"  Current EE Position:    {current_ee_pos_debug.tolist()}")
                    print(f"  Object Position:        {object_pos_debug.tolist()}")
                    print(f"  EE-Object Distance:      {np.linalg.norm(current_ee_pos_debug - object_pos_debug):.3f}")
                    print(f"  =========================================\n")

                # Fill action buffers with first exec_horizon actions
                for env_id in range(num_envs):
                    if len(action_buffers[env_id]) == 0:
                        # Store exec_horizon actions in buffer (as list for easy pop)
                        for t in range(min(exec_horizon, pred_horizon)):
                            action_buffers[env_id].append(
                                action_trajectory[env_id, t, :]
                            )

            # Pop the first action from each buffer and apply smoothing (EMA)
            actions_list = []
            for env_id in range(num_envs):
                if len(action_buffers[env_id]) > 0:
                    # Get raw action from buffer (model's desired action)
                    raw_action = action_buffers[env_id].pop(0)
                    
                    # Apply exponential moving average (EMA) smoothing to reduce jitter
                    # Formula: smoothed_action = (1 - α) * last_action + α * raw_action
                    # This prevents sudden action changes while still following model predictions
                    if last_actions[env_id] is None:
                        # First action: use raw action directly
                        smoothed_action = raw_action
                    else:
                        # Apply EMA smoothing: blend between last action and new prediction
                        smoothed_action = (1 - smoothing_factor) * last_actions[env_id] + smoothing_factor * raw_action
                    
                    # Update last action for next iteration
                    last_actions[env_id] = smoothed_action
                    actions_list.append(smoothed_action)
                else:
                    # Fallback: should not happen if logic is correct
                    print(
                        f"[Play] WARNING: Empty action buffer for env {env_id}, using zeros!"
                    )
                    zero_action = torch.zeros(action_dim, device=device)
                    # Apply smoothing even for zero action
                    if last_actions[env_id] is not None:
                        zero_action = (1 - smoothing_factor) * last_actions[env_id] + smoothing_factor * zero_action
                    last_actions[env_id] = zero_action
                    actions_list.append(zero_action)

            # Stack into batch tensors
            actions = torch.stack(actions_list, dim=0)  # [num_envs, action_dim] - ABSOLUTE actions (for env.step)
            
            # Store ABSOLUTE target actions in history buffer
            # Training: action_history[i, j] = actions[j] (absolute positions)
            # Inference: Store target actions (from model predictions) in history buffer
            # Extract target action (first action from trajectory, which is what we're executing)
            # Store ABSOLUTE target actions in history buffer
            action_history_buffers.update(actions)  # Store target actions (absolute positions)
            node_history_buffers.update(ee_node_current, object_node_current)  # [num_envs, 7] each
            if joint_dim is not None and joint_state_history_buffers is not None:
                joint_state_history_buffers.update(joint_states_current)  # [num_envs, joint_dim]
            
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
                done_env_ids = torch.where(done)[0]  # Get all done environment IDs
                
                # 🔧 FIX: Reset history buffers for done environments
                # First reset (clear to zeros)
                action_history_buffers.reset(env_ids=done_env_ids)
                node_history_buffers.reset(env_ids=done_env_ids)
                if (
                    joint_dim is not None
                    and joint_state_history_buffers is not None
                ):
                    joint_state_history_buffers.reset(env_ids=done_env_ids)
                
                # 🔧 FIX: Re-initialize with new observation (matches training behavior!)
                # Get the new observation for done environments (obs is already updated after env.step)
                # Need to process the new obs first (obs is the new observation after env.step)
                new_obs_tensor = process_obs_to_tensor(obs)  # Process new obs after env.step
                done_obs_tensor = new_obs_tensor[done_env_ids]  # [num_done_envs, obs_dim]
                initialize_buffers_from_obs(done_obs_tensor, env_ids=done_env_ids)
                
                # Reset action smoothing state for done environments
                for env_id in done_env_ids:
                    env_id_int = env_id.item() if hasattr(env_id, 'item') else env_id
                    last_actions[env_id_int] = None

                for i in range(num_envs):
                    if done[i]:
                        episode_rewards.append(current_episode_rewards[i].item())
                        episode_count += 1
                        current_episode_rewards[i] = 0.0
                        
                        # CRITICAL: Clear action buffer on episode reset!
                        # This forces re-planning at the start of each new episode
                        action_buffers[i].clear()
                        
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
    parser = argparse.ArgumentParser(description="Play Graph-DiT Policy")
    
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Policy checkpoint path"
    )
    parser.add_argument(
        "--num_envs", type=int, default=64, help="Number of parallel environments"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--num_diffusion_steps",
        type=int,
        default=None,
        help="Number of diffusion steps for inference (None = auto based on mode, DDPM: 50, Flow Matching: 10)",
    )
    
    args = parser.parse_args()
    
    # Run playback
    play_graph_dit_policy(
        task_name=args.task,
        checkpoint_path=args.checkpoint,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        device=args.device,
        num_diffusion_steps=args.num_diffusion_steps,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()

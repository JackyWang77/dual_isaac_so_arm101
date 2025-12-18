# Copyright (c) 2024-2025, SO-ARM100 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback script for trained Graph-DiT Policy.

This script loads a trained Graph-DiT policy and runs it in an Isaac Lab environment.

Usage:
    ./isaaclab.sh -p scripts/graph_dit/play.py \
        --task SO-ARM100-Pick-Place-DualArm-IK-Abs-v0 \
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

import numpy as np
import torch

import gymnasium as gym
import SO_100.tasks  # noqa: F401  # Register environments
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from SO_100.policies.graph_dit_policy import GraphDiTPolicy


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
        if hasattr(space, 'shape') and space.shape is not None:
            # Box space with shape
            if isinstance(space.shape, tuple):
                # For vectorized env, shape might be (num_envs, dim), take the last dim
                if len(space.shape) > 1:
                    return space.shape[-1]  # Last dimension is the actual obs dim
                else:
                    return space.shape[0]
            else:
                return space.shape if isinstance(space.shape, int) else 1
        elif hasattr(space, 'spaces'):
            # Dict or Tuple space
            if isinstance(space.spaces, dict):
                # For Isaac Lab, if 'policy' key exists, use only that (training uses policy obs)
                if 'policy' in space.spaces:
                    return get_obs_dim(space.spaces['policy'])
                else:
                    # Otherwise, sum all sub-spaces
                    return sum(get_obs_dim(sub_space) for sub_space in space.spaces.values())
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
                    if 'policy' in sample:
                        policy_sample = sample['policy']
                        if hasattr(policy_sample, 'size'):
                            return policy_sample.size if isinstance(policy_sample.size, int) else policy_sample.numel()
                        elif hasattr(policy_sample, '__len__'):
                            return len(policy_sample) if isinstance(policy_sample, (list, tuple)) else policy_sample.shape[-1] if hasattr(policy_sample, 'shape') else 1
                    else:
                        return sum(
                            s.size if hasattr(s, 'size') and isinstance(s.size, int) else 
                            s.numel() if hasattr(s, 'numel') else 
                            len(s) if hasattr(s, '__len__') and isinstance(s, (list, tuple)) else 1
                            for s in sample.values()
                        )
                elif hasattr(sample, 'size'):
                    return sample.size if isinstance(sample.size, int) else sample.numel()
                elif hasattr(sample, '__len__'):
                    if isinstance(sample, (list, tuple)):
                        return len(sample)
                    elif hasattr(sample, 'shape'):
                        return sample.shape[-1] if len(sample.shape) > 0 else 1
                    return len(sample)
                else:
                    return 1
            except Exception as e:
                print(f"[Play] Warning: Could not compute obs_dim from space, using default. Error: {e}")
                return 39  # Default fallback for reach task
    
    obs_dim = get_obs_dim(obs_space)
    
    # For action space, handle vectorized case
    if hasattr(action_space, 'shape') and action_space.shape is not None:
        if isinstance(action_space.shape, tuple) and len(action_space.shape) > 1:
            action_dim = action_space.shape[-1]  # Last dimension is the actual action dim
        else:
            action_dim = action_space.shape[0] if isinstance(action_space.shape, tuple) else action_space.shape
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
    cfg = checkpoint.get('cfg', None)
    if cfg is not None:
        mode = getattr(cfg, 'mode', 'ddpm')
        print(f"[Play] Policy mode: {mode.upper()}")
        # Set default steps based on mode if not provided
        if num_diffusion_steps is None:
            if mode == "flow_matching":
                num_diffusion_steps = 10  # Flow Matching: 1-10 steps
            else:
                num_diffusion_steps = 50  # DDPM: 50-100 steps
            print(f"[Play] Using default diffusion steps for {mode}: {num_diffusion_steps}")
    else:
        if num_diffusion_steps is None:
            num_diffusion_steps = 50  # Default fallback
            print(f"[Play] Using default diffusion steps: {num_diffusion_steps}")
    
    # Load normalization stats (if available)
    obs_stats = checkpoint.get('obs_stats', None)
    action_stats = checkpoint.get('action_stats', None)
    
    if obs_stats is not None:
        print(f"[Play] Loaded observation normalization stats")
        # Handle both numpy arrays and torch tensors
        if isinstance(obs_stats['mean'], np.ndarray):
            obs_mean = torch.from_numpy(obs_stats['mean']).squeeze().to(device)
            obs_std = torch.from_numpy(obs_stats['std']).squeeze().to(device)
        else:
            obs_mean = obs_stats['mean'].squeeze().to(device)
            obs_std = obs_stats['std'].squeeze().to(device)
    else:
        print(f"[Play] Warning: No observation stats found, skipping normalization")
        obs_mean = None
        obs_std = None
    
    if action_stats is not None:
        print(f"[Play] Loaded action normalization stats")
        # Handle both numpy arrays and torch tensors
        if isinstance(action_stats['mean'], np.ndarray):
            action_mean = torch.from_numpy(action_stats['mean']).squeeze().to(device)
            action_std = torch.from_numpy(action_stats['std']).squeeze().to(device)
        else:
            action_mean = action_stats['mean'].squeeze().to(device)
            action_std = action_stats['std'].squeeze().to(device)
    else:
        print(f"[Play] Warning: No action stats found, skipping denormalization")
        action_mean = None
        action_std = None
    
    # Get action_history_length from policy config
    action_history_length = policy.cfg.action_history_length if hasattr(policy.cfg, 'action_history_length') else 4
    
    # Run episodes
    print(f"\n[Play] Running {num_episodes} episodes...")
    print(f"[Play] Action history length: {action_history_length}")
    
    obs, info = env.reset()
    episode_count = 0
    episode_rewards = []
    current_episode_rewards = torch.zeros(num_envs, device=device)
    
    # Initialize action and node history buffers for each environment
    action_history_buffers = [torch.zeros(action_history_length, action_dim, device=device) for _ in range(num_envs)]
    ee_node_history_buffers = [torch.zeros(action_history_length, 7, device=device) for _ in range(num_envs)]
    object_node_history_buffers = [torch.zeros(action_history_length, 7, device=device) for _ in range(num_envs)]
    
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
        ee_pos = obs_tensor[:, 19:22]   # [batch, 3] - ee_position
        ee_ori = obs_tensor[:, 22:26]   # [batch, 4] - ee_orientation
        
        ee_node = torch.cat([ee_pos, ee_ori], dim=-1)  # [batch, 7]
        object_node = torch.cat([obj_pos, obj_ori], dim=-1)  # [batch, 7]
        return ee_node, object_node
    
    with torch.inference_mode():
        while simulation_app.is_running() and episode_count < num_episodes:
            # Process observations
            if isinstance(obs, dict):
                # Check if there's a 'policy' key (Isaac Lab wrapper format)
                if 'policy' in obs:
                    # Direct policy observations (already concatenated)
                    # But it might still contain target_object_position (39 dims)
                    # We need to extract only the keys we used during training (32 dims)
                    obs_val = obs['policy']
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
                        obs_tensor = torch.cat([
                            obs_tensor_raw[:, :26],  # Everything before target_object_position
                            obs_tensor_raw[:, 33:]   # Everything after target_object_position (actions)
                        ], dim=1)  # Should be 32 dims
                    elif obs_tensor_raw.shape[1] == 32:
                        # Already correct (training format)
                        obs_tensor = obs_tensor_raw
                    else:
                        # Unknown dimension, use as is (might cause issues)
                        print(f"[Play] Warning: Unexpected obs dim {obs_tensor_raw.shape[1]}, expected 32 or 39")
                        obs_tensor = obs_tensor_raw
                else:
                    # Concatenate dictionary observations in the correct order
                    # Skip target_object_position even if it exists (redundant with object_position + object_orientation)
                    obs_keys_order = [
                        "joint_pos", "joint_vel",
                        "object_position", "object_orientation",
                        "ee_position", "ee_orientation",
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
                                obs_list.append(torch.from_numpy(obs_val).flatten(start_dim=1))
                            else:
                                obs_list.append(torch.tensor(obs_val).flatten(start_dim=1))
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
                                obs_list.append(torch.from_numpy(val).flatten(start_dim=1))
                            else:
                                obs_list.append(torch.tensor(val).flatten(start_dim=1))
                        obs_tensor = torch.cat(obs_list, dim=1).to(device)
            else:
                obs_tensor = torch.from_numpy(obs).to(device) if isinstance(obs, np.ndarray) else obs
                if len(obs_tensor.shape) == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                elif len(obs_tensor.shape) > 2:
                    obs_tensor = obs_tensor.view(obs_tensor.shape[0], -1)
            
            # Normalize observations (if stats available)
            # Ensure dimensions match (obs_tensor should be 32 dims after removing target_object_position)
            if obs_mean is not None and obs_std is not None:
                if obs_tensor.shape[1] != obs_mean.shape[0]:
                    print(f"[Play] ERROR: Obs dimension mismatch! obs_tensor: {obs_tensor.shape[1]}, obs_mean: {obs_mean.shape[0]}")
                    print(f"[Play] This means target_object_position was not properly removed!")
                    raise RuntimeError(f"Observation dimension mismatch: {obs_tensor.shape[1]} vs {obs_mean.shape[0]}")
                obs_tensor_normalized = (obs_tensor - obs_mean) / obs_std
            else:
                obs_tensor_normalized = obs_tensor
            
            # Extract current node features (before normalization for node history)
            # obs_tensor is now 32 dims (after removing target_object_position)
            # Indices: joint_pos[0:6], joint_vel[6:12], object_position[12:15], object_orientation[15:19],
            #          ee_position[19:22], ee_orientation[22:26], actions[26:32]
            ee_node_current, object_node_current = _extract_node_features_from_obs(obs_tensor)  # [num_envs, 7]
            
            # Build action and node histories for batch
            action_history_batch = []
            ee_node_history_batch = []
            object_node_history_batch = []
            
            for env_id in range(num_envs):
                # Get histories for this environment
                action_history_batch.append(action_history_buffers[env_id].clone())  # [history_length, action_dim]
                ee_node_history_batch.append(ee_node_history_buffers[env_id].clone())  # [history_length, 7]
                object_node_history_batch.append(object_node_history_buffers[env_id].clone())  # [history_length, 7]
            
            # Stack into batch format
            action_history_tensor = torch.stack(action_history_batch, dim=0).to(device)  # [num_envs, history_length, action_dim]
            ee_node_history_tensor = torch.stack(ee_node_history_batch, dim=0).to(device)  # [num_envs, history_length, 7]
            object_node_history_tensor = torch.stack(object_node_history_batch, dim=0).to(device)  # [num_envs, history_length, 7]
            
            # Get subtask condition (optional)
            subtask_condition = None
            
            # Get actions from policy (output is normalized)
            actions_normalized = policy.predict(
                obs_tensor_normalized,
                action_history=action_history_tensor,
                ee_node_history=ee_node_history_tensor,
                object_node_history=object_node_history_tensor,
                subtask_condition=subtask_condition,
                num_diffusion_steps=num_diffusion_steps,
                deterministic=True
            )  # [num_envs, action_dim]
            
            # Denormalize actions (if stats available)
            # Keep as torch.Tensor (Isaac Lab expects tensor, not numpy array)
            if action_mean is not None and action_std is not None:
                actions = actions_normalized * action_std + action_mean  # [num_envs, action_dim]
            else:
                actions = actions_normalized  # [num_envs, action_dim]
                if step_count == 0:
                    print(f"[Play] Warning: No action normalization stats, using normalized actions directly")
            
            # Update history buffers (shift and add new)
            # IMPORTANT: Store normalized actions in history buffer, as policy expects normalized action_history
            for env_id in range(num_envs):
                # Shift action history (use normalized actions, not denormalized!)
                action_history_buffers[env_id] = torch.cat([
                    action_history_buffers[env_id][1:],
                    actions_normalized[env_id:env_id+1]  # [1, action_dim] - Use normalized actions!
                ], dim=0)
                
                # Shift node histories
                ee_node_history_buffers[env_id] = torch.cat([
                    ee_node_history_buffers[env_id][1:],
                    ee_node_current[env_id:env_id+1]
                ], dim=0)
                
                object_node_history_buffers[env_id] = torch.cat([
                    object_node_history_buffers[env_id][1:],
                    object_node_current[env_id:env_id+1]
                ], dim=0)
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Accumulate rewards (rewards might be numpy or tensor)
            if isinstance(rewards, np.ndarray):
                current_episode_rewards += torch.from_numpy(rewards).to(device)
            else:
                current_episode_rewards += rewards.to(device) if hasattr(rewards, 'to') else torch.tensor(rewards, device=device)
            
            # Check for episode completion and reset history buffers
            done = terminated | truncated
            if done.any():
                for i in range(num_envs):
                    if done[i]:
                        episode_rewards.append(current_episode_rewards[i].item())
                        episode_count += 1
                        current_episode_rewards[i] = 0.0
                        
                        # Reset history buffers for this environment
                        action_history_buffers[i].zero_()
                        ee_node_history_buffers[i].zero_()
                        object_node_history_buffers[i].zero_()
                        
                        if episode_count >= num_episodes:
                            break
            
            step_count += 1
            
            # Print progress
            if step_count % 100 == 0:
                avg_reward = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10) if episode_rewards else 0.0
                print(f"[Play] Step: {step_count}, Episodes: {episode_count}/{num_episodes}, "
                      f"Avg reward (last 10): {avg_reward:.3f}")
    
    # Print final statistics
    if episode_rewards:
        print(f"\n[Play] ===== Final Statistics =====")
        print(f"[Play] Total episodes: {len(episode_rewards)}")
        print(f"[Play] Average reward: {sum(episode_rewards) / len(episode_rewards):.3f}")
        print(f"[Play] Max reward: {max(episode_rewards):.3f}")
        print(f"[Play] Min reward: {min(episode_rewards):.3f}")
    
    # Close environment
    env.close()
    print(f"\n[Play] Playback completed!")


def main():
    """Main playback function."""
    parser = argparse.ArgumentParser(description="Play Graph-DiT Policy")
    
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint path")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num_diffusion_steps", type=int, default=None, help="Number of diffusion steps for inference (None = auto based on mode, DDPM: 50, Flow Matching: 10)")
    
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



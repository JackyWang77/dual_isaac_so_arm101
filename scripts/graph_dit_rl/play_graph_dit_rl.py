#!/usr/bin/env python3
"""
Playback script for trained GraphDiT + Residual RL Policy.

Usage:
    ./isaaclab.sh -p scripts/graph_dit_rl/play_graph_dit_rl.py \
        --task SO-ARM101-Lift-Cube-v0 \
        --checkpoint ./logs/graph_dit_rl/policy_final.pt \
        --pretrained_checkpoint ./logs/graph_dit/best_model.pt \
        --num_envs 64
"""

from isaaclab.app import AppLauncher

# CLI args for AppLauncher (parse early to get headless flag)
import sys
import argparse
parser_launcher = argparse.ArgumentParser(description="Play GraphDiT + Residual RL Policy")
AppLauncher.add_app_launcher_args(parser_launcher)
args_launcher, _ = parser_launcher.parse_known_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_launcher)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import numpy as np

import gymnasium as gym
import SO_101.tasks  # noqa: F401  # Register environments
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from SO_101.policies.graph_dit_policy import GraphDiTPolicy
from SO_101.policies.graph_dit_residual_rl_policy import (
    GraphDiTBackboneAdapter,
    GraphDiTResidualRLPolicy,
)


def play_graph_dit_rl_policy(
    task_name: str,
    checkpoint_path: str,
    pretrained_checkpoint: str,
    num_envs: int = 64,
    num_episodes: int = 10,
    device: str = "cuda",
    deterministic: bool = True,
):
    """Play trained GraphDiT + Residual RL policy.

    Args:
        task_name: Environment task name.
        checkpoint_path: Path to trained RL policy checkpoint.
        pretrained_checkpoint: Path to pretrained GraphDiT checkpoint (for backbone).
        num_envs: Number of parallel environments.
        num_episodes: Number of episodes to run.
        device: Device to run on.
        deterministic: If True, use deterministic actions.
    """

    print(f"[Play] ===== GraphDiT + Residual RL Policy Playback =====")
    print(f"[Play] Task: {task_name}")
    print(f"[Play] RL Checkpoint: {checkpoint_path}")
    print(f"[Play] GraphDiT Checkpoint: {pretrained_checkpoint}")
    print(f"[Play] Num envs: {num_envs}")

    # Load pretrained GraphDiT backbone
    print(f"\n[Play] Loading pretrained GraphDiT: {pretrained_checkpoint}")
    graph_dit = GraphDiTPolicy.load(pretrained_checkpoint, device=device)
    graph_dit.eval()
    for p in graph_dit.parameters():
        p.requires_grad = False
    print(f"[Play] GraphDiT loaded and frozen")

    # Create backbone adapter
    backbone = GraphDiTBackboneAdapter(graph_dit)

    # Load RL policy
    print(f"\n[Play] Loading RL policy: {checkpoint_path}")
    policy = GraphDiTResidualRLPolicy.load(checkpoint_path, backbone=backbone, device=device)
    policy.eval()
    print(f"[Play] RL policy loaded")

    # Load normalization stats from GraphDiT checkpoint (same as train_graph_dit_rl.py)
    checkpoint = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    
    # Load obs stats
    obs_stats = checkpoint.get("obs_stats", None)
    obs_mean, obs_std = None, None
    if obs_stats is not None:
        if isinstance(obs_stats["mean"], np.ndarray):
            obs_mean = torch.from_numpy(obs_stats["mean"]).squeeze().to(device)
            obs_std = torch.from_numpy(obs_stats["std"]).squeeze().to(device)
        else:
            obs_mean = obs_stats["mean"].squeeze().to(device)
            obs_std = obs_stats["std"].squeeze().to(device)
        print(f"[Play] Loaded obs normalization stats")
    
    # Load node stats
    node_stats = checkpoint.get("node_stats", None)
    ee_node_mean, ee_node_std = None, None
    object_node_mean, object_node_std = None, None
    if node_stats is not None:
        if isinstance(node_stats["ee_mean"], np.ndarray):
            ee_node_mean = torch.from_numpy(node_stats["ee_mean"]).squeeze().to(device)
            ee_node_std = torch.from_numpy(node_stats["ee_std"]).squeeze().to(device)
            object_node_mean = torch.from_numpy(node_stats["object_mean"]).squeeze().to(device)
            object_node_std = torch.from_numpy(node_stats["object_std"]).squeeze().to(device)
        else:
            ee_node_mean = node_stats["ee_mean"].squeeze().to(device)
            ee_node_std = node_stats["ee_std"].squeeze().to(device)
            object_node_mean = node_stats["object_mean"].squeeze().to(device)
            object_node_std = node_stats["object_std"].squeeze().to(device)
        print(f"[Play] Loaded node normalization stats")
    
    # Load action stats
    action_stats = checkpoint.get("action_stats", None)
    action_mean, action_std = None, None
    if action_stats is not None:
        if isinstance(action_stats["mean"], np.ndarray):
            action_mean = torch.from_numpy(action_stats["mean"]).squeeze().to(device)
            action_std = torch.from_numpy(action_stats["std"]).squeeze().to(device)
        else:
            action_mean = action_stats["mean"].squeeze().to(device)
            action_std = action_stats["std"].squeeze().to(device)
        print(f"[Play] Loaded action normalization stats")
    
    # Load joint stats
    joint_stats = checkpoint.get("joint_stats", None)
    joint_mean, joint_std = None, None
    if joint_stats is not None:
        if isinstance(joint_stats["mean"], np.ndarray):
            joint_mean = torch.from_numpy(joint_stats["mean"]).squeeze().to(device)
            joint_std = torch.from_numpy(joint_stats["std"]).squeeze().to(device)
        else:
            joint_mean = joint_stats["mean"].squeeze().to(device)
            joint_std = joint_stats["std"].squeeze().to(device)
        print(f"[Play] Loaded joint normalization stats")
    
    # Set normalization stats in policy (same as train_graph_dit_rl.py)
    policy.set_normalization_stats(
        ee_node_mean=ee_node_mean,
        ee_node_std=ee_node_std,
        object_node_mean=object_node_mean,
        object_node_std=object_node_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    print(f"[Play] Set normalization stats in policy")

    # Create environment
    print(f"\n[Play] Creating environment...")
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    env = gym.make(task_name, cfg=env_cfg)

    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space

    print(f"[Play] Observation space: {obs_space}")
    print(f"[Play] Action space: {action_space}")

    # Initialize policy buffers
    policy.init_env_buffers(num_envs)

    # History buffers for Graph-DiT (optimized: single tensor per history type)
    action_history_length = getattr(graph_dit.cfg, 'action_history_length', 4)
    # NOTE: GraphDiT expects 6 dimensions for joint states (was trained with 6)
    # The residual RL config's joint_dim=5 is only for EMA smoothing, not for joint state extraction
    graph_dit_joint_dim = 6  # Always 6 for GraphDiT compatibility
    action_dim = policy.cfg.action_dim if hasattr(policy, 'cfg') and hasattr(policy.cfg, 'action_dim') else 6
    
    # Use single tensor [num_envs, history_len, dim] for efficiency
    action_history = torch.zeros(num_envs, action_history_length, action_dim, device=device)
    ee_node_history = torch.zeros(num_envs, action_history_length, 7, device=device)
    object_node_history = torch.zeros(num_envs, action_history_length, 7, device=device)
    joint_state_history = torch.zeros(num_envs, action_history_length, graph_dit_joint_dim, device=device)

    # Reset environment
    obs, info = env.reset()
    obs = _process_obs(obs, device)

    # Stats
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    completed_episodes = 0

    print(f"\n[Play] Starting playback (deterministic={deterministic})...")
    print(f"[Play] Press Ctrl+C to stop\n")

    # Main loop
    step_count = 0
    try:
        while simulation_app.is_running() and completed_episodes < num_episodes:
            # CRITICAL: Normalize observations (same as train_graph_dit_rl.py)
            if obs_mean is not None and obs_std is not None:
                obs_norm = (obs - obs_mean) / obs_std
            else:
                obs_norm = obs
            
            # Extract current node features and joint states from obs
            ee_node_current, object_node_current = policy._extract_nodes_from_obs(obs)
            # NOTE: GraphDiT expects 6 dimensions (was trained with 6), regardless of residual RL config
            joint_states_current = obs[:, :6]  # [B, 6] - always 6 for GraphDiT compatibility
            
            # CRITICAL: Normalize node and joint histories (same as train_graph_dit_rl.py)
            # History tensors are already in batch format [num_envs, history_len, dim]
            action_history_norm = action_history  # [B, H, action_dim] - already normalized (stored as normalized)
            
            # Normalize node histories
            ee_node_history_norm = ee_node_history.clone()  # [B, H, 7]
            object_node_history_norm = object_node_history.clone()  # [B, H, 7]
            if ee_node_mean is not None and ee_node_std is not None:
                ee_node_history_norm = (ee_node_history_norm - ee_node_mean) / ee_node_std
            if object_node_mean is not None and object_node_std is not None:
                object_node_history_norm = (object_node_history_norm - object_node_mean) / object_node_std
            
            # Normalize joint history
            joint_state_history_norm = joint_state_history.clone()  # [B, H, 6]
            if joint_mean is not None and joint_std is not None:
                joint_state_history_norm = (joint_state_history_norm - joint_mean) / joint_std
            
            # Get action from policy
            with torch.no_grad():
                action, policy_info = policy.act(
                    obs_raw=obs,
                    obs_norm=obs_norm,  # Use normalized obs
                    action_history=action_history_norm,  # [num_envs, history_len, action_dim] - normalized
                    ee_node_history=ee_node_history_norm,  # [num_envs, history_len, 7] - normalized
                    object_node_history=object_node_history_norm,  # [num_envs, history_len, 7] - normalized
                    joint_states_history=joint_state_history_norm,  # [num_envs, history_len, joint_dim] - normalized
                    deterministic=deterministic
                )

            # Clip action to action space bounds
            if hasattr(env, 'action_space'):
                if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                    action = torch.clamp(
                        action,
                        torch.tensor(env.action_space.low, device=action.device, dtype=action.dtype),
                        torch.tensor(env.action_space.high, device=action.device, dtype=action.dtype)
                    )
            
            # DEBUG: Print all joints and gripper base action, residual, and final action (same as train_graph_dit_rl.py)
            action_dim = action.shape[-1]
            if action_dim >= 6:
                a_base = policy_info.get("a_base", None)  # [B, action_dim] - denormalized base action
                delta = policy_info.get("delta", None)  # [B, action_dim] - denormalized delta
                delta_norm = policy_info.get("delta_norm", None)  # [B, action_dim] - normalized delta (Actor output)
                alpha_val = policy_info.get("alpha", None)
                if alpha_val is not None:
                    if isinstance(alpha_val, torch.Tensor):
                        alpha = alpha_val.item()
                    else:
                        alpha = float(alpha_val)
                else:
                    alpha = policy.alpha.item() if hasattr(policy, 'alpha') else 0.1
                
                if a_base is not None and delta is not None:
                    # Print for first env only, every 10 steps
                    if step_count % 10 == 0:
                        print(f"\n[Play Step {step_count:3d}] Action Debug (env 0):")
                        print(f"  alpha: {alpha:.4f}")
                        print(f"\n  Joints (0-4):")
                        for i in range(min(5, action_dim)):
                            a_base_i = a_base[0, i].item()
                            delta_i = delta[0, i].item()
                            final_i = action[0, i].item()
                            print(f"    Joint {i}: base={a_base_i:7.4f}, delta={delta_i:7.4f}, final={final_i:7.4f} (base + {alpha:.2f}*delta = {a_base_i + alpha * delta_i:.4f})")
                        
                        # Gripper (index 5) - before binarization
                        if action_dim >= 6:
                            a_base_gripper = a_base[0, 5].item()
                            delta_gripper = delta[0, 5].item()
                            gripper_continuous_before_bin = action[0, 5].item()
                            print(f"\n  Gripper (5) - Before Binarization:")
                            print(f"    base: {a_base_gripper:7.4f}")
                            print(f"    delta: {delta_gripper:7.4f}")
                            print(f"    continuous (base + {alpha:.2f}*delta): {gripper_continuous_before_bin:7.4f}")
                            print(f"    threshold: -0.2")
                        
                        # Statistics across all envs
                        print(f"\n  Delta Statistics (all envs):")
                        delta_mean = delta.mean(dim=0)  # [action_dim]
                        delta_std = delta.std(dim=0)  # [action_dim]
                        delta_min = delta.min(dim=0)[0]  # [action_dim]
                        delta_max = delta.max(dim=0)[0]  # [action_dim]
                        delta_positive_ratio = (delta > 0).float().mean(dim=0)  # [action_dim]
                        
                        print(f"    Joints (0-4):")
                        for i in range(min(5, action_dim)):
                            print(f"      Joint {i}: mean={delta_mean[i].item():7.4f}, std={delta_std[i].item():7.4f}, "
                                  f"min={delta_min[i].item():7.4f}, max={delta_max[i].item():7.4f}, "
                                  f"positive={delta_positive_ratio[i].item():.1%}")
                        
                        if action_dim >= 6:
                            print(f"    Gripper (5): mean={delta_mean[5].item():7.4f}, std={delta_std[5].item():7.4f}, "
                                  f"min={delta_min[5].item():7.4f}, max={delta_max[5].item():7.4f}, "
                                  f"positive={delta_positive_ratio[5].item():.1%}")
                        
                        if delta_norm is not None:
                            delta_norm_mean = delta_norm.mean(dim=0)  # [action_dim]
                            delta_norm_std = delta_norm.std(dim=0)  # [action_dim]
                            print(f"\n  Delta_norm Statistics (normalized, all envs):")
                            print(f"    Joints (0-4):")
                            for i in range(min(5, action_dim)):
                                print(f"      Joint {i}: mean={delta_norm_mean[i].item():7.4f}, std={delta_norm_std[i].item():7.4f}")
                            if action_dim >= 6:
                                print(f"    Gripper (5): mean={delta_norm_mean[5].item():7.4f}, std={delta_norm_std[5].item():7.4f}")
                
                # CRITICAL: Binarize gripper action (same as play.py)
                # Action is already: a_base + alpha * delta (continuous value for all dimensions including gripper)
                # Isaac Sim expects gripper: > -0.2 -> 1.0 (open), <= -0.2 -> -1.0 (close)
                # This allows RL to learn fine-tuning of gripper timing, especially when base policy
                # outputs values near -0.2 threshold
                gripper_continuous = action[:, 5]  # [num_envs] - continuous value: a_base_gripper + alpha * delta_gripper
                gripper_binary = torch.where(
                    gripper_continuous > -0.2,
                    torch.tensor(1.0, device=action.device, dtype=action.dtype),
                    torch.tensor(-1.0, device=action.device, dtype=action.dtype)
                )
                action[:, 5] = gripper_binary
                
                # Print gripper binarization result (for first env, every 10 steps)
                if step_count % 10 == 0 and a_base is not None and delta is not None:
                    gripper_binary_val = gripper_binary[0].item()
                    print(f"  Gripper (5) - After Binarization:")
                    print(f"    binary (final): {gripper_binary_val:7.4f} ({'open' if gripper_binary_val > 0 else 'close'})")

            # Step environment
            next_obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated | truncated

            # Process
            next_obs = _process_obs(next_obs, device)
            reward = reward.to(device).float()
            done = done.to(device).float()

            # Update history buffers
            # NOTE: For test_dit_only, Graph-DiT's action is already in normalized space
            # So we can directly use action for history
            action_for_history = action
            
            # Roll all histories: shift left by 1, new data goes to last position
            action_history = torch.roll(action_history, -1, dims=1)
            action_history[:, -1, :] = action_for_history  # [num_envs, action_dim]
            
            ee_node_history = torch.roll(ee_node_history, -1, dims=1)
            ee_node_history[:, -1, :] = ee_node_current  # [num_envs, 7]
            
            object_node_history = torch.roll(object_node_history, -1, dims=1)
            object_node_history[:, -1, :] = object_node_current  # [num_envs, 7]
            
            joint_state_history = torch.roll(joint_state_history, -1, dims=1)
            joint_state_history[:, -1, :] = joint_states_current  # [num_envs, joint_dim]

            # Track stats
            episode_rewards += reward
            episode_lengths += 1

            # Handle resets
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_envs) > 0:
                policy.reset_envs(done_envs)
                # Reset history buffers for done environments
                for i in done_envs.tolist():
                    completed_episodes += 1
                    # Compute gate entropy from gate_w
                    gate_w = policy_info["gate_w"][i]
                    gate_entropy = -(gate_w * torch.log(gate_w + 1e-8)).sum().item()
                    print(
                        f"[Play] Episode {completed_episodes:3d} | "
                        f"Reward: {episode_rewards[i].item():7.2f} | "
                        f"Length: {episode_lengths[i].item():4.0f} | "
                        f"Alpha: {policy_info['alpha'].item():.3f} | "
                        f"Gate Entropy: {gate_entropy:.3f}"
                    )
                
                # Vectorized reset: zero out all done environments at once
                action_history[done_envs] = 0
                ee_node_history[done_envs] = 0
                object_node_history[done_envs] = 0
                joint_state_history[done_envs] = 0
                episode_rewards[done_envs] = 0
                episode_lengths[done_envs] = 0

            obs = next_obs
            step_count += 1

    except KeyboardInterrupt:
        print("\n[Play] Interrupted by user")

    print(f"\n[Play] Playback complete!")
    print(f"[Play] Total steps: {step_count}")
    print(f"[Play] Completed episodes: {completed_episodes}")

    # Cleanup
    env.close()


def test_dit_only(
    task_name: str,
    pretrained_checkpoint: str,
    num_envs: int = 64,
    num_steps: int = 200,
    device: str = "cuda",
):
    """Test Graph-DiT base_action only, without residual RL.
    
    This function tests if the Graph-DiT backbone can produce reasonable
    base actions on its own, without any residual corrections.
    
    Args:
        task_name: Environment task name.
        pretrained_checkpoint: Path to pretrained GraphDiT checkpoint.
        num_envs: Number of parallel environments.
        num_steps: Number of steps to run.
        device: Device to run on.
    """
    print(f"\n[Test DiT Only] ===== Testing Graph-DiT Base Action Only =====")
    print(f"[Test DiT Only] Task: {task_name}")
    print(f"[Test DiT Only] GraphDiT Checkpoint: {pretrained_checkpoint}")
    print(f"[Test DiT Only] Num envs: {num_envs}")
    print(f"[Test DiT Only] Num steps: {num_steps}")
    
    # Load pretrained GraphDiT backbone
    print(f"\n[Test DiT Only] Loading pretrained GraphDiT: {pretrained_checkpoint}")
    graph_dit = GraphDiTPolicy.load(pretrained_checkpoint, device=device)
    graph_dit.eval()
    for p in graph_dit.parameters():
        p.requires_grad = False
    print(f"[Test DiT Only] GraphDiT loaded and frozen")
    
    # Create backbone adapter
    backbone = GraphDiTBackboneAdapter(graph_dit)
    
    # Create environment first to get obs/action dims
    print(f"\n[Test DiT Only] Creating environment...")
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    env = gym.make(task_name, cfg=env_cfg)
    
    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space
    print(f"[Test DiT Only] Observation space: {obs_space}")
    print(f"[Test DiT Only] Action space: {action_space}")
    
    # Get dimensions (handle Dict observation space and vectorized action space)
    if isinstance(obs_space, gym.spaces.Dict):
        # Extract from 'policy' key if available
        if 'policy' in obs_space.spaces:
            policy_obs_space = obs_space['policy']
            if hasattr(policy_obs_space, 'shape') and policy_obs_space.shape is not None:
                # Handle vectorized: shape is (num_envs, obs_dim)
                if len(policy_obs_space.shape) == 2:
                    obs_dim = policy_obs_space.shape[1]
                elif len(policy_obs_space.shape) == 1:
                    obs_dim = policy_obs_space.shape[0]
                else:
                    obs_dim = 32  # Fallback
            else:
                obs_dim = 32  # Fallback
        else:
            obs_dim = 32  # Fallback
    elif hasattr(obs_space, 'shape') and obs_space.shape is not None:
        # Handle vectorized: shape is (num_envs, obs_dim)
        if len(obs_space.shape) == 2:
            obs_dim = obs_space.shape[1]
        elif len(obs_space.shape) == 1:
            obs_dim = obs_space.shape[0]
        else:
            obs_dim = 32  # Fallback
    else:
        obs_dim = 32  # Default fallback
    
    # Get action dimension (handle vectorized action space)
    if hasattr(action_space, 'shape') and action_space.shape is not None:
        # Handle vectorized: shape is (num_envs, action_dim)
        if len(action_space.shape) == 2:
            action_dim = action_space.shape[1]
        elif len(action_space.shape) == 1:
            action_dim = action_space.shape[0]
        else:
            action_dim = 6  # Fallback
    else:
        action_dim = 6  # Default fallback
    
    print(f"[Test DiT Only] Extracted obs_dim: {obs_dim}, action_dim: {action_dim}")
    
    # Create a minimal policy just to use its helper methods
    print(f"\n[Test DiT Only] Creating minimal policy for helper methods...")
    from SO_101.policies.graph_dit_residual_rl_policy import GraphDiTResidualRLCfg
    cfg = GraphDiTResidualRLCfg(
        obs_dim=obs_dim,
        action_dim=action_dim,
        z_dim=graph_dit.cfg.z_dim,
        num_layers=graph_dit.cfg.num_layers,
        obs_structure=getattr(graph_dit.cfg, "obs_structure", None),
        robot_state_dim=6,
    )
    policy = GraphDiTResidualRLPolicy(cfg, backbone=backbone)
    policy.eval()
    
    # Load normalization stats from GraphDiT checkpoint
    checkpoint = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    if "node_stats" in checkpoint:
        node_stats = checkpoint["node_stats"]
        policy.set_normalization_stats(
            ee_node_mean=node_stats.get("ee_mean"),
            ee_node_std=node_stats.get("ee_std"),
            object_node_mean=node_stats.get("object_mean"),
            object_node_std=node_stats.get("object_std"),
        )
        print(f"[Test DiT Only] Loaded normalization stats")
    
    # Initialize history buffers (same as in trainer)
    action_history_length = getattr(graph_dit.cfg, 'action_history_length', 16)
    action_dim = 6  # Assuming 6D action space
    graph_dit_joint_dim = 6  # GraphDiT expects 6 dimensions for joint states
    
    # Use single tensor [num_envs, history_len, dim] for efficiency
    action_history = torch.zeros(num_envs, action_history_length, action_dim, device=device)
    ee_node_history = torch.zeros(num_envs, action_history_length, 7, device=device)
    object_node_history = torch.zeros(num_envs, action_history_length, 7, device=device)
    joint_state_history = torch.zeros(num_envs, action_history_length, graph_dit_joint_dim, device=device)
    
    # Reset environment
    obs, info = env.reset()
    obs = _process_obs(obs, device)
    
    # Stats
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    completed_episodes = 0
    
    print(f"\n[Test DiT Only] Starting test (DiT base_action only, no residual)...")
    print(f"[Test DiT Only] Press Ctrl+C to stop\n")
    
    # Main loop
    step_count = 0
    try:
        while simulation_app.is_running() and step_count < num_steps:
            # Extract current state
            ee_node_current, object_node_current = policy._extract_nodes_from_obs(obs)
            # GraphDiT expects 6 dimensions (was trained with 6)
            joint_states_current = obs[:, :6]  # [B, 6]
            
            # Get base_action only (no residual!)
            with torch.no_grad():
                a_base = policy.get_base_action(
                    obs_norm=obs,
                    action_history=action_history,
                    ee_node_history=ee_node_history,
                    object_node_history=object_node_history,
                    joint_states_history=joint_state_history,
                    deterministic=True,
                )
            
            # Use base_action directly, no residual!
            action = a_base
            
            # Print first env's action every 10 steps
            if step_count % 10 == 0:
                print(f"[{step_count:3d}] a_base: {a_base[0].cpu().numpy().round(3)}")
            
            # Clip action to action space bounds
            if hasattr(env, 'action_space'):
                if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                    action = torch.clamp(
                        action,
                        torch.tensor(env.action_space.low, device=action.device, dtype=action.dtype),
                        torch.tensor(env.action_space.high, device=action.device, dtype=action.dtype)
                    )
            
            # CRITICAL: Binarize gripper action (same as play.py)
            # Action is already: a_base + alpha * delta (continuous value for all dimensions including gripper)
            # Isaac Sim expects gripper: > -0.2 -> 1.0 (open), <= -0.2 -> -1.0 (close)
            # This allows RL to learn fine-tuning of gripper timing, especially when base policy
            # outputs values near -0.2 threshold
            action_dim = action.shape[-1]
            if action_dim >= 6:
                gripper_continuous = action[:, 5]  # [num_envs] - continuous value: a_base_gripper + alpha * delta_gripper
                action[:, 5] = torch.where(
                    gripper_continuous > -0.2,
                    torch.tensor(1.0, device=action.device, dtype=action.dtype),
                    torch.tensor(-1.0, device=action.device, dtype=action.dtype)
                )
            
            # Step environment
            next_obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated | truncated
            
            # Process
            next_obs = _process_obs(next_obs, device)
            reward = reward.to(device).float()
            done = done.to(device).float()
            
            # Update history buffers
            # NOTE: For test_dit_only, Graph-DiT's action is already in normalized space
            # So we can directly use action for history
            action_for_history = action
            
            # Roll all histories: shift left by 1, new data goes to last position
            action_history = torch.roll(action_history, -1, dims=1)
            action_history[:, -1, :] = action_for_history  # [num_envs, action_dim]
            
            ee_node_history = torch.roll(ee_node_history, -1, dims=1)
            ee_node_history[:, -1, :] = ee_node_current  # [num_envs, 7]
            
            object_node_history = torch.roll(object_node_history, -1, dims=1)
            object_node_history[:, -1, :] = object_node_current  # [num_envs, 7]
            
            joint_state_history = torch.roll(joint_state_history, -1, dims=1)
            joint_state_history[:, -1, :] = joint_states_current  # [num_envs, joint_dim]
            
            # Track stats
            episode_rewards += reward
            episode_lengths += 1
            
            # Handle resets
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_envs) > 0:
                # Reset history buffers for done environments
                for i in done_envs.tolist():
                    completed_episodes += 1
                    print(
                        f"[Test DiT Only] Episode {completed_episodes:3d} | "
                        f"Reward: {episode_rewards[i].item():7.2f} | "
                        f"Length: {episode_lengths[i].item():4.0f}"
                    )
                
                # Vectorized reset: zero out all done environments at once
                action_history[done_envs] = 0
                ee_node_history[done_envs] = 0
                object_node_history[done_envs] = 0
                joint_state_history[done_envs] = 0
                episode_rewards[done_envs] = 0
                episode_lengths[done_envs] = 0
            
            obs = next_obs
            step_count += 1
            
            # Small delay for visualization
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n[Test DiT Only] Interrupted by user")
    
    print(f"\n[Test DiT Only] Test complete!")
    print(f"[Test DiT Only] Total steps: {step_count}")
    print(f"[Test DiT Only] Completed episodes: {completed_episodes}")
    if completed_episodes > 0:
        print(f"[Test DiT Only] Average reward: {episode_rewards.sum().item() / completed_episodes:.2f}")
    
    # Cleanup
    env.close()


def _process_obs(obs, device: str) -> torch.Tensor:
    """处理 observation"""
    if isinstance(obs, dict):
        if "policy" in obs:
            obs = obs["policy"]
        else:
            obs = torch.cat([v.flatten(start_dim=1) for v in obs.values()], dim=-1)
    return obs.to(device).float()


def main():
    parser = argparse.ArgumentParser(description="Play GraphDiT + Residual RL Policy")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--checkpoint", type=str, default=None, help="RL policy checkpoint path (required if not testing DiT only)")
    parser.add_argument(
        "--pretrained_checkpoint", type=str, required=True, help="GraphDiT checkpoint path"
    )
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of steps (for test_dit_only mode)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic actions"
    )
    parser.add_argument(
        "--test_dit_only", action="store_true", 
        help="Test Graph-DiT base_action only (no residual RL). Requires --pretrained_checkpoint only."
    )

    args = parser.parse_args()

    if args.test_dit_only:
        # Test DiT only mode
        test_dit_only(
            task_name=args.task,
            pretrained_checkpoint=args.pretrained_checkpoint,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            device=args.device,
        )
    else:
        # Normal playback mode
        if args.checkpoint is None:
            parser.error("--checkpoint is required when not using --test_dit_only")
        
        play_graph_dit_rl_policy(
            task_name=args.task,
            checkpoint_path=args.checkpoint,
            pretrained_checkpoint=args.pretrained_checkpoint,
            num_envs=args.num_envs,
            num_episodes=args.num_episodes,
            device=args.device,
            deterministic=args.deterministic,
        )


if __name__ == "__main__":
    main()
    simulation_app.close()

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

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse
from collections import deque

import gymnasium as gym
import numpy as np
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
        print(f"[Play] Loaded normalization stats")

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
            # Extract current node features and joint states from obs
            ee_node_current, object_node_current = policy._extract_nodes_from_obs(obs)
            # NOTE: GraphDiT expects 6 dimensions (was trained with 6), regardless of residual RL config
            joint_states_current = obs[:, :6]  # [B, 6] - always 6 for GraphDiT compatibility
            
            # History tensors are already in batch format [num_envs, history_len, dim]
            # No need to stack, just use directly
            
            # Get action from policy
            with torch.no_grad():
                action, policy_info = policy.act(
                    obs_raw=obs,
                    obs_norm=obs,
                    action_history=action_history,  # [num_envs, history_len, action_dim]
                    ee_node_history=ee_node_history,  # [num_envs, history_len, 7]
                    object_node_history=object_node_history,  # [num_envs, history_len, 7]
                    joint_states_history=joint_state_history,  # [num_envs, history_len, joint_dim]
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

            # Step environment
            next_obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated | truncated

            # Process
            next_obs = _process_obs(next_obs, device)
            reward = reward.to(device).float()
            done = done.to(device).float()

            # Update history buffers (optimized: use torch.roll for efficiency)
            action_normalized = action  # Assuming action is already in normalized space
            
            # Roll all histories: shift left by 1, new data goes to last position
            action_history = torch.roll(action_history, -1, dims=1)
            action_history[:, -1, :] = action_normalized  # [num_envs, action_dim]
            
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
    parser.add_argument("--checkpoint", type=str, required=True, help="RL policy checkpoint path")
    parser.add_argument(
        "--pretrained_checkpoint", type=str, required=True, help="GraphDiT checkpoint path"
    )
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic actions"
    )

    args = parser.parse_args()

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

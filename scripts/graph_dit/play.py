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
import torch

import gymnasium as gym
import SO_100.tasks  # noqa: F401  # Register environments
from SO_100.policies.graph_dit_policy import GraphDiTPolicy


def play_graph_dit_policy(
    task_name: str,
    checkpoint_path: str,
    num_envs: int = 64,
    num_episodes: int = 10,
    device: str = "cuda",
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
    env_cfg = None  # Auto-load from task
    env = gym.make(task_name, cfg=env_cfg, num_envs=num_envs)
    
    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space
    
    print(f"[Play] Observation space: {obs_space}")
    print(f"[Play] Action space: {action_space}")
    
    # Compute observation dimension
    if hasattr(obs_space, 'shape'):
        obs_dim = sum(obs_space.shape) if isinstance(obs_space.shape, tuple) else obs_space.shape[0]
    else:
        # Dictionary space: need to compute
        if isinstance(obs_space, dict):
            obs_dim = sum(
                sum(space.shape) if hasattr(space, 'shape') else 1
                for space in obs_space.values()
            )
        else:
            obs_dim = 72  # Default fallback
    
    action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else 8
    
    print(f"[Play] Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Load policy
    print(f"\n[Play] Loading policy from: {checkpoint_path}")
    policy = GraphDiTPolicy.load(checkpoint_path, device=device)
    policy.eval()
    
    # Run episodes
    print(f"\n[Play] Running {num_episodes} episodes...")
    
    obs, info = env.reset()
    episode_count = 0
    episode_rewards = []
    current_episode_rewards = torch.zeros(num_envs, device=device)
    
    step_count = 0
    
    with torch.inference_mode():
        while simulation_app.is_running() and episode_count < num_episodes:
            # Process observations
            if isinstance(obs, dict):
                # Concatenate dictionary observations
                obs_list = []
                for key in sorted(obs.keys()):
                    obs_val = obs[key]
                    if isinstance(obs_val, torch.Tensor):
                        obs_list.append(obs_val.flatten(start_dim=1))
                    else:
                        obs_list.append(torch.from_numpy(obs_val).flatten(start_dim=1))
                obs_tensor = torch.cat(obs_list, dim=1).to(device)
            else:
                obs_tensor = torch.from_numpy(obs).to(device)
                if len(obs_tensor.shape) == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
            
            # Get actions from policy
            actions = policy.predict(obs_tensor, deterministic=True)
            actions = actions.cpu().numpy()
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Accumulate rewards
            current_episode_rewards += torch.from_numpy(rewards).to(device)
            
            # Check for episode completion
            done = terminated | truncated
            if done.any():
                for i in range(num_envs):
                    if done[i]:
                        episode_rewards.append(current_episode_rewards[i].item())
                        episode_count += 1
                        current_episode_rewards[i] = 0.0
                        
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
    
    args = parser.parse_args()
    
    # Run playback
    play_graph_dit_policy(
        task_name=args.task,
        checkpoint_path=args.checkpoint,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        device=args.device,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()



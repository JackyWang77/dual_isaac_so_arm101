# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback script for trained Graph DiT RL Policy (fine-tuned with RL).

This script loads a trained Graph DiT RL policy and runs it in an Isaac Lab environment.

Usage:
    ./isaaclab.sh -p scripts/graph_dit_rl/play_rl.py \
        --task SO-ARM101-Reach-Cube-v0 \
        --checkpoint ./logs/graph_dit_rl/SO-ARM101-Reach-Cube-v0/2025-01-10_15-30-45/final_checkpoint.pt \
        --num_envs 64
"""

"""Launch Isaac Sim Simulator first."""

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
from SO_101.policies.graph_dit_rl_policy import GraphDiTRLPolicy


def play_graph_dit_rl_policy(
    task_name: str,
    checkpoint_path: str,
    num_envs: int = 64,
    num_episodes: int = 10,
    device: str = "cuda",
    deterministic: bool = True,
):
    """Play trained Graph DiT RL policy.

    Args:
        task_name: Environment task name.
        checkpoint_path: Path to trained RL policy checkpoint.
        num_envs: Number of parallel environments.
        num_episodes: Number of episodes to run.
        device: Device to run on.
        deterministic: If True, use deterministic actions (mean), else sample from distribution.
    """
    print(f"[Play] ===== Graph DiT RL Policy Playback =====")
    print(f"[Play] Task: {task_name}")
    print(f"[Play] Checkpoint: {checkpoint_path}")
    print(f"[Play] Num envs: {num_envs}")
    print(f"[Play] Deterministic: {deterministic}")

    # Load policy
    print(f"\n[Play] Loading policy...")
    policy = GraphDiTRLPolicy.load(checkpoint_path, device=device, weights_only=False)
    policy.eval()

    # Create environment
    print(f"\n[Play] Creating environment...")
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    env = gym.make(task_name, cfg=env_cfg)

    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space

    print(f"[Play] Observation space: {obs_space}")
    print(f"[Play] Action space: {action_space}")

    # Helper to extract observation from dict
    def extract_obs(obs_dict):
        """Extract observation vector from dict."""
        if isinstance(obs_dict, dict):
            # Isaac Lab typically uses 'policy' key for observations
            if "policy" in obs_dict:
                obs = obs_dict["policy"]
                # Check if we need to remove target_object_position (if present)
                if isinstance(obs, torch.Tensor):
                    if obs.shape[-1] == 39:  # Has target_object_position
                        # Remove dimensions 26:33 (7 dims for target_object_position)
                        obs = torch.cat([obs[..., :26], obs[..., 33:]], dim=-1)
                    return obs
                elif isinstance(obs, np.ndarray):
                    if obs.shape[-1] == 39:
                        obs = np.concatenate([obs[..., :26], obs[..., 33:]], axis=-1)
                    return obs
            else:
                # Concatenate all observations
                obs_list = []
                for key in sorted(obs_dict.keys()):
                    obs_val = obs_dict[key]
                    if isinstance(obs_val, (torch.Tensor, np.ndarray)):
                        if len(obs_val.shape) > 1:
                            obs_list.append(obs_val.flatten())
                        else:
                            obs_list.append(obs_val)
                if obs_list:
                    if isinstance(obs_list[0], torch.Tensor):
                        return torch.cat(obs_list, dim=-1)
                    else:
                        return np.concatenate(obs_list, axis=-1)
        return obs_dict

    # Reset environment
    obs, info = env.reset()
    obs = extract_obs(obs)

    # Convert to tensor if needed
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float().to(device)
    elif not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

    # Statistics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    current_episode_rewards = torch.zeros(num_envs, device=device)
    current_episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.int32)
    total_episodes = 0

    print(f"\n[Play] Starting playback...")
    print(f"[Play] Press Ctrl+C to stop\n")

    try:
        step = 0
        while total_episodes < num_episodes:
            # Get actions from policy
            with torch.no_grad():
                actions, _ = policy.act(obs, deterministic=deterministic)

            # Convert actions to numpy if needed
            if isinstance(actions, torch.Tensor):
                actions_np = actions.cpu().numpy()
            else:
                actions_np = actions

            # Step environment
            next_obs, rewards, terminated, truncated, info = env.step(actions_np)
            done = terminated | truncated

            # Update statistics
            if isinstance(rewards, torch.Tensor):
                current_episode_rewards += rewards.cpu()
            else:
                current_episode_rewards += torch.from_numpy(rewards).float()
            current_episode_lengths += 1

            # Handle completed episodes
            if done.any():
                for i in range(num_envs):
                    if done[i]:
                        total_episodes += 1
                        reward = current_episode_rewards[i].item()
                        length = current_episode_lengths[i].item()
                        episode_rewards.append(reward)
                        episode_lengths.append(length)

                        print(
                            f"[Play] Episode {total_episodes:3d} | "
                            f"Reward: {reward:7.2f} | "
                            f"Length: {length:4d} steps | "
                            f"Avg reward (last 100): {np.mean(episode_rewards):7.2f}"
                        )

                        current_episode_rewards[i] = 0
                        current_episode_lengths[i] = 0

            # Update observation
            obs = extract_obs(next_obs)
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float().to(device)
            elif not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32, device=device)

            step += 1

    except KeyboardInterrupt:
        print("\n[Play] Interrupted by user")

    # Final statistics
    print(f"\n[Play] ===== Playback Complete =====")
    if episode_rewards:
        print(f"[Play] Total episodes: {total_episodes}")
        print(
            f"[Play] Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
        )
        print(
            f"[Play] Average length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}"
        )
        print(f"[Play] Max reward: {np.max(episode_rewards):.2f}")
        print(f"[Play] Min reward: {np.min(episode_rewards):.2f}")

    env.close()


def main():
    """Main playback function."""
    parser = argparse.ArgumentParser(description="Play Graph DiT RL Policy")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to RL policy checkpoint"
    )
    parser.add_argument(
        "--num_envs", type=int, default=64, help="Number of environments"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of episodes to run"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (mean)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (sample from distribution)",
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    args = parser.parse_args()

    # Handle deterministic vs stochastic
    deterministic = not args.stochastic if args.stochastic else args.deterministic

    play_graph_dit_rl_policy(
        task_name=args.task,
        checkpoint_path=args.checkpoint,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        device=args.device,
        deterministic=deterministic,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()

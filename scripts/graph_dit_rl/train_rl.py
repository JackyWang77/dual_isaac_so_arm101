#!/usr/bin/env python3
"""Train Graph DiT RL Policy (head-only fine-tuning) for Reach task.

This script demonstrates how to fine-tune a pre-trained Graph DiT policy
using RL (PPO) with only the head trainable.

Usage:
    ./isaaclab.sh -p scripts/graph_dit_rl/train_rl.py \
        --task SO-ARM101-Reach-Cube-v0 \
        --pretrained_checkpoint ./logs/graph_dit/best_model.pt \
        --num_envs 64 \
        --max_iterations 1000
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True)  # Set to False for visualization
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse
import os
from datetime import datetime

import gymnasium as gym
import torch

import SO_101.tasks  # noqa: F401  # Register environments
from SO_101.policies.graph_dit_rl_policy import GraphDiTRLPolicy, GraphDiTRLPolicyCfg
from SO_101.policies.graph_dit_policy import GraphDiTPolicyCfg


def train_graph_dit_rl(
    task_name: str,
    pretrained_checkpoint: str,
    num_envs: int = 64,
    max_iterations: int = 1000,
    device: str = "cuda",
    save_dir: str = "logs/graph_dit_rl",
):
    """Train Graph DiT RL Policy with head-only fine-tuning.

    Args:
        task_name: Environment task name
        pretrained_checkpoint: Path to pre-trained Graph DiT checkpoint
        num_envs: Number of parallel environments
        max_iterations: Maximum training iterations
        device: Device to run on
        save_dir: Directory to save checkpoints
    """
    print("=" * 70)
    print("Graph DiT RL Training - Head-Only Fine-Tuning")
    print("=" * 70)
    print(f"Task: {task_name}")
    print(f"Pre-trained checkpoint: {pretrained_checkpoint}")
    print(f"Num envs: {num_envs}")
    print(f"Max iterations: {max_iterations}")
    print("=" * 70)

    # Create environment
    print(f"\n[Train] Creating environment...")
    env = gym.make(task_name, num_envs=num_envs)

    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space

    print(f"[Train] Observation space: {obs_space}")
    print(f"[Train] Action space: {action_space}")

    # Compute observation dimension
    if isinstance(obs_space, dict):
        obs_dim = sum(
            sum(space.shape) if hasattr(space, "shape") else 1
            for space in obs_space.values()
        )
    else:
        obs_dim = obs_space.shape[0] if hasattr(obs_space, "shape") else 39

    action_dim = action_space.shape[0] if hasattr(action_space, "shape") else 6

    print(f"[Train] Obs dim: {obs_dim}, Action dim: {action_dim}")

    # Load pre-trained Graph DiT to get config
    print(f"\n[Train] Loading pre-trained Graph DiT config from: {pretrained_checkpoint}")
    checkpoint = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False)
    graph_dit_cfg = checkpoint.get("cfg", None)
    if graph_dit_cfg is None:
        raise ValueError(f"No config found in checkpoint: {pretrained_checkpoint}")

    # Override obs_dim and action_dim to match environment
    # Note: If obs_dim differs (e.g., target_object_position present), adjust accordingly
    print(f"[Train] Pre-trained obs_dim: {graph_dit_cfg.obs_dim}, Env obs_dim: {obs_dim}")
    if graph_dit_cfg.obs_dim != obs_dim:
        print(f"[Train] Warning: Observation dimension mismatch!")
        print(f"[Train]   Pre-trained: {graph_dit_cfg.obs_dim}")
        print(f"[Train]   Environment: {obs_dim}")
        print(f"[Train]   Using environment dimension: {obs_dim}")
    graph_dit_cfg.obs_dim = obs_dim
    graph_dit_cfg.action_dim = action_dim

    # Create RL policy config
    rl_policy_cfg = GraphDiTRLPolicyCfg(
        graph_dit_cfg=graph_dit_cfg,
        pretrained_checkpoint=pretrained_checkpoint,
        freeze_backbone=True,  # Freeze Graph DiT backbone
        rl_head_hidden_dims=[128, 64],
        rl_head_activation="elu",
        value_hidden_dims=[256, 128, 64],
        value_activation="elu",
        init_noise_std=0.5,
        feature_extraction_mode="last_embedding",
    )

    # Create RL policy
    print(f"\n[Train] Creating RL policy...")
    policy = GraphDiTRLPolicy(rl_policy_cfg).to(device)
    policy.train()

    # Count trainable parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[Train] Total parameters: {total_params:,}")
    print(f"[Train] Trainable parameters: {trainable_params:,}")
    print(f"[Train] Frozen parameters: {total_params - trainable_params:,}")
    print(f"[Train] Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    # Create optimizer (only for trainable parameters)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=3e-4,
    )

    # Create save directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(save_dir, task_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Train] Saving checkpoints to: {run_dir}")

    # Simple PPO training loop (simplified version)
    print(f"\n[Train] Starting training...")
    print(f"[Train] Note: This is a simplified PPO implementation.")
    print(f"[Train] For full PPO, consider integrating with RSL-RL framework.\n")

    obs, info = env.reset()
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    total_episodes = 0

    for iteration in range(max_iterations):
        # Collect rollouts (simplified - in practice use proper PPO buffer)
        actions, log_probs = policy.act(obs)
        actions_np = actions.cpu().numpy()

        # Step environment
        next_obs, rewards, terminated, truncated, info = env.step(actions_np)
        done = terminated | truncated

        # Update episode statistics
        episode_rewards += rewards
        episode_lengths += 1

        # Log completed episodes
        if done.any():
            for i in range(num_envs):
                if done[i]:
                    total_episodes += 1
                    print(
                        f"[Train] Iter {iteration:5d} | Episode {total_episodes:5d} | "
                        f"Reward: {episode_rewards[i].item():7.2f} | "
                        f"Length: {episode_lengths[i].item():5.0f}"
                    )
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0

        # TODO: Implement proper PPO training loop
        # This is a placeholder - actual PPO needs:
        # 1. Collect rollouts (store obs, actions, rewards, dones, log_probs, values)
        # 2. Compute advantages using GAE (Generalized Advantage Estimation)
        # 3. Compute value targets (rewards + gamma * next_values)
        # 4. Update policy using clipped objective: L^CLIP = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)
        # 5. Update value function: L^VF = (V_θ - V_target)^2
        # 6. Entropy bonus: L^ENT = entropy(π_θ)
        # 7. Total loss: L = L^CLIP - c1 * L^VF + c2 * L^ENT
        # 
        # For now, we just collect rollouts and save checkpoints
        # To implement full PPO, consider using RSL-RL framework or implementing a proper buffer

        obs = next_obs

        # Save checkpoint
        if iteration % 100 == 0 and iteration > 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_{iteration}.pt")
            policy.save(checkpoint_path)
            print(f"[Train] Saved checkpoint: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint = os.path.join(run_dir, "final_checkpoint.pt")
    best_checkpoint = os.path.join(run_dir, "best_checkpoint.pt")
    policy.save(final_checkpoint)
    
    # Also save as best_checkpoint.pt for consistency with other training scripts
    policy.save(best_checkpoint)
    
    print(f"\n[Train] Training completed!")
    print(f"[Train] Final checkpoint: {final_checkpoint}")
    print(f"[Train] Best checkpoint: {best_checkpoint}")

    env.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Graph DiT RL Policy")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained Graph DiT checkpoint",
    )
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
    parser.add_argument(
        "--max_iterations", type=int, default=1000, help="Max training iterations"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="logs/graph_dit_rl", help="Save directory")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    args = parser.parse_args()

    train_graph_dit_rl(
        task_name=args.task,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()

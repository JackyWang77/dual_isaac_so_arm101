#!/usr/bin/env python3
"""
Train GraphDiT + Residual RL Policy (OUR METHOD)

不使用 RSL-RL，自己实现训练循环：
- Rollout 收集
- GAE 计算
- Advantage-weighted regression 更新

Usage:
    python scripts/graph_dit_rl/train_graph_dit_rl.py \
        --task SO-ARM101-Lift-Cube-v0 \
        --pretrained_checkpoint ./logs/graph_dit/best_model.pt \
        --num_envs 64 \
        --max_iterations 500
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser(description="Train GraphDiT + Residual RL Policy")
parser.add_argument("--task", type=str, default="SO-ARM101-Lift-Cube-v0")
parser.add_argument("--pretrained_checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
# Note: --device is added by AppLauncher, don't add it manually
parser.add_argument("--log_dir", type=str, default="./logs/graph_dit_rl")
parser.add_argument("--save_interval", type=int, default=50)

# Rollout config
parser.add_argument("--steps_per_env", type=int, default=24, help="Steps per env per iteration")
parser.add_argument("--num_epochs", type=int, default=5, help="Epochs per iteration")
parser.add_argument("--mini_batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")

# AppLauncher
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ============================================================
# Imports after AppLauncher
# ============================================================
import os
from datetime import datetime
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import SO_101.tasks  # noqa: F401 Register envs
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from SO_101.policies.graph_dit_policy import GraphDiTPolicy
from SO_101.policies.graph_dit_residual_rl_policy import (
    GraphDiTBackboneAdapter,
    GraphDiTResidualRLCfg,
    GraphDiTResidualRLPolicy,
    compute_gae,
)


# ============================================================
# Rollout Buffer
# ============================================================
class RolloutBuffer:
    """存储 rollout 数据"""

    def __init__(self, num_envs: int, steps_per_env: int, device: str):
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.device = device
        self.ptr = 0

        # 预分配空间（在第一次 add 时确定 shape）
        self._initialized = False
        self.obs_dim = None
        self.action_dim = None
        self.z_dim = None
        self.num_layers = None

    def _init_buffers(self, obs_dim: int, action_dim: int, z_dim: int, num_layers: int):
        """延迟初始化 buffer"""
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.num_layers = num_layers

        N, T = self.num_envs, self.steps_per_env
        device = self.device

        self.obs = torch.zeros(T, N, obs_dim, device=device)
        self.actions = torch.zeros(T, N, action_dim, device=device)
        self.rewards = torch.zeros(T, N, device=device)
        self.dones = torch.zeros(T, N, device=device)
        self.values = torch.zeros(T, N, device=device)
        self.log_probs = torch.zeros(T, N, device=device)

        # GraphDiT specific
        self.a_base = torch.zeros(T, N, action_dim, device=device)
        self.z_layers = torch.zeros(T, N, num_layers, z_dim, device=device)
        self.z_bar = torch.zeros(T, N, z_dim, device=device)
        self.gate_w = torch.zeros(T, N, num_layers, device=device)

        self._initialized = True

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        info: Dict[str, torch.Tensor],
    ):
        """添加一步数据"""
        if not self._initialized:
            self._init_buffers(
                obs_dim=obs.shape[-1],
                action_dim=action.shape[-1],
                z_dim=info["z_bar"].shape[-1],
                num_layers=info["z_layers"].shape[1],
            )

        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = action
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = info["v_bar"]
        self.log_probs[t] = info["log_prob"]
        self.a_base[t] = info["a_base"]
        self.z_layers[t] = info["z_layers"]
        self.z_bar[t] = info["z_bar"]
        self.gate_w[t] = info["gate_w"]

        self.ptr += 1

    def compute_returns(self, last_value: torch.Tensor, gamma: float, lam: float):
        """计算 GAE advantages 和 returns"""
        self.advantages, self.returns = compute_gae(
            self.rewards, self.dones, self.values, last_value, gamma, lam
        )

    def get_batches(self, mini_batch_size: int, num_epochs: int):
        """生成 mini-batch 迭代器"""
        T, N = self.steps_per_env, self.num_envs
        total_samples = T * N

        # Flatten
        obs_flat = self.obs.reshape(total_samples, -1)
        actions_flat = self.actions.reshape(total_samples, -1)
        returns_flat = self.returns.reshape(total_samples)
        adv_flat = self.advantages.reshape(total_samples)
        a_base_flat = self.a_base.reshape(total_samples, -1)
        z_layers_flat = self.z_layers.reshape(total_samples, self.num_layers, -1)
        z_bar_flat = self.z_bar.reshape(total_samples, -1)

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        for _ in range(num_epochs):
            indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, mini_batch_size):
                end = min(start + mini_batch_size, total_samples)
                idx = indices[start:end]

                yield {
                    "obs": obs_flat[idx],  # 原始 obs（用于重新计算 z_layers）
                    "obs_norm": obs_flat[idx],  # 归一化 obs（如果不需要归一化，可以相同）
                    "action": actions_flat[idx],
                    "returns": returns_flat[idx],
                    "adv": adv_flat[idx],
                    "a_base": a_base_flat[idx],
                    "z_layers": z_layers_flat[idx],
                    "z_bar": z_bar_flat[idx],
                }

    def reset(self):
        """重置 buffer"""
        self.ptr = 0


# ============================================================
# Trainer
# ============================================================
class GraphDiTRLTrainer:
    """GraphDiT + Residual RL Trainer"""

    def __init__(
        self,
        env,
        policy: GraphDiTResidualRLPolicy,
        cfg: GraphDiTResidualRLCfg,
        device: str = "cuda",
        log_dir: str = "./logs",
        # Training params
        steps_per_env: int = 24,
        num_epochs: int = 5,
        mini_batch_size: int = 64,
        lr: float = 3e-4,
        max_grad_norm: float = 1.0,
    ):
        self.env = env
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.log_dir = log_dir

        self.steps_per_env = steps_per_env
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm

        # Get num_envs from env (handle wrapped environments)
        if hasattr(env, "num_envs"):
            self.num_envs = env.num_envs
        elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "num_envs"):
            self.num_envs = env.unwrapped.num_envs
        elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "cfg"):
            # Try to get from config
            self.num_envs = env.unwrapped.cfg.scene.num_envs
        else:
            # Fallback: try to infer from observation space shape
            obs_shape = env.observation_space.shape if hasattr(env.observation_space, "shape") else None
            if obs_shape and len(obs_shape) > 1:
                self.num_envs = obs_shape[0]
            else:
                raise AttributeError(f"Cannot determine num_envs from environment: {type(env)}")

        # Buffer
        self.buffer = RolloutBuffer(self.num_envs, steps_per_env, device)

        # Optimizer (只优化 trainable 参数)
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=lr)

        # Logger
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Stats
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []

        # Initialize env buffers
        policy.init_env_buffers(self.num_envs)

        print(f"[Trainer] num_envs: {self.num_envs}")
        print(f"[Trainer] steps_per_env: {steps_per_env}")
        print(f"[Trainer] batch_size: {self.num_envs * steps_per_env}")
        print(f"[Trainer] trainable params: {sum(p.numel() for p in trainable_params):,}")

    def collect_rollout(self) -> Dict[str, float]:
        """收集一个 rollout"""
        self.policy.eval()
        self.buffer.reset()

        obs, _ = self.env.reset()
        obs = self._process_obs(obs)

        ep_rewards = torch.zeros(self.num_envs, device=self.device)
        ep_lengths = torch.zeros(self.num_envs, device=self.device)

        for step in range(self.steps_per_env):
            with torch.no_grad():
                action, info = self.policy.act(obs, deterministic=False)

            # Step env
            next_obs, reward, terminated, truncated, env_info = self.env.step(action)
            done = terminated | truncated

            # Process
            next_obs = self._process_obs(next_obs)
            reward = reward.to(self.device).float()
            done = done.to(self.device).float()

            # Store
            self.buffer.add(obs, action, reward, done, info)

            # Track episode stats
            ep_rewards += reward
            ep_lengths += 1

            # Handle resets
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_envs) > 0:
                self.policy.reset_envs(done_envs)
                for i in done_envs.tolist():
                    self.episode_rewards.append(ep_rewards[i].item())
                    self.episode_lengths.append(ep_lengths[i].item())
                ep_rewards[done_envs] = 0
                ep_lengths[done_envs] = 0

            obs = next_obs
            self.total_steps += self.num_envs

        # Compute last value for GAE
        with torch.no_grad():
            _, last_info = self.policy.act(obs, deterministic=True)
            last_value = last_info["v_bar"]

        self.buffer.compute_returns(last_value, self.cfg.gamma, self.cfg.lam)

        # Return stats
        stats = {}
        if len(self.episode_rewards) > 0:
            stats["ep_reward_mean"] = np.mean(self.episode_rewards[-100:])
            stats["ep_length_mean"] = np.mean(self.episode_lengths[-100:])
        return stats

    def update(self) -> Dict[str, float]:
        """更新 policy"""
        self.policy.train()

        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_gate_entropy = 0
        num_updates = 0

        for batch in self.buffer.get_batches(self.mini_batch_size, self.num_epochs):
            losses = self.policy.compute_loss(batch)

            self.optimizer.zero_grad()
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += losses["loss_total"].item()
            total_actor_loss += losses["loss_actor"].item()
            total_critic_loss += losses["loss_critic_layers"].item()
            total_gate_entropy += losses["gate_entropy"].item()
            num_updates += 1

        return {
            "loss_total": total_loss / num_updates,
            "loss_actor": total_actor_loss / num_updates,
            "loss_critic": total_critic_loss / num_updates,
            "gate_entropy": total_gate_entropy / num_updates,
            "alpha": self.policy.alpha.item(),
        }

    def train(self, max_iterations: int, save_interval: int = 50):
        """主训练循环"""
        print(f"\n[Trainer] Starting training for {max_iterations} iterations...")

        for iteration in range(1, max_iterations + 1):
            # Collect
            rollout_stats = self.collect_rollout()

            # Update
            update_stats = self.update()

            # Log
            self._log(iteration, rollout_stats, update_stats)

            # Save
            if iteration % save_interval == 0:
                self._save(iteration)

        # Final save
        self._save(max_iterations, final=True)
        self.writer.close()
        print(f"\n[Trainer] Training complete!")

    def _process_obs(self, obs) -> torch.Tensor:
        """处理 observation"""
        if isinstance(obs, dict):
            if "policy" in obs:
                obs = obs["policy"]
            else:
                obs = torch.cat([v.flatten(start_dim=1) for v in obs.values()], dim=-1)
        return obs.to(self.device).float()

    def _log(self, iteration: int, rollout_stats: Dict, update_stats: Dict):
        """记录日志"""
        # TensorBoard
        for k, v in rollout_stats.items():
            self.writer.add_scalar(f"rollout/{k}", v, self.total_steps)
        for k, v in update_stats.items():
            self.writer.add_scalar(f"train/{k}", v, self.total_steps)
        self.writer.add_scalar("train/total_steps", self.total_steps, iteration)

        # Console
        ep_reward = rollout_stats.get("ep_reward_mean", 0)
        print(
            f"[Iter {iteration:4d}] "
            f"steps={self.total_steps:7d} | "
            f"reward={ep_reward:7.2f} | "
            f"loss={update_stats['loss_total']:7.4f} | "
            f"alpha={update_stats['alpha']:.3f} | "
            f"gate_ent={update_stats['gate_entropy']:.3f}"
        )

    def _save(self, iteration: int, final: bool = False):
        """保存 checkpoint"""
        suffix = "final" if final else f"iter_{iteration}"
        path = os.path.join(self.log_dir, f"policy_{suffix}.pt")
        self.policy.save(path)
        print(f"[Trainer] Saved checkpoint: {path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("GraphDiT + Residual RL Training (OUR METHOD)")
    print("=" * 70)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get device from AppLauncher (or use default)
    device = getattr(args, "device", "cuda")

    # Load pretrained GraphDiT
    print(f"\n[Main] Loading pretrained GraphDiT: {args.pretrained_checkpoint}")
    graph_dit = GraphDiTPolicy.load(args.pretrained_checkpoint, device=device)
    graph_dit.eval()
    for p in graph_dit.parameters():
        p.requires_grad = False
    print(f"[Main] GraphDiT loaded and frozen")

    # Create backbone adapter
    backbone = GraphDiTBackboneAdapter(graph_dit)

    # Create environment
    print(f"\n[Main] Creating environment: {args.task}")
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    # Get dimensions
    obs_space = env.observation_space
    action_space = env.action_space

    def get_obs_dim(space):
        """Recursively compute observation dimension from space."""
        if hasattr(space, "shape") and space.shape is not None:
            if isinstance(space.shape, tuple):
                if len(space.shape) > 1:
                    return space.shape[-1]
                else:
                    return space.shape[0]
            else:
                return space.shape if isinstance(space.shape, int) else 1
        elif hasattr(space, "spaces"):
            if isinstance(space.spaces, dict):
                if "policy" in space.spaces:
                    return get_obs_dim(space.spaces["policy"])
                else:
                    return sum(get_obs_dim(s) for s in space.spaces.values())
            else:
                return sum(get_obs_dim(s) for s in space.spaces)
        return 1

    obs_dim = get_obs_dim(obs_space)

    # Get action_dim (handle vectorized environments)
    def get_action_dim(space):
        """Recursively compute action dimension from space."""
        if hasattr(space, "shape") and space.shape is not None:
            if isinstance(space.shape, tuple):
                if len(space.shape) > 1:
                    # Vectorized: (num_envs, action_dim) -> take last dim
                    return space.shape[-1]
                else:
                    # Non-vectorized: (action_dim,) -> take first dim
                    return space.shape[0]
            else:
                return space.shape if isinstance(space.shape, int) else 1
        elif hasattr(space, "spaces"):
            if isinstance(space.spaces, dict):
                if "policy" in space.spaces:
                    return get_action_dim(space.spaces["policy"])
                else:
                    return sum(get_action_dim(s) for s in space.spaces.values())
            else:
                return sum(get_action_dim(s) for s in space.spaces)
        # Fallback: try to sample
        try:
            sample = space.sample()
            if isinstance(sample, dict):
                if "policy" in sample:
                    sample = sample["policy"]
            if hasattr(sample, "shape"):
                return sample.shape[-1] if len(sample.shape) > 1 else sample.shape[0]
            elif hasattr(sample, "__len__"):
                return len(sample) if isinstance(sample, (list, tuple)) else 1
        except:
            pass
        return 1

    action_dim = get_action_dim(action_space)

    print(f"[Main] obs_dim: {obs_dim}, action_dim: {action_dim}")

    # Create policy config
    policy_cfg = GraphDiTResidualRLCfg(
        obs_dim=obs_dim,
        action_dim=action_dim,
        z_dim=graph_dit.cfg.z_dim,
        num_layers=graph_dit.cfg.num_layers,
        device=device,
        # 从 graph_dit 获取 obs_structure (如果有)
        obs_structure=getattr(graph_dit.cfg, "obs_structure", None),
        robot_state_dim=6,  # 根据你的机器人调整
    )

    # Create policy
    policy = GraphDiTResidualRLPolicy(
        cfg=policy_cfg,
        backbone=backbone,
        pred_horizon=getattr(graph_dit.cfg, "pred_horizon", 16),
        exec_horizon=getattr(graph_dit.cfg, "exec_horizon", 8),
    )
    policy.to(device)

    # Load normalization stats from checkpoint
    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
    if "node_stats" in checkpoint:
        node_stats = checkpoint["node_stats"]
        policy.set_normalization_stats(
            ee_node_mean=node_stats.get("ee_mean"),
            ee_node_std=node_stats.get("ee_std"),
            object_node_mean=node_stats.get("object_mean"),
            object_node_std=node_stats.get("object_std"),
        )
        print(f"[Main] Loaded normalization stats from checkpoint")

    # Create log dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.task, timestamp)

    # Get num_envs from env_cfg (more reliable)
    num_envs = env_cfg.scene.num_envs if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs") else args.num_envs
    print(f"[Main] Using num_envs: {num_envs}")

    # Create trainer
    trainer = GraphDiTRLTrainer(
        env=env,
        policy=policy,
        cfg=policy_cfg,
        device=device,
        log_dir=log_dir,
        steps_per_env=args.steps_per_env,
        num_epochs=args.num_epochs,
        mini_batch_size=args.mini_batch_size,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
    )

    # Train
    trainer.train(
        max_iterations=args.max_iterations,
        save_interval=args.save_interval,
    )

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

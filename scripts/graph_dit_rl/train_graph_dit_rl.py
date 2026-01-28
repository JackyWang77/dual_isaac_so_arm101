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
parser.add_argument("--gripper-model", type=str, default=None, help="Path to gripper model checkpoint (optional)")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
# Note: --device is added by AppLauncher, don't add it manually
parser.add_argument("--log_dir", type=str, default="./logs/graph_dit_rl")
parser.add_argument("--save_interval", type=int, default=50)

# Rollout config
parser.add_argument("--steps_per_env", type=int, default=130, help="Steps per env per iteration")
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
import json
from datetime import datetime
from typing import Dict  # List, Optional removed (unused)

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

# Import gripper model (same directory structure as play.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
gripper_model_dir = os.path.join(os.path.dirname(current_dir), "graph_dit")
sys.path.insert(0, gripper_model_dir)
try:
    from gripper_model import GripperPredictor
    GRIPPER_MODEL_AVAILABLE = True
except ImportError:
    GRIPPER_MODEL_AVAILABLE = False
    print("[Train] Warning: gripper_model not found, will use Graph-DiT for gripper")


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

        self.obs = torch.zeros(T, N, obs_dim, device=device)  # Raw obs (for recomputing z_layers)
        self.obs_norm = torch.zeros(T, N, obs_dim, device=device)  # Normalized obs (for Actor/Critic)
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
        obs_norm: torch.Tensor,  # Add normalized obs
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
        self.obs[t] = obs  # Raw obs (for recomputing z_layers)
        self.obs_norm[t] = obs_norm  # Normalized obs (for Actor/Critic)
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
        obs_flat = self.obs.reshape(total_samples, -1)  # Raw obs
        obs_norm_flat = self.obs_norm.reshape(total_samples, -1)  # Normalized obs
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
                    "obs": obs_flat[idx],  # Raw obs (for recomputing z_layers)
                    "obs_norm": obs_norm_flat[idx],  # Normalized obs (for Actor/Critic)
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
        steps_per_env: int = 130,
        num_epochs: int = 5,
        mini_batch_size: int = 64,
        lr: float = 3e-4,
        max_grad_norm: float = 1.0,
        action_history_length: int = 4,  # From Graph-DiT config
        action_dim_env: int = 6,  # Environment action dim (usually 6, includes gripper)
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
        
        # CRITICAL: Also include node_to_z parameters if it's trainable
        # node_to_z is in backbone.backbone.node_to_z, not in policy.parameters()
        node_to_z_params = []
        if hasattr(policy, 'backbone') and hasattr(policy.backbone, 'backbone'):
            graph_dit = policy.backbone.backbone
            if hasattr(graph_dit, 'node_to_z'):
                node_to_z_params = [p for p in graph_dit.node_to_z.parameters() if p.requires_grad]
                trainable_params.extend(node_to_z_params)
                if len(node_to_z_params) > 0:
                    print(f"[Trainer] Including {len(node_to_z_params)} node_to_z parameters in optimizer")
        
        # Verify all trainable components
        policy_param_names = [name for name, param in policy.named_parameters() if param.requires_grad]
        print(f"[Trainer] Policy trainable components: {len([n for n in policy_param_names if 'z_adapter' in n])} z_adapter, "
              f"{len([n for n in policy_param_names if 'gate' in n])} gate, "
              f"{len([n for n in policy_param_names if 'actor' in n])} actor, "
              f"{len([n for n in policy_param_names if 'critic' in n])} critic")
        print(f"[Trainer] Total trainable params: {len(trainable_params)} (policy: {len(trainable_params) - len(node_to_z_params)}, node_to_z: {len(node_to_z_params)})")
        
        self.optimizer = optim.AdamW(trainable_params, lr=lr)

        # Logger
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Debug log file
        self.debug_log_path = os.path.join(log_dir, "debug_log.jsonl")
        self.debug_interval = 10  # Debug every N iterations

        # Stats
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []

        # Initialize env buffers
        policy.init_env_buffers(self.num_envs)
        
        # History buffers for Graph-DiT (optimized: single tensor per history type)
        # NOTE: GraphDiT expects 6 dimensions for joint states (was trained with 6)
        # The residual RL config's joint_dim=5 is only for EMA smoothing, not for joint state extraction
        graph_dit_joint_dim = 6  # Always 6 for GraphDiT compatibility
        
        # Use provided action_history_length (from Graph-DiT config)
        # CRITICAL: action_history must use action_dim_env (6), not cfg.action_dim (5)
        # because Graph-DiT expects 6D action history, and we store the actual executed actions (6D)
        # cfg.action_dim=5 is only for Actor output (RL fine-tunes 5 dims), but executed actions are 6D
        self.action_history = torch.zeros(self.num_envs, action_history_length, action_dim_env, device=device)
        self.ee_node_history = torch.zeros(self.num_envs, action_history_length, 7, device=device)
        self.object_node_history = torch.zeros(self.num_envs, action_history_length, 7, device=device)
        self.joint_state_history = torch.zeros(self.num_envs, action_history_length, graph_dit_joint_dim, device=device)

        # Normalization stats (will be set from main)
        self.obs_mean = None
        self.obs_std = None
        self.ee_node_mean = None
        self.ee_node_std = None
        self.object_node_mean = None
        self.object_node_std = None
        self.joint_mean = None
        self.joint_std = None
        self.action_mean = None
        self.action_std = None

        print(f"[Trainer] num_envs: {self.num_envs}")
        print(f"[Trainer] steps_per_env: {steps_per_env}")
        print(f"[Trainer] batch_size: {self.num_envs * steps_per_env}")
        print(f"[Trainer] trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    def set_normalization_stats(
        self,
        obs_mean=None, obs_std=None,
        ee_node_mean=None, ee_node_std=None,
        object_node_mean=None, object_node_std=None,
        joint_mean=None, joint_std=None,
        action_mean=None, action_std=None,
    ):
        """Set normalization statistics (same as Graph-DiT play.py)"""
        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.ee_node_mean = ee_node_mean
        self.ee_node_std = ee_node_std
        self.object_node_mean = object_node_mean
        self.object_node_std = object_node_std
        self.joint_mean = joint_mean
        self.joint_std = joint_std
        self.action_mean = action_mean
        self.action_std = action_std

    def collect_rollout(self) -> Dict[str, float]:
        """收集一个 rollout"""
        self.policy.eval()
        self.buffer.reset()

        obs, _ = self.env.reset()
        obs = self._process_obs(obs)

        ep_rewards = torch.zeros(self.num_envs, device=self.device)
        ep_lengths = torch.zeros(self.num_envs, device=self.device)
        
        # Track individual reward terms
        reward_terms_dict = {}  # {reward_name: list of [N] tensors}

        for step in range(self.steps_per_env):
            # CRITICAL: Normalize observations (same as Graph-DiT play.py)
            if self.obs_mean is not None and self.obs_std is not None:
                obs_norm = (obs - self.obs_mean) / self.obs_std
            else:
                obs_norm = obs
            
            # Extract current node features and joint states from RAW obs (before normalization for history)
            ee_node_current, object_node_current = self.policy._extract_nodes_from_obs(obs)
            
            # Extract joint states (joint position only, from first 6 dims of obs)
            # NOTE: GraphDiT expects 6 dimensions (was trained with 6), regardless of residual RL config
            # The residual RL config's joint_dim=5 is only for EMA smoothing, not for joint state extraction
            joint_states_current = obs[:, :6]  # [B, 6] - always 6 for GraphDiT compatibility
            
            # CRITICAL: Normalize node and joint histories (same as Graph-DiT play.py)
            # History tensors are already in batch format [num_envs, history_len, dim]
            action_history = self.action_history  # [B, H, action_dim] - already normalized
            
            # Normalize node histories
            ee_node_history = self.ee_node_history.clone()  # [B, H, 7]
            object_node_history = self.object_node_history.clone()  # [B, H, 7]
            if self.ee_node_mean is not None and self.ee_node_std is not None:
                ee_node_history = (ee_node_history - self.ee_node_mean) / self.ee_node_std
            if self.object_node_mean is not None and self.object_node_std is not None:
                object_node_history = (object_node_history - self.object_node_mean) / self.object_node_std
            
            # Normalize joint history
            joint_states_history = self.joint_state_history.clone()  # [B, H, 6]
            if self.joint_mean is not None and self.joint_std is not None:
                joint_states_history = (joint_states_history - self.joint_mean) / self.joint_std
            
            with torch.no_grad():
                action, info = self.policy.act(
                    obs_raw=obs,
                    obs_norm=obs_norm,  # Use normalized obs
                    action_history=action_history,
                    ee_node_history=ee_node_history,
                    object_node_history=object_node_history,
                    joint_states_history=joint_states_history,
                    deterministic=False
                )
                
                # Clip action to action space bounds (critical for stability!)
                if hasattr(self.env, 'action_space'):
                    if hasattr(self.env.action_space, 'low') and hasattr(self.env.action_space, 'high'):
                        action = torch.clamp(
                            action,
                            torch.tensor(self.env.action_space.low, device=action.device, dtype=action.dtype),
                            torch.tensor(self.env.action_space.high, device=action.device, dtype=action.dtype)
                        )
                
                # DEBUG: Print all joints and gripper base action, residual, and final action
                action_dim = action.shape[-1]
                if action_dim >= 6:
                    a_base = info.get("a_base", None)  # [B, action_dim] - denormalized base action
                    delta = info.get("delta", None)  # [B, action_dim] - denormalized delta
                    delta_norm = info.get("delta_norm", None)  # [B, action_dim] - normalized delta (Actor output)
                    alpha_val = info.get("alpha", None)
                    if alpha_val is not None:
                        if isinstance(alpha_val, torch.Tensor):
                            alpha = alpha_val.item()
                        else:
                            alpha = float(alpha_val)
                    else:
                        alpha = self.policy.alpha.item() if hasattr(self.policy, 'alpha') else 0.1
                    
                    if a_base is not None and delta is not None:
                        # Print for first env only, every 10 steps
                        if step % 10 == 0:
                            print(f"\n[Step {step:3d}] Action Debug (env 0):")
                            print(f"  alpha: {alpha:.4f}")
                            print(f"\n  Joints (0-4):")
                            for i in range(min(5, action_dim)):
                                a_base_i = a_base[0, i].item()
                                delta_i = delta[0, i].item()
                                final_i = action[0, i].item()
                                print(f"    Joint {i}: base={a_base_i:7.4f}, delta={delta_i:7.4f}, final={final_i:7.4f} (base + {alpha:.2f}*delta = {a_base_i + alpha * delta_i:.4f})")
                            
                            # Gripper (index 5)
                            if action_dim >= 6:
                                a_base_gripper = a_base[0, 5].item()
                                delta_gripper = delta[0, 5].item()
                                gripper_continuous = action[0, 5].item()
                                gripper_binary = 1.0 if gripper_continuous > 0 else -1.0
                                print(f"\n  Gripper (5):")
                                print(f"    base: {a_base_gripper:7.4f}")
                                print(f"    delta: {delta_gripper:7.4f}")
                                print(f"    continuous (base + {alpha:.2f}*delta): {gripper_continuous:7.4f}")
                                print(f"    threshold: -0.2")
                                print(f"    binary (final): {gripper_binary:7.4f} ({'open' if gripper_binary > 0 else 'close'})")
                            
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
                if action_dim >= 6:
                    gripper_continuous = action[:, 5]  # [num_envs] - continuous value: a_base_gripper + alpha * delta_gripper
                    action[:, 5] = torch.where(
                        gripper_continuous > -0.2,
                        torch.tensor(1.0, device=action.device, dtype=action.dtype),
                        torch.tensor(-1.0, device=action.device, dtype=action.dtype)
                    )

            # Step env
            next_obs, reward, terminated, truncated, env_info = self.env.step(action)
            done = terminated | truncated

            # Process
            next_obs = self._process_obs(next_obs)
            reward = reward.to(self.device).float()
            done = done.to(self.device).float()
            
            # Extract individual reward terms from reward manager (if available)
            if hasattr(self.env, 'rew_manager'):
                rew_manager = self.env.rew_manager
                # Try to get reward terms from the manager
                # Isaac Lab stores computed rewards in rew_manager.reward_buf
                if hasattr(rew_manager, 'reward_buf'):
                    reward_buf = rew_manager.reward_buf
                    for reward_name in ['grasp_behavior', 'object_grasped', 'task_complete', 'gripper_smooth']:
                        if reward_name in reward_buf:
                            reward_value = reward_buf[reward_name]
                            if isinstance(reward_value, torch.Tensor):
                                if reward_name not in reward_terms_dict:
                                    # Initialize list for this reward term
                                    reward_terms_dict[reward_name] = []
                                # Store reward value for this timestep
                                reward_terms_dict[reward_name].append(reward_value.clone().to(self.device))

            # Store (obs is raw, obs_norm is normalized)
            self.buffer.add(obs, obs_norm, action, reward, done, info)

            # Track episode stats
            ep_rewards += reward
            ep_lengths += 1

            # Update history buffers (optimized: use torch.roll for efficiency)
            # ============================================================
            # FIX: Store normalized final executed action (not a_base_norm)
            # This matches play.py and Graph-DiT training where action_history
            # contains the actual executed actions (normalized)
            # ============================================================
            if self.action_mean is not None and self.action_std is not None:
                # action is final executed (denormalized), normalize for history
                action_for_history = (action - self.action_mean) / (self.action_std + 1e-8)
            else:
                action_for_history = action
            
            # Roll all histories: shift left by 1, new data goes to last position
            self.action_history = torch.roll(self.action_history, -1, dims=1)
            self.action_history[:, -1, :] = action_for_history  # [num_envs, action_dim] - normalized final action
            
            self.ee_node_history = torch.roll(self.ee_node_history, -1, dims=1)
            self.ee_node_history[:, -1, :] = ee_node_current  # [num_envs, 7]
            
            self.object_node_history = torch.roll(self.object_node_history, -1, dims=1)
            self.object_node_history[:, -1, :] = object_node_current  # [num_envs, 7]
            
            self.joint_state_history = torch.roll(self.joint_state_history, -1, dims=1)
            self.joint_state_history[:, -1, :] = joint_states_current  # [num_envs, joint_dim]
            
            # Handle resets
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_envs) > 0:
                self.policy.reset_envs(done_envs)
                # Reset history buffers for done environments (optimized: vectorized)
                done_env_ids = done_envs.tolist()
                for i in done_env_ids:
                    self.episode_rewards.append(ep_rewards[i].item())
                    self.episode_lengths.append(ep_lengths[i].item())
                
                # Vectorized reset: zero out all done environments at once
                self.action_history[done_envs] = 0
                self.ee_node_history[done_envs] = 0
                self.object_node_history[done_envs] = 0
                self.joint_state_history[done_envs] = 0
                
                ep_rewards[done_envs] = 0
                ep_lengths[done_envs] = 0

            obs = next_obs
            self.total_steps += self.num_envs

        # At the end of rollout, record any incomplete episodes (if they have accumulated reward)
        # This helps track progress even when episodes don't complete within steps_per_env
        incomplete_ep_rewards = ep_rewards[ep_rewards != 0]  # Non-zero accumulated rewards
        incomplete_ep_lengths = ep_lengths[ep_lengths != 0]  # Non-zero lengths
        if len(incomplete_ep_rewards) > 0:
            # These are ongoing episodes, not completed ones, but still useful for tracking
            for i in range(len(incomplete_ep_rewards)):
                self.episode_rewards.append(incomplete_ep_rewards[i].item())
                self.episode_lengths.append(incomplete_ep_lengths[i].item())

        # Compute last value for GAE (history tensors are already in batch format)
        # CRITICAL: Normalize obs and histories (same as Graph-DiT play.py)
        if self.obs_mean is not None and self.obs_std is not None:
            obs_norm = (obs - self.obs_mean) / self.obs_std
        else:
            obs_norm = obs
        
        action_history = self.action_history  # [B, H, action_dim] - already normalized
        
        # Normalize node histories
        ee_node_history = self.ee_node_history.clone()  # [B, H, 7]
        object_node_history = self.object_node_history.clone()  # [B, H, 7]
        if self.ee_node_mean is not None and self.ee_node_std is not None:
            ee_node_history = (ee_node_history - self.ee_node_mean) / self.ee_node_std
        if self.object_node_mean is not None and self.object_node_std is not None:
            object_node_history = (object_node_history - self.object_node_mean) / self.object_node_std
        
        # Normalize joint history
        joint_states_history = self.joint_state_history.clone()  # [B, H, 6]
        if self.joint_mean is not None and self.joint_std is not None:
            joint_states_history = (joint_states_history - self.joint_mean) / self.joint_std
        
        # Compute last value for GAE
        with torch.no_grad():
            _, last_info = self.policy.act(
                obs_raw=obs,
                obs_norm=obs_norm,  # Use normalized obs
                action_history=action_history,
                ee_node_history=ee_node_history,
                object_node_history=object_node_history,
                joint_states_history=joint_states_history,
                deterministic=True
            )
            last_value = last_info["v_bar"]

        self.buffer.compute_returns(last_value, self.cfg.gamma, self.cfg.lam)

        # Return stats
        stats = {}
        if len(self.episode_rewards) > 0:
            stats["ep_reward_mean"] = np.mean(self.episode_rewards[-100:])
            stats["ep_length_mean"] = np.mean(self.episode_lengths[-100:])
            stats["ep_reward_max"] = np.max(self.episode_rewards[-100:])
            stats["ep_reward_min"] = np.min(self.episode_rewards[-100:])
        else:
            # No episodes completed in this rollout - this is a problem!
            stats["ep_reward_mean"] = 0.0
            stats["ep_length_mean"] = 0.0
            stats["warning"] = "No episodes completed in this rollout!"
        
        # Track individual reward terms collected during rollout
        # reward_terms_dict contains lists of tensors for each reward term
        # Convert to stacked tensors and compute statistics
        for reward_name, reward_list in reward_terms_dict.items():
            if len(reward_list) > 0:
                # Stack all timesteps: [T, N]
                reward_tensor = torch.stack(reward_list, dim=0)
                stats[f"reward_{reward_name}_mean"] = reward_tensor.mean().item()
                stats[f"reward_{reward_name}_sum"] = reward_tensor.sum().item()
                stats[f"reward_{reward_name}_max"] = reward_tensor.max().item()
                stats[f"reward_{reward_name}_min"] = reward_tensor.min().item()
        
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
            # Clip gradients for all trainable parameters (including node_to_z)
            trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
            if hasattr(self.policy, 'backbone') and hasattr(self.policy.backbone, 'backbone'):
                graph_dit = self.policy.backbone.backbone
                if hasattr(graph_dit, 'node_to_z'):
                    trainable_params.extend([p for p in graph_dit.node_to_z.parameters() if p.requires_grad])
            torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
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

            # Debug (every N iterations)
            if iteration % self.debug_interval == 0:
                self.debug_iteration(iteration)

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
        # TensorBoard (only numeric values)
        # Filter out unwanted metrics
        excluded_keys = {
            "num_episodes",
            "rollout_done_pct",
            "rollout_early_done_pct",
            "rollout_nonzero_reward_pct",
            "rollout_reward_max",
            "rollout_reward_mean",
            "rollout_reward_min",
            "rollout_reward_sum",
        }
        
        for k, v in rollout_stats.items():
            # Skip excluded keys and non-numeric values (like 'warning' string)
            if k in excluded_keys:
                continue
            if isinstance(v, (int, float)) or (hasattr(v, 'item') and hasattr(v, 'dtype')):
                try:
                    if hasattr(v, 'item'):
                        v = v.item()
                    self.writer.add_scalar(f"rollout/{k}", v, self.total_steps)
                except (ValueError, TypeError):
                    pass  # Skip if cannot convert to float
        
        for k, v in update_stats.items():
            # Skip non-numeric values
            if isinstance(v, (int, float)) or (hasattr(v, 'item') and hasattr(v, 'dtype')):
                try:
                    if hasattr(v, 'item'):
                        v = v.item()
                    self.writer.add_scalar(f"train/{k}", v, self.total_steps)
                except (ValueError, TypeError):
                    pass  # Skip if cannot convert to float
        
        self.writer.add_scalar("train/total_steps", self.total_steps, iteration)

        # Console
        ep_reward = rollout_stats.get("ep_reward_mean", 0)
        
        # Get individual reward terms for display
        reward_grasp = rollout_stats.get("reward_grasp_behavior_mean", 0)
        reward_grasped = rollout_stats.get("reward_object_grasped_mean", 0)
        reward_success = rollout_stats.get("reward_task_complete_mean", 0)
        reward_gripper_smooth = rollout_stats.get("reward_gripper_smooth_mean", 0)
        
        warning = rollout_stats.get("warning", "")
        if warning:
            print(f"[Iter {iteration:4d}] ⚠️  {warning}")
        
        print(
            f"[Iter {iteration:4d}] "
            f"steps={self.total_steps:7d} | "
            f"ep_reward={ep_reward:7.2f} | "
            f"grasp={reward_grasp:6.3f} | "
            f"grasped={reward_grasped:6.3f} | "
            f"success={reward_success:6.3f} | "
            f"gripper_smooth={reward_gripper_smooth:6.3f} | "
            f"loss={update_stats['loss_total']:7.4f} | "
            f"alpha={update_stats['alpha']:.3f}"
        )

    def debug_iteration(self, iteration: int):
        """Debug iteration: analyze batch and save to file"""
        # Get a sample batch from buffer
        sample_batch = None
        for batch in self.buffer.get_batches(self.mini_batch_size, 1):
            sample_batch = batch
            break  # Just get the first batch
        
        if sample_batch is None:
            return
        
        obs = sample_batch["obs"]
        a_base = sample_batch["a_base"]
        action = sample_batch["action"]
        adv = sample_batch["adv"]
        returns = sample_batch.get("returns", None)
        
        alpha = torch.clamp(self.policy.alpha, 0.0, 0.3).item()
        delta = (action - a_base) / (alpha + 1e-8)
        
        # Check if base action is reasonable (not all zeros or very small)
        base_action_norm = torch.norm(a_base, dim=-1).mean().item()
        action_norm = torch.norm(action, dim=-1).mean().item()
        delta_norm = torch.norm(delta, dim=-1).mean().item()
        
        # Apply residual mask if exists
        if self.policy.residual_action_mask is not None:
            delta = delta * self.policy.residual_action_mask
        
        # Get joint_dim from config
        joint_dim = self.cfg.joint_dim if hasattr(self.cfg, 'joint_dim') else 5
        
        # ===== 1. Action Decomposition =====
        joint_base = a_base[:, :joint_dim]
        joint_delta = delta[:, :joint_dim]
        joint_final = action[:, :joint_dim]
        
        # Gripper (remaining dimensions)
        if action.shape[-1] > joint_dim:
            grip_base = a_base[:, joint_dim:]
            grip_delta = delta[:, joint_dim:]
            grip_final = action[:, joint_dim:]
        else:
            grip_base = None
            grip_delta = None
            grip_final = None
        
        # ===== 2. Advantage Distribution =====
        adv_mean = adv.mean().item()
        adv_std = adv.std().item()
        adv_min = adv.min().item()
        adv_max = adv.max().item()
        adv_positive_pct = (adv > 0).float().mean().item() * 100
        
        # ===== 3. AWR Weights =====
        w = torch.exp(adv / max(self.cfg.beta, 1e-8))
        w = torch.clamp(w, 0.0, self.cfg.weight_clip_max)
        w = w / (w.mean() + 1e-8)
        w_mean = w.mean().item()
        w_std = w.std().item()
        w_min = w.min().item()
        w_max = w.max().item()
        w_high_pct = (w > 2.0).float().mean().item() * 100
        
        # ===== 4. Task Progress (if obs contains position info) =====
        task_info = {}
        if obs.shape[-1] >= 16:
            try:
                ee_pos = obs[:, 13:16]
                obj_pos = obs[:, 6:9]
                distance = torch.norm(ee_pos - obj_pos, dim=-1)
                obj_height = obj_pos[:, 2]
                
                task_info = {
                    "ee_obj_distance_mean": distance.mean().item(),
                    "ee_obj_distance_min": distance.min().item(),
                    "object_height_mean": obj_height.mean().item(),
                    "object_height_max": obj_height.max().item(),
                }
            except:
                pass
        
        # ===== 5. Reward Statistics =====
        reward_info = {}
        if returns is not None:
            reward_info = {
                "returns_mean": returns.mean().item(),
                "returns_std": returns.std().item(),
                "returns_min": returns.min().item(),
                "returns_max": returns.max().item(),
            }
        
        # Compile debug info
        debug_info = {
            "iteration": iteration,
            "total_steps": self.total_steps,
            "alpha": alpha,
            "action_norms": {
                "base_action_norm": base_action_norm,
                "action_norm": action_norm,
                "delta_norm": delta_norm,
            },
            "action_decomposition": {
                "joints": {
                    "base_mean": joint_base.mean().item(),
                    "base_std": joint_base.std().item(),
                    "delta_mean": joint_delta.mean().item(),
                    "delta_std": joint_delta.std().item(),
                    "final_mean": joint_final.mean().item(),
                    "final_std": joint_final.std().item(),
                },
            },
            "advantage": {
                "mean": adv_mean,
                "std": adv_std,
                "min": adv_min,
                "max": adv_max,
                "positive_pct": adv_positive_pct,
            },
            "awr_weights": {
                "mean": w_mean,
                "std": w_std,
                "min": w_min,
                "max": w_max,
                "high_pct": w_high_pct,
            },
        }
        
        if grip_base is not None:
            debug_info["action_decomposition"]["gripper"] = {
                "base_mean": grip_base.mean().item(),
                "base_std": grip_base.std().item(),
                "delta_mean": grip_delta.mean().item(),
                "delta_std": grip_delta.std().item(),
                "final_mean": grip_final.mean().item(),
                "final_std": grip_final.std().item(),
            }
        
        if task_info:
            debug_info["task_progress"] = task_info
        
        if reward_info:
            debug_info["rewards"] = reward_info
        
        # Write to JSONL file (one JSON object per line)
        with open(self.debug_log_path, "a") as f:
            f.write(json.dumps(debug_info) + "\n")
        
        print(f"[DEBUG] Iteration {iteration}: Debug info saved to {self.debug_log_path}")

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
    # CRITICAL: Make node_to_z trainable (it was never trained in Graph-DiT training)
    # because it's not in the gradient path of noise_pred loss
    if hasattr(graph_dit, 'node_to_z'):
        for p in graph_dit.node_to_z.parameters():
            p.requires_grad = True
        n2z_trainable = sum(1 for p in graph_dit.node_to_z.parameters() if p.requires_grad)
        n2z_total = len(list(graph_dit.node_to_z.parameters()))
        print(f"[Main] node_to_z made trainable: {n2z_trainable}/{n2z_total} params trainable (was not trained in Graph-DiT)")
    else:
        print(f"[Main] ⚠️  WARNING: node_to_z not found in Graph-DiT!")
    print(f"[Main] GraphDiT loaded and frozen (except node_to_z)")

    # Load gripper model (if provided)
    gripper_model = None
    gripper_input_mean = None
    gripper_input_std = None
    gripper_states = None  # State machine: 0=OPEN, 1=CLOSING, 2=CLOSED, 3=OPENING
    
    if args.gripper_model is not None and GRIPPER_MODEL_AVAILABLE:
        gripper_model_path = args.gripper_model
        if os.path.exists(gripper_model_path):
            print(f"[Main] Loading gripper model from: {gripper_model_path}")
            gripper_checkpoint = torch.load(gripper_model_path, map_location=device, weights_only=False)
            
            # Get model config
            hidden_dims = gripper_checkpoint.get('hidden_dims', [128, 128, 64])
            dropout = gripper_checkpoint.get('dropout', 0.1)
            
            # Load model
            gripper_model = GripperPredictor(hidden_dims=hidden_dims, dropout=dropout).to(device)
            gripper_model.load_state_dict(gripper_checkpoint['model_state_dict'])
            gripper_model.eval()
            
            # Load normalization stats
            input_mean = gripper_checkpoint['input_mean']
            input_std = gripper_checkpoint['input_std']
            
            # Ensure numpy arrays
            if isinstance(input_mean, torch.Tensor):
                input_mean = input_mean.cpu().numpy()
            if isinstance(input_std, torch.Tensor):
                input_std = input_std.cpu().numpy()
            
            # Convert to torch tensors
            gripper_input_mean = torch.from_numpy(input_mean).float().to(device)
            gripper_input_std = torch.from_numpy(input_std).float().to(device)
            
            # Initialize gripper state machine (0=OPEN for all envs)
            gripper_states = torch.zeros(args.num_envs, dtype=torch.long, device=device)
            
            print(f"[Main] Gripper model loaded successfully")
            print(f"[Main] Gripper input_mean shape: {gripper_input_mean.shape}, input_std shape: {gripper_input_std.shape}")
        else:
            print(f"[Main] Warning: Gripper model path provided but file not found: {gripper_model_path}")
    else:
        if args.gripper_model is not None:
            print(f"[Main] Warning: Gripper model path provided but gripper_model module not available")
        print(f"[Main] Using Graph-DiT for all action dimensions (including gripper)")

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

    action_dim_env = get_action_dim(action_space)  # Environment action dim (usually 6)
    action_dim_rl = 5  # RL only fine-tunes arm joints (5 dims), gripper excluded

    print(f"[Main] obs_dim: {obs_dim}, action_dim_env: {action_dim_env}, action_dim_rl: {action_dim_rl}")
    print(f"[Main] RL will only fine-tune first 5 dimensions (arm joints), gripper excluded")

    # Create policy config
    # CRITICAL: Get obs_structure from Graph-DiT config to ensure consistency
    obs_structure = getattr(graph_dit.cfg, "obs_structure", None)
    print(f"[Main] Graph-DiT obs_structure: {obs_structure}")
    
    # Create residual action mask: [1, 1, 1, 1, 1, 0] (mask gripper, index 5)
    # This ensures gripper residual is always 0, even if somehow computed
    residual_action_mask = torch.ones(action_dim_env, device=device)  # [6]
    residual_action_mask[5] = 0.0  # Mask gripper (index 5)
    
    policy_cfg = GraphDiTResidualRLCfg(
        obs_dim=obs_dim,
        action_dim=action_dim_rl,  # RL only outputs 5 dims (arm joints)
        z_dim=graph_dit.cfg.z_dim,
        num_layers=graph_dit.cfg.num_layers,
        device=device,
        obs_structure=obs_structure,  # CRITICAL: Use same obs_structure as Graph-DiT
        robot_state_dim=6,  # 根据你的机器人调整
        residual_action_mask=residual_action_mask,  # Mask gripper (index 5) from residual
    )
    print(f"[Main] residual_action_mask: {residual_action_mask.tolist()} (gripper masked)")
    print(f"[Main] Residual RL obs_structure: {policy_cfg.obs_structure}")
    
    # Verify obs_structure consistency
    if getattr(graph_dit.cfg, "obs_structure", None) != policy_cfg.obs_structure:
        print("[Main] ⚠️  WARNING: obs_structure MISMATCH between Graph-DiT and Residual RL!")
    else:
        print("[Main] ✅ obs_structure consistent between Graph-DiT and Residual RL")

    # Create policy (with gripper model if available)
    policy = GraphDiTResidualRLPolicy(
        cfg=policy_cfg,
        backbone=backbone,
        pred_horizon=getattr(graph_dit.cfg, "pred_horizon", 16),
        exec_horizon=getattr(graph_dit.cfg, "exec_horizon", 8),
        gripper_model=gripper_model,
        gripper_input_mean=gripper_input_mean,
        gripper_input_std=gripper_input_std,
    )
    policy.to(device)

    # Load normalization stats from checkpoint (same as Graph-DiT play.py)
    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
    
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
        print(f"[Main] Loaded obs normalization stats")
    
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
        print(f"[Main] Loaded action normalization stats")
    
    # Set normalization stats in policy (including action stats)
    policy.set_normalization_stats(
        ee_node_mean=ee_node_mean,
        ee_node_std=ee_node_std,
        object_node_mean=object_node_mean,
        object_node_std=object_node_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    if node_stats is not None:
        print(f"[Main] Loaded node normalization stats")
    
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
        print(f"[Main] Loaded joint normalization stats")
    
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
        print(f"[Main] Loaded action normalization stats")

    # Create log dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.task, timestamp)

    # Get num_envs from env_cfg (more reliable)
    num_envs = env_cfg.scene.num_envs if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs") else args.num_envs
    print(f"[Main] Using num_envs: {num_envs}")

    # Get action_history_length from Graph-DiT config
    action_history_length = getattr(graph_dit.cfg, 'action_history_length', 4)
    
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
        action_history_length=action_history_length,
        action_dim_env=action_dim_env,  # Pass environment action dim (6) for history buffer
    )
    
    # Set normalization stats in trainer (same as Graph-DiT play.py)
    trainer.set_normalization_stats(
        obs_mean=obs_mean, obs_std=obs_std,
        ee_node_mean=ee_node_mean, ee_node_std=ee_node_std,
        object_node_mean=object_node_mean, object_node_std=object_node_std,
        joint_mean=joint_mean, joint_std=joint_std,
        action_mean=action_mean, action_std=action_std,
    )
    print(f"[Main] Set normalization stats in trainer")
    
    # Verify normalization stats
    print(f"[Main] Normalization stats verification:")
    for name, val in [('obs_mean', obs_mean), ('obs_std', obs_std),
                      ('action_mean', action_mean), ('action_std', action_std),
                      ('ee_node_mean', ee_node_mean), ('ee_node_std', ee_node_std),
                      ('object_node_mean', object_node_mean), ('object_node_std', object_node_std),
                      ('joint_mean', joint_mean), ('joint_std', joint_std)]:
        shape = val.shape if val is not None else 'None'
        print(f"    {name}: {shape}")
        if val is None:
            print(f"      ⚠️  WARNING: {name} is None!")

    # ============================================================
    # PRE-TRAINING VERIFICATION
    # ============================================================
    print("\n" + "=" * 60)
    print("PRE-TRAINING VERIFICATION")
    print("=" * 60)
    
    # 1. obs_structure 检查
    print(f"\n[1] obs_structure:")
    graph_dit_obs_struct = getattr(graph_dit.cfg, 'obs_structure', None)
    print(f"    Graph-DiT: {graph_dit_obs_struct}")
    print(f"    Residual RL: {policy_cfg.obs_structure}")
    if graph_dit_obs_struct != policy_cfg.obs_structure:
        print("    ⚠️  WARNING: obs_structure MISMATCH!")
    else:
        print("    ✅ obs_structure consistent")
    
    # 2. 可训练参数检查
    print(f"\n[2] Trainable parameters:")
    policy_trainable = [(name, param.shape) for name, param in policy.named_parameters() if param.requires_grad]
    print(f"    Policy trainable params: {len(policy_trainable)}")
    for name, shape in policy_trainable[:5]:  # Show first 5
        print(f"      {name}: {shape}")
    if len(policy_trainable) > 5:
        print(f"      ... and {len(policy_trainable) - 5} more")
    
    # 3. node_to_z 检查
    print(f"\n[3] node_to_z:")
    if hasattr(graph_dit, 'node_to_z'):
        n2z_trainable = sum(1 for p in graph_dit.node_to_z.parameters() if p.requires_grad)
        n2z_total = len(list(graph_dit.node_to_z.parameters()))
        print(f"    Trainable params: {n2z_trainable}/{n2z_total}")
        if n2z_trainable == 0:
            print("    ⚠️  WARNING: node_to_z has NO trainable params!")
        else:
            print("    ✅ node_to_z is trainable")
    else:
        print("    ⚠️  WARNING: node_to_z not found!")
    
    # 4. 归一化统计量检查
    print(f"\n[4] Normalization stats:")
    stats_ok = True
    for name, val in [('obs_mean', obs_mean), ('action_mean', action_mean),
                      ('ee_node_mean', ee_node_mean), ('joint_mean', joint_mean)]:
        if val is not None:
            print(f"    {name}: {val.shape} ✅")
        else:
            print(f"    {name}: None ⚠️  WARNING!")
            stats_ok = False
    if stats_ok:
        print("    ✅ All normalization stats loaded")
    else:
        print("    ⚠️  WARNING: Some normalization stats are missing!")
    
    # 5. 测试 forward pass
    print(f"\n[5] Test forward pass:")
    try:
        test_obs = torch.randn(2, obs_dim, device=device)
        policy.init_env_buffers(2)
        # Test z extraction
        z = policy._get_z_layers_fast(test_obs)
        print(f"    z_layers shape: {z.shape} ✅")
        print(f"    Forward pass test: OK")
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60 + "\n")

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

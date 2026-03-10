#!/usr/bin/env python3
"""
Train GraphDiT + Residual RL Policy (OUR METHOD) - FIXED VERSION

修复:
1. 添加 Success Rate 统计
2. 添加 Explained Variance 统计
3. 移除过多 DEBUG 打印
4. 修复除零保护
5. 改进日志输出

Usage:
    python scripts/graph_dit_rl/train_graph_rl.py \
        --task SO-ARM101-Lift-Cube-v0 \
        --pretrained_checkpoint ./logs/graph_dit/best_model.pt \
        --num_envs 128 \
        --max_iterations 500
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser(description="Train GraphDiT + Residual RL Policy")
parser.add_argument("--task", type=str, default="SO-ARM101-Dual-Cube-Stack-RL-v0",
                    help="SO-ARM101-Dual-Cube-Stack-RL-v0 (dual arm RL), SO-ARM101-Dual-Cube-Stack-v0 (original rewards)")
parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                    help="Pretrained Graph-Unet checkpoint (residual RL is Unet-only)")
parser.add_argument("--policy_type", type=str, default="dual_arm_gated",
                    choices=["unet", "graph_unet", "dual_arm", "dual_arm_gated"],
                    help="Backbone policy (auto-detected from checkpoint if dual arm)")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
# Note: --device is added by AppLauncher, don't add it manually
parser.add_argument("--log_dir", type=str, default="./logs/graph_unet_rl", help="Log directory for residual RL (Unet)")
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--best_sr_window", type=int, default=200,
                    help="Rolling window for best-model SR (100: ±10%% CI, 200: ±7%%)")
parser.add_argument("--best_sr_require_consistent", action="store_true",
                    help="Only save best when current SR >= sr_window - 15%% (avoid lucky flukes)")

# Rollout config
parser.add_argument("--steps_per_env", type=int, default=400, help="Steps per env per iteration (must >= env max_episode_steps=400 to avoid premature rollout truncation)")
parser.add_argument("--num_epochs", type=int, default=5, help="Epochs per iteration")
parser.add_argument("--mini_batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
parser.add_argument("--c_delta_reg", type=float, default=2.0, help="Delta (residual) regularization weight; higher = smoother, RL 'don't move unless reward'")
parser.add_argument("--c_ent", type=float, default=0.01, help="Entropy coefficient; encourages exploration")
parser.add_argument("--beta", type=float, default=1.0, help="AWR beta: w=exp(adv/beta); higher=softer weighting, lower=sharper on high-adv samples")
parser.add_argument("--alpha_init", type=float, default=0.10, help="Residual alpha init: a=base+alpha*delta; arm uses clamped(0,0.5), gripper=1.0")
parser.add_argument("--expectile_tau", type=float, default=0.7, help="Expectile τ for value loss (IQL-style); τ=0.7 → optimistic; 0.5=MSE")
parser.add_argument("--use_adaptive_alpha", action="store_true", default=True, help="Use state-dependent alpha_net (default: True)")
parser.add_argument("--no_adaptive_alpha", action="store_false", dest="use_adaptive_alpha", help="Use scalar alpha instead")
parser.add_argument("--debug_alpha", action="store_true", help="Print alpha/delta stats in act() to diagnose 0%% SR")
parser.add_argument("--no_adaptive_entropy", action="store_true", help="Use fixed cEnt instead of Adaptive Entropy")
parser.add_argument("--c_ent_bad", type=float, default=0.02, help="High entropy weight for adv<0 (failed steps)")
parser.add_argument("--c_ent_good", type=float, default=0.005, help="Low entropy weight for adv>0 (success steps)")

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
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import SO_101.tasks  # noqa: F401 Register envs
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from SO_101.policies.graph_unet_policy import GraphUnetPolicy, UnetPolicy
from SO_101.policies.dual_arm_unet_policy import DualArmDisentangledPolicy
from SO_101.policies.dual_arm_unet_policy_gated import DualArmDisentangledPolicyGated
from SO_101.policies.graph_unet_residual_rl_policy import (
    GraphUnetBackboneAdapter,
    GraphUnetResidualRLCfg,
    GraphUnetResidualRLPolicy,
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

        self.obs = torch.zeros(T, N, obs_dim, device=device)  # Raw obs (for recomputing z_layers)
        self.obs_norm = torch.zeros(T, N, obs_dim, device=device)  # Normalized obs (for Actor/Critic)
        self.actions = torch.zeros(T, N, action_dim, device=device)
        self.rewards = torch.zeros(T, N, device=device)
        self.dones = torch.zeros(T, N, device=device)
        self.timeouts = torch.zeros(T, N, device=device)  # Truncated (timeout) for GAE bootstrap
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
        truncated: torch.Tensor,  # Timeout only; for GAE bootstrap (nonterminal=1 when truncated)
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
        self.timeouts[t] = truncated
        self.values[t] = info["v_bar"]
        self.log_probs[t] = info["log_prob"]
        self.a_base[t] = info["a_base"]
        self.z_layers[t] = info["z_layers"]
        self.z_bar[t] = info["z_bar"]
        self.gate_w[t] = info["gate_w"]

        self.ptr += 1

    def compute_returns(self, last_value: torch.Tensor, gamma: float, lam: float):
        """计算 GAE advantages 和 returns（truncated 时 bootstrap next_value）"""
        self.advantages, self.returns = compute_gae(
            self.rewards, self.dones, self.values, last_value, gamma, lam, timeouts=self.timeouts
        )

    def get_batches(self, mini_batch_size: int, num_epochs: int):
        """生成 mini-batch 迭代器"""
        T, N = self.steps_per_env, self.num_envs
        total_samples = T * N

        # Flatten
        obs_flat = self.obs.reshape(total_samples, -1)
        obs_norm_flat = self.obs_norm.reshape(total_samples, -1)
        actions_flat = self.actions.reshape(total_samples, -1)
        returns_flat = self.returns.reshape(total_samples)
        adv_flat = self.advantages.reshape(total_samples)
        a_base_flat = self.a_base.reshape(total_samples, -1)
        z_layers_flat = self.z_layers.reshape(total_samples, self.num_layers, -1)
        z_bar_flat = self.z_bar.reshape(total_samples, -1)
        values_flat = self.values.reshape(total_samples)

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        for _ in range(num_epochs):
            indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, mini_batch_size):
                end = min(start + mini_batch_size, total_samples)
                idx = indices[start:end]

                yield {
                    "obs": obs_flat[idx],
                    "obs_norm": obs_norm_flat[idx],
                    "action": actions_flat[idx],
                    "returns": returns_flat[idx],
                    "adv": adv_flat[idx],
                    "a_base": a_base_flat[idx],
                    "z_layers": z_layers_flat[idx],
                    "z_bar": z_bar_flat[idx],
                    "values": values_flat[idx],
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
        policy: GraphUnetResidualRLPolicy,
        cfg: GraphUnetResidualRLCfg,
        device: str = "cuda",
        log_dir: str = "./logs",
        # Training params
        steps_per_env: int = 160,
        num_epochs: int = 5,
        mini_batch_size: int = 64,
        lr: float = 3e-4,
        max_grad_norm: float = 1.0,
        action_history_length: int = 4,  # From Graph-DiT config
        action_dim_env: int = 6,  # Environment action dim (usually 6, includes gripper)
        best_sr_window: int = 200,  # Rolling window for best-model selection (200: ±7% CI)
        best_sr_require_consistent: bool = False,  # Require current SR >= sr_window - 15%
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

        # EMA smoothing for joints
        self._ema_smoothed_joints = None
        # Dual arm detection
        self._is_dual_arm = (cfg.action_dim == 12)
        self._robot_state_dim = cfg.robot_state_dim

        # Multi-env fix: cache env_origins for world→local conversion.
        # Isaac Lab observation terms (ee_position_w, object_position_w) return
        # world-frame positions. BC normalization stats are in local frame.
        _base = env.unwrapped if hasattr(env, "unwrapped") else env
        self._env_origins = _base.scene.env_origins.clone().to(device=device, dtype=torch.float32)
        self._pos_keys = [
            "left_ee_position", "right_ee_position",
            "cube_1_pos", "cube_2_pos",
            "fork_pos", "knife_pos",
        ]
        _n_fixed = sum(1 for k in self._pos_keys if k in (cfg.obs_structure or {}))
        print(f"[Trainer] env_origins shape: {self._env_origins.shape}, fixing {_n_fixed} pos keys to local frame")

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
        
        self.optimizer = optim.AdamW(trainable_params, lr=lr)

        # Logger
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Stats
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.best_sr = -1.0  # For best-model-by-SR saving
        self.best_sr_window = best_sr_window
        self.best_sr_require_consistent = best_sr_require_consistent

        # Initialize env buffers
        policy.init_env_buffers(self.num_envs)
        
        # History buffers for Graph-DiT (optimized: single tensor per history type)
        # Joint state dim: 6 for single arm, 12 for dual arm
        graph_dit_joint_dim = cfg.robot_state_dim
        
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

        print(f"[Trainer] num_envs: {self.num_envs}, steps_per_env: {steps_per_env}")
        if steps_per_env < 400:
            print(
                f"[Trainer] ⚠️ WARNING: steps_per_env={steps_per_env} < 400 (env max_episode_steps). "
                "Episodes may be truncated by rollout before env termination! "
                "Dataset mean=251, P95=339, env_max=400. Recommend steps_per_env >= 400."
            )
        print(f"[Trainer] trainable params: {sum(p.numel() for p in trainable_params):,}")

        # Success detection: use env termination signals from info dict
        # Isaac Lab returns Episode_Termination/success, Episode_Termination/stack_success etc.
        print(f"[Trainer] Success: using env termination signals (Episode_Termination/success)")
        print(f"[Trainer] truncated=timeout=failure, terminated+success_signal=success")
        print(f"[Trainer] Best-model: SR_{self.best_sr_window}ep window, consistent_check={self.best_sr_require_consistent}")

    def _check_success_from_info(self, env_info, env_idx: int) -> bool:
        """Check success from termination manager's per-env signals.
        
        Isaac Lab extras['log']['Episode_Termination/X'] are scalar averages,
        NOT per-env flags. We access termination_manager._last_episode_dones directly.
        """
        try:
            base = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            tm = base.termination_manager
            names = list(tm._term_names)
            for sname in ("success", "stack_success"):
                if sname in names:
                    idx = names.index(sname)
                    return bool(tm._last_episode_dones[env_idx, idx].item())
        except Exception:
            pass
        return False

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
        """收集 rollout (FIXED: 添加 success rate，除零保护，移除 DEBUG 打印)"""
        self.policy.eval()
        self.buffer.reset()

        raw_obs, _ = self.env.reset()
        obs = self._process_obs(raw_obs)
        subtask_cond = self._extract_subtask_condition(raw_obs, obs)

        # Pre-fill histories with first observation (matches BC training padding)
        self._prefill_histories_from_obs(obs)

        if not hasattr(self, "_subtask_logged"):
            self._subtask_logged = True
            if subtask_cond is not None:
                print(f"[Trainer] subtask_condition enabled: shape={subtask_cond.shape}, env0={subtask_cond[0].cpu().tolist()}")
            else:
                _bb = self.policy.backbone
                _actual_bb = _bb.backbone if hasattr(_bb, "backbone") else _bb
                _ns = getattr(getattr(_actual_bb, "cfg", None), "num_subtasks", 0)
                print(f"[Trainer] subtask_condition: None (backbone num_subtasks={_ns})")

        ep_rewards = torch.zeros(self.num_envs, device=self.device)
        ep_lengths = torch.zeros(self.num_envs, device=self.device)
        rollout_successes = []
        delta_norms_all = []

        for step in range(self.steps_per_env):
            # FIXED: 1e-8 除零保护
            if self.obs_mean is not None and self.obs_std is not None:
                obs_norm = (obs - self.obs_mean) / (self.obs_std + 1e-8)
            else:
                obs_norm = obs

            ee_node_current, object_node_current = self.policy._extract_nodes_from_obs(obs)
            obs_struct = getattr(self.cfg, "obs_structure", None)
            if obs_struct is not None and "left_joint_pos" in obs_struct:
                joint_states_current = torch.cat([
                    obs[:, obs_struct["left_joint_pos"][0]:obs_struct["left_joint_pos"][1]],
                    obs[:, obs_struct["right_joint_pos"][0]:obs_struct["right_joint_pos"][1]],
                ], dim=-1)
            else:
                joint_states_current = obs[:, :self._robot_state_dim]

            action_history = self.action_history
            ee_node_history = self.ee_node_history.clone()
            object_node_history = self.object_node_history.clone()
            if self.ee_node_mean is not None and self.ee_node_std is not None:
                ee_node_history = (ee_node_history - self.ee_node_mean) / (self.ee_node_std + 1e-8)
            if self.object_node_mean is not None and self.object_node_std is not None:
                object_node_history = (object_node_history - self.object_node_mean) / (self.object_node_std + 1e-8)

            joint_states_history = self.joint_state_history.clone()
            if self.joint_mean is not None and self.joint_std is not None:
                joint_states_history = (joint_states_history - self.joint_mean) / (self.joint_std + 1e-8)

            with torch.no_grad():
                action, info = self.policy.act(
                    obs_raw=obs,
                    obs_norm=obs_norm,
                    action_history=action_history,
                    ee_node_history=ee_node_history,
                    object_node_history=object_node_history,
                    joint_states_history=joint_states_history,
                    subtask_condition=subtask_cond,
                    deterministic=False
                )

                if hasattr(self.env, 'action_space'):
                    if hasattr(self.env.action_space, 'low') and hasattr(self.env.action_space, 'high'):
                        action = torch.clamp(
                            action,
                            torch.tensor(self.env.action_space.low, device=action.device, dtype=action.dtype),
                            torch.tensor(self.env.action_space.high, device=action.device, dtype=action.dtype)
                        )

                delta = info.get("delta", None)
                if delta is not None:
                    # Log joint delta norms (exclude grippers)
                    joint_delta = torch.cat([delta[:, :5], delta[:, 6:11]], dim=-1) if self._is_dual_arm else delta[:, :5]
                    delta_norms_all.append(torch.norm(joint_delta, dim=-1).mean().item())

                action_dim = action.shape[-1]
                action_for_sim = action.clone()

                # CRITICAL: Backbone outputs [left_6, right_6], env expects [right_6, left_6]
                # Must swap before env.step (matching play.py line 1581)
                if action_dim == 12:
                    action_for_sim = torch.cat([action_for_sim[:, 6:12], action_for_sim[:, 0:6]], dim=1)

                # Process each arm block: [5 joints, 1 gripper] × num_arms (now in env order)
                arm_block = 6  # 5 joints + 1 gripper
                for arm_i in range(action_dim // arm_block):
                    base = arm_i * arm_block
                    joints_slice = slice(base, base + 5)
                    if self._ema_smoothed_joints is None:
                        self._ema_smoothed_joints = action_for_sim[:, :action_dim].clone()
                    ema_alpha = 1.0
                    self._ema_smoothed_joints[:, joints_slice] = (
                        ema_alpha * action_for_sim[:, joints_slice]
                        + (1 - ema_alpha) * self._ema_smoothed_joints[:, joints_slice]
                    )
                    action_for_sim[:, joints_slice] = self._ema_smoothed_joints[:, joints_slice]
                    # Gripper: apply linear mapping (matching play.py gripper_a/b for Stack)
                    # then threshold for BinaryJointPositionAction (+1=open, -1=close)
                    gripper_idx = base + 5
                    mapped = 1.7150 * action_for_sim[:, gripper_idx] + 0.8579
                    action_for_sim[:, gripper_idx] = torch.where(
                        mapped > 0.2,
                        torch.tensor(1.0, device=action.device, dtype=action.dtype),
                        torch.tensor(-1.0, device=action.device, dtype=action.dtype),
                    )

            if step < 3:
                sc_str = subtask_cond[0].cpu().tolist() if subtask_cond is not None else "None"
                print(f"[Rollout step={step}] subtask_cond env0={sc_str}")
                print(f"[Rollout step={step}] action_for_sim env0 = {action_for_sim[0].cpu().tolist()}")
                print(f"[Rollout step={step}] reward env0 = ...(pending)")

            raw_next_obs, reward, terminated, truncated, env_info = self.env.step(action_for_sim)
            done = terminated | truncated

            if step < 3:
                print(f"[Rollout step={step}] reward env0 = {reward[0].item():.4f}")

            next_obs = self._process_obs(raw_next_obs)
            next_subtask_cond = self._extract_subtask_condition(raw_next_obs, next_obs)
            reward = reward.to(self.device).float()
            done = done.to(self.device).float()
            truncated_tensor = truncated.to(self.device).float()

            self.buffer.add(obs, obs_norm, action, reward, done, truncated_tensor, info)

            # Track episode stats
            ep_rewards += reward
            ep_lengths += 1

            if self.action_mean is not None and self.action_std is not None:
                action_for_history = (action - self.action_mean) / (self.action_std + 1e-8)
            else:
                action_for_history = action

            self.action_history = torch.roll(self.action_history, -1, dims=1)
            self.action_history[:, -1, :] = action_for_history
            self.ee_node_history = torch.roll(self.ee_node_history, -1, dims=1)
            self.ee_node_history[:, -1, :] = ee_node_current
            self.object_node_history = torch.roll(self.object_node_history, -1, dims=1)
            self.object_node_history[:, -1, :] = object_node_current
            self.joint_state_history = torch.roll(self.joint_state_history, -1, dims=1)
            self.joint_state_history[:, -1, :] = joint_states_current

            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_envs) > 0:
                self.policy.reset_envs(done_envs)
                for i in done_envs.tolist():
                    self.episode_rewards.append(ep_rewards[i].item())
                    self.episode_lengths.append(ep_lengths[i].item())
                    is_truncated = bool(truncated[i].item() if hasattr(truncated[i], "item") else truncated[i])
                    if is_truncated:
                        is_success = False
                    else:
                        is_success = self._check_success_from_info(env_info, i)
                    rollout_successes.append(float(is_success))
                    self.episode_successes.append(float(is_success))

                # Pre-fill histories with post-reset obs (matches BC training padding)
                self._prefill_histories_from_obs(next_obs, env_ids=done_envs)
                ep_rewards[done_envs] = 0
                ep_lengths[done_envs] = 0
                if self._ema_smoothed_joints is not None:
                    self._ema_smoothed_joints[done_envs] = 0

            obs = next_obs
            subtask_cond = next_subtask_cond
            self.total_steps += self.num_envs

        # Compute last value for GAE
        if self.obs_mean is not None and self.obs_std is not None:
            obs_norm = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        else:
            obs_norm = obs

        ee_node_history = self.ee_node_history.clone()
        object_node_history = self.object_node_history.clone()
        if self.ee_node_mean is not None and self.ee_node_std is not None:
            ee_node_history = (ee_node_history - self.ee_node_mean) / (self.ee_node_std + 1e-8)
        if self.object_node_mean is not None and self.object_node_std is not None:
            object_node_history = (object_node_history - self.object_node_mean) / (self.object_node_std + 1e-8)

        joint_states_history = self.joint_state_history.clone()
        if self.joint_mean is not None and self.joint_std is not None:
            joint_states_history = (joint_states_history - self.joint_mean) / (self.joint_std + 1e-8)

        with torch.no_grad():
            _, last_info = self.policy.act(
                obs_raw=obs,
                obs_norm=obs_norm,
                action_history=self.action_history,
                ee_node_history=ee_node_history,
                object_node_history=object_node_history,
                joint_states_history=joint_states_history,
                subtask_condition=subtask_cond,
                deterministic=True
            )
            last_value = last_info["v_bar"]

        self.buffer.compute_returns(last_value, self.cfg.gamma, self.cfg.lam)

        stats = {}
        if len(self.episode_rewards) > 0:
            stats["ep_reward_mean"] = np.mean(self.episode_rewards[-100:])
            stats["ep_length_mean"] = np.mean(self.episode_lengths[-100:])
            stats["ep_reward_max"] = np.max(self.episode_rewards[-100:])
            stats["ep_reward_min"] = np.min(self.episode_rewards[-100:])
        else:
            stats["ep_reward_mean"] = 0.0
            stats["ep_length_mean"] = 0.0
            stats["warning"] = "No episodes completed!"

        if len(rollout_successes) > 0:
            stats["success_rate"] = np.mean(rollout_successes)
            stats["num_episodes"] = len(rollout_successes)
        else:
            stats["success_rate"] = 0.0
            stats["num_episodes"] = 0

        if len(self.episode_successes) > 0:
            stats["success_rate_100"] = np.mean(self.episode_successes[-100:])
            stats["success_rate_window"] = np.mean(self.episode_successes[-self.best_sr_window:])
        else:
            stats["success_rate_100"] = 0.0
            stats["success_rate_window"] = 0.0

        if len(delta_norms_all) > 0:
            stats["delta_norm_mean"] = np.mean(delta_norms_all)
            stats["delta_norm_max"] = np.max(delta_norms_all)
        else:
            stats["delta_norm_mean"] = 0.0
            stats["delta_norm_max"] = 0.0

        return stats

    def update(self) -> Dict[str, float]:
        """更新 policy (FIXED: 添加 Explained Variance)"""
        self.policy.train()

        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss_delta_reg = 0
        total_loss_critic_bar = 0
        total_entropy = 0
        total_alpha = 0.0
        num_updates = 0
        all_returns = []
        all_values = []

        for batch in self.buffer.get_batches(self.mini_batch_size, self.num_epochs):
            losses = self.policy.compute_loss(batch)

            self.optimizer.zero_grad()
            losses["loss_total"].backward()
            trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
            if hasattr(self.policy, 'backbone') and hasattr(self.policy.backbone, 'backbone'):
                graph_dit = self.policy.backbone.backbone
                if hasattr(graph_dit, 'node_to_z'):
                    trainable_params.extend([p for p in graph_dit.node_to_z.parameters() if p.requires_grad])
            # Gradient Clipping: 防小 batch 梯度爆炸，max_norm 通常 0.5~1.0
            torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
            self.optimizer.step()

            total_loss += losses["loss_total"].item()
            total_actor_loss += losses["loss_actor"].item()
            total_critic_loss += losses["loss_critic_layers"].item()
            total_loss_delta_reg += losses["loss_delta_reg"].item()
            total_loss_critic_bar += losses["loss_critic_bar"].item()
            total_entropy += losses["entropy"].item()
            alpha_val = losses.get("alpha")
            if alpha_val is not None:
                total_alpha += alpha_val.item() if hasattr(alpha_val, "item") else float(alpha_val)
            num_updates += 1
            all_returns.append(batch["returns"])
            all_values.append(batch["values"])

        n = num_updates
        all_returns = torch.cat(all_returns)
        all_values = torch.cat(all_values)
        var_returns = torch.var(all_returns)
        var_residual = torch.var(all_returns - all_values)
        explained_variance = (1.0 - var_residual / (var_returns + 1e-8)).item()

        return {
            "loss_total": total_loss / n,
            "loss_actor": total_actor_loss / n,
            "loss_critic": total_critic_loss / n,
            "loss_critic_bar": total_loss_critic_bar / n,
            "loss_delta_reg": total_loss_delta_reg / n,
            "entropy": total_entropy / n,
            "alpha": total_alpha / num_updates if num_updates else 0.0,
            "explained_variance": explained_variance,
        }

    def train(self, max_iterations: int, save_interval: int = 50):
        """主训练循环"""
        print(f"\n[Trainer] Starting training for {max_iterations} iterations...")

        # Step Decay LR Scheduler: 0-25% 1.0, 25-50% 0.8, 50-75% 0.6, 75-100% 0.4 (final lr = 2e-4 when base=5e-4)
        def lr_lambda(epoch):
            progress = (epoch + 1) / max_iterations
            if progress < 0.25:
                return 1.0
            elif progress < 0.50:
                return 0.8
            elif progress < 0.75:
                return 0.6
            else:
                return 0.4

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        for iteration in range(1, max_iterations + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]
            # Collect
            rollout_stats = self.collect_rollout()

            # Update
            update_stats = self.update()
            update_stats["lr"] = current_lr

            # Log
            self._log(iteration, rollout_stats, update_stats)

            # Save best by success rate (use rolling N-ep SR; 200ep: ±7% CI)
            sr_window = rollout_stats.get("success_rate_window", 0.0)
            sr_current = rollout_stats.get("success_rate", 0.0)
            # Require enough episodes in buffer for window to be valid
            n_eps = len(self.episode_successes)
            window_valid = n_eps >= self.best_sr_window
            # Optional: require current SR not far below window (avoid saving lucky fluke)
            consistent = (not self.best_sr_require_consistent) or (sr_current >= sr_window - 0.15)
            if window_valid and consistent and sr_window > self.best_sr:
                self.best_sr = sr_window
                self._save_best(iteration)

            # Save when current iteration SR hits new high (captures peak performance)
            if not hasattr(self, "_best_iter_sr"):
                self._best_iter_sr = -1.0
            if sr_current > self._best_iter_sr and sr_current > 0:
                self._best_iter_sr = sr_current
                path = os.path.join(self.log_dir, f"policy_peak_sr{sr_current*100:.0f}_iter{iteration}.pt")
                self.policy.save(path)
                print(f"[Trainer] Peak SR saved (iter {iteration}, SR={sr_current*100:.1f}%): {path}")

            # Save interval
            if iteration % save_interval == 0:
                self._save(iteration)

            scheduler.step()

        # Final save
        self._save(max_iterations, final=True)
        self.writer.close()
        print(f"\n[Trainer] Training complete!")

    def _extract_subtask_condition(self, raw_obs, obs_local: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Extract subtask condition from env obs.

        Priority:
        1. raw_obs["subtask_terms"] dict (if env provides it)
        2. env.get_subtask_term_signals() API
        3. Compute from positions in obs_local (cube-stack or table-setting)
        4. Default: first subtask active [1,0,...] (matches play.py fallback)
        """
        bb = self.policy.backbone
        actual_bb = bb.backbone if hasattr(bb, "backbone") else bb
        num_subtasks = getattr(getattr(actual_bb, "cfg", None), "num_subtasks", 0)
        if num_subtasks <= 0:
            return None

        obs_struct = getattr(self.cfg, "obs_structure", None)

        # --- Path 1: subtask_terms in obs dict ---
        if isinstance(raw_obs, dict) and "subtask_terms" in raw_obs:
            st = raw_obs["subtask_terms"]
            if isinstance(st, dict):
                pick = st.get("pick_cube", None)
                if pick is None:
                    pick = st.get("place_fork", None)
                stack = st.get("stack_cube", None)
                if stack is None:
                    stack = st.get("place_knife", None)
                if pick is not None and stack is not None:
                    return self._build_subtask_cond(pick, stack, num_subtasks)

        # --- Path 2: env signal API ---
        try:
            base = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            if hasattr(base, "get_subtask_term_signals"):
                signals = base.get_subtask_term_signals()
                pick = signals.get("pick_cube", None)
                if pick is None:
                    pick = signals.get("place_fork", None)
                stack = signals.get("stack_cube", None)
                if stack is None:
                    stack = signals.get("place_knife", None)
                if pick is not None and stack is not None:
                    return self._build_subtask_cond(pick, stack, num_subtasks)
        except Exception:
            pass

        # --- Path 3: compute from obs positions (matches play.py obs_mimic fallback) ---
        if obs_local is not None and obs_struct is not None and num_subtasks >= 2:
            cond = self._subtask_from_positions(obs_local, obs_struct, num_subtasks)
            if cond is not None:
                return cond

        # --- Path 4: default first subtask active ---
        B = obs_local.shape[0] if obs_local is not None else self.num_envs
        cond = torch.zeros(B, num_subtasks, device=self.device)
        cond[:, 0] = 1.0
        return cond

    def _build_subtask_cond(self, pick, stack, num_subtasks: int) -> torch.Tensor:
        pick = pick.to(self.device).float().bool()
        stack = stack.to(self.device).float().bool()
        B = pick.shape[0]
        cond = torch.zeros(B, num_subtasks, device=self.device)
        cond[~pick, 0] = 1.0
        cond[pick & ~stack, 1] = 1.0
        cond[pick & stack, 1] = 1.0
        return cond

    def _subtask_from_positions(self, obs: torch.Tensor, obs_struct: dict, num_subtasks: int) -> Optional[torch.Tensor]:
        """Compute subtask condition from object positions (cube-stack or table-setting)."""
        B = obs.shape[0]
        dev = obs.device

        # Cube stack: check pick (cube at target) + stack (cubes stacked)
        if "cube_1_pos" in obs_struct and "cube_2_pos" in obs_struct:
            PICK_XY = (0.1623, -0.023)
            PICK_Z = 0.006
            PICK_EPS_XY = 0.0603
            PICK_EPS_Z = 0.002
            STACK_H = 0.018
            STACK_EPS_Z = 0.001
            STACK_EPS_XY = 0.0073

            s1 = obs_struct["cube_1_pos"][0]
            s2 = obs_struct["cube_2_pos"][0]
            c1 = obs[:, s1:s1+3]
            c2 = obs[:, s2:s2+3]
            tgt_xy = torch.tensor([PICK_XY[0], PICK_XY[1]], device=dev, dtype=obs.dtype).unsqueeze(0)

            at1 = (torch.norm(c1[:,:2] - tgt_xy, dim=1) < PICK_EPS_XY) & (torch.abs(c1[:,2] - PICK_Z) < PICK_EPS_Z)
            at2 = (torch.norm(c2[:,:2] - tgt_xy, dim=1) < PICK_EPS_XY) & (torch.abs(c2[:,2] - PICK_Z) < PICK_EPS_Z)
            pick_done = at1 | at2

            za = (torch.abs((c1[:,2] - c2[:,2]) - STACK_H) < STACK_EPS_Z) & (torch.norm(c1[:,:2] - c2[:,:2], dim=1) < STACK_EPS_XY)
            zb = (torch.abs((c2[:,2] - c1[:,2]) - STACK_H) < STACK_EPS_Z) & (torch.norm(c2[:,:2] - c1[:,:2], dim=1) < STACK_EPS_XY)
            stack_done = za | zb

            cond = torch.zeros(B, num_subtasks, device=dev)
            cond[~pick_done, 0] = 1.0
            cond[pick_done & ~stack_done, 1] = 1.0
            cond[pick_done & stack_done, 1] = 1.0
            return cond

        # Table setting: check fork placed + knife placed
        if "fork_pos" in obs_struct and "knife_pos" in obs_struct:
            FORK_XY = (0.255, 0.18)
            FORK_Z = 0.0076
            KNIFE_XY = (0.2366, -0.1288)
            KNIFE_Z = 0.0068
            EPS_XY = 0.05
            EPS_Z = 0.01

            sf = obs_struct["fork_pos"][0]
            sk = obs_struct["knife_pos"][0]
            fp = obs[:, sf:sf+3]
            kp = obs[:, sk:sk+3]
            fork_tgt = torch.tensor([FORK_XY[0], FORK_XY[1]], device=dev, dtype=obs.dtype).unsqueeze(0)
            knife_tgt = torch.tensor([KNIFE_XY[0], KNIFE_XY[1]], device=dev, dtype=obs.dtype).unsqueeze(0)

            fork_done = (torch.norm(fp[:,:2] - fork_tgt, dim=1) < EPS_XY) & (torch.abs(fp[:,2] - FORK_Z) < EPS_Z)
            knife_done = (torch.norm(kp[:,:2] - knife_tgt, dim=1) < EPS_XY) & (torch.abs(kp[:,2] - KNIFE_Z) < EPS_Z)

            cond = torch.zeros(B, num_subtasks, device=dev)
            cond[~fork_done, 0] = 1.0
            cond[fork_done & ~knife_done, 1] = 1.0
            cond[fork_done & knife_done, 1] = 1.0
            return cond

        return None

    def _process_obs(self, obs) -> torch.Tensor:
        """处理 observation: flatten dict → tensor, then convert world→local frame."""
        if isinstance(obs, dict):
            if "policy" in obs:
                obs = obs["policy"]
            else:
                obs = torch.cat([v.flatten(start_dim=1) for v in obs.values()], dim=-1)
        obs = obs.to(self.device).float()
        return self._obs_to_local(obs)

    def _obs_to_local(self, obs: torch.Tensor) -> torch.Tensor:
        """Subtract env_origins from position observation keys (world→local frame)."""
        obs_struct = getattr(self.cfg, "obs_structure", None)
        if obs_struct is None:
            return obs
        origins = self._env_origins[:obs.shape[0], :3].to(obs.device)
        for pk in self._pos_keys:
            if pk in obs_struct:
                s, e = obs_struct[pk]
                obs[:, s:e] -= origins
        return obs

    def _prefill_histories_from_obs(self, obs: torch.Tensor, env_ids=None):
        """Pre-fill history buffers with first-frame values (matches BC training padding).

        BC training pads early-episode histories with copies of the first real
        frame, NOT zeros.  Zero-histories give out-of-distribution inputs.
        """
        if env_ids is None:
            env_ids = torch.arange(obs.shape[0], device=obs.device)

        ee_node, obj_node = self.policy._extract_nodes_from_obs(obs)
        obs_struct = getattr(self.cfg, "obs_structure", None)
        if obs_struct is not None and "left_joint_pos" in obs_struct:
            joint_states = torch.cat([
                obs[:, obs_struct["left_joint_pos"][0]:obs_struct["left_joint_pos"][1]],
                obs[:, obs_struct["right_joint_pos"][0]:obs_struct["right_joint_pos"][1]],
            ], dim=-1)
        else:
            joint_states = obs[:, :self._robot_state_dim]

        H = self.ee_node_history.shape[1]
        for i in env_ids.tolist():
            for t in range(H):
                self.ee_node_history[i, t] = ee_node[i]
                self.object_node_history[i, t] = obj_node[i]
                self.joint_state_history[i, t] = joint_states[i]
            if self.action_mean is not None and self.action_std is not None:
                init_action_norm = (joint_states[i] - self.action_mean) / (self.action_std + 1e-8)
                for t in range(H):
                    self.action_history[i, t, :init_action_norm.shape[0]] = init_action_norm
            else:
                self.action_history[i] = 0

        # Pre-fill multi-node temporal history in policy (matches play.py padding)
        self.policy.prefill_node_history(obs, env_ids=env_ids)

    def _log(self, iteration: int, rollout_stats: Dict, update_stats: Dict):
        """记录日志 (FIXED: 关键指标一目了然，只打印一行)"""
        def _scalar(tag: str, value, step: int):
            if isinstance(value, (int, float)):
                self.writer.add_scalar(tag, value, step)
            elif hasattr(value, "item"):
                try:
                    self.writer.add_scalar(tag, value.item(), step)
                except (ValueError, TypeError):
                    pass

        step = self.total_steps

        _scalar("main/success_rate", rollout_stats.get("success_rate", 0), step)
        _scalar("main/success_rate_100", rollout_stats.get("success_rate_100", 0), step)
        _scalar("main/success_rate_window", rollout_stats.get("success_rate_window", 0), step)
        _scalar("main/num_episodes", rollout_stats.get("num_episodes", 0), step)
        _scalar("main/ep_reward_mean", rollout_stats.get("ep_reward_mean", 0), step)
        _scalar("main/ep_length_mean", rollout_stats.get("ep_length_mean", 0), step)
        _scalar("main/explained_variance", update_stats.get("explained_variance", 0), step)
        _scalar("main/delta_norm_mean", rollout_stats.get("delta_norm_mean", 0), step)
        _scalar("main/alpha", update_stats.get("alpha", 0), step)
        _scalar("main/loss_total", update_stats.get("loss_total", 0), step)
        _scalar("main/loss_actor", update_stats.get("loss_actor", 0), step)
        _scalar("main/loss_critic_bar", update_stats.get("loss_critic_bar", 0), step)
        _scalar("main/entropy", update_stats.get("entropy", 0), step)
        _scalar("main/lr", update_stats.get("lr", 0), step)

        sr = rollout_stats.get("success_rate", 0) * 100
        sr_100 = rollout_stats.get("success_rate_100", 0) * 100
        sr_win = rollout_stats.get("success_rate_window", 0) * 100
        num_eps = rollout_stats.get("num_episodes", 0)
        reward = rollout_stats.get("ep_reward_mean", 0)
        ev = update_stats.get("explained_variance", -999)
        delta = rollout_stats.get("delta_norm_mean", 0)
        alpha = update_stats.get("alpha", 0)
        loss = update_stats.get("loss_total", 0)
        lr = update_stats.get("lr", 0)
        warning = rollout_stats.get("warning", "")
        if warning:
            print(f"[Iter {iteration:4d}] ⚠️  {warning}")
        print(
            f"[Iter {iteration:4d}] "
            f"SR={sr:5.1f}% (100:{sr_100:4.1f}% {self.best_sr_window}ep:{sr_win:4.1f}%) [{num_eps:2d}ep] | "
            f"Rew={reward:6.1f} | "
            f"EV={ev:5.2f} | "
            f"Δ={delta:.3f} | "
            f"α={alpha:.2f} | "
            f"L={loss:.3f} | "
            f"LR={lr:.1e}"
        )

    def _save_best(self, iteration: int):
        """按 success_rate_window（最近 N episode 平均）保存当前最佳模型"""
        path = os.path.join(self.log_dir, "best_model.pt")
        self.policy.save(path)
        print(f"[Trainer] Best model saved (SR_{self.best_sr_window}ep={self.best_sr*100:.1f}%): {path}")

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
    print("GraphDiT + Residual RL Training (FIXED VERSION)")
    print("=" * 70)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get device from AppLauncher (or use default)
    device = getattr(args, "device", "cuda")

    # Load pretrained backbone (auto-detect from checkpoint or CLI flag)
    _backbone_classes = {
        "graph_unet": GraphUnetPolicy,
        "unet": UnetPolicy,
        "dual_arm_gated": DualArmDisentangledPolicyGated,
        "dual_arm": DualArmDisentangledPolicy,
    }
    # Auto-detect: if checkpoint cfg has arm_action_dim → dual arm
    _ckpt_preview = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
    _ckpt_cfg = _ckpt_preview.get("cfg", None)
    if _ckpt_cfg is not None and getattr(_ckpt_cfg, "arm_action_dim", None) is not None:
        # Dual arm checkpoint - detect gated vs non-gated
        _sd = _ckpt_preview.get("policy_state_dict", _ckpt_preview.get("model_state_dict", {}))
        if "graph_gate_logit" in _sd:
            args.policy_type = "dual_arm_gated"
        else:
            args.policy_type = "dual_arm"
        print(f"[Main] Auto-detected dual arm checkpoint: policy_type={args.policy_type}")
    del _ckpt_preview

    BackboneClass = _backbone_classes.get(args.policy_type, GraphUnetPolicy)
    print(f"\n[Main] Loading pretrained {BackboneClass.__name__}: {args.pretrained_checkpoint}")
    backbone_policy = BackboneClass.load(args.pretrained_checkpoint, device=device)
    backbone_policy.eval()
    for p in backbone_policy.parameters():
        p.requires_grad = False
    if hasattr(backbone_policy, "node_to_z"):
        for p in backbone_policy.node_to_z.parameters():
            p.requires_grad = True
        n2z_trainable = sum(1 for p in backbone_policy.node_to_z.parameters() if p.requires_grad)
        n2z_total = len(list(backbone_policy.node_to_z.parameters()))
        print(f"[Main] node_to_z made trainable: {n2z_trainable}/{n2z_total} params trainable")
    else:
        print(f"[Main] ⚠️  WARNING: node_to_z not found in backbone!")
    print(f"[Main] Graph-Unet loaded and frozen (except node_to_z)")

    _backbone_action_dim = getattr(backbone_policy.cfg, "action_dim", 6)
    print(f"[Main] Using backbone for all {_backbone_action_dim} action dimensions (including gripper)")

    # Create backbone adapter
    backbone = GraphUnetBackboneAdapter(backbone_policy)

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

    action_dim_env = get_action_dim(action_space)  # Environment action dim (6 single, 12 dual)
    action_dim_rl = action_dim_env  # RL outputs same dim as env (6 single, 12 dual)

    print(f"[Main] obs_dim: {obs_dim}, action_dim_env: {action_dim_env}, action_dim_rl: {action_dim_rl}")
    num_arms = max(1, action_dim_rl // 6)
    print(f"[Main] RL outputs {action_dim_rl}D ({num_arms} arm(s)); arm uses alpha_learned, gripper uses 1.0")

    # Create policy config
    obs_structure = getattr(backbone_policy.cfg, "obs_structure", None)
    print(f"[Main] Backbone obs_structure: {obs_structure}")

    # No mask: RL controls all 6 dims; per-channel alpha in policy (arm low, gripper full)
    residual_action_mask = torch.ones(action_dim_env, device=device)

    is_dual_arm = (action_dim_rl == 12)
    _robot_state_dim = 12 if is_dual_arm else 6
    _joint_dim = 10 if is_dual_arm else 5

    policy_cfg = GraphUnetResidualRLCfg(
        obs_dim=obs_dim,
        action_dim=action_dim_rl,
        z_dim=backbone_policy.cfg.z_dim,
        num_layers=backbone_policy.cfg.num_layers,
        device=device,
        obs_structure=obs_structure,
        robot_state_dim=_robot_state_dim,
        joint_dim=_joint_dim,
        residual_action_mask=residual_action_mask,
        c_delta_reg=args.c_delta_reg,
        cEnt=args.c_ent,
        beta=args.beta,
        alpha_init=args.alpha_init,
        use_expectile_value=True,
        expectile_tau=args.expectile_tau,
        use_adaptive_alpha=args.use_adaptive_alpha,
        debug_alpha=getattr(args, "debug_alpha", False),
        use_adaptive_entropy=not getattr(args, "no_adaptive_entropy", False),
        c_ent_bad=args.c_ent_bad,
        c_ent_good=args.c_ent_good,
    )
    print(f"[Main] residual_action_mask: {residual_action_mask.tolist()} (all 1s, no mask)")
    print(f"[Main] Residual RL obs_structure: {policy_cfg.obs_structure}")

    if getattr(backbone_policy.cfg, "obs_structure", None) != policy_cfg.obs_structure:
        print("[Main] ⚠️  WARNING: obs_structure MISMATCH between backbone and Residual RL!")
    else:
        print("[Main] ✅ obs_structure consistent")
    ent_mode = f"Adaptive Entropy (bad={getattr(policy_cfg,'c_ent_bad',0.1)}, good={getattr(policy_cfg,'c_ent_good',0.005)})" if getattr(policy_cfg, "use_adaptive_entropy", False) else f"Fixed cEnt={policy_cfg.cEnt}"
    print(f"[Main] Entropy: {ent_mode}")

    # Create policy (gripper from backbone 6th dim only)
    policy = GraphUnetResidualRLPolicy(
        cfg=policy_cfg,
        backbone=backbone,
        pred_horizon=getattr(backbone_policy.cfg, "pred_horizon", 16),
        exec_horizon=getattr(backbone_policy.cfg, "exec_horizon", 8),
        num_diffusion_steps=15,
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

    # Run folder name from key hyperparams only (no timestamp)
    use_adapt_ent = not getattr(args, "no_adaptive_entropy", False)
    run_name = (
        f"env{args.num_envs}_epochs{args.num_epochs}_beta{args.beta}_"
        f"cEnt{args.c_ent}_cDelta{args.c_delta_reg}_adaptEnt{use_adapt_ent}"
    )
    run_name = run_name.replace(" ", "").replace("/", "-")
    log_dir = os.path.join(args.log_dir, args.task, run_name)

    # Get num_envs from env_cfg (more reliable)
    num_envs = env_cfg.scene.num_envs if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs") else args.num_envs
    print(f"[Main] Using num_envs: {num_envs}")

    # Get action_history_length from backbone config
    action_history_length = getattr(backbone_policy.cfg, "action_history_length", 4)
    
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
        action_dim_env=action_dim_env,  # Pass environment action dim (6 or 12) for history buffer
        best_sr_window=getattr(args, "best_sr_window", 200),
        best_sr_require_consistent=getattr(args, "best_sr_require_consistent", False),
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
    backbone_obs_struct = getattr(backbone_policy.cfg, "obs_structure", None)
    print(f"    Backbone: {backbone_obs_struct}")
    print(f"    Residual RL: {policy_cfg.obs_structure}")
    if backbone_obs_struct != policy_cfg.obs_structure:
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
    if hasattr(backbone_policy, "node_to_z"):
        n2z_trainable = sum(1 for p in backbone_policy.node_to_z.parameters() if p.requires_grad)
        n2z_total = len(list(backbone_policy.node_to_z.parameters()))
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
        z = policy._get_z_layers_fast(test_obs)
        print(f"    z_layers shape: {z.shape} ✅")
        print(f"    Forward pass test: OK")
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
    # Restore env buffers to actual num_envs (test above used size=2)
    policy.init_env_buffers(num_envs)
    
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

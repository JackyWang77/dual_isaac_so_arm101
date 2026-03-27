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
parser.add_argument("--run_name", type=str, default=None, help="Override run folder name (e.g. for ablation: reg_0.5)")
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
parser.add_argument("--max_delta_norm", type=float, default=0.0, help="Hard clamp on ||delta||_2; 0=disabled. Projects delta to L2 ball.")
parser.add_argument("--expectile_tau", type=float, default=0.7, help="Expectile τ for value loss (IQL-style); τ=0.7 → optimistic; 0.5=MSE")
parser.add_argument("--no_adaptive_entropy", action="store_true", help="Use fixed cEnt instead of Adaptive Entropy")
parser.add_argument("--c_ent_bad", type=float, default=0.02, help="High entropy weight for adv<0 (failed steps)")
parser.add_argument("--c_ent_good", type=float, default=0.005, help="Low entropy weight for adv>0 (success steps)")
parser.add_argument("--critic_warmup_iters", type=int, default=5,
    help="First N iters: only train critic (freeze actor/z_adapter). 'Don't move before critic learns'.")
parser.add_argument("--resume", type=str, default=None,
    help="Resume from RL checkpoint (policy_iter_X.pt or policy_final.pt). Continue training to max_iterations.")
parser.add_argument("--use_counterfactual_q", action="store_true", default=False,
    help="Use Q(s,a) critic with counterfactual advantage A_res = Q(s,a_total) - Q(s,a_base)")
parser.add_argument("--counterfactual_log_tau", type=float, default=0.5,
    help="Log compression temperature for counterfactual advantage: sign(A)*log(1+|A|/tau)*tau")

# SAC-style adaptive parameters (data-driven, no manual tuning)
parser.add_argument("--use_adaptive_delta_reg", action="store_true", default=True,
    help="SAC-style learnable delta_reg: auto-adjusts to keep ||δ|| near target (default: True)")
parser.add_argument("--no_adaptive_delta_reg", action="store_false", dest="use_adaptive_delta_reg",
    help="Use fixed c_delta_reg instead of adaptive")
parser.add_argument("--target_delta_norm", type=float, default=0.15,
    help="Target ||δ|| for adaptive delta_reg (default: 0.15)")
parser.add_argument("--c_delta_reg_init", type=float, default=5.0,
    help="Initial c_delta_reg for adaptive mode (log-space learnable)")
parser.add_argument("--use_auto_entropy", action="store_true", default=True,
    help="SAC-style learnable entropy coefficient: auto-adjusts to target entropy (default: True)")
parser.add_argument("--no_auto_entropy", action="store_false", dest="use_auto_entropy",
    help="Use manual entropy (adaptive or fixed) instead of auto")
parser.add_argument("--target_entropy", type=float, default=-6.0,
    help="Target entropy for auto entropy tuning (default: -action_dim/2 ≈ -6)")
parser.add_argument("--c_ent_init", type=float, default=0.01,
    help="Initial entropy coefficient for auto mode (log-space learnable)")
parser.add_argument("--use_adaptive_beta", action="store_true", default=True,
    help="SAC-style learnable AWR beta: auto-adjusts to target effective sample ratio (default: True)")
parser.add_argument("--no_adaptive_beta", action="store_false", dest="use_adaptive_beta",
    help="Use fixed beta instead of adaptive")
parser.add_argument("--target_eff_ratio", type=float, default=0.4,
    help="Target effective sample ratio for adaptive beta (0.4 = ~40%% of batch effectively used)")
parser.add_argument("--beta_init", type=float, default=0.3,
    help="Initial AWR beta for adaptive mode (log-space learnable)")

# Expert Intervention (Jacobian correction + DAgger)
parser.add_argument("--use_expert_intervention", action="store_true", default=False,
    help="Enable expert Jacobian intervention during rollout (DAgger-style)")
parser.add_argument("--expert_intervention_ratio", type=float, default=1.0,
    help="Initial expert intervention probability [0,1]; decays per iteration")
parser.add_argument("--expert_intervention_decay", type=float, default=0.95,
    help="Per-iteration decay rate for expert intervention ratio")

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
import re
import json
import time
from typing import Dict, Optional, Tuple

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
        # Q(s, a_base) for counterfactual advantage (only used when use_counterfactual_q=True)
        self.q_base = torch.zeros(T, N, device=device)

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
        if info.get("q_base") is not None:
            self.q_base[t] = info["q_base"].detach()

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
        q_base_flat = self.q_base.reshape(total_samples)  # Q(s, a_base) for counterfactual

        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Normalize returns AND values with same stats for critic stability
        # (prevents MSE loss explosion; values must use same scale for EV to be meaningful)
        returns_mean = returns_flat.mean()
        returns_std = returns_flat.std() + 1e-8
        returns_flat = (returns_flat - returns_mean) / returns_std
        values_flat = (values_flat - returns_mean) / returns_std
        q_base_flat = (q_base_flat - returns_mean) / returns_std  # Same scale as q_total for counterfactual

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
                    "q_total": values_flat[idx],  # values = Q(s, a_total) in Q mode
                    "q_base": q_base_flat[idx],   # Q(s, a_base) for counterfactual
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
        critic_warmup_iters: int = 0,  # First N iters: only train critic; 0 = disabled
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

        # Optimizer (只优化 trainable 参数: actor, critic, z_adapter, alpha)
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable_params, lr=lr)

        # Logger
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Stats
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []  # NOTE: biased by episode completion order (short=success first)
        self.iter_success_rates = []  # Per-iteration SR (unbiased, one entry per rollout)
        self.best_sr = -1.0  # For best-model-by-SR saving
        self.best_sr_window = best_sr_window
        self.best_sr_require_consistent = best_sr_require_consistent
        self.critic_warmup_iters = critic_warmup_iters

        # Success detection: set up termination manager refs ONCE (matching play.py)
        self._term_mgr = _base.termination_manager
        self._term_names = list(self._term_mgr._term_names)
        self._success_term_idx = None
        for _sname in ("success", "stack_success"):
            if _sname in self._term_names:
                self._success_term_idx = self._term_names.index(_sname)
                break
        self._drop_term_indices = []
        for _dname in ("object_dropping", "cube_1_dropping", "cube_2_dropping",
                        "fork_dropping", "knife_dropping"):
            if _dname in self._term_names:
                self._drop_term_indices.append(self._term_names.index(_dname))
        print(f"[Trainer] Termination terms: {self._term_names}")
        print(f"[Trainer] Success term idx: {self._success_term_idx} "
              f"({self._term_names[self._success_term_idx] if self._success_term_idx is not None else 'NONE'})")
        print(f"[Trainer] Drop term indices: {[(i, self._term_names[i]) for i in self._drop_term_indices]}")

        # Expert Intervention (Jacobian correction + DAgger)
        self.use_expert_intervention = getattr(cfg, 'use_expert_intervention', False)
        self.expert_ratio = getattr(cfg, 'expert_intervention_ratio', 1.0)
        self.expert_decay = getattr(cfg, 'expert_intervention_decay', 0.95)
        self._expert_initialized = False

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
        env_max_steps = 400  # CubeStackRL: episode_length_s=8.0 @ 50Hz
        if steps_per_env < env_max_steps:
            print(
                f"[Trainer] ⚠️ WARNING: steps_per_env={steps_per_env} < {env_max_steps} (env max_episode_steps). "
                "Rollout stops before timeout → timeout failures are NEVER counted. SR will be inflated. "
                f"Use steps_per_env >= {env_max_steps}."
            )
        print(f"[Trainer] trainable params: {sum(p.numel() for p in trainable_params):,}")

        # Success detection: use env termination signals from info dict
        # Isaac Lab returns Episode_Termination/success, Episode_Termination/stack_success etc.
        print(f"[Trainer] Success: using env termination signals (Episode_Termination/success)")
        print(f"[Trainer] truncated=timeout=failure, terminated+success_signal=success")
        print(f"[Trainer] Best-model: SR_{self.best_sr_window}ep window, consistent_check={self.best_sr_require_consistent}")

    def _init_expert_intervention(self):
        """Initialize Jacobian infrastructure for expert correction (lazy, called on first rollout)."""
        base = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        self._right_arm = base.scene["right_arm"]
        # EE is last body in kinematic chain
        # Jacobian: [N, num_bodies, 6, num_joints]
        # We want last body (-1), xyz rows (:3), arm joint cols (:5)
        self._expert_initialized = True
        if self.use_expert_intervention:
            print(f"[Trainer] Expert intervention initialized (ratio={self.expert_ratio:.2f}, decay={self.expert_decay})")

    def _compute_expert_delta(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute expert delta via Jacobian pseudoinverse when xy misalignment detected.

        Returns:
            expert_delta: [B, action_dim] expert correction (0 where no intervention)
            intervene_mask: [B] bool, True for envs where expert intervened
        """
        B = obs.shape[0]
        act_dim = self.cfg.action_dim
        expert_delta = torch.zeros(B, act_dim, device=self.device)
        intervene_mask = torch.zeros(B, dtype=torch.bool, device=self.device)

        s = self.cfg.obs_structure
        if s is None or "cube_1_pos" not in s or "cube_2_pos" not in s:
            return expert_delta, intervene_mask
        if "right_ee_position" not in s:
            return expert_delta, intervene_mask

        # Extract cube positions (local frame, from obs)
        c1 = obs[:, s["cube_1_pos"][0]:s["cube_1_pos"][1]]  # [B, 3]
        c2 = obs[:, s["cube_2_pos"][0]:s["cube_2_pos"][1]]  # [B, 3]
        r_ee = obs[:, s["right_ee_position"][0]:s["right_ee_position"][1]]  # [B, 3]

        # Determine target (lower z = on table) and held cube (higher z = being moved)
        # cube edge = 12mm, table z ~ 0. Stacked center z_diff = 12mm
        c1_z = c1[:, 2]
        c2_z = c2[:, 2]

        # Target = lower cube, held = higher cube
        target_xy = torch.where((c1_z < c2_z).unsqueeze(-1), c1[:, :2], c2[:, :2])
        held_z = torch.max(c1_z, c2_z)

        # Held cube xy vs target (base) cube xy
        held_xy = torch.where((c1_z < c2_z).unsqueeze(-1), c2[:, :2], c1[:, :2])
        xy_error = held_xy - target_xy  # [B, 2]
        xy_error_norm = xy_error.norm(dim=-1)

        # Z diff between held cube and target cube
        z_diff = torch.abs(held_z - torch.min(c1_z, c2_z))

        # Only intervene when held cube is near target cube:
        # - held cube above table (z > 15mm)
        # - xy_error in [3mm, 20mm]: close enough for final alignment,
        #   but not already perfect. >20mm = still transporting, let backbone handle.
        near_target = (held_z > 0.015) & (xy_error_norm > 0.003) & (xy_error_norm < 0.02)
        needs_correction = near_target | ((held_z > 0.015) & (xy_error_norm < 0.02) & (z_diff > 0.005))

        if not needs_correction.any():
            return expert_delta, intervene_mask

        # Get Jacobian from physics
        try:
            jac = self._right_arm.root_physx_view.get_jacobians()  # [N, num_bodies, 6, num_joints]
            # Last body (EE), xyz position rows, first 5 arm joints
            J_xyz = jac[:, -1, :3, :5]  # [B, 3, 5]

            # Two-phase strategy to avoid friction when cubes touch:
            # Phase 1: XY not aligned → move XY, lift Z slightly above target
            # Phase 2: XY aligned → descend Z to stack
            xy_aligned = xy_error_norm < 0.003  # 3mm = aligned enough to descend
            hover_height = 0.025  # hover 25mm above base cube (cube=18mm + 7mm margin)

            max_step = 0.001  # 1mm per step, per axis (x, y, z independent)
            dx = torch.zeros(B, 3, device=self.device)

            # Phase 1: correct XY, maintain hover height
            phase1 = needs_correction & (~xy_aligned)
            dx[phase1, 0] = -xy_error[phase1, 0]
            dx[phase1, 1] = -xy_error[phase1, 1]
            # If held cube is too low during XY correction, lift up
            target_hover_z = torch.min(c1_z, c2_z) + hover_height
            z_lift = target_hover_z - held_z  # positive = need to go up
            dx[phase1, 2] = torch.clamp(z_lift[phase1], min=0.0, max=max_step)

            # Phase 2: XY aligned, descend to stack (target z_diff = 0.018)
            phase2 = needs_correction & xy_aligned
            target_stack_z = torch.min(c1_z, c2_z) + 0.018  # exact stacking height
            z_descend = target_stack_z - held_z  # negative = need to go down
            dx[phase2, 0] = -xy_error[phase2, 0]  # keep correcting XY during descent
            dx[phase2, 1] = -xy_error[phase2, 1]
            dx[phase2, 2] = torch.clamp(z_descend[phase2], min=-max_step, max=max_step)

            # Clamp each axis independently: x, y, z each get ±1mm budget
            dx = torch.clamp(dx, -max_step, max_step)

            # Pseudoinverse: dq = J_pinv @ dx
            J_pinv = torch.linalg.pinv(J_xyz)  # [B, 5, 3]
            dq = torch.bmm(J_pinv, dx.unsqueeze(-1)).squeeze(-1)  # [B, 5]

            # Safety clamp on joint delta
            dq = torch.clamp(dq, -0.05, 0.05)

            # Pre-scale by 1/alpha so that after alpha*delta the correction is exact
            # (alpha=0.4 for arm joints, so expert_delta = dq/0.4)
            alpha_vec, _ = self.policy._compute_alpha_vec(B, self.device)
            arm_alpha = alpha_vec[0, 6] if act_dim == 12 else alpha_vec[0, 0]  # arm joint alpha
            if arm_alpha > 0:
                dq = dq / arm_alpha

            # Place into right arm joint slots
            # Backbone order: [left_6, right_6], right arm joints = indices 6:11
            if act_dim == 12:
                expert_delta[needs_correction, 6:11] = dq[needs_correction]
            else:
                expert_delta[needs_correction, 0:5] = dq[needs_correction]

            intervene_mask = needs_correction

        except Exception as e:
            pass  # silently ignore expert errors

        return expert_delta, intervene_mask

    def _get_success_flags(self) -> torch.Tensor | None:
        """Per-env success flags from termination manager's _last_episode_dones.

        Exact same logic as play.py _get_success_flags() — vectorized, no try/except.
        Returns bool tensor [num_envs] or None if no success term exists.
        """
        if self._success_term_idx is None:
            return None
        led = self._term_mgr._last_episode_dones  # [num_envs, num_terms]
        success_mask = led[:, self._success_term_idx].bool()
        for di in self._drop_term_indices:
            drop_mask = led[:, di].bool()
            success_mask = success_mask & (~drop_mask)
        return success_mask

    def _check_success_from_info(self, env_info, env_idx: int, obs_before_step: torch.Tensor = None) -> bool:
        """Check success using termination manager signal (matching play.py exactly).

        Primary: _last_episode_dones vectorized check (same as play.py _get_success_flags)
        Fallback: position-based check from pre-step obs
        """
        # Primary: termination manager signal (play.py approach — no silent exception!)
        success_flags = self._get_success_flags()
        if success_flags is not None:
            is_success = bool(success_flags[env_idx].item())
            if is_success:
                return True
            # Diagnostic: print which terms fired (limited count)
            if not hasattr(self, "_l1_diag_count"):
                self._l1_diag_count = 0
            if self._l1_diag_count < 10:
                self._l1_diag_count += 1
                led = self._term_mgr._last_episode_dones
                fired = [self._term_names[j] for j in range(len(self._term_names))
                         if bool(led[env_idx, j].item())]
                print(f"  [SR diag] env {env_idx}: fired={fired}, success_idx={self._success_term_idx}")

        # Fallback: position-based check from pre-step obs (cubes stacked but term didn't fire)
        if obs_before_step is not None:
            s = getattr(self.cfg, "obs_structure", None)
            if s is not None and "cube_1_pos" in s and "cube_2_pos" in s:
                c1 = obs_before_step[env_idx, s["cube_1_pos"][0]:s["cube_1_pos"][1]]
                c2 = obs_before_step[env_idx, s["cube_2_pos"][0]:s["cube_2_pos"][1]]
                z_diff_a = torch.abs((c1[2] - c2[2]) - 0.018)
                xy_dist_a = torch.norm(c1[:2] - c2[:2])
                z_diff_b = torch.abs((c2[2] - c1[2]) - 0.018)
                xy_dist_b = torch.norm(c2[:2] - c1[:2])
                stack_ok = (z_diff_a < 0.003 and xy_dist_a < 0.009) or \
                           (z_diff_b < 0.003 and xy_dist_b < 0.009)
                if stack_ok:
                    return True

        return False

    def _check_position_success(self, obs: torch.Tensor, env_idx: int) -> bool:
        """Position-based success check using env DoneTerm thresholds (no gripper requirement).
        
        Used for truncated (timeout) episodes where cubes may be stacked but grippers
        not released, so the env's success DoneTerm didn't fire.
        """
        s = getattr(self.cfg, "obs_structure", None)
        if s is None:
            return False
        # Cube stack check
        if "cube_1_pos" in s and "cube_2_pos" in s:
            c1 = obs[env_idx, s["cube_1_pos"][0]:s["cube_1_pos"][1]]
            c2 = obs[env_idx, s["cube_2_pos"][0]:s["cube_2_pos"][1]]
            z_diff_a = torch.abs((c1[2] - c2[2]) - 0.018)
            xy_dist_a = torch.norm(c1[:2] - c2[:2])
            z_diff_b = torch.abs((c2[2] - c1[2]) - 0.018)
            xy_dist_b = torch.norm(c2[:2] - c1[:2])
            return bool(
                (z_diff_a < 0.003 and xy_dist_a < 0.009) or
                (z_diff_b < 0.003 and xy_dist_b < 0.009)
            )
        # Table setting check
        if "fork_pos" in s and "knife_pos" in s:
            fp = obs[env_idx, s["fork_pos"][0]:s["fork_pos"][1]]
            kp = obs[env_idx, s["knife_pos"][0]:s["knife_pos"][1]]
            fork_tgt = torch.tensor([0.255, 0.18], device=obs.device, dtype=obs.dtype)
            knife_tgt = torch.tensor([0.2366, -0.1288], device=obs.device, dtype=obs.dtype)
            fork_ok = bool(torch.norm(fp[:2] - fork_tgt) < 0.05 and torch.abs(fp[2] - 0.0076) < 0.01)
            knife_ok = bool(torch.norm(kp[:2] - knife_tgt) < 0.05 and torch.abs(kp[2] - 0.0068) < 0.01)
            return fork_ok and knife_ok
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

    def collect_rollout(self, iteration: int = 0) -> Dict[str, float]:
        """收集 rollout (FIXED: 添加 success rate，除零保护，移除 DEBUG 打印)
        iteration: current iter (1-based); when in critic warmup, use zero_residual (pure backbone).
        """
        self.policy.eval()
        zero_residual = (
            self.critic_warmup_iters > 0
            and iteration > 0
            and iteration <= self.critic_warmup_iters
        )
        self.buffer.reset()

        if self.use_expert_intervention and not self._expert_initialized:
            self._init_expert_intervention()

        raw_obs, _ = self.env.reset()
        obs = self._process_obs(raw_obs)
        subtask_cond = self._extract_subtask_condition(raw_obs, obs)

        # CRITICAL: Reset policy buffers for ALL envs (fresh rollout)
        # Otherwise stale RHC/EMA from prev iter corrupts base_action -> SR drops across iters
        self.policy.reset_envs(torch.arange(self.num_envs, device=self.device))

        # Pre-fill histories with first observation (matches BC training padding)
        self._prefill_histories_from_obs(obs)

        # Save home obs for fixing stale EE after mid-rollout auto-reset.
        # env.reset() calls sim.forward() so this obs is correct.
        # step() auto-reset does NOT call sim.forward(), so EE positions are stale.
        self._home_obs = obs.clone()

        ep_rewards = torch.zeros(self.num_envs, device=self.device)
        ep_lengths = torch.zeros(self.num_envs, device=self.device)
        rollout_successes = []
        delta_norms_all = []
        _alignment_steps = 0  # count steps where RL residual was active
        _total_env_steps = 0  # total env-steps for ratio
        _n_expert_intervened = 0  # count expert intervention env-steps

        # Track per-term reward accumulation
        _reward_term_accum = None
        _reward_term_names = None
        _reward_term_episodes = []

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
                    deterministic=False,
                    zero_residual=zero_residual,
                )

                # --- Alignment phase gating ---
                # RL residual only active during final stacking alignment,
                # same range as expert: held cube above ground, xy_error < 10mm.
                # Outside this phase, use pure backbone (delta=0).
                if not zero_residual and self._is_dual_arm:
                    s = getattr(self.cfg, "obs_structure", None)
                    if s is not None and "cube_1_pos" in s and "cube_2_pos" in s:
                        c1 = obs[:, s["cube_1_pos"][0]:s["cube_1_pos"][1]]  # [B, 3]
                        c2 = obs[:, s["cube_2_pos"][0]:s["cube_2_pos"][1]]  # [B, 3]
                        c1_z, c2_z = c1[:, 2], c2[:, 2]
                        # held cube = the higher one; target = the lower one
                        held_xy = torch.where((c1_z < c2_z).unsqueeze(-1), c2[:, :2], c1[:, :2])
                        held_z = torch.max(c1_z, c2_z)
                        # target_xy = lower cube's XY (same as expert)
                        target_xy = torch.where((c1_z < c2_z).unsqueeze(-1), c1[:, :2], c2[:, :2])
                        xy_error_norm = (held_xy - target_xy).norm(dim=-1)
                        z_diff = torch.abs(c1_z - c2_z)
                        # Match expert range: held above ground, xy < 10mm, z_diff > 5mm
                        in_alignment = (held_z > 0.015) & (xy_error_norm < 0.01) & (z_diff > 0.005)
                        _alignment_steps += in_alignment.sum().item()
                        _total_env_steps += obs.shape[0]
                        not_aligned = ~in_alignment
                        if not_aligned.any():
                            a_base = info["a_base"]
                            action[not_aligned] = a_base[not_aligned]
                            if info.get("delta") is not None:
                                info["delta"][not_aligned] = 0.0

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

                # --- Expert Intervention (Jacobian correction + DAgger) ---
                if self.use_expert_intervention and self._expert_initialized and not zero_residual:
                    B = obs.shape[0]
                    expert_delta, intervene_mask = self._compute_expert_delta(obs)
                    # Stochastic DAgger: only intervene with probability expert_ratio
                    dagger_coin = torch.rand(B, device=self.device) < self.expert_ratio
                    intervene_mask = intervene_mask & dagger_coin

                    if intervene_mask.any():
                        # Expert delta goes through same path: a = a_base + alpha * expert_delta
                        alpha_vec, _ = self.policy._compute_alpha_vec(B, self.device)
                        a_base = info["a_base"]
                        expert_action = a_base + alpha_vec * expert_delta
                        action[intervene_mask] = expert_action[intervene_mask]
                        # Update info so buffer stores expert's delta
                        info["delta"][intervene_mask] = expert_delta[intervene_mask]
                        _n_expert_intervened += intervene_mask.sum().item()

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
                    # Gripper: direct -1/1 mapping (policy outputs continuous, env expects +1/-1)
                    # Stack: -0.25 (train_disentangled_graph_gated); Table: -0.12
                    gripper_idx = base + 5
                    gripper_threshold = -0.25  # stack (table uses -0.12)
                    action_for_sim[:, gripper_idx] = torch.where(
                        action_for_sim[:, gripper_idx] > gripper_threshold,
                        torch.tensor(1.0, device=action.device, dtype=action.dtype),
                        torch.tensor(-1.0, device=action.device, dtype=action.dtype),
                    )

            raw_next_obs, reward, terminated, truncated, env_info = self.env.step(action_for_sim)
            done = terminated | truncated

            next_obs = self._process_obs(raw_next_obs)
            next_subtask_cond = self._extract_subtask_condition(raw_next_obs, next_obs)
            reward = reward.to(self.device).float()
            done = done.to(self.device).float()
            truncated_tensor = truncated.to(self.device).float()

            self.buffer.add(obs, obs_norm, action, reward, done, truncated_tensor, info)

            # Track episode stats
            ep_rewards += reward
            ep_lengths += 1

            # Accumulate per-term rewards (_step_reward stores value/dt, multiply back by dt)
            try:
                unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
                if hasattr(unwrapped, 'reward_manager'):
                    rm = unwrapped.reward_manager
                    if _reward_term_names is None:
                        _reward_term_names = list(rm._term_names)
                        _reward_term_accum = torch.zeros(self.num_envs, len(_reward_term_names), device=self.device)
                        _reward_dt = unwrapped.step_dt  # sim.dt * decimation = 0.02
                    _reward_term_accum += rm._step_reward * _reward_dt
            except Exception as e:
                if not hasattr(self, '_reward_accum_err_logged'):
                    print(f"  [WARN] reward_term_accum error: {e}")
                    self._reward_accum_err_logged = True

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
                n_terminated = 0
                n_truncated = 0
                n_success = 0
                # Vectorized success check (matching play.py _get_success_flags)
                success_flags = self._get_success_flags()
                for i in done_envs.tolist():
                    self.episode_rewards.append(ep_rewards[i].item())
                    self.episode_lengths.append(ep_lengths[i].item())
                    is_truncated = bool(truncated[i].item() if hasattr(truncated[i], "item") else truncated[i])
                    is_terminated = bool(terminated[i].item() if hasattr(terminated[i], "item") else terminated[i])

                    # Matching play.py: truncated=failure, terminated=check signal
                    if is_truncated and not is_terminated:
                        is_success = False
                        n_truncated += 1
                    elif success_flags is not None:
                        is_success = bool(success_flags[i].item())
                        if is_terminated:
                            n_terminated += 1
                        else:
                            n_truncated += 1
                    else:
                        # No success term in env — fallback to position check
                        is_success = self._check_success_from_info(env_info, i, obs_before_step=obs)
                        n_terminated += 1

                    # Debug: log episode outcomes with full diagnostics
                    if not hasattr(self, "_ep_debug_count"):
                        self._ep_debug_count = 0
                    _should_log = self._ep_debug_count < 10
                    if _should_log:
                        self._ep_debug_count += 1
                        led = self._term_mgr._last_episode_dones
                        fired = [self._term_names[j] for j in range(len(self._term_names))
                                 if bool(led[i, j].item())]
                        _diag = ""
                        try:
                            s = getattr(self.cfg, "obs_structure", None)
                            if s is not None and "cube_1_pos" in s and "cube_2_pos" in s:
                                c1 = obs[i, s["cube_1_pos"][0]:s["cube_1_pos"][1]]
                                c2 = obs[i, s["cube_2_pos"][0]:s["cube_2_pos"][1]]
                                z_diff_ab = (c1[2] - c2[2]).item()
                                z_diff_ba = (c2[2] - c1[2]).item()
                                xy_dist = torch.norm(c1[:2] - c2[:2]).item()
                                z_ok = abs(z_diff_ab - 0.018) < 0.003 or abs(z_diff_ba - 0.018) < 0.003
                                xy_ok = xy_dist < 0.009
                                _diag = (f" | z={z_diff_ab*1000:.1f}mm xy={xy_dist*1000:.1f}mm "
                                         f"[z={'OK' if z_ok else 'FAIL'} xy={'OK' if xy_ok else 'FAIL'}]")
                            if s is not None and "right_joint_pos" in s:
                                rj = obs[i, s["right_joint_pos"][0]:s["right_joint_pos"][1]]
                                lj = obs[i, s["left_joint_pos"][0]:s["left_joint_pos"][1]]
                                r_grip_rel = rj[-1].item()
                                l_grip_rel = lj[-1].item()
                                r_grip_abs = r_grip_rel + 0.4
                                l_grip_abs = l_grip_rel + 0.4
                                grip_ok = r_grip_abs > 0.1 and l_grip_abs > 0.1
                                _diag += f" Rg={r_grip_abs:.3f} Lg={l_grip_abs:.3f} [grip={'OK' if grip_ok else 'FAIL'}]"
                        except Exception as e:
                            _diag = f" | diag_err={e}"
                        # Per-episode reward breakdown for this env
                        _rew_str = ""
                        if _reward_term_accum is not None and _reward_term_names is not None:
                            terms = _reward_term_accum[i]
                            _rew_parts = [f"{n}={v.item():.2f}" for n, v in zip(_reward_term_names, terms) if abs(v.item()) > 0.001]
                            _rew_str = f" | rew_terms=[{', '.join(_rew_parts)}]"
                    if is_success:
                        n_success += 1
                    rollout_successes.append(float(is_success))
                    self.episode_successes.append(float(is_success))

                # Record per-term episode rewards
                if _reward_term_accum is not None:
                    for i in done_envs.tolist():
                        _reward_term_episodes.append(_reward_term_accum[i].clone())
                    _reward_term_accum[done_envs] = 0

                # FIX stale EE positions after auto-reset.
                # step() auto-reset calls _reset_idx() but NOT sim.forward(),
                # so EE positions in next_obs are stale (from previous episode's final state).
                # This causes backbone to output wrong actions for reset envs.
                # Fix: overwrite stale EE positions with saved home positions.
                if hasattr(self, '_home_obs') and self._home_obs is not None:
                    obs_struct = getattr(self.cfg, "obs_structure", None)
                    if obs_struct is not None:
                        _ee_keys = ["left_ee_position", "left_ee_orientation",
                                    "right_ee_position", "right_ee_orientation"]
                        for _hk in _ee_keys:
                            if _hk in obs_struct:
                                _s, _e = obs_struct[_hk]
                                for idx in done_envs.tolist():
                                    next_obs[idx, _s:_e] = self._home_obs[idx, _s:_e]

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
                deterministic=True,
                zero_residual=zero_residual,
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
            sr_this = np.mean(rollout_successes)
            stats["success_rate"] = sr_this
            stats["num_episodes"] = len(rollout_successes)
            self.iter_success_rates.append(sr_this)
        else:
            stats["success_rate"] = 0.0
            stats["num_episodes"] = 0

        # Use per-iteration SR (unbiased) instead of per-episode list
        # (per-episode list is biased: short=success episodes complete first,
        #  so [-100:] always contains only timeout=failure episodes)
        if len(self.iter_success_rates) > 0:
            stats["success_rate_100"] = np.mean(self.iter_success_rates[-3:])  # ~1500 eps
            n_window_iters = max(1, self.best_sr_window // max(1, len(rollout_successes)))
            stats["success_rate_window"] = np.mean(self.iter_success_rates[-n_window_iters:]) if n_window_iters <= len(self.iter_success_rates) else np.mean(self.iter_success_rates)
        else:
            stats["success_rate_100"] = 0.0
            stats["success_rate_window"] = 0.0

        if len(delta_norms_all) > 0:
            stats["delta_norm_mean"] = np.mean(delta_norms_all)
            stats["delta_norm_max"] = np.max(delta_norms_all)
        else:
            stats["delta_norm_mean"] = 0.0
            stats["delta_norm_max"] = 0.0

        if self.use_expert_intervention:
            stats["expert_ratio"] = self.expert_ratio

        return stats

    def update(self, iteration: int = 0) -> Dict[str, float]:
        """更新 policy (FIXED: 添加 Explained Variance)
        iteration: current iter (1-based); used for critic warmup.
        """
        self.policy.train()

        total_loss = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss_delta_reg = 0
        total_loss_critic_bar = 0
        total_entropy = 0
        total_alpha = 0.0
        total_c_delta_reg = 0.0
        total_c_ent = 0.0
        total_beta = 0.0
        total_eff_ratio = 0.0
        num_updates = 0
        all_returns = []
        all_values = []

        critic_warmup = (
            self.critic_warmup_iters > 0
            and iteration > 0
            and iteration <= self.critic_warmup_iters
        )

        for batch in self.buffer.get_batches(self.mini_batch_size, self.num_epochs):
            losses = self.policy.compute_loss(batch)

            self.optimizer.zero_grad()
            if critic_warmup:
                # Only train critic: don't let actor/z_adapter move before advantage estimates are stable
                loss_to_backward = self.cfg.cV * 0.5 * losses["loss_critic_bar_raw"]
                loss_to_backward.backward()
            else:
                losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
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
            c_delta_val = losses.get("c_delta_reg")
            if c_delta_val is not None:
                total_c_delta_reg += c_delta_val.item() if hasattr(c_delta_val, "item") else float(c_delta_val)
            c_ent_val = losses.get("c_ent")
            if c_ent_val is not None:
                total_c_ent += c_ent_val.item() if hasattr(c_ent_val, "item") else float(c_ent_val)
            beta_val = losses.get("beta")
            if beta_val is not None:
                total_beta += beta_val.item() if hasattr(beta_val, "item") else float(beta_val)
            eff_val = losses.get("eff_ratio")
            if eff_val is not None:
                total_eff_ratio += eff_val.item() if hasattr(eff_val, "item") else float(eff_val)
            num_updates += 1
            all_returns.append(batch["returns"])
            all_values.append(batch["values"])

        n = num_updates
        all_returns = torch.cat(all_returns)
        all_values = torch.cat(all_values)
        var_returns = torch.var(all_returns)
        var_residual = torch.var(all_returns - all_values)
        explained_variance = (1.0 - var_residual / (var_returns + 1e-8)).item()

        # During warmup, rollout used zero_residual (alpha=0); report that for clarity
        alpha_reported = 0.0 if critic_warmup else (total_alpha / num_updates if num_updates else 0.0)
        return {
            "critic_warmup": critic_warmup,
            "loss_total": total_loss / n,
            "loss_actor": total_actor_loss / n,
            "loss_critic": total_critic_loss / n,
            "loss_critic_bar": total_loss_critic_bar / n,
            "loss_delta_reg": total_loss_delta_reg / n,
            "entropy": total_entropy / n,
            "alpha": alpha_reported,
            "explained_variance": explained_variance,
            "c_delta_reg": total_c_delta_reg / n if n > 0 else 0.0,
            "c_ent": total_c_ent / n if n > 0 else 0.0,
            "beta": total_beta / n if n > 0 else 0.0,
            "eff_ratio": total_eff_ratio / n if n > 0 else 0.0,
        }

    def train(self, max_iterations: int, save_interval: int = 50, start_iteration: int = 1):
        """主训练循环。start_iteration: 从第几 iter 开始（resume 时 >1）"""
        self._start_iteration = start_iteration
        if start_iteration > 1:
            print(f"\n[Trainer] Resuming from iteration {start_iteration} to {max_iterations}...")
        else:
            print(f"\n[Trainer] Starting training for {max_iterations} iterations...")

        # Constant LR (no decay for short 50-iter runs)
        def lr_lambda(epoch):
            return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self._train_start = time.time()
        self._max_iterations = max_iterations

        start_iter = getattr(self, "_start_iteration", 1)
        for iteration in range(start_iter, max_iterations + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]
            # Collect
            rollout_stats = self.collect_rollout(iteration=iteration)

            # Update
            update_stats = self.update(iteration=iteration)
            update_stats["lr"] = current_lr

            # DAgger: decay expert intervention ratio (only after warmup ends)
            critic_warmup = (
                self.critic_warmup_iters > 0
                and iteration <= self.critic_warmup_iters
            )
            # Expert decay disabled – keep expert_ratio=1.0 for diagnostics
            # if self.use_expert_intervention and iteration > 0 and not critic_warmup:
            #     self.expert_ratio *= self.expert_decay
            #     self.expert_ratio = max(self.expert_ratio, 0.0)

            # Log
            self._log(iteration, rollout_stats, update_stats)

            critic_warmup = (
                self.critic_warmup_iters > 0
                and iteration <= self.critic_warmup_iters
            )

            # Save best by success rate (use rolling N-ep SR; 200ep: ±7% CI)
            # Skip during critic warmup: policy unchanged, SR variance is noise
            sr_window = rollout_stats.get("success_rate_window", 0.0)
            sr_current = rollout_stats.get("success_rate", 0.0)
            n_eps = len(self.episode_successes)
            window_valid = n_eps >= self.best_sr_window
            consistent = (not self.best_sr_require_consistent) or (sr_current >= sr_window - 0.15)
            if not critic_warmup and window_valid and consistent and sr_window > self.best_sr:
                self.best_sr = sr_window
                self._save_best(iteration)

            # Save when current iteration SR hits new high (captures peak performance)
            # During warmup: save iter 1 baseline, don't overwrite with variance
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
        step_iter = iteration  # By-iteration step for smooth curve (warmup iters 1-5 visible)

        _scalar("main/success_rate", rollout_stats.get("success_rate", 0), step)
        _scalar("main/success_rate_by_iter", rollout_stats.get("success_rate", 0), step_iter)
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
        _scalar("main/c_delta_reg", update_stats.get("c_delta_reg", 0), step)
        _scalar("main/c_ent", update_stats.get("c_ent", 0), step)
        _scalar("main/beta", update_stats.get("beta", 0), step)
        _scalar("main/eff_ratio", update_stats.get("eff_ratio", 0), step)
        _scalar("main/expert_ratio", rollout_stats.get("expert_ratio", 0), step)

        sr = rollout_stats.get("success_rate", 0) * 100
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

        # Progress / ETA
        elapsed = time.time() - self._train_start if hasattr(self, "_train_start") else 0
        max_iter = self._max_iterations if hasattr(self, "_max_iterations") else iteration
        warmup_tag = " [critic_warmup]" if update_stats.get("critic_warmup", False) else ""
        if elapsed > 0 and iteration > 0:
            eta_s = elapsed / iteration * (max_iter - iteration)
            eta_m, eta_sec = divmod(int(eta_s), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            elapsed_m = int(elapsed) // 60
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h else f"{eta_m}m{eta_sec:02d}s"
            progress_str = f"[{iteration}/{max_iter} | {elapsed_m}m | ETA {eta_str}]{warmup_tag}"
        else:
            progress_str = f"[{iteration}/{max_iter}]{warmup_tag}"

        c_dreg = update_stats.get("c_delta_reg", 0)
        c_ent_v = update_stats.get("c_ent", 0)
        beta_v = update_stats.get("beta", 0)
        eff_r = update_stats.get("eff_ratio", 0)
        print(
            f"{progress_str} "
            f"SR={sr:5.1f}% [{num_eps}ep] | "
            f"Rew={reward:6.1f} | "
            f"EV={ev:5.2f} | "
            f"Δ={delta:.3f} | "
            f"α={alpha:.2f} β={beta_v:.3f} | "
            f"cΔ={c_dreg:.1f} cH={c_ent_v:.4f} eff={eff_r:.2f} | "
            f"L={loss:.3f}"
            + (f" | Exp={rollout_stats.get('expert_ratio', 0):.2f}" if self.use_expert_intervention else "")
        )

    def _save_best(self, iteration: int):
        """按 success_rate_window（最近 N episode 平均）保存当前最佳模型"""
        path = os.path.join(self.log_dir, "best_model.pt")
        self.policy.save(path, iteration=iteration)
        print(f"[Trainer] Best model saved (SR_{self.best_sr_window}ep={self.best_sr*100:.1f}%): {path}")

    def _save(self, iteration: int, final: bool = False):
        """保存 checkpoint (includes iteration for resume)"""
        suffix = "final" if final else f"iter_{iteration}"
        path = os.path.join(self.log_dir, f"policy_{suffix}.pt")
        self.policy.save(path, iteration=iteration)
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
            p.requires_grad = False
        print(f"[Main] node_to_z FROZEN (z_adapter provides adaptation capacity)")
    else:
        print(f"[Main] ⚠️  WARNING: node_to_z not found in backbone!")
    print(f"[Main] Graph-Unet loaded and fully frozen (z_adapter is the only adaptation layer)")

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
        max_delta_norm=getattr(args, "max_delta_norm", 0.0),
        use_expectile_value=True,
        expectile_tau=args.expectile_tau,
        use_expert_intervention=getattr(args, "use_expert_intervention", False),
        expert_intervention_ratio=getattr(args, "expert_intervention_ratio", 1.0),
        expert_intervention_decay=getattr(args, "expert_intervention_decay", 0.95),
        use_adaptive_entropy=not getattr(args, "no_adaptive_entropy", False),
        c_ent_bad=args.c_ent_bad,
        c_ent_good=args.c_ent_good,
        use_counterfactual_q=getattr(args, "use_counterfactual_q", False),
        counterfactual_log_tau=getattr(args, "counterfactual_log_tau", 0.5),
        # SAC-style adaptive parameters
        use_adaptive_delta_reg=getattr(args, "use_adaptive_delta_reg", True),
        target_delta_norm=getattr(args, "target_delta_norm", 0.15),
        c_delta_reg_init=getattr(args, "c_delta_reg_init", 5.0),
        use_auto_entropy=getattr(args, "use_auto_entropy", True),
        target_entropy=getattr(args, "target_entropy", -6.0),
        c_ent_init=getattr(args, "c_ent_init", 0.01),
        use_adaptive_beta=getattr(args, "use_adaptive_beta", True),
        target_eff_ratio=getattr(args, "target_eff_ratio", 0.4),
        beta_init=getattr(args, "beta_init", 0.3),
    )
    print(f"[Main] residual_action_mask: {residual_action_mask.tolist()} (all 1s, no mask)")
    print(f"[Main] Residual RL obs_structure: {policy_cfg.obs_structure}")

    if getattr(backbone_policy.cfg, "obs_structure", None) != policy_cfg.obs_structure:
        print("[Main] ⚠️  WARNING: obs_structure MISMATCH between backbone and Residual RL!")
    else:
        print("[Main] ✅ obs_structure consistent")
    # Entropy mode
    if getattr(policy_cfg, "use_auto_entropy", False):
        ent_mode = f"Auto Entropy (SAC-style, c_init={getattr(policy_cfg,'c_ent_init',0.01)}, target_H={getattr(policy_cfg,'target_entropy',-6.0)})"
    elif getattr(policy_cfg, "use_adaptive_entropy", False):
        ent_mode = f"Adaptive Entropy (bad={getattr(policy_cfg,'c_ent_bad',0.1)}, good={getattr(policy_cfg,'c_ent_good',0.005)})"
    else:
        ent_mode = f"Fixed cEnt={policy_cfg.cEnt}"
    print(f"[Main] Entropy: {ent_mode}")
    # Delta reg mode
    if getattr(policy_cfg, "use_adaptive_delta_reg", False):
        dreg_mode = f"Adaptive delta_reg (SAC-style, c_init={getattr(policy_cfg,'c_delta_reg_init',5.0)}, target_δ={getattr(policy_cfg,'target_delta_norm',0.15)})"
    else:
        dreg_mode = f"Fixed c_delta_reg={policy_cfg.c_delta_reg}"
    print(f"[Main] Delta Reg: {dreg_mode}")
    # Beta mode
    if getattr(policy_cfg, "use_adaptive_beta", False):
        beta_mode = f"Adaptive beta (SAC-style, init={getattr(policy_cfg,'beta_init',0.3)}, target_eff={getattr(policy_cfg,'target_eff_ratio',0.4)})"
    else:
        beta_mode = f"Fixed beta={policy_cfg.beta}"
    print(f"[Main] AWR Beta: {beta_mode}")
    # Expert intervention mode
    if getattr(policy_cfg, "use_expert_intervention", False):
        print(f"[Main] Expert Intervention: ENABLED (ratio={policy_cfg.expert_intervention_ratio}, decay={policy_cfg.expert_intervention_decay})")
    else:
        print(f"[Main] Expert Intervention: disabled")
    if policy_cfg.use_counterfactual_q:
        print(f"[Main] Q(s,a) counterfactual baseline ENABLED (log_tau={policy_cfg.counterfactual_log_tau})")
    else:
        print(f"[Main] V(s) baseline (standard)")

    # Create or load policy
    start_iteration = 1
    if getattr(args, "resume", None) and os.path.isfile(args.resume):
        print(f"\n[Main] Resuming from RL checkpoint: {args.resume}")
        _resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        iter_from_ckpt = _resume_ckpt.get("iteration", None)
        if iter_from_ckpt is not None:
            start_iteration = int(iter_from_ckpt) + 1
        else:
            # Fallback: parse from filename policy_iter_100.pt -> 100
            basename = os.path.basename(args.resume)
            m = re.search(r"policy_iter_(\d+)\.pt", basename)
            if m:
                start_iteration = int(m.group(1)) + 1
            # policy_final.pt / best_model.pt: no iter in file, cannot infer; start from 1
        print(f"[Main] Resuming from iteration {start_iteration} (checkpoint had iter={start_iteration-1})")
        policy = GraphUnetResidualRLPolicy.load(args.resume, backbone=backbone, device=device)
        policy.num_diffusion_steps = 15
        policy_cfg = policy.cfg  # use cfg from checkpoint when resuming
    else:
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

    # Run folder name: --run_name override (ablation) or auto from hyperparams
    if getattr(args, "run_name", None):
        run_name = args.run_name.replace(" ", "").replace("/", "-")
    else:
        use_adapt_ent = not getattr(args, "no_adaptive_entropy", False)
        run_name = (
            f"env{args.num_envs}_epochs{args.num_epochs}_beta{args.beta}_"
            f"cEnt{args.c_ent}_cDelta{args.c_delta_reg}_adaptEnt{use_adapt_ent}"
        )
        run_name = run_name.replace(" ", "").replace("/", "-")
    # When resuming, save to same folder as checkpoint
    if getattr(args, "resume", None) and os.path.isfile(args.resume):
        log_dir = os.path.dirname(os.path.abspath(args.resume))
        print(f"[Main] Resume mode: saving to same folder as checkpoint: {log_dir}")
    else:
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
        critic_warmup_iters=getattr(args, "critic_warmup_iters", 0),
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
            print("    ✅ node_to_z is FROZEN (backbone fully frozen, z_adapter handles adaptation)")
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
        start_iteration=start_iteration,
    )

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

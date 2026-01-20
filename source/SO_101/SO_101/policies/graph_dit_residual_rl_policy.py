# Copyright (c) 2024-2026, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
GraphDiT + Residual RL Policy (OUR METHOD) - High-Frequency Z Version

Key improvements:
1) base_action: Low-frequency (obtain trajectory from DiT every exec_horizon steps)
2) z_layers: High-frequency (call extract_z_fast every step, only run Graph-Attention)
3) z_adapter: Trainable (transform frozen z to RL-friendly z)
4) GateNet + DeepValueCritic: Unchanged
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


TensorOrArray = Union[torch.Tensor, np.ndarray]


# =========================================================
# Utilities (unchanged)
# =========================================================
def to_tensor(x: Any, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return torch.tensor(x, device=device)


def flatten_obs(obs: Any, device: str) -> torch.Tensor:
    """IsaacLab obs often is dict with key 'policy'. We prioritize it."""
    if isinstance(obs, dict):
        if "policy" in obs:
            t = to_tensor(obs["policy"], device=device).float()
        else:
            parts = []
            for k in sorted(obs.keys()):
                v = to_tensor(obs[k], device=device).float()
                if v.ndim == 1:
                    v = v.unsqueeze(0)
                parts.append(v.flatten(start_dim=1))
            t = torch.cat(parts, dim=1)
    else:
        t = to_tensor(obs, device=device).float()

    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.ndim > 2:
        t = t.view(t.shape[0], -1)
    return t


def orthogonal_init(m: nn.Module, gain: float = 1.0) -> None:
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            nn.init.orthogonal_(mod.weight, gain=gain)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)


def mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, act: str = "silu") -> nn.Sequential:
    acts = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }
    Act = acts.get(act.lower(), nn.SiLU)
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), Act()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


# =========================================================
# GAE (unchanged)
# =========================================================
@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t].float()
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nonterminal * next_value - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae

    ret = adv + values
    return adv, ret


# =========================================================
# Config (added obs_structure related configuration)
# =========================================================
@dataclass
class GraphDiTResidualRLCfg:
    # dims
    obs_dim: int
    action_dim: int
    z_dim: int = 128
    num_layers: int = 6  # K

    # observation structure (used to extract ee_node, obj_node from obs)
    # If None, use default structure
    obs_structure: Optional[Dict[str, Tuple[int, int]]] = None
    robot_state_dim: int = 6  # joint_pos dimension

    # Actor/Critic obs selection: which parts of obs to use
    # Default: "robot_state" - only joint position (recommended, since z_bar already contains scene info)
    # Options:
    #   - "robot_state": use only joint position (obs[:, :robot_state_dim], default)
    #   - "full": use all obs (may be redundant with z_bar)
    #   - Tuple[int, int]: use obs[:, start:end] slice
    actor_obs_mode: str = "robot_state"  # Only joint position: obs[:, :6]
    critic_obs_mode: str = "robot_state"  # Only joint position: obs[:, :6]

    # actor input: [obs_selected, a_base, z_bar]
    actor_hidden: Tuple[int, ...] = (256, 256)
    critic_hidden: Tuple[int, ...] = (256, 256)
    gate_hidden: Tuple[int, ...] = (256,)
    
    # NEW: z_adapter hidden dims
    z_adapter_hidden: Tuple[int, ...] = (256,)

    # residual distribution
    log_std_init: float = -0.7
    log_std_min: float = -10.0
    log_std_max: float = 1.0

    # residual scaling
    alpha_init: float = 0.10

    # EMA smoothing for base_action (joints only, gripper excluded)
    ema_alpha: float = 0.5  # EMA weight: higher = more responsive, lower = smoother
    joint_dim: int = 5  # Number of joint dimensions (gripper is excluded from EMA)

    # advantage weighted regression
    beta: float = 1.0
    weight_clip_max: float = 20.0

    # losses
    cV: float = 1.0
    cEnt: float = 0.0
    cGate: float = 0.02

    # gae
    gamma: float = 0.99
    lam: float = 0.95

    # init
    orthogonal_init: bool = True
    activation: str = "silu"

    # device
    device: str = "cuda"

    # residual mask
    residual_action_mask: Optional[torch.Tensor] = None


# =========================================================
# Modules: Gate / Actor / Critic (unchanged)
# =========================================================
class LayerWiseGateNet(nn.Module):
    """
    Input: z_layers [B, K, z_dim]
    Output: w [B, K] softmax weights
    """
    def __init__(self, K: int, z_dim: int, hidden: Tuple[int, ...], act: str = "silu"):
        super().__init__()
        self.K = K
        self.z_dim = z_dim
        self.net = mlp(K * z_dim, hidden, K, act=act)

    def forward(self, z_layers: torch.Tensor) -> torch.Tensor:
        B, K, D = z_layers.shape
        assert K == self.K and D == self.z_dim
        x = z_layers.reshape(B, K * D)
        logits = self.net(x)
        w = torch.softmax(logits, dim=-1)
        return w


class ResidualGaussianActor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        hidden: Tuple[int, ...],
        act: str,
        log_std_init: float,
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        self.body = mlp(in_dim, hidden, hidden[-1], act=act)
        self.mu = nn.Linear(hidden[-1], action_dim)

        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def dist(self, x: torch.Tensor) -> Normal:
        h = self.body(x)
        raw_mu = self.mu(h)
        # CRITICAL: Apply tanh to bound mu (delta mean) to [-1, 1]
        # This ensures delta is a small, bounded adjustment
        # Without this, delta can be unbounded (e.g., 4.5 ~ 5.5), causing issues
        mu = torch.tanh(raw_mu)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mu)
        return Normal(mu, std)


class DeepValueCritic(nn.Module):
    def __init__(self, K: int, z_dim: int, obs_dim: int, hidden: Tuple[int, ...], act: str):
        super().__init__()
        self.K = K
        self.z_dim = z_dim

        # Layer heads: only use z^k (keep deep supervision simple)
        self.trunk = mlp(z_dim, hidden, hidden[-1], act=act)
        self.layer_heads = nn.ModuleList([nn.Linear(hidden[-1], 1) for _ in range(K)])

        # Bar head: use z_bar + obs (more accurate baseline)
        self.bar_trunk = mlp(z_dim + obs_dim, hidden, hidden[-1], act=act)
        self.bar_head = nn.Linear(hidden[-1], 1)

    def forward_layers(self, z_layers: torch.Tensor) -> torch.Tensor:
        B, K, D = z_layers.shape
        assert K == self.K and D == self.z_dim

        Vs = []
        for k in range(K):
            zk = z_layers[:, k, :]
            hk = self.trunk(zk)
            vk = self.layer_heads[k](hk)
            Vs.append(vk)
        V_layers = torch.cat(Vs, dim=1)
        return V_layers

    def forward_bar(self, z_bar: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_bar, obs], dim=-1)
        h = self.bar_trunk(x)
        return self.bar_head(h).squeeze(-1)


# =========================================================
# NEW: Z Adapter (transform frozen z to RL-friendly z)
# =========================================================
class ZAdapter(nn.Module):
    """
    Trainable adapter: frozen DiT z → RL-friendly z
    
    DiT's z is trained for denoising, which may not be optimal for RL representation.
    Adapter allows RL gradients to fine-tune how z is used.
    """
    def __init__(self, z_dim: int, hidden: Tuple[int, ...], act: str = "silu"):
        super().__init__()
        
        if len(hidden) == 0:
            # Direct linear transformation
            self.net = nn.Linear(z_dim, z_dim)
        else:
            layers: List[nn.Module] = []
            acts = {"relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU, 
                    "tanh": nn.Tanh, "silu": nn.SiLU}
            Act = acts.get(act.lower(), nn.SiLU)
            
            last = z_dim
            for h in hidden:
                layers += [nn.Linear(last, h), nn.LayerNorm(h), Act()]
                last = h
            layers.append(nn.Linear(last, z_dim))
            
            self.net = nn.Sequential(*layers)
        
        # Initialize to approximate identity mapping
        self._init_near_identity()
    
    def _init_near_identity(self):
        """Initialize to approximate identity mapping, adapter barely changes z at early training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)  # Small gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Residual connection: z_out = z + adapter(z)
        # With this initialization, adapter ≈ 0, so z_out ≈ z
        return z + self.net(z)


# =========================================================
# Backbone Adapter (modified: added extract_z_fast)
# =========================================================
class GraphDiTBackboneAdapter:
    """
    Adapter interface for GraphDiT backbone.
    
    Methods:
    - predict_base_trajectory(): Low-frequency, returns full trajectory
    - extract_z_fast(): High-frequency, only runs Graph-Attention
    """
    def __init__(self, graph_dit_policy: nn.Module):
        self.backbone = graph_dit_policy

    @torch.no_grad()
    def predict_base_trajectory(self, **kwargs) -> torch.Tensor:
        """Low-frequency: predict full trajectory [B, pred_horizon, act_dim]"""
        return self.backbone.predict(**kwargs)

    @torch.no_grad()
    def extract_z_fast(
        self, 
        ee_node: torch.Tensor, 
        obj_node: torch.Tensor
    ) -> torch.Tensor:
        """
        High-frequency: only run Graph-Attention, return z_layers [B, K, z_dim]
        
        Args:
            ee_node: [B, 7] - EE position(3) + orientation(4)
            obj_node: [B, 7] - Object position(3) + orientation(4)
        
        Returns:
            z_layers: [B, K, z_dim]
        """
        if not hasattr(self.backbone, "extract_z_fast"):
            raise NotImplementedError(
                "GraphDiTPolicy must implement extract_z_fast(ee_node, obj_node) -> [B,K,z_dim]"
            )
        return self.backbone.extract_z_fast(ee_node, obj_node)
    
    @property
    def node_stats(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get normalization stats (if available)"""
        stats = {}
        if hasattr(self.backbone, 'cfg'):
            # Try to load stats from checkpoint
            pass
        return stats


# =========================================================
# OUR POLICY: GraphDiT + Residual RL Head (high-frequency z version)
# =========================================================
class GraphDiTResidualRLPolicy(nn.Module):
    """
    Online inference:
        1. base_action_t = pop from RHC buffer (low-frequency, updated every exec_horizon steps)
        2. z_layers_t = extract_z_fast(ee_node, obj_node) (high-frequency, updated every step!)
        3. z_adapted = ZAdapter(z_layers) (trainable)
        4. z_bar_t = GateNet(z_adapted)
        5. delta_a_t ~ Actor(obs, base_action, z_bar)
        6. a_t = base + alpha * delta
    
    Key improvements:
    - z updated every step (high-frequency), reflects current EE-Object relationship
    - base_action updated at low-frequency, leverages DiT's trajectory planning capability
    - ZAdapter allows frozen z to be fine-tuned by RL gradients
    """

    def __init__(
        self,
        cfg: GraphDiTResidualRLCfg,
        backbone: GraphDiTBackboneAdapter,
        pred_horizon: int = 16,
        exec_horizon: int = 8,
        num_diffusion_steps: int = 2,
    ):
        super().__init__()
        self.cfg = cfg
        self.device_str = cfg.device

        self.backbone = backbone
        self.pred_horizon = pred_horizon
        self.exec_horizon = exec_horizon
        self.num_diffusion_steps = num_diffusion_steps

        # ============================================================
        # NEW: Z Adapter (trainable)
        # ============================================================
        self.z_adapter = ZAdapter(
            z_dim=cfg.z_dim,
            hidden=cfg.z_adapter_hidden,
            act=cfg.activation,
        )

        # Gate (softmax aggregation of adapted z)
        self.gate = LayerWiseGateNet(
            K=cfg.num_layers,
            z_dim=cfg.z_dim,
            hidden=cfg.gate_hidden,
            act=cfg.activation,
        )

        # Compute actual obs dim for Actor/Critic (based on obs_mode)
        def get_obs_dim_for_mode(mode: str) -> int:
            if mode == "full":
                return cfg.obs_dim
            elif mode == "robot_state" or mode == "robot_state_only":
                return cfg.robot_state_dim
            elif isinstance(mode, tuple) and len(mode) == 2:
                start, end = mode
                return end - start
            else:
                raise ValueError(f"Unknown obs_mode: {mode}")
        
        actor_obs_dim = get_obs_dim_for_mode(cfg.actor_obs_mode)
        critic_obs_dim = get_obs_dim_for_mode(cfg.critic_obs_mode)

        # Actor input dim: obs_selected + base_action + z_bar
        actor_in_dim = actor_obs_dim + cfg.action_dim + cfg.z_dim
        self.actor = ResidualGaussianActor(
            in_dim=actor_in_dim,
            action_dim=cfg.action_dim,
            hidden=cfg.actor_hidden,
            act=cfg.activation,
            log_std_init=cfg.log_std_init,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        )

        # Critic (deep supervision on adapted z)
        self.critic = DeepValueCritic(
            K=cfg.num_layers,
            z_dim=cfg.z_dim,
            obs_dim=critic_obs_dim,  # Use selected obs dim
            hidden=cfg.critic_hidden,
            act=cfg.activation,
        )

        # Residual scaling
        self.alpha = nn.Parameter(torch.tensor(cfg.alpha_init, dtype=torch.float32))

        # Residual mask
        if cfg.residual_action_mask is not None:
            self.register_buffer(
                "residual_action_mask",
                cfg.residual_action_mask.float().view(1, -1),
            )
        else:
            self.residual_action_mask = None

        # ============================================================
        # RHC buffers (for base_action, low-frequency)
        # ============================================================
        self._base_action_buffers: Optional[List[List[torch.Tensor]]] = None
        self._num_envs: Optional[int] = None

        # ============================================================
        # EMA smoothing for base_action (joints only)
        # ============================================================
        self.ema_alpha = cfg.ema_alpha
        self.joint_dim = cfg.joint_dim
        self._ema_smoothed_joints: Optional[torch.Tensor] = None  # [num_envs, joint_dim]

        # ============================================================
        # Normalization stats (obtained from backbone or manually set)
        # ============================================================
        self.register_buffer('norm_ee_node_mean', None, persistent=False)
        self.register_buffer('norm_ee_node_std', None, persistent=False)
        self.register_buffer('norm_object_node_mean', None, persistent=False)
        self.register_buffer('norm_object_node_std', None, persistent=False)
        
        # Action normalization stats (for denormalizing base_action and delta)
        self.register_buffer('action_mean', None, persistent=False)
        self.register_buffer('action_std', None, persistent=False)

        # Init
        if cfg.orthogonal_init:
            orthogonal_init(self.gate, gain=np.sqrt(2))
            orthogonal_init(self.actor, gain=np.sqrt(2))
            orthogonal_init(self.critic, gain=np.sqrt(2))
            # CRITICAL: Zero initialize Actor's mu layer (last layer)
            # This ensures initial delta ≈ 0, so action ≈ a_base at start of training
            # Without this, random initialization can cause large deltas that break base policy
            nn.init.zeros_(self.actor.mu.weight)
            nn.init.zeros_(self.actor.mu.bias)

    # -----------------------------
    # Normalization stats setter
    # -----------------------------
    def set_normalization_stats(
        self,
        ee_node_mean: Optional[torch.Tensor] = None,
        ee_node_std: Optional[torch.Tensor] = None,
        object_node_mean: Optional[torch.Tensor] = None,
        object_node_std: Optional[torch.Tensor] = None,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
    ):
        """Set normalization stats (call after loading from checkpoint)"""
        def _to_tensor(x):
            """Convert numpy array or tensor to torch.Tensor"""
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            if isinstance(x, torch.Tensor):
                return x.float()
            return torch.tensor(x, dtype=torch.float32)

        if ee_node_mean is not None:
            self.norm_ee_node_mean = _to_tensor(ee_node_mean)
            self.norm_ee_node_std = _to_tensor(ee_node_std)
        if object_node_mean is not None:
            self.norm_object_node_mean = _to_tensor(object_node_mean)
            self.norm_object_node_std = _to_tensor(object_node_std)
        if action_mean is not None:
            self.action_mean = _to_tensor(action_mean)
            self.action_std = _to_tensor(action_std)

    # -----------------------------
    # Buffer management
    # -----------------------------
    def init_env_buffers(self, num_envs: int) -> None:
        """Initialize RHC buffers and EMA state"""
        self._base_action_buffers = [[] for _ in range(num_envs)]
        self._num_envs = num_envs
        # EMA state will be initialized lazily in get_base_action()
        self._ema_smoothed_joints = None

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset buffers and EMA state for specified envs"""
        if self._base_action_buffers is None:
            return
        for i in env_ids.tolist():
            if i < len(self._base_action_buffers):
                self._base_action_buffers[i].clear()
        # Reset EMA state for reset environments
        if self._ema_smoothed_joints is not None:
            self._ema_smoothed_joints[env_ids] = 0.0

    # -----------------------------
    # Node extraction from obs
    # -----------------------------
    def _extract_nodes_from_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ee_node and obj_node from obs
        
        Args:
            obs: [B, obs_dim]
        
        Returns:
            ee_node: [B, 7] - position(3) + orientation(4)
            obj_node: [B, 7] - position(3) + orientation(4)
        """
        cfg = self.cfg
        
        if cfg.obs_structure is not None:
            obs_struct = cfg.obs_structure
            obj_pos = obs[:, obs_struct["object_position"][0]:obs_struct["object_position"][1]]
            obj_ori = obs[:, obs_struct["object_orientation"][0]:obs_struct["object_orientation"][1]]
            ee_pos = obs[:, obs_struct["ee_position"][0]:obs_struct["ee_position"][1]]
            ee_ori = obs[:, obs_struct["ee_orientation"][0]:obs_struct["ee_orientation"][1]]
        else:
            # Default structure: [joint_pos(6), obj_pos(3), obj_ori(4), ee_pos(3), ee_ori(4), ...]
            robot_state_dim = cfg.robot_state_dim
            obj_pos = obs[:, robot_state_dim:robot_state_dim + 3]
            obj_ori = obs[:, robot_state_dim + 3:robot_state_dim + 7]
            ee_pos = obs[:, robot_state_dim + 7:robot_state_dim + 10]
            ee_ori = obs[:, robot_state_dim + 10:robot_state_dim + 14]
        
        ee_node = torch.cat([ee_pos, ee_ori], dim=-1)      # [B, 7]
        obj_node = torch.cat([obj_pos, obj_ori], dim=-1)   # [B, 7]
        
        return ee_node, obj_node

    def _select_obs_for_rl(self, obs: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Select which parts of obs to use for Actor/Critic
        
        Args:
            obs: [B, obs_dim] - full observation
            mode: "full", "robot_state", or Tuple[int, int] for slice
        
        Returns:
            obs_selected: [B, selected_dim]
        """
        if mode == "full":
            return obs
        elif mode == "robot_state" or mode == "robot_state_only":
            # Use only robot_state (joint_pos) part
            robot_state_dim = self.cfg.robot_state_dim
            return obs[:, :robot_state_dim]
        elif isinstance(mode, tuple) and len(mode) == 2:
            # Custom slice: obs[:, start:end]
            start, end = mode
            return obs[:, start:end]
        else:
            raise ValueError(f"Unknown obs_mode: {mode}. Use 'full', 'robot_state', or (start, end) tuple.")

    def _normalize_nodes(
        self, 
        ee_node: torch.Tensor, 
        obj_node: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize nodes (if stats available)"""
        if self.norm_ee_node_mean is not None:
            ee_node = (ee_node - self.norm_ee_node_mean.to(ee_node.device)) / (self.norm_ee_node_std.to(ee_node.device) + 1e-8)
        if self.norm_object_node_mean is not None:
            obj_node = (obj_node - self.norm_object_node_mean.to(obj_node.device)) / (self.norm_object_node_std.to(obj_node.device) + 1e-8)
        return ee_node, obj_node

    # -----------------------------
    # High-frequency z extraction
    # -----------------------------
    def _get_z_layers_fast(self, obs: torch.Tensor) -> torch.Tensor:
        """
        High-frequency z_layers extraction: called every step
        
        1. Extract current ee_node, obj_node from obs
        2. Normalize
        3. Call backbone.extract_z_fast() (frozen)
        4. Pass through z_adapter (trainable)
        
        Returns:
            z_layers_adapted: [B, K, z_dim]
        """
        # 1. Extract nodes
        ee_node, obj_node = self._extract_nodes_from_obs(obs)
        
        # 2. Normalize
        ee_node_norm, obj_node_norm = self._normalize_nodes(ee_node, obj_node)
        
        # 3. Frozen z extraction
        z_layers_frozen = self.backbone.extract_z_fast(ee_node_norm, obj_node_norm)  # [B, K, z_dim]
        
        # 4. Adapter (trainable)
        B, K, D = z_layers_frozen.shape
        z_flat = z_layers_frozen.reshape(B * K, D)
        z_adapted_flat = self.z_adapter(z_flat)
        z_layers_adapted = z_adapted_flat.reshape(B, K, D)
        
        return z_layers_adapted

    # -----------------------------
    # Latent aggregation
    # -----------------------------
    def aggregate_latent(self, z_layers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z_layers: [B, K, z_dim]
        returns:
            z_bar: [B, z_dim]
            w:     [B, K]
        """
        w = self.gate(z_layers)                                 # [B, K]
        z_bar = torch.sum(w.unsqueeze(-1) * z_layers, dim=1)    # [B, z_dim]
        return z_bar, w

    # -----------------------------
    # Base action (low-frequency, RHC)
    # -----------------------------
    @torch.no_grad()
    def get_base_action(
        self,
        obs_norm: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        ee_node_history: Optional[torch.Tensor] = None,
        object_node_history: Optional[torch.Tensor] = None,
        joint_states_history: Optional[torch.Tensor] = None,
        subtask_condition: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """
        Return base action [B, act_dim] using RHC buffer.
        Low-frequency: only call DiT when buffer is empty
        """
        B = obs_norm.shape[0]
        
        if self._base_action_buffers is None or len(self._base_action_buffers) != B:
            # Offline mode: direct prediction
            traj = self.backbone.predict_base_trajectory(
                obs=obs_norm,
                action_history=action_history,
                ee_node_history=ee_node_history,
                object_node_history=object_node_history,
                joint_states_history=joint_states_history,
                subtask_condition=subtask_condition,
                num_diffusion_steps=self.num_diffusion_steps,
                deterministic=deterministic,
            )
            return traj[:, 0, :]

        # Online mode: use buffer
        need_replan = any(len(self._base_action_buffers[i]) == 0 for i in range(B))
        
        if need_replan:
            traj = self.backbone.predict_base_trajectory(
                obs=obs_norm,
                action_history=action_history,
                ee_node_history=ee_node_history,
                object_node_history=object_node_history,
                joint_states_history=joint_states_history,
                subtask_condition=subtask_condition,
                num_diffusion_steps=self.num_diffusion_steps,
                deterministic=deterministic,
            )

            for i in range(B):
                if len(self._base_action_buffers[i]) == 0:
                    for t in range(min(self.exec_horizon, traj.shape[1])):
                        self._base_action_buffers[i].append(traj[i, t, :].detach())

        base_actions = []
        for i in range(B):
            base_actions.append(self._base_action_buffers[i].pop(0))
        base_action = torch.stack(base_actions, dim=0)  # [B, action_dim]

        # Apply EMA smoothing to joints only (gripper excluded)
        if self.joint_dim < base_action.shape[-1]:
            joints = base_action[:, : self.joint_dim]  # [B, joint_dim]
            gripper = base_action[:, self.joint_dim :]  # [B, action_dim - joint_dim]

            # Initialize EMA state if needed
            if self._ema_smoothed_joints is None:
                self._ema_smoothed_joints = joints.clone()
            elif self._ema_smoothed_joints.shape[0] != B:
                # Resize if batch size changed
                old_smoothed = self._ema_smoothed_joints
                self._ema_smoothed_joints = torch.zeros(
                    B, self.joint_dim, device=joints.device, dtype=joints.dtype
                )
                if old_smoothed.shape[0] <= B:
                    self._ema_smoothed_joints[: old_smoothed.shape[0]] = old_smoothed

            # Apply EMA: smoothed = alpha * new + (1 - alpha) * old
            self._ema_smoothed_joints = (
                self.ema_alpha * joints + (1 - self.ema_alpha) * self._ema_smoothed_joints
            )

            # Concatenate smoothed joints with gripper
            base_action = torch.cat([self._ema_smoothed_joints, gripper], dim=-1)

        return base_action

    # -----------------------------
    # Main act()
    # -----------------------------
    @torch.no_grad()
    def act(
        self,
        obs_raw: Any,
        obs_norm: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
        ee_node_history: Optional[torch.Tensor] = None,
        object_node_history: Optional[torch.Tensor] = None,
        joint_states_history: Optional[torch.Tensor] = None,
        subtask_condition: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            action: [B, act_dim]
            info: dict for training
        """
        obs = flatten_obs(obs_raw, device=self.device_str)

        if obs_norm is None:
            obs_norm = obs

        # ============================================================
        # 1. Base action (low-frequency, from buffer)
        # ============================================================
        # get_base_action returns NORMALIZED actions (from Graph-DiT predict)
        a_base_norm = self.get_base_action(
            obs_norm=obs_norm,
            action_history=action_history,
            ee_node_history=ee_node_history,
            object_node_history=object_node_history,
            joint_states_history=joint_states_history,
            subtask_condition=subtask_condition,
            deterministic=True,
        )
        
        # Denormalize base_action for execution (but keep normalized version for Actor input)
        if self.action_mean is not None and self.action_std is not None:
            a_base = a_base_norm * self.action_std + self.action_mean
        else:
            a_base = a_base_norm

        # ============================================================
        # 2. z_layers (high-frequency! called every step via extract_z_fast)
        # ============================================================
        z_layers = self._get_z_layers_fast(obs)  # [B, K, z_dim], already passed through adapter

        # 3. GateNet aggregation
        z_bar, w = self.aggregate_latent(z_layers)

        # 4. Actor: select obs based on actor_obs_mode
        # CRITICAL: Actor input should be NORMALIZED (obs_norm, a_base_norm, z_bar)
        obs_actor = self._select_obs_for_rl(obs_norm, self.cfg.actor_obs_mode)
        x = torch.cat([obs_actor, a_base_norm, z_bar], dim=-1)  # Use normalized a_base
        dist = self.actor.dist(x)
        
        if deterministic:
            delta_norm = dist.mean  # Actor outputs NORMALIZED delta (mu, already tanh-bounded to [-1, 1])
        else:
            delta_norm = dist.rsample()  # Actor outputs NORMALIZED delta (sampled from Normal(mu, std))
            # CRITICAL: Clamp sampled delta_norm to [-1, 1] to ensure bounded residual
            # Even though mu is tanh-bounded, sampling can produce values outside [-1, 1] when std > 0
            delta_norm = torch.clamp(delta_norm, -1.0, 1.0)

        # Residual mask
        if self.residual_action_mask is not None:
            delta_norm = delta_norm * self.residual_action_mask

        # Denormalize delta for execution
        if self.action_mean is not None and self.action_std is not None:
            delta = delta_norm * self.action_std  # Only scale, no shift (delta is residual)
        else:
            delta = delta_norm

        # 5. Final action (both a_base and delta are DENORMALIZED)
        alpha = torch.clamp(self.alpha, 0.0, 0.3)  # Limit alpha to 30%
        a = a_base + alpha * delta

        # ============================================================
        # FIX: log_prob uses normalized delta (Actor's output space)
        # Actor distribution is defined in normalized space, so log_prob
        # must be computed in the same space
        # ============================================================
        log_prob = dist.log_prob(delta_norm).sum(dim=-1)  # ✅ Use normalized delta
        entropy = dist.entropy().sum(dim=-1)
        # Critic: select obs based on critic_obs_mode
        obs_critic = self._select_obs_for_rl(obs_norm, self.cfg.critic_obs_mode)
        v_bar = self.critic.forward_bar(z_bar, obs_critic)

        info = {
            "obs": obs,  # Raw obs (for recomputing z_layers during training to get gradients)
            "obs_norm": obs_norm,
            "a_base": a_base,  # Denormalized (for execution)
            "a_base_norm": a_base_norm,  # Normalized (for history storage)
            "delta": delta,  # Denormalized (for execution)
            "delta_norm": delta_norm,  # Normalized (for training)
            "log_prob": log_prob,
            "entropy": entropy,
            "z_layers": z_layers,  # Note: z_layers during rollout have no gradients (due to @torch.no_grad)
            "z_bar": z_bar,
            "gate_w": w,
            "v_bar": v_bar,
            "alpha": alpha.detach(),
        }
        return a, info

    # -----------------------------
    # Loss (unchanged, but z_layers is now high-frequency)
    # -----------------------------
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        batch keys:
            obs_norm:   [B, obs_dim]
            a_base:     [B, act_dim]
            action:     [B, act_dim]
            obs:        [B, obs_dim]  # Raw obs (for recomputing z_layers to get gradients)
            returns:    [B]
            adv:        [B]
        
        Note: z_layers need to be recomputed (not taken from batch),
        because act() has @torch.no_grad() during rollout, cannot get z_adapter gradients.
        """
        obs_norm = batch["obs_norm"]
        obs = batch.get("obs", obs_norm)  # If raw obs not provided, use obs_norm
        a_base = batch["a_base"]  # Denormalized (from rollout)
        action = batch["action"]  # Denormalized (final executed action)
        returns = batch["returns"]
        adv = batch["adv"]

        # ============================================================
        # FIX 1: Normalize a_base to match act() where Actor sees normalized input
        # ============================================================
        if self.action_mean is not None and self.action_std is not None:
            a_base_norm = (a_base - self.action_mean) / (self.action_std + 1e-8)
        else:
            a_base_norm = a_base

        # Recompute z_layers with gradients (for training z_adapter)
        z_layers = self._get_z_layers_fast(obs)  # [B, K, z_dim], with gradients!

        # Recompute gate + z_bar (for grad)
        z_bar_new, w = self.aggregate_latent(z_layers)

        # ============================================================
        # FIX 2: Actor input uses normalized values (consistent with act())
        # ============================================================
        obs_actor = self._select_obs_for_rl(obs_norm, self.cfg.actor_obs_mode)
        x = torch.cat([obs_actor, a_base_norm, z_bar_new], dim=-1)  # ✅ Use normalized a_base_norm
        dist = self.actor.dist(x)

        # Recover delta in denormalized space
        alpha = torch.clamp(self.alpha, 0.0, 0.3)  # Limit alpha to 30%
        delta = (action - a_base) / (alpha + 1e-8)  # Denormalized delta

        if self.residual_action_mask is not None:
            delta = delta * self.residual_action_mask

        # ============================================================
        # FIX 3: Normalize delta to match Actor output space (normalized)
        # ============================================================
        if self.action_mean is not None and self.action_std is not None:
            delta_norm = delta / (self.action_std + 1e-8)  # ✅ Actor outputs normalized delta
        else:
            delta_norm = delta

        # ============================================================
        # FIX 4: log_prob uses normalized delta (Actor's output space)
        # ============================================================
        log_prob = dist.log_prob(delta_norm).sum(dim=-1)  # ✅ Use normalized delta
        entropy = dist.entropy().sum(dim=-1)

        # Delta regularization: encourage small residuals (in normalized space)
        loss_delta_reg = (delta_norm ** 2).mean()  # ✅ Use normalized delta

        # AWR weights
        w_adv = torch.exp(adv / max(self.cfg.beta, 1e-8))
        w_adv = torch.clamp(w_adv, 0.0, self.cfg.weight_clip_max)
        w_adv = w_adv / (w_adv.mean() + 1e-8)

        # Actor loss
        loss_actor = -(w_adv.detach() * log_prob).mean()

        # Deep critic supervision
        V_layers = self.critic.forward_layers(z_layers)
        K = V_layers.shape[1]
        omega = torch.arange(1, K + 1, device=V_layers.device, dtype=torch.float32)
        omega = omega / omega.sum()
        
        returns_expand = returns.unsqueeze(-1).expand_as(V_layers)
        loss_v_layers = ((V_layers - returns_expand) ** 2)
        loss_critic = (loss_v_layers * omega.view(1, -1)).mean()

        # Baseline head: select obs based on critic_obs_mode
        obs_critic = self._select_obs_for_rl(obs_norm, self.cfg.critic_obs_mode)
        V_bar = self.critic.forward_bar(z_bar_new, obs_critic)
        loss_v_bar = F.mse_loss(V_bar, returns)

        # Gate entropy
        gate_entropy = -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()
        loss_gate = -gate_entropy

        # Total
        loss_total = (
            loss_actor
            + self.cfg.cV * (loss_critic + 0.5 * loss_v_bar)
            + self.cfg.cGate * loss_gate
            - self.cfg.cEnt * entropy.mean()
            + 0.1 * loss_delta_reg  # Delta regularization: encourage small residuals
        )

        return {
            "loss_total": loss_total,
            "loss_actor": loss_actor.detach(),
            "loss_critic_layers": loss_critic.detach(),
            "loss_critic_bar": loss_v_bar.detach(),
            "loss_gate": loss_gate.detach(),
            "gate_entropy": gate_entropy.detach(),
            "entropy": entropy.mean().detach(),
            "loss_delta_reg": loss_delta_reg.detach(),
            "alpha": alpha.detach(),
        }
    
    # -----------------------------
    # Save / Load
    # -----------------------------
    def save(self, path: str):
        """Save policy"""
        torch.save({
            "policy_state_dict": self.state_dict(),
            "cfg": self.cfg,
        }, path)
        print(f"[GraphDiTResidualRLPolicy] Saved to: {path}")
    
    @classmethod
    def load(cls, path: str, backbone: GraphDiTBackboneAdapter, device: str = "cuda"):
        """Load policy"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint["cfg"]
        
        policy = cls(cfg, backbone)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.to(device)
        
        print(f"[GraphDiTResidualRLPolicy] Loaded from: {path}")
        return policy

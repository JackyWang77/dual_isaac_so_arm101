# Copyright (c) 2024-2026, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Graph-Unet + Residual RL Policy (OUR METHOD) - High-Frequency Z Version

Key improvements:
1) base_action: Low-frequency (obtain trajectory from Unet every exec_horizon steps)
2) z: High-frequency (call extract_z every step, only run graph backbone)
3) Residual head: Trainable (frozen backbone z + RL head)
4) DeepValueCritic: Single head (z_bar)
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
# Unified History Buffer (for RL online interaction)
# =========================================================
class UnifiedHistoryBuffer:
    """
    Unified History Buffer manager for RL online interaction.
    
    Manages all temporal histories:
    - Node history (EE + Object): [num_envs, H, 7]
    - Action history: [num_envs, H, action_dim]
    - Joint states history: [num_envs, H, joint_dim]
    
    Uses Ring Buffer implementation: O(1) update, no memory allocation.
    """
    
    def __init__(
        self,
        num_envs: int,
        history_length: int,
        node_dim: int = 7,
        action_dim: int = 6,
        joint_dim: int = 6,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.num_envs = num_envs
        self.history_length = history_length
        self.node_dim = node_dim
        self.action_dim = action_dim
        self.joint_dim = joint_dim
        self.device = device
        self.dtype = dtype
        
        # Ring buffers: [num_envs, history_length, dim]
        self.ee_node = torch.zeros(num_envs, history_length, node_dim, device=device, dtype=dtype)
        self.obj_node = torch.zeros(num_envs, history_length, node_dim, device=device, dtype=dtype)
        self.action = torch.zeros(num_envs, history_length, action_dim, device=device, dtype=dtype)
        self.joint_states = torch.zeros(num_envs, history_length, joint_dim, device=device, dtype=dtype)
        
        # Ring buffer indices
        self.write_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.filled = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    def update(
        self,
        ee_node: torch.Tensor,       # [B, 7]
        obj_node: torch.Tensor,      # [B, 7]
        action: torch.Tensor,        # [B, action_dim]
        joint_states: torch.Tensor,  # [B, joint_dim]
    ):
        """Update all buffers with new data (O(1), no memory allocation)"""
        B = ee_node.shape[0]
        batch_idx = torch.arange(B, device=self.device)
        idx = self.write_idx[:B]
        
        # Write to current position
        self.ee_node[batch_idx, idx] = ee_node
        self.obj_node[batch_idx, idx] = obj_node
        self.action[batch_idx, idx] = action
        self.joint_states[batch_idx, idx] = joint_states
        
        # Update ring buffer state
        self.write_idx[:B] = (self.write_idx[:B] + 1) % self.history_length
        self.filled[:B] = self.filled[:B] | (self.write_idx[:B] == 0)
    
    def get_history(self, num_envs: int | None = None) -> dict:
        """
        Get all histories in chronological order (oldest to newest)
        
        Args:
            num_envs: Number of environments to get history for (default: all)
        
        Returns:
            dict with keys:
            - ee_node_history: [B, H, 7]
            - object_node_history: [B, H, 7]
            - action_history: [B, H, action_dim]
            - joint_states_history: [B, H, joint_dim]
        """
        B = num_envs if num_envs is not None else self.num_envs
        H = self.history_length
        
        batch_idx = torch.arange(B, device=self.device)
        
        # Determine start index for chronological order
        start = torch.where(
            self.filled[:B], 
            self.write_idx[:B], 
            torch.zeros(B, dtype=torch.long, device=self.device)
        )
        
        # Create reordering indices
        base = torch.arange(H, device=self.device).unsqueeze(0)  # [1, H]
        indices = (base + start.unsqueeze(1)) % H  # [B, H]
        
        # Gather in chronological order
        return {
            "ee_node_history": self.ee_node[batch_idx.unsqueeze(1), indices],       # [B, H, 7]
            "object_node_history": self.obj_node[batch_idx.unsqueeze(1), indices],  # [B, H, 7]
            "action_history": self.action[batch_idx.unsqueeze(1), indices],         # [B, H, action_dim]
            "joint_states_history": self.joint_states[batch_idx.unsqueeze(1), indices],  # [B, H, joint_dim]
        }
    
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset buffers for specified envs (or all if None)"""
        if env_ids is None:
            self.ee_node.zero_()
            self.obj_node.zero_()
            self.action.zero_()
            self.joint_states.zero_()
            self.write_idx.zero_()
            self.filled.zero_()
        else:
            if env_ids.dim() == 0:
                env_ids = env_ids.unsqueeze(0)
            self.ee_node[env_ids] = 0
            self.obj_node[env_ids] = 0
            self.action[env_ids] = 0
            self.joint_states[env_ids] = 0
            self.write_idx[env_ids] = 0
            self.filled[env_ids] = False
    
    def to(self, device: str | torch.device):
        """Move buffer to device"""
        self.device = device if isinstance(device, str) else str(device)
        self.ee_node = self.ee_node.to(device)
        self.obj_node = self.obj_node.to(device)
        self.action = self.action.to(device)
        self.joint_states = self.joint_states.to(device)
        self.write_idx = self.write_idx.to(device)
        self.filled = self.filled.to(device)
        return self


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
# GAE (with timeout bootstrap: truncated -> nonterminal=1)
# =========================================================
@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    timeouts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If timeouts is provided: done=1 & timeout=1 (truncated) -> nonterminal=1 (bootstrap next_value).
    Terminated (real done) -> nonterminal=0; truncated (timeout) -> nonterminal=1; running -> nonterminal=1.
    """
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        if timeouts is not None:
            # done=1 & timeout=1 (truncated) -> nonterminal=1 (bootstrap); done=1 & timeout=0 (terminated) -> nonterminal=0
            nonterminal = (1.0 - dones[t].float() + timeouts[t].float()).clamp(0.0, 1.0)
        else:
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
class GraphUnetResidualRLCfg:
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

    # z_adapter hidden dims
    z_adapter_hidden: Tuple[int, ...] = (256,)

    # residual distribution (log_std_init small = less exploration at start)
    log_std_init: float = -2.0
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

    # losses (residual RL is U-Net only: no gate, bar head only)
    cV: float = 1.0
    cEnt: float = 0.01
    cGate: float = 0.0  # Unused (no gate)
    c_delta_reg: float = 2.0  # Delta (residual) regularization: higher = smoother, RL "don't move unless reward"

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
    def __init__(
        self,
        K: int,
        z_dim: int,
        obs_dim: int,
        hidden: Tuple[int, ...],
        act: str,
        use_layer_heads: bool = True,
    ):
        super().__init__()
        self.K = K
        self.z_dim = z_dim
        self.use_layer_heads = use_layer_heads

        if use_layer_heads:
            self.trunk = mlp(z_dim, hidden, hidden[-1], act=act)
            self.layer_heads = nn.ModuleList([nn.Linear(hidden[-1], 1) for _ in range(K)])
        else:
            self.trunk = None
            self.layer_heads = None

        self.bar_trunk = mlp(z_dim + obs_dim, hidden, hidden[-1], act=act)
        self.bar_head = nn.Linear(hidden[-1], 1)

    def forward_layers(self, z_layers: torch.Tensor) -> torch.Tensor:
        if not self.use_layer_heads:
            B = z_layers.shape[0]
            return torch.zeros(B, self.K, device=z_layers.device, dtype=z_layers.dtype)
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
class GraphUnetBackboneAdapter:
    """
    Adapter interface for Graph-Unet (or compatible) backbone.

    Methods:
    - predict_base_trajectory(): Low-frequency, returns full trajectory
    - extract_z(): High-frequency, extracts z with temporal history support
    """
    def __init__(self, backbone_policy: nn.Module):
        self.backbone = backbone_policy

    @torch.no_grad()
    def predict_base_trajectory(self, **kwargs) -> torch.Tensor:
        """Low-frequency: predict full trajectory [B, pred_horizon, act_dim]"""
        return self.backbone.predict(**kwargs)

    @torch.no_grad()
    def extract_z(
        self, 
        ee_node_history: torch.Tensor,   # [B, H, 7] or [B, 7]
        obj_node_history: torch.Tensor,  # [B, H, 7] or [B, 7]
    ) -> torch.Tensor:
        """
        High-frequency: extract z with temporal history support.
        
        Automatically detects temporal mode:
        - Input [B, H, 7] with H > 1 → temporal mode (uses edge modulation)
        - Input [B, 7] → single frame mode (simplified computation)
        
        Args:
            ee_node_history: EE node history [B, H, 7] or current [B, 7]
            obj_node_history: Object node history [B, H, 7] or current [B, 7]
        
        Returns:
            z_layers: [B, K, z_dim]
        """
        if not hasattr(self.backbone, "extract_z"):
            raise NotImplementedError(
                "Backbone policy must implement extract_z(ee_node_history, obj_node_history) -> [B,z_dim]"
            )
        return self.backbone.extract_z(ee_node_history, obj_node_history)
    
    @property
    def node_stats(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get normalization stats (if available)"""
        stats = {}
        if hasattr(self.backbone, 'cfg'):
            # Try to load stats from checkpoint
            pass
        return stats


# =========================================================
# OUR POLICY: Graph-Unet + Residual RL Head (high-frequency z version)
# =========================================================
class GraphUnetResidualRLPolicy(nn.Module):
    """
    Online inference:
        1. base_action_t = pop from RHC buffer (low-frequency, updated every exec_horizon steps)
        2. z_t = backbone.extract_z(...) (high-frequency, updated every step!)
        3. z_bar_t = z (single latent for Unet)
        4. delta_a_t ~ Actor(obs, base_action, z_bar)
        5. a_t = base + alpha * delta

    Key improvements:
    - z updated every step (high-frequency), reflects current EE-Object relationship
    - base_action updated at low-frequency, leverages Unet's trajectory planning capability
    - Residual head allows frozen backbone to be fine-tuned by RL gradients
    """

    def __init__(
        self,
        cfg: GraphUnetResidualRLCfg,
        backbone: GraphUnetBackboneAdapter,
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

        # Residual RL is U-Net only: single z, no gate
        self.gate = None

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

        # Actor input dim: obs_selected + base_action (6 dims) + z_bar
        # Actor outputs 6 dims (arm + gripper); Zero Init so initial delta ≈ 0 (safe start)
        base_action_dim_for_actor = 6
        actor_in_dim = actor_obs_dim + base_action_dim_for_actor + cfg.z_dim
        self.actor = ResidualGaussianActor(
            in_dim=actor_in_dim,
            action_dim=6,  # Full 6D: arm + gripper (RL controls gripper with per-channel alpha)
            hidden=cfg.actor_hidden,
            act=cfg.activation,
            log_std_init=cfg.log_std_init,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        )
        self.actor_action_dim = 6

        # Critic: bar head only (U-Net single z)
        self.critic = DeepValueCritic(
            K=cfg.num_layers,
            z_dim=cfg.z_dim,
            obs_dim=critic_obs_dim,
            hidden=cfg.critic_hidden,
            act=cfg.activation,
            use_layer_heads=False,
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
        
        # History length (should match Graph-DiT's action_history_length)
        self.history_length = getattr(cfg, 'history_length', 10)
        
        # Unified history buffer (lazy initialization)
        self.history_buffer: UnifiedHistoryBuffer | None = None

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
            # Default structure: [joint_pos(6), joint_vel(6), obj_pos(3), obj_ori(4), ee_pos(3), ee_ori(4), ...]
            # WARNING: This default assumes joint_pos(6) + joint_vel(6) = 12 before object_position
            # This matches graph_unet_policy default: object_position = obs[:, 12:15]
            # If obs_structure is None, both should use the same default indices
            # CRITICAL: Always pass obs_structure from Graph-DiT config to avoid inconsistency!
            import warnings
            warnings.warn(
                "obs_structure is None! Using default indices. "
                "This may be inconsistent with Graph-DiT default indices. "
                "Please ensure obs_structure is passed from Graph-DiT config.",
                UserWarning
            )
            # Default: [joint_pos(6), joint_vel(6), obj_pos(3), obj_ori(4), ee_pos(3), ee_ori(4), ...]
            # object_position starts at index 12 (6 + 6)
            object_start_idx = 12  # joint_pos(6) + joint_vel(6)
            obj_pos = obs[:, object_start_idx:object_start_idx + 3]
            obj_ori = obs[:, object_start_idx + 3:object_start_idx + 7]
            ee_pos = obs[:, object_start_idx + 7:object_start_idx + 10]
            ee_ori = obs[:, object_start_idx + 10:object_start_idx + 14]
        
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
        High-frequency z_layers extraction: called every step with temporal history
        
        1. Extract current ee_node, obj_node from obs
        2. Get history from buffer (BEFORE updating buffer)
        3. Normalize history
        4. Call backbone.extract_z() with history (frozen)
        5. Pass through z_adapter (trainable)
        
        Returns:
            z_layers_adapted: [B, K, z_dim]
        """
        B = obs.shape[0]
        
        # 1. Extract current nodes
        ee_node, obj_node = self._extract_nodes_from_obs(obs)
        
        # 2. Get history from buffer (BEFORE updating buffer)
        if self.history_buffer is not None:
            histories = self.history_buffer.get_history(B)
            ee_node_history = histories["ee_node_history"]      # [B, H, 7]
            obj_node_history = histories["object_node_history"]  # [B, H, 7]
        else:
            # Fallback: no history, use current only
            ee_node_history = ee_node.unsqueeze(1)   # [B, 1, 7]
            obj_node_history = obj_node.unsqueeze(1)  # [B, 1, 7]
        
        # 3. Normalize history
        if self.norm_ee_node_mean is not None:
            ee_node_history = (ee_node_history - self.norm_ee_node_mean) / (self.norm_ee_node_std + 1e-8)
        if self.norm_object_node_mean is not None:
            obj_node_history = (obj_node_history - self.norm_object_node_mean) / (self.norm_object_node_std + 1e-8)
        
        # 4. Frozen z extraction with temporal history
        z_layers_frozen = self.backbone.extract_z(
            ee_node_history, 
            obj_node_history
        )  # [B, K, z_dim]
        
        # 5. Adapter (trainable)
        B, K, D = z_layers_frozen.shape
        z_flat = z_layers_frozen.reshape(B * K, D)
        z_adapted_flat = self.z_adapter(z_flat)
        z_layers_adapted = z_adapted_flat.reshape(B, K, D)
        
        return z_layers_adapted

    # -----------------------------
    # Latent aggregation
    # -----------------------------
    def aggregate_latent(self, z_layers: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        U-Net only: single z (all layers same), no gate.
        z_layers: [B, K, z_dim] -> z_bar: [B, z_dim], w: None
        """
        z_bar = z_layers[:, -1, :]
        return z_bar, None

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
        base_action_5d = torch.stack(base_actions, dim=0)  # [B, 5] - Graph-DiT outputs 5D (arm joints only)

        # Apply EMA smoothing to joints only (gripper excluded)
        # Graph-DiT outputs 5D, so base_action_5d is already [B, 5] (all arm joints)
        joints = base_action_5d  # [B, 5] - all are arm joints

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

        # Graph-DiT outputs 5D, so we return 5D (gripper is handled separately or padded later)
        # Return normalized 5D action (arm joints only)
        return self._ema_smoothed_joints  # [B, 5] - smoothed arm joints

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
        # 1. Extract current state and get histories
        # ============================================================
        B = obs.shape[0]
        ee_node, obj_node = self._extract_nodes_from_obs(obs)
        joint_states = obs[:, :self.cfg.robot_state_dim]
        
        # Get histories from buffer (for predict() call)
        if self.history_buffer is not None:
            histories = self.history_buffer.get_history(B)
            # Use buffer histories if available, otherwise fallback to provided args
            action_history = action_history if action_history is not None else histories["action_history"]
            ee_node_history = ee_node_history if ee_node_history is not None else histories["ee_node_history"]
            object_node_history = object_node_history if object_node_history is not None else histories["object_node_history"]
            joint_states_history = joint_states_history if joint_states_history is not None else histories["joint_states_history"]
        else:
            # Fallback: use provided args or create empty history
            if action_history is None:
                action_history = torch.zeros(B, 1, self.cfg.action_dim, device=obs.device)
            if ee_node_history is None:
                ee_node_history = ee_node.unsqueeze(1)
            if object_node_history is None:
                object_node_history = obj_node.unsqueeze(1)
            if joint_states_history is None:
                joint_states_history = joint_states.unsqueeze(1)
        
        # ============================================================
        # 2. Base action (low-frequency, from buffer)
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
        
        # Denormalize base_action for execution (arm 5D; gripper from backbone 6th dim or fallback)
        a_base_norm_5d = a_base_norm[:, :5]  # [B, 5] - arm only
        if self.action_mean is not None and self.action_std is not None:
            a_base_5d = a_base_norm_5d * self.action_std[:5] + self.action_mean[:5]  # [B, 5]
        else:
            a_base_5d = a_base_norm_5d  # [B, 5]
        
        # Gripper: backbone outputs 6D → use 6th dim (denormalized); 5D → pad 0
        if a_base_norm.shape[-1] >= 6:
            if self.action_mean is not None and self.action_std is not None:
                a_base_gripper = a_base_norm[:, 5:6] * self.action_std[5:6] + self.action_mean[5:6]  # [B, 1]
            else:
                a_base_gripper = a_base_norm[:, 5:6]  # [B, 1]
        else:
            a_base_gripper = torch.zeros(B, 1, device=a_base_5d.device, dtype=a_base_5d.dtype)
        a_base = torch.cat([a_base_5d, a_base_gripper], dim=-1)  # [B, 6]

        # ============================================================
        # 3. z_layers (high-frequency! called every step with temporal history)
        # ============================================================
        z_layers = self._get_z_layers_fast(obs)  # [B, K, z_dim], already passed through adapter

        # 4. GateNet aggregation
        z_bar, w = self.aggregate_latent(z_layers)

        # 5. Actor: select obs based on actor_obs_mode
        # Actor input: NORMALIZED (obs_norm, a_base_norm full 6D including gripper, z_bar)
        obs_actor = self._select_obs_for_rl(obs_norm, self.cfg.actor_obs_mode)
        # FIX: Expand a_base_norm to 6D for Actor input (backbone may return 5D arm-only)
        if a_base_norm.shape[-1] == 5:
            gripper_pad = torch.zeros(B, 1, device=a_base_norm.device, dtype=a_base_norm.dtype)
            a_base_norm_6d = torch.cat([a_base_norm, gripper_pad], dim=-1)  # [B, 6]
        else:
            a_base_norm_6d = a_base_norm
        x = torch.cat([obs_actor, a_base_norm_6d, z_bar], dim=-1)  # [B, obs_dim + 6 + z_dim]
        dist = self.actor.dist(x)
        
        # Actor outputs 6D delta (arm + gripper)
        if deterministic:
            delta_norm = dist.mean  # [B, 6]
        else:
            delta_norm = dist.rsample()  # [B, 6]
            delta_norm = torch.clamp(delta_norm, -1.0, 1.0)

        delta_norm = torch.clamp(delta_norm, -1.0, 1.0)

        # Denormalize delta for execution
        if self.action_mean is not None and self.action_std is not None:
            delta = delta_norm * self.action_std
        else:
            delta = delta_norm

        # Optional mask (if all 1s, no effect; if gripper was masked, would zero it)
        if self.residual_action_mask is not None:
            delta = delta * self.residual_action_mask

        # Per-channel alpha: Arm (0-4) low authority, Gripper (5) full authority
        alpha_learned = torch.clamp(self.alpha, 0.0, 0.3)
        alpha_vec = torch.ones_like(delta, device=delta.device, dtype=delta.dtype) * alpha_learned
        alpha_vec[:, 5] = 1.0  # Gripper: full override

        a = a_base + alpha_vec * delta  # [B, 6]
        alpha = alpha_learned  # for info
        
        # 7. Update history buffer (AFTER computing action)
        if self.history_buffer is not None:
            # Use normalized action for history (matches training data format)
            # Graph-DiT outputs 5D, but history buffer expects 6D (with gripper)
            # We need to pad with 0 for gripper, or use the final action 'a' which is 6D
            # Actually, we should store the final action 'a' (6D) after normalization
            # But for now, let's pad a_base_norm to 6D if needed
            action_for_history = a_base_norm
            if action_for_history.shape[-1] == 5:
                # Pad with 0 for gripper (Graph-DiT doesn't output gripper)
                gripper_pad = torch.zeros(B, 1, device=action_for_history.device, dtype=action_for_history.dtype)
                action_for_history = torch.cat([action_for_history, gripper_pad], dim=-1)  # [B, 6]
            self.history_buffer.update(
                ee_node=ee_node,
                obj_node=obj_node,
                action=action_for_history,  # Store normalized action [B, 6]
                joint_states=joint_states,
            )

        # Log prob / entropy over full 6D (Actor outputs 6D)
        log_prob = dist.log_prob(delta_norm).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        # Critic: select obs based on critic_obs_mode
        obs_critic = self._select_obs_for_rl(obs_norm, self.cfg.critic_obs_mode)
        v_bar = self.critic.forward_bar(z_bar, obs_critic)

        # gate_w: U-Net has no gate; use uniform placeholder for buffer compatibility
        K = self.cfg.num_layers
        gate_w = torch.ones(B, K, device=z_bar.device, dtype=z_bar.dtype) / K

        info = {
            "obs": obs,
            "obs_norm": obs_norm,
            "a_base": a_base,
            "a_base_norm": a_base_norm,
            "delta": delta,
            "delta_norm": delta_norm,
            "log_prob": log_prob,
            "entropy": entropy,
            "z_layers": z_layers,
            "z_bar": z_bar,
            "gate_w": gate_w,  # [B, K]; placeholder when no gate (U-Net)
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
        # Pad a_base_norm to 6D if needed (rollout may store 5D from backbone)
        # ============================================================
        obs_actor = self._select_obs_for_rl(obs_norm, self.cfg.actor_obs_mode)
        B_loss = obs_norm.shape[0]
        if a_base_norm.shape[-1] == 5:
            gripper_pad = torch.zeros(B_loss, 1, device=a_base_norm.device, dtype=a_base_norm.dtype)
            a_base_norm_6d = torch.cat([a_base_norm, gripper_pad], dim=-1)
        else:
            a_base_norm_6d = a_base_norm
        x = torch.cat([obs_actor, a_base_norm_6d, z_bar_new], dim=-1)
        dist = self.actor.dist(x)

        # Recover delta in denormalized space
        # Recover delta with per-channel alpha (same as act(): arm=alpha_learned, gripper=1.0)
        alpha_learned = torch.clamp(self.alpha, 0.0, 0.3)
        B_loss = obs_norm.shape[0]
        alpha_vec = torch.ones(B_loss, 6, device=action.device, dtype=action.dtype) * alpha_learned
        alpha_vec[:, 5] = 1.0
        delta = (action - a_base) / (alpha_vec + 1e-8)

        if self.residual_action_mask is not None:
            delta = delta * self.residual_action_mask

        if self.action_mean is not None and self.action_std is not None:
            delta_norm = delta / (self.action_std + 1e-8)
        else:
            delta_norm = delta
        delta_norm = torch.clamp(delta_norm, -1.0, 1.0)
        log_prob = dist.log_prob(delta_norm).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        loss_delta_reg = (delta_norm ** 2).mean()
        alpha = alpha_learned  # for return dict

        # AWR weights
        w_adv = torch.exp(adv / max(self.cfg.beta, 1e-8))
        w_adv = torch.clamp(w_adv, 0.0, self.cfg.weight_clip_max)
        w_adv = w_adv / (w_adv.mean() + 1e-8)

        # Actor loss
        loss_actor = -(w_adv.detach() * log_prob).mean()

        # Critic: bar head only (no layer heads)
        loss_critic = torch.tensor(0.0, device=obs.device)

        # Baseline head: select obs based on critic_obs_mode
        obs_critic = self._select_obs_for_rl(obs_norm, self.cfg.critic_obs_mode)
        V_bar = self.critic.forward_bar(z_bar_new, obs_critic)
        loss_v_bar = F.mse_loss(V_bar, returns)

        # No gate (U-Net only)
        gate_entropy = torch.tensor(0.0, device=obs.device)
        loss_gate = torch.tensor(0.0, device=obs.device)

        # Total
        loss_total = (
            loss_actor
            + self.cfg.cV * (loss_critic + 0.5 * loss_v_bar)
            + self.cfg.cGate * loss_gate
            - self.cfg.cEnt * entropy.mean()
            + self.cfg.c_delta_reg * loss_delta_reg  # Delta regularization: encourage small residuals (smoother)
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
        """Save policy including trainable node_to_z from backbone"""
        policy_state_dict = self.state_dict()
        
        # CRITICAL: Also save node_to_z weights from backbone if it's trainable
        # node_to_z is in backbone.backbone.node_to_z, not in policy.parameters()
        node_to_z_state_dict = {}
        if hasattr(self, 'backbone') and hasattr(self.backbone, 'backbone'):
            backbone_model = self.backbone.backbone
            if hasattr(backbone_model, 'node_to_z'):
                # Extract node_to_z weights from backbone (if present, e.g. Graph-DiT)
                for name, param in backbone_model.node_to_z.named_parameters():
                    node_to_z_state_dict[f"backbone.backbone.node_to_z.{name}"] = param.data.clone()
                if len(node_to_z_state_dict) > 0:
                    print(f"[GraphUnetResidualRLPolicy] Saving {len(node_to_z_state_dict)} node_to_z parameters")
                    print(f"[GraphUnetResidualRLPolicy] node_to_z_state_dict keys: {list(node_to_z_state_dict.keys())[:3]}...")
                else:
                    print("[GraphUnetResidualRLPolicy] ⚠️  WARNING: node_to_z_state_dict is EMPTY!")
            else:
                print("[GraphUnetResidualRLPolicy] ⚠️  WARNING: backbone has no node_to_z attribute!")
        else:
            print("[GraphUnetResidualRLPolicy] ⚠️  WARNING: Cannot access backbone.backbone!")
        
        torch.save({
            "policy_state_dict": policy_state_dict,
            "node_to_z_state_dict": node_to_z_state_dict if len(node_to_z_state_dict) > 0 else None,
            "cfg": self.cfg,
        }, path)
        print(f"[GraphUnetResidualRLPolicy] Saved to: {path}")
    
    @classmethod
    def load(cls, path: str, backbone: GraphUnetBackboneAdapter, device: str = "cuda"):
        """Load policy including trainable node_to_z from backbone"""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint["cfg"]
        
        policy = cls(cfg, backbone)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        
        # CRITICAL: Also load node_to_z weights from checkpoint if available
        # This ensures we load the trained node_to_z, not the original Graph-DiT node_to_z
        node_to_z_state_dict = checkpoint.get("node_to_z_state_dict", None)
        if node_to_z_state_dict is not None and len(node_to_z_state_dict) > 0:
            if hasattr(backbone, 'backbone'):
                backbone_model = backbone.backbone
                if hasattr(backbone_model, 'node_to_z'):
                    # Load node_to_z weights into backbone (if present)
                    node_to_z_dict = {}
                    for key, value in node_to_z_state_dict.items():
                        # Remove "backbone.backbone." prefix
                        if key.startswith("backbone.backbone.node_to_z."):
                            param_name = key[len("backbone.backbone.node_to_z."):]
                            node_to_z_dict[param_name] = value
                    
                    missing_keys, unexpected_keys = backbone_model.node_to_z.load_state_dict(node_to_z_dict, strict=False)
                    if missing_keys:
                        print(f"[GraphUnetResidualRLPolicy] Warning: Missing node_to_z keys: {missing_keys}")
                    if unexpected_keys:
                        print(f"[GraphUnetResidualRLPolicy] Warning: Unexpected node_to_z keys: {unexpected_keys}")
                    print(f"[GraphUnetResidualRLPolicy] Loaded {len(node_to_z_dict)} node_to_z parameters from checkpoint")
                else:
                    print("[GraphUnetResidualRLPolicy] Warning: node_to_z_state_dict found but backbone has no node_to_z")
            else:
                print("[GraphUnetResidualRLPolicy] Warning: node_to_z_state_dict found but backbone has no backbone")
        else:
            print("[GraphUnetResidualRLPolicy] No node_to_z_state_dict in checkpoint (using original backbone node_to_z)")
        
        policy.to(device)
        
        print(f"[GraphUnetResidualRLPolicy] Loaded from: {path}")
        return policy

# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Residual RL Policy - PPO residual fine-tuning on top of Graph-DiT.

This module implements a Residual RL policy where:
- Graph-DiT (Teacher) provides: (1) Base action, (2) Scene understanding features (Graph Embedding)
- PPO (Student) uses: Graph Embedding + Robot State → Residual action
- Final action = Base action + Residual action

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Graph-DiT (Frozen Teacher)                   │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
    │  │ Node Embed  │    │ Edge Embed  │    │ Graph Attention     │  │
    │  │ (EE, Obj)   │───▶│ (dist, ori) │───▶│ (Scene Understanding)│  │
    │  └─────────────┘    └─────────────┘    └──────────┬──────────┘  │
    │                                                    │             │
    │                                         ┌─────────▼─────────┐   │
    │                                         │ Graph Embedding   │   │
    │                                         │ (Scene Features)  │   │
    │                                         └─────────┬─────────┘   │
    │                                                    │             │
    │  ┌─────────────────────────────────────────────────┼───────────┐│
    │  │ Diffusion Process                               │           ││
    │  │ (denoise noise → action trajectory)             │           ││
    │  │                                                 │           ││
    │  │  └──────────────▶ Base Action (a_base)          │           ││
    │  └─────────────────────────────────────────────────┼───────────┘│
    └────────────────────────────────────────────────────┼────────────┘
                                                         │
    ┌────────────────────────────────────────────────────┼────────────┐
    │                    PPO (Trainable Student)         │            │
    │                                                    │            │
    │  ┌─────────────┐         ┌─────────────────────────▼─────────┐  │
    │  │ Robot State │         │                                   │  │
    │  │ (joint_pos, │────────▶│    MLP Actor-Critic               │  │
    │  │  joint_vel) │         │ Input: [Robot_State, Graph_Embed] │  │
    │  └─────────────┘         │ Output: Residual (a_residual)     │  │
    │                          └─────────────────┬─────────────────┘  │
    └────────────────────────────────────────────┼────────────────────┘
                                                 │
                                      ┌──────────▼──────────┐
                                      │ Final Action        │
                                      │ a = a_base + α*a_res│
                                      └─────────────────────┘

Key advantages:
1. PPO doesn't need to learn scene understanding - it reuses Graph-DiT's features
2. Residual learning is easier - only needs to learn "corrections"
3. Stable baseline - DiT provides a good starting point
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import torch
import torch.nn as nn
import torch.nn.functional as F
from isaaclab.utils import configclass

from .graph_dit_policy import GraphDiTPolicy, GraphDiTPolicyCfg


@configclass
class ResidualRLPolicyCfg:
    """Configuration for Residual RL Policy."""

    # Graph DiT backbone config (should match pre-trained model)
    graph_dit_cfg: GraphDiTPolicyCfg = MISSING
    """Configuration for the frozen Graph DiT backbone."""

    # Path to pre-trained Graph DiT checkpoint
    pretrained_checkpoint: str | None = None
    """Path to pre-trained Graph DiT checkpoint. Required for residual RL."""

    # PPO Residual Network configuration
    residual_hidden_dims: list[int] = [256, 128, 64]
    """Hidden dimensions for residual actor network."""

    residual_activation: str = "elu"
    """Activation function for residual networks."""

    # Value network configuration
    value_hidden_dims: list[int] = [256, 128, 64]
    """Hidden dimensions for value network."""

    value_activation: str = "elu"
    """Activation function for value network."""

    # Input dimensions
    robot_state_dim: int = 12
    """Robot state dimension (joint_pos + joint_vel, e.g., 6+6=12)."""

    value_obs_dim: int | None = None
    """Input dimension for value network. If None, computed automatically."""

    # Residual scaling
    # NOTE: Start VERY small for Residual RL! PPO's initial noise can ruin DiT's good actions
    residual_scale: float = 0.05
    """Initial scale for residual actions (start small for stability)."""

    max_residual_scale: float = 0.3
    """Maximum scale for residual actions."""

    # RL training parameters
    # NOTE: Smaller noise is critical for Residual RL! Default 1.0 is too large.
    init_noise_std: float = 0.1
    """Initial standard deviation for action noise (PPO). Keep small for Residual RL!"""

    # Feature extraction settings
    use_graph_embedding: bool = True
    """Whether to use Graph Embedding from DiT as PPO input."""

    use_node_features: bool = False
    """Whether to include node features (EE, Object) in PPO input."""

    use_edge_features: bool = False
    """Whether to include edge features (distance, alignment) in PPO input."""

    # Freeze settings
    freeze_backbone: bool = True
    """Whether to freeze the Graph DiT backbone (recommended)."""


class ResidualRLPolicy(nn.Module):
    """Residual RL Policy - PPO fine-tuning on top of Graph-DiT.

    This policy uses:
    1. Frozen Graph-DiT to get base action + scene understanding features
    2. Trainable PPO to output residual corrections
    3. Final action = base_action + residual_scale * residual_action

    The PPO agent sees [Robot_State, Graph_Embedding] and learns to output
    small corrections to the DiT's base action.
    """

    def __init__(self, cfg: ResidualRLPolicyCfg):
        super().__init__()
        self.cfg = cfg

        # Load pre-trained Graph DiT backbone FIRST to get actual config
        if cfg.pretrained_checkpoint is not None:
            print(
                f"[ResidualRLPolicy] Loading pre-trained Graph DiT from: {cfg.pretrained_checkpoint}"
            )
            self.graph_dit = GraphDiTPolicy.load(
                cfg.pretrained_checkpoint, device="cpu"
            )
            # CRITICAL: Use the config from loaded checkpoint, not from cfg!
            self.graph_dit_cfg = self.graph_dit.cfg
            print(
                f"[ResidualRLPolicy] Loaded Graph DiT config: hidden_dim={self.graph_dit_cfg.hidden_dim}, "
                f"action_dim={self.graph_dit_cfg.action_dim}, obs_dim={self.graph_dit_cfg.obs_dim}"
            )

            # ============================================================
            # CRITICAL: Load normalization stats from checkpoint!
            # Graph DiT was trained with normalized data, so we must:
            # 1. Normalize obs before feeding to Graph DiT
            # 2. Denormalize action output from Graph DiT
            # ============================================================
            checkpoint = torch.load(
                cfg.pretrained_checkpoint, map_location="cpu", weights_only=False
            )
            self._load_normalization_stats(checkpoint)
        else:
            # Fallback: use provided config
            print(
                "[ResidualRLPolicy] Warning: No pretrained_checkpoint specified. Using provided config."
            )
            graph_dit_cfg = cfg.graph_dit_cfg
            if isinstance(graph_dit_cfg, dict):
                from .graph_dit_policy import GraphDiTPolicyCfg

                graph_dit_cfg = GraphDiTPolicyCfg(**graph_dit_cfg)
            self.graph_dit_cfg = graph_dit_cfg
            self.graph_dit = GraphDiTPolicy(graph_dit_cfg)
            # No normalization stats available (use norm_ prefix!)
            self.norm_obs_mean = None
            self.norm_obs_std = None
            self.norm_action_mean = None
            self.norm_action_std = None
            print(
                "[ResidualRLPolicy] Warning: No normalization stats available (no checkpoint)"
            )

        # Freeze Graph DiT backbone
        if cfg.freeze_backbone:
            for param in self.graph_dit.parameters():
                param.requires_grad = False
            self.graph_dit.eval()
            print("[ResidualRLPolicy] Graph DiT backbone frozen")

        # Compute input dimension for PPO using ACTUAL config from loaded model
        hidden_dim = self.graph_dit_cfg.hidden_dim
        ppo_input_dim = cfg.robot_state_dim  # Start with robot state

        if cfg.use_graph_embedding:
            ppo_input_dim += hidden_dim  # Add graph embedding
        if cfg.use_node_features:
            ppo_input_dim += 2 * hidden_dim  # Add EE + Object node features
        if cfg.use_edge_features:
            ppo_input_dim += hidden_dim  # Add edge features (after embedding)

        self.ppo_input_dim = ppo_input_dim
        print(
            f"[ResidualRLPolicy] PPO input dim: {ppo_input_dim} "
            f"(robot_state={cfg.robot_state_dim}, graph_embed={hidden_dim if cfg.use_graph_embedding else 0})"
        )

        # Build activation function
        activation_fn = self._get_activation(cfg.residual_activation)

        # ============================================================
        # PPO Input = [Robot_State, Graph_Feature, Base_Action_t] → Residual_t
        # ============================================================
        action_dim = self.graph_dit_cfg.action_dim

        # Add base_action_t dimension to PPO input
        ppo_input_dim_with_base = ppo_input_dim + action_dim
        self.ppo_input_dim = ppo_input_dim_with_base
        print(
            f"[ResidualRLPolicy] PPO input dim (with base_action): {ppo_input_dim_with_base} "
            f"= robot_state({cfg.robot_state_dim}) + graph_embed({hidden_dim}) + base_action({action_dim})"
        )

        actor_layers = []
        input_dim = ppo_input_dim_with_base
        for hidden_dim_layer in cfg.residual_hidden_dims:
            actor_layers.append(nn.Linear(input_dim, hidden_dim_layer))
            actor_layers.append(nn.LayerNorm(hidden_dim_layer))
            actor_layers.append(activation_fn())
            input_dim = hidden_dim_layer
        actor_layers.append(
            nn.Linear(input_dim, action_dim)
        )  # Output: single step residual

        self.residual_actor = nn.Sequential(*actor_layers)

        # Log std for action noise (learnable) - single action
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * math.log(cfg.init_noise_std), requires_grad=True
        )

        # Residual scale (learnable, starts small)
        self.residual_scale = nn.Parameter(
            torch.tensor(cfg.residual_scale), requires_grad=True
        )

        # Value Network: obs → value (independent from actor)
        value_activation_fn = self._get_activation(cfg.value_activation)
        value_layers = []
        # Value network uses full observation (can see everything)
        obs_dim = (
            cfg.value_obs_dim
            if cfg.value_obs_dim is not None
            else self.graph_dit_cfg.obs_dim
        )
        input_dim = obs_dim
        for hidden_dim_value in cfg.value_hidden_dims:
            value_layers.append(nn.Linear(input_dim, hidden_dim_value))
            value_layers.append(nn.LayerNorm(hidden_dim_value))
            value_layers.append(value_activation_fn())
            input_dim = hidden_dim_value
        value_layers.append(nn.Linear(input_dim, 1))

        self.value_network = nn.Sequential(*value_layers)

        # Observation normalizer (optional)
        self.obs_normalizer = None

        # Distribution (for RSL-RL compatibility)
        self.distribution = None

        # Cache for base action (to avoid recomputing in same step)
        self._cached_base_action = None
        self._cached_features = None

        # ============================================================
        # CRITICAL: History buffers for Graph DiT
        # Graph DiT needs action history and node history to work properly!
        # ============================================================
        self.action_history_length = self.graph_dit_cfg.action_history_length
        self._num_envs = None  # Will be set on first forward pass
        self._action_history = None  # [num_envs, history_length, action_dim]
        self._ee_node_history = None  # [num_envs, history_length, 7]
        self._object_node_history = None  # [num_envs, history_length, 7]
        self._last_action = None  # [num_envs, action_dim]
        print(f"[ResidualRLPolicy] Action history length: {self.action_history_length}")

        # ============================================================
        # ACTION CHUNKING: Buffer for BASE action trajectory (DiT output only)
        # ============================================================
        self.pred_horizon = self.graph_dit_cfg.pred_horizon
        self.exec_horizon = self.graph_dit_cfg.exec_horizon
        self.action_dim = action_dim
        self._base_action_buffer = (
            None  # [num_envs, exec_horizon, action_dim] - DiT's base trajectory
        )
        self._buffer_indices = None  # [num_envs] - current index in buffer for each env
        print(
            f"[ResidualRLPolicy] Action Chunking: DiT predicts every {self.exec_horizon} steps, PPO corrects every step"
        )

        # Initialize weights
        self._init_weights()

    def _load_normalization_stats(self, checkpoint: dict):
        """Load normalization statistics from checkpoint.

        Graph DiT was trained with normalized data, so we must:
        1. Normalize obs before feeding to Graph DiT
        2. Denormalize action output from Graph DiT
        3. Normalize action_history before feeding to Graph DiT
        4. Normalize ee_node_history and object_node_history before feeding to Graph DiT

        CRITICAL: Use norm_ prefix to avoid conflict with @property action_mean!

        Args:
            checkpoint: Loaded checkpoint dictionary
        """
        import numpy as np

        obs_stats = checkpoint.get("obs_stats", {})
        action_stats = checkpoint.get("action_stats", {})
        node_stats = checkpoint.get("node_stats", {})

        # Load obs stats (norm_ prefix to avoid property conflict)
        if obs_stats and "mean" in obs_stats and "std" in obs_stats:
            obs_mean = obs_stats["mean"]
            obs_std = obs_stats["std"]
            # Convert numpy to tensor if needed
            if isinstance(obs_mean, np.ndarray):
                obs_mean = torch.from_numpy(obs_mean).float().squeeze()
                obs_std = torch.from_numpy(obs_std).float().squeeze()
            self.norm_obs_mean = obs_mean
            self.norm_obs_std = obs_std
            print(
                f"[ResidualRLPolicy] Loaded obs_stats: mean shape={obs_mean.shape}, std shape={obs_std.shape}"
            )
        else:
            self.norm_obs_mean = None
            self.norm_obs_std = None
            print("[ResidualRLPolicy] Warning: No obs_stats found in checkpoint!")

        # Load action stats (norm_ prefix to avoid conflict with @property action_mean!)
        if action_stats and "mean" in action_stats and "std" in action_stats:
            action_mean = action_stats["mean"]
            action_std = action_stats["std"]
            # Convert numpy to tensor if needed
            if isinstance(action_mean, np.ndarray):
                action_mean = torch.from_numpy(action_mean).float().squeeze()
                action_std = torch.from_numpy(action_std).float().squeeze()
            self.norm_action_mean = action_mean
            self.norm_action_std = action_std
            print(
                f"[ResidualRLPolicy] Loaded action_stats: mean shape={action_mean.shape}, std shape={action_std.shape}"
            )
        else:
            self.norm_action_mean = None
            self.norm_action_std = None
            print("[ResidualRLPolicy] Warning: No action_stats found in checkpoint!")

        # Load node stats (CRITICAL: ee_node and object_node need normalization too!)
        if node_stats:
            # EE node stats
            if "ee_mean" in node_stats and "ee_std" in node_stats:
                ee_mean = node_stats["ee_mean"]
                ee_std = node_stats["ee_std"]
                if isinstance(ee_mean, np.ndarray):
                    ee_mean = torch.from_numpy(ee_mean).float().squeeze()
                    ee_std = torch.from_numpy(ee_std).float().squeeze()
                self.norm_ee_node_mean = ee_mean
                self.norm_ee_node_std = ee_std
                print(
                    f"[ResidualRLPolicy] Loaded ee_node_stats: mean shape={ee_mean.shape}"
                )
            else:
                self.norm_ee_node_mean = None
                self.norm_ee_node_std = None

            # Object node stats
            if "object_mean" in node_stats and "object_std" in node_stats:
                obj_mean = node_stats["object_mean"]
                obj_std = node_stats["object_std"]
                if isinstance(obj_mean, np.ndarray):
                    obj_mean = torch.from_numpy(obj_mean).float().squeeze()
                    obj_std = torch.from_numpy(obj_std).float().squeeze()
                self.norm_object_node_mean = obj_mean
                self.norm_object_node_std = obj_std
                print(
                    f"[ResidualRLPolicy] Loaded object_node_stats: mean shape={obj_mean.shape}"
                )
            else:
                self.norm_object_node_mean = None
                self.norm_object_node_std = None
        else:
            self.norm_ee_node_mean = None
            self.norm_ee_node_std = None
            self.norm_object_node_mean = None
            self.norm_object_node_std = None
            print("[ResidualRLPolicy] Warning: No node_stats found in checkpoint!")

        # Load joint_stats (CRITICAL: joint_states_history needs normalization too!)
        joint_stats = checkpoint.get("joint_stats", {})
        if joint_stats and "mean" in joint_stats and "std" in joint_stats:
            joint_mean = joint_stats["mean"]
            joint_std = joint_stats["std"]
            if isinstance(joint_mean, np.ndarray):
                joint_mean = torch.from_numpy(joint_mean).float().squeeze()
                joint_std = torch.from_numpy(joint_std).float().squeeze()
            self.norm_joint_mean = joint_mean
            self.norm_joint_std = joint_std
            print(
                f"[ResidualRLPolicy] Loaded joint_stats: mean shape={joint_mean.shape}"
            )
        else:
            self.norm_joint_mean = None
            self.norm_joint_std = None
            print("[ResidualRLPolicy] Warning: No joint_stats found in checkpoint!")

    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observation for Graph DiT input.

        Args:
            obs: [batch, obs_dim] - Raw observation from environment

        Returns:
            Normalized observation
        """
        if self.norm_obs_mean is None or self.norm_obs_std is None:
            return obs

        # Move stats to same device as obs
        mean = self.norm_obs_mean.to(obs.device)
        std = self.norm_obs_std.to(obs.device)

        return (obs - mean) / std

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action from Graph DiT output.

        Args:
            action: [batch, action_dim] or [batch, horizon, action_dim] - Normalized action

        Returns:
            Denormalized action in environment scale
        """
        if self.norm_action_mean is None or self.norm_action_std is None:
            return action

        # Move stats to same device as action
        mean = self.norm_action_mean.to(action.device)
        std = self.norm_action_std.to(action.device)

        # Handle trajectory shape [batch, horizon, action_dim]
        if len(action.shape) == 3:
            mean = mean.unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
            std = std.unsqueeze(0).unsqueeze(0)

        return action * std + mean

    def _get_activation(self, activation_name: str):
        """Get activation function class."""
        activations = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        if activation_name.lower() not in activations:
            raise ValueError(f"Unknown activation: {activation_name}")
        return activations[activation_name.lower()]

    def _init_weights(self):
        """Initialize network weights."""
        # Initialize residual actor with small weights (start with near-zero residual)
        for module in self.residual_actor:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(
                    module.weight, gain=0.01
                )  # Small gain for stability
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize value network
        for module in self.value_network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _init_history_buffers(self, num_envs: int, device: torch.device):
        """Initialize history buffers for Graph DiT.

        Args:
            num_envs: Number of parallel environments
            device: Device to create buffers on
        """
        if self._num_envs == num_envs and self._action_history is not None:
            return  # Already initialized

        self._num_envs = num_envs
        action_dim = self.graph_dit_cfg.action_dim
        history_len = self.action_history_length

        # Get joint_dim from config or default to robot_state_dim
        joint_dim = (
            getattr(self.graph_dit_cfg, "joint_dim", None) or self.cfg.robot_state_dim
        )

        # Initialize with zeros
        self._action_history = torch.zeros(
            num_envs, history_len, action_dim, device=device, dtype=torch.float32
        )
        self._ee_node_history = torch.zeros(
            num_envs, history_len, 7, device=device, dtype=torch.float32
        )
        self._object_node_history = torch.zeros(
            num_envs, history_len, 7, device=device, dtype=torch.float32
        )
        # CRITICAL: Add joint_states_history for State-Action Self-Attention in GraphDiTUnit
        self._joint_states_history = torch.zeros(
            num_envs, history_len, joint_dim, device=device, dtype=torch.float32
        )
        self._last_action = torch.zeros(
            num_envs, action_dim, device=device, dtype=torch.float32
        )

        # ACTION CHUNKING: Initialize BASE action buffer (DiT output only)
        # Buffer stores DiT's base trajectory, PPO computes residual each step
        self._base_action_buffer = torch.zeros(
            num_envs, self.exec_horizon, action_dim, device=device, dtype=torch.float32
        )
        self._buffer_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Start with exec_horizon so first step triggers DiT prediction
        self._buffer_indices.fill_(self.exec_horizon)

        print(
            f"[ResidualRLPolicy] Initialized history buffers: num_envs={num_envs}, "
            f"history_len={history_len}, action_dim={action_dim}, joint_dim={joint_dim}"
        )
        print(
            f"[ResidualRLPolicy] Initialized BASE action buffer: exec_horizon={self.exec_horizon} "
            f"(DiT predicts, PPO corrects each step)"
        )

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action for storing in history buffer.

        CRITICAL: DiT expects normalized action_history (same as training data)!

        Args:
            action: [batch, action_dim] - Denormalized action

        Returns:
            Normalized action
        """
        if self.norm_action_mean is None or self.norm_action_std is None:
            return action

        mean = self.norm_action_mean.to(action.device)
        std = self.norm_action_std.to(action.device)

        return (action - mean) / std

    def _normalize_ee_node_history(self, ee_node_history: torch.Tensor) -> torch.Tensor:
        """Normalize EE node history for DiT input.

        CRITICAL: DiT was trained with normalized node features!

        Args:
            ee_node_history: [batch, history_len, 7] - Raw EE node features

        Returns:
            Normalized EE node history
        """
        if self.norm_ee_node_mean is None or self.norm_ee_node_std is None:
            return ee_node_history

        mean = self.norm_ee_node_mean.to(ee_node_history.device)
        std = self.norm_ee_node_std.to(ee_node_history.device)

        return (ee_node_history - mean) / std

    def _normalize_object_node_history(
        self, object_node_history: torch.Tensor
    ) -> torch.Tensor:
        """Normalize Object node history for DiT input.

        CRITICAL: DiT was trained with normalized node features!

        Args:
            object_node_history: [batch, history_len, 7] - Raw object node features

        Returns:
            Normalized object node history
        """
        if self.norm_object_node_mean is None or self.norm_object_node_std is None:
            return object_node_history

        mean = self.norm_object_node_mean.to(object_node_history.device)
        std = self.norm_object_node_std.to(object_node_history.device)

        return (object_node_history - mean) / std

    def _normalize_joint_states_history(
        self, joint_states_history: torch.Tensor
    ) -> torch.Tensor:
        """Normalize Joint states history for DiT input.

        CRITICAL: DiT was trained with normalized joint states!
        This is used for the State-Action Self-Attention in GraphDiTUnit.

        Args:
            joint_states_history: [batch, history_len, joint_dim] - Raw joint states (pos + vel)

        Returns:
            Normalized joint states history
        """
        if self.norm_joint_mean is None or self.norm_joint_std is None:
            return joint_states_history

        mean = self.norm_joint_mean.to(joint_states_history.device)
        std = self.norm_joint_std.to(joint_states_history.device)

        return (joint_states_history - mean) / std

    def _update_history_buffers(self, obs: torch.Tensor, action: torch.Tensor):
        """Update history buffers with new observation and action.

        CRITICAL: action_history must store NORMALIZED actions!
        DiT was trained with normalized action_history.

        Args:
            obs: Current observation [num_envs, obs_dim]
            action: Action just taken [num_envs, action_dim] (denormalized)
        """
        if self._action_history is None:
            return

        # Roll history (drop oldest, add new)
        self._action_history = torch.roll(self._action_history, shifts=-1, dims=1)
        # CRITICAL: Store NORMALIZED action in history buffer!
        action_normalized = self._normalize_action(action)
        self._action_history[:, -1, :] = action_normalized

        # Extract EE and Object node features from observation
        # Use obs_structure if available (more robust), otherwise fallback to hardcoded offsets
        robot_state_dim = self.cfg.robot_state_dim  # Usually 12 (joint_pos + joint_vel)

        # Extract joint_states (joint_pos + joint_vel) for State-Action Self-Attention
        # joint_states is the first robot_state_dim elements of obs
        joint_states = obs[:, :robot_state_dim]  # [num_envs, robot_state_dim]

        # Try to use obs_structure from Graph DiT config (more robust)
        if (
            hasattr(self.graph_dit_cfg, "obs_structure")
            and self.graph_dit_cfg.obs_structure is not None
        ):
            obs_struct = self.graph_dit_cfg.obs_structure
            # Extract using obs_structure dict
            if "object_position" in obs_struct:
                obj_start, obj_end = obs_struct["object_position"]
                obj_pos = obs[:, obj_start:obj_end]
            else:
                # Fallback: assume object_position starts at robot_state_dim
                obj_pos = obs[:, robot_state_dim : robot_state_dim + 3]

            if "object_orientation" in obs_struct:
                obj_ori_start, obj_ori_end = obs_struct["object_orientation"]
                obj_ori = obs[:, obj_ori_start:obj_ori_end]
            else:
                # Fallback: assume object_orientation starts at robot_state_dim + 3
                obj_ori = obs[:, robot_state_dim + 3 : robot_state_dim + 7]

            if "ee_position" in obs_struct:
                ee_pos_start, ee_pos_end = obs_struct["ee_position"]
                ee_pos = obs[:, ee_pos_start:ee_pos_end]
            else:
                # Fallback: assume ee_position starts at robot_state_dim + 7
                ee_pos = obs[:, robot_state_dim + 7 : robot_state_dim + 10]

            if "ee_orientation" in obs_struct:
                ee_ori_start, ee_ori_end = obs_struct["ee_orientation"]
                ee_ori = obs[:, ee_ori_start:ee_ori_end]
            else:
                # Fallback: assume ee_orientation starts at robot_state_dim + 10
                ee_ori = obs[:, robot_state_dim + 10 : robot_state_dim + 14]
        else:
            # Fallback: Hardcoded offsets (original behavior)
            # Assuming obs layout: [joint_pos(6), joint_vel(6), obj_pos(3), obj_ori(4), ee_pos(3), ee_ori(4), ...]
            obj_pos = obs[:, robot_state_dim : robot_state_dim + 3]
            obj_ori = obs[:, robot_state_dim + 3 : robot_state_dim + 7]
            ee_pos = obs[:, robot_state_dim + 7 : robot_state_dim + 10]
            ee_ori = obs[:, robot_state_dim + 10 : robot_state_dim + 14]

        object_node = torch.cat([obj_pos, obj_ori], dim=-1)  # [num_envs, 7]
        ee_node = torch.cat([ee_pos, ee_ori], dim=-1)  # [num_envs, 7]

        # Update node histories
        self._ee_node_history = torch.roll(self._ee_node_history, shifts=-1, dims=1)
        self._ee_node_history[:, -1, :] = ee_node

        self._object_node_history = torch.roll(
            self._object_node_history, shifts=-1, dims=1
        )
        self._object_node_history[:, -1, :] = object_node

        # Update joint_states_history (CRITICAL for State-Action Self-Attention!)
        # Note: Store RAW joint_states, normalization is done when passing to DiT
        self._joint_states_history = torch.roll(
            self._joint_states_history, shifts=-1, dims=1
        )
        self._joint_states_history[:, -1, :] = joint_states

        self._last_action = action

    def reset_history(self, env_ids: torch.Tensor | None = None):
        """Reset history buffers for specified environments.

        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        if self._action_history is None:
            return

        if env_ids is None:
            self._action_history.zero_()
            self._ee_node_history.zero_()
            self._object_node_history.zero_()
            self._joint_states_history.zero_()  # Reset joint_states_history too!
            self._last_action.zero_()
            # Reset action buffer - force DiT re-prediction on next step
            if self._base_action_buffer is not None:
                self._base_action_buffer.zero_()
                self._buffer_indices.fill_(self.exec_horizon)
        else:
            self._action_history[env_ids].zero_()
            self._ee_node_history[env_ids].zero_()
            self._object_node_history[env_ids].zero_()
            self._joint_states_history[
                env_ids
            ].zero_()  # Reset joint_states_history too!
            self._last_action[env_ids].zero_()
            # Reset action buffer for specific envs - force DiT re-prediction
            if self._base_action_buffer is not None:
                self._base_action_buffer[env_ids].zero_()
                self._buffer_indices[env_ids] = self.exec_horizon

    def _extract_robot_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract robot state (joint_pos, joint_vel) from observation.

        Args:
            obs: [batch, obs_dim] - Full observation

        Returns:
            robot_state: [batch, robot_state_dim] - Joint positions and velocities
        """
        # Default: first robot_state_dim elements are joint_pos + joint_vel
        return obs[:, : self.cfg.robot_state_dim]

    def _build_ppo_input(
        self,
        robot_state: torch.Tensor,
        features: dict[str, torch.Tensor],
        base_action_t: torch.Tensor,
    ) -> torch.Tensor:
        """Build input for PPO residual network.

        CRITICAL: PPO Input must include Base_Action_t for proper correction
        PPO Input = [Robot_State, Graph_Feature, Base_Action_t]
        - Robot_State: Current robot state (joint positions and velocities)
        - Graph_Feature: Scene understanding from Graph DiT (graph embedding)
        - Base_Action_t: Teacher's intended action for this step (from DiT buffer)

        Args:
            robot_state: [batch, robot_state_dim] - Joint positions and velocities
            features: Dict from graph_dit.extract_features()
            base_action_t: [batch, action_dim] - Current step's base action from DiT

        Returns:
            ppo_input: [batch, ppo_input_dim] - Concatenated features
        """
        components = [robot_state]

        if self.cfg.use_graph_embedding:
            graph_embed = features["graph_embedding"]
            # Ensure graph_embedding has correct shape [batch, hidden_dim]
            if len(graph_embed.shape) == 3:
                # [batch, seq, hidden] -> [batch, hidden] (take last or mean)
                graph_embed = (
                    graph_embed[:, -1, :]
                    if graph_embed.shape[1] > 0
                    else graph_embed.squeeze(1)
                )
            elif len(graph_embed.shape) == 1:
                # [hidden] -> [1, hidden]
                graph_embed = graph_embed.unsqueeze(0)
            components.append(graph_embed)

        if self.cfg.use_node_features:
            # Flatten node features: [batch, 2, hidden_dim] → [batch, 2*hidden_dim]
            node_feat = features["node_features"].flatten(start_dim=1)
            components.append(node_feat)

        if self.cfg.use_edge_features:
            components.append(features["edge_features"])

        # CRITICAL: Add base_action_t so PPO knows "what teacher wants this step"
        components.append(base_action_t)

        ppo_input = torch.cat(components, dim=-1)
        return ppo_input

    def forward(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for PPO training.

        Args:
            obs: Observations [batch, obs_dim]
            action_history: Action history [batch, history_length, action_dim] (optional)
            ee_node_history: EE node history [batch, history_length, 7] (optional)
            object_node_history: Object node history [batch, history_length, 7] (optional)
            subtask_condition: Subtask condition [batch, num_subtasks] (optional)

        Returns:
            actions: [batch, action_dim] - Final actions (base + residual)
            log_probs: [batch] - Log probabilities of residual
            values: [batch] - Value estimates
        """
        batch_size = obs.shape[0]

        # Normalize ALL inputs for Graph DiT
        obs_normalized = self._normalize_obs(obs)
        # Note: action_history should already be normalized if from internal buffer
        ee_node_history_normalized = (
            self._normalize_ee_node_history(ee_node_history)
            if ee_node_history is not None
            else None
        )
        object_node_history_normalized = (
            self._normalize_object_node_history(object_node_history)
            if object_node_history is not None
            else None
        )
        # Get joint_states_history from internal buffer and normalize
        joint_states_history_normalized = (
            self._normalize_joint_states_history(self._joint_states_history)
            if self._joint_states_history is not None
            else None
        )

        # 1. Get base action and features from Graph DiT (frozen)
        with torch.no_grad():
            # Extract features (scene understanding) - all inputs normalized, including joint_states_history
            features = self.graph_dit.extract_features(
                obs_normalized,
                action_history,
                ee_node_history_normalized,
                object_node_history_normalized,
                joint_states_history_normalized,
                subtask_condition,
            )

            # Get base action from diffusion - all inputs already normalized!
            # CRITICAL: Set normalize=False to avoid double normalization!
            base_action_normalized = self.graph_dit.get_base_action(
                obs_normalized,
                action_history,
                ee_node_history_normalized,
                object_node_history_normalized,
                joint_states_history_normalized,
                subtask_condition,
                normalize=False,  # ✅ CRITICAL: Inputs already normalized!
            )
            # Denormalize base action (get_base_action returns normalized when normalize=False)
            base_action = self._denormalize_action(base_action_normalized)

        # 2. Build PPO input: [Robot_State, Graph_Embedding, Base_Action_t]
        robot_state = self._extract_robot_state(obs)
        ppo_input = self._build_ppo_input(robot_state, features, base_action)

        # 3. PPO residual network: features → residual action
        residual_mean = self.residual_actor(ppo_input)  # [batch, action_dim]

        # Clamp residual scale
        scale = torch.clamp(self.residual_scale, 0.01, self.cfg.max_residual_scale)

        # 4. Sample residual action
        residual_std = torch.exp(self.log_std).expand(batch_size, -1)
        residual_dist = torch.distributions.Normal(residual_mean, residual_std)
        residual_action = residual_dist.sample()  # [batch, action_dim]
        log_probs = residual_dist.log_prob(residual_action).sum(dim=-1)  # [batch]

        # 5. Combine: final_action = base_action + scale * residual_action
        final_action = base_action + scale * residual_action

        # 6. Value network
        obs_for_value = obs
        if self.obs_normalizer is not None:
            obs_for_value = self.obs_normalizer(obs_for_value)
        values = self.value_network(obs_for_value).squeeze(-1)  # [batch]

        return final_action, log_probs, values

    def act(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get actions for rollout (PPO).

        Args:
            obs: Observations [batch, obs_dim]
            deterministic: If True, return mean actions (no sampling)

        Returns:
            actions: [batch, action_dim] - Final actions
            log_probs: [batch] - Log probabilities of residual
        """
        batch_size = obs.shape[0]

        # Normalize ALL inputs for Graph DiT
        obs_normalized = self._normalize_obs(obs)
        ee_node_history_normalized = (
            self._normalize_ee_node_history(ee_node_history)
            if ee_node_history is not None
            else None
        )
        object_node_history_normalized = (
            self._normalize_object_node_history(object_node_history)
            if object_node_history is not None
            else None
        )
        # Get joint_states_history from internal buffer and normalize
        joint_states_history_normalized = (
            self._normalize_joint_states_history(self._joint_states_history)
            if self._joint_states_history is not None
            else None
        )

        # 1. Get base action and features from Graph DiT
        with torch.no_grad():
            features = self.graph_dit.extract_features(
                obs_normalized,
                action_history,
                ee_node_history_normalized,
                object_node_history_normalized,
                joint_states_history_normalized,
                subtask_condition,
            )
            # Get base action from diffusion - all inputs already normalized!
            # CRITICAL: Set normalize=False to avoid double normalization!
            base_action_normalized = self.graph_dit.get_base_action(
                obs_normalized,
                action_history,
                ee_node_history_normalized,
                object_node_history_normalized,
                joint_states_history_normalized,
                subtask_condition,
                normalize=False,  # ✅ CRITICAL: Inputs already normalized!
            )
            # Denormalize base action (get_base_action returns normalized when normalize=False)
            base_action = self._denormalize_action(base_action_normalized)

        # Cache for later (in case evaluate is called)
        self._cached_base_action = base_action.detach()
        self._cached_features = {k: v.detach() for k, v in features.items()}

        # 2. Build PPO input with base_action
        robot_state = self._extract_robot_state(obs)
        ppo_input = self._build_ppo_input(robot_state, features, base_action)

        # 3. Get residual action
        residual_mean = self.residual_actor(ppo_input)
        scale = torch.clamp(self.residual_scale, 0.01, self.cfg.max_residual_scale)

        if deterministic:
            residual_action = residual_mean
            residual_std = torch.exp(self.log_std).expand(batch_size, -1)
            residual_dist = torch.distributions.Normal(residual_mean, residual_std)
            log_probs = residual_dist.log_prob(residual_action).sum(dim=-1)
        else:
            residual_std = torch.exp(self.log_std).expand(batch_size, -1)
            residual_dist = torch.distributions.Normal(residual_mean, residual_std)
            residual_action = residual_dist.sample()
            log_probs = residual_dist.log_prob(residual_action).sum(dim=-1)

        # 4. Combine
        final_action = base_action + scale * residual_action

        # [CRITICAL FIX] Update buffer indices after execution!
        # Without this, DiT would always predict step 0 or get stuck
        if self._buffer_indices is not None:
            self._buffer_indices += 1
            # Update history buffers
            self._update_history_buffers(obs, final_action.detach())

        return final_action, log_probs

    def update_distribution(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
    ):
        """
        Update action distribution (for RSL-RL compatibility).

        Correct Design: DiT predicts in chunks, PPO corrects step-by-step
        =========================================================
        DiT (Teacher): Predicts base_trajectory every exec_horizon steps, stores in buffer
        PPO (Student): Computes residual at every step based on current observation

        PPO Input = [Robot_State, Graph_Feature, Base_Action_t]
        - Robot_State: Current robot state
        - Graph_Feature: Scene understanding
        - Base_Action_t: Teacher's intended action for this step (from buffer)

        This provides true closed-loop control!
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Initialize history buffers if needed
        self._init_history_buffers(batch_size, device)

        # Use internal history buffers if not provided
        if action_history is None:
            action_history = self._action_history
        if ee_node_history is None:
            ee_node_history = self._ee_node_history
        if object_node_history is None:
            object_node_history = self._object_node_history
        # Get joint_states_history from internal buffer
        joint_states_history = self._joint_states_history

        # ============================================================
        # CRITICAL: Normalize ALL inputs for Graph DiT!
        # Graph DiT was trained with ALL data normalized:
        # - obs: normalized
        # - action_history: normalized (already stored normalized in buffer)
        # - ee_node_history: normalized
        # - object_node_history: normalized
        # - joint_states_history: normalized (CRITICAL for State-Action Self-Attention!)
        # ============================================================
        obs_normalized = self._normalize_obs(obs)
        # Note: action_history is already stored normalized in _update_history_buffers
        ee_node_history_normalized = self._normalize_ee_node_history(ee_node_history)
        object_node_history_normalized = self._normalize_object_node_history(
            object_node_history
        )
        joint_states_history_normalized = self._normalize_joint_states_history(
            joint_states_history
        )

        # ============================================================
        # STEP 1: DiT predicts in chunks - Check if we need DiT to predict new trajectory
        # ============================================================
        needs_replan = self._buffer_indices >= self.exec_horizon

        if needs_replan.any():
            # DiT predicts base trajectory (only when buffer exhausted)
            # CRITICAL: All inputs are already normalized manually above!
            # Must set normalize=False to avoid double normalization!
            with torch.no_grad():
                base_trajectory_normalized = self.graph_dit.predict(
                    obs_normalized,
                    action_history,
                    ee_node_history_normalized,
                    object_node_history_normalized,
                    joint_states_history_normalized,
                    subtask_condition,
                    deterministic=True,
                    normalize=False,  # ✅ CRITICAL: Inputs already normalized!
                )  # [batch, pred_horizon, action_dim] - normalized (not denormalized by predict)

            # Denormalize action trajectory for execution
            base_trajectory = self._denormalize_action(base_trajectory_normalized)

            # Store base trajectory in buffer (denormalized for execution!)
            self._base_action_buffer[needs_replan] = base_trajectory[
                needs_replan, : self.exec_horizon, :
            ]
            self._buffer_indices[needs_replan] = 0

        # ============================================================
        # STEP 2: Get current base_action_t from buffer
        # ============================================================
        env_indices = torch.arange(batch_size, device=device)
        base_action_t = self._base_action_buffer[
            env_indices, self._buffer_indices, :
        ]  # [batch, action_dim]

        # ============================================================
        # STEP 3: PPO corrects step-by-step - Compute residual based on CURRENT obs
        # Get features from Graph DiT (always compute for current obs)
        # NOTE: All inputs must be normalized, including joint_states_history!
        # ============================================================
        with torch.no_grad():
            features = self.graph_dit.extract_features(
                obs_normalized,
                action_history,
                ee_node_history_normalized,
                object_node_history_normalized,
                joint_states_history_normalized,
                subtask_condition,
            )

        # Cache features
        self._cached_features = {k: v.detach() for k, v in features.items()}

        # ============================================================
        # STEP 4: Build PPO input with base_action_t (CRITICAL for proper correction!)
        # PPO Input = [Robot_State, Graph_Feature, Base_Action_t]
        # ============================================================
        robot_state = self._extract_robot_state(obs)
        ppo_input = self._build_ppo_input(robot_state, features, base_action_t)

        # PPO outputs residual for THIS step (not entire trajectory)
        residual_mean = self.residual_actor(ppo_input)  # [batch, action_dim]

        # Compute residual std
        log_std = self.log_std.to(ppo_input.device)
        log_std_clamped = torch.clamp(log_std, min=-20.0, max=2.0)
        residual_std = torch.exp(log_std_clamped).expand(batch_size, -1)
        residual_std = torch.clamp(residual_std, min=1e-6)

        # Store distribution (for residual, not final action)
        from torch.distributions import Normal

        self.distribution = Normal(residual_mean, residual_std)

        # Store base_action_t for combining in act()
        self._base_action_for_dist = base_action_t

        # Debug: print dimensions and VALUES on first call
        if not hasattr(self, "_debug_printed"):
            print(
                f"[ResidualRLPolicy] === DiT predicts in chunks + PPO corrects step-by-step ==="
            )
            print(f"[ResidualRLPolicy] obs shape: {obs.shape}")
            print(f"[ResidualRLPolicy] obs_normalized shape: {obs_normalized.shape}")
            print(
                f"[ResidualRLPolicy] Normalization enabled: norm_obs_mean={self.norm_obs_mean is not None}, norm_action_mean={self.norm_action_mean is not None}"
            )

            # ============================================================
            # DEBUG: Check for double normalization!
            # If obs values are near 0 (already normalized by RSL-RL), we have a problem!
            # ============================================================
            print(
                f"[ResidualRLPolicy] DEBUG - Raw obs first 6 values: {obs[0, :6].tolist()}"
            )
            print(
                f"[ResidualRLPolicy] DEBUG - Raw obs stats: min={obs[0].min():.4f}, max={obs[0].max():.4f}, mean={obs[0].mean():.4f}"
            )
            if self.norm_obs_mean is not None:
                print(
                    f"[ResidualRLPolicy] DEBUG - Checkpoint norm_obs_mean first 6: {self.norm_obs_mean[:6].tolist()}"
                )
                print(
                    f"[ResidualRLPolicy] DEBUG - Checkpoint norm_obs_std first 6: {self.norm_obs_std[:6].tolist()}"
                )
            print(
                f"[ResidualRLPolicy] DEBUG - Normalized obs first 6 values: {obs_normalized[0, :6].tolist()}"
            )
            print(
                f"[ResidualRLPolicy] DEBUG - base_action_t first 6: {base_action_t[0].tolist()}"
            )

            print(f"[ResidualRLPolicy] robot_state: {robot_state.shape}")
            print(
                f"[ResidualRLPolicy] graph_embedding: {features['graph_embedding'].shape}"
            )
            print(
                f"[ResidualRLPolicy] base_action_t (denormalized): {base_action_t.shape}"
            )
            print(f"[ResidualRLPolicy] ppo_input (with base_action): {ppo_input.shape}")
            print(f"[ResidualRLPolicy] residual_mean: {residual_mean.shape}")
            print(
                f"[ResidualRLPolicy] DiT predicts every {self.exec_horizon} steps, PPO corrects EVERY step"
            )
            self._debug_printed = True

    @property
    def action_mean(self) -> torch.Tensor:
        """Get mean of final action distribution."""
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call update_distribution() first."
            )
        scale = torch.clamp(self.residual_scale, 0.01, self.cfg.max_residual_scale)
        return self._base_action_for_dist + scale * self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Get std of residual action distribution."""
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call update_distribution() first."
            )
        scale = torch.clamp(self.residual_scale, 0.01, self.cfg.max_residual_scale)
        return scale * self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Get entropy of residual distribution."""
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call update_distribution() first."
            )
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities of actions (for PPO).

        Note: This computes log_prob of the RESIDUAL, not the final action.
        """
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call update_distribution() first."
            )

        # Recover residual from final action
        scale = torch.clamp(self.residual_scale, 0.01, self.cfg.max_residual_scale)
        residual = (actions - self._base_action_for_dist) / scale

        return self.distribution.log_prob(residual).sum(dim=-1)

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for PPO training).

        Args:
            obs: Observations [batch, obs_dim]
            actions: Final actions to evaluate [batch, action_dim]

        Returns:
            log_probs: [batch] - Log probabilities
            values: [batch] - Value estimates
            entropy: [batch] - Entropy of distribution
        """
        batch_size = obs.shape[0]

        # Normalize ALL inputs for Graph DiT
        obs_normalized = self._normalize_obs(obs)
        ee_node_history_normalized = (
            self._normalize_ee_node_history(ee_node_history)
            if ee_node_history is not None
            else None
        )
        object_node_history_normalized = (
            self._normalize_object_node_history(object_node_history)
            if object_node_history is not None
            else None
        )
        # Get joint_states_history from internal buffer and normalize
        joint_states_history_normalized = (
            self._normalize_joint_states_history(self._joint_states_history)
            if self._joint_states_history is not None
            else None
        )

        # Get base action and features
        # NOTE: This is an approximation! During rollout, we used chunked base_action
        # from buffer, but RSL-RL doesn't let us store extra info. Here we re-predict
        # "what would DiT do now" which may differ slightly from the historical chunked
        # action. This is acceptable for Residual Learning (PPO learns to correct
        # "current DiT's intention" rather than "historical DiT's intention").
        # Ensure deterministic=True for consistency.
        with torch.no_grad():
            features = self.graph_dit.extract_features(
                obs_normalized,
                action_history,
                ee_node_history_normalized,
                object_node_history_normalized,
                joint_states_history_normalized,
                subtask_condition,
            )
            # Get base action from diffusion - all inputs already normalized!
            # CRITICAL: Set normalize=False to avoid double normalization!
            base_action_normalized = self.graph_dit.get_base_action(
                obs_normalized,
                action_history,
                ee_node_history_normalized,
                object_node_history_normalized,
                joint_states_history_normalized,
                subtask_condition,
                normalize=False,  # ✅ CRITICAL: Inputs already normalized!
            )  # deterministic=True by default
            # Denormalize base action (get_base_action returns normalized when normalize=False)
            base_action = self._denormalize_action(base_action_normalized)

        # Build PPO input with base_action
        robot_state = self._extract_robot_state(obs)
        ppo_input = self._build_ppo_input(robot_state, features, base_action)

        # Get residual distribution
        residual_mean = self.residual_actor(ppo_input)
        residual_std = torch.exp(self.log_std).expand(batch_size, -1)
        residual_dist = torch.distributions.Normal(residual_mean, residual_std)

        # Recover residual from final action
        scale = torch.clamp(self.residual_scale, 0.01, self.cfg.max_residual_scale)
        residual = (actions - base_action) / scale

        # Compute log prob and entropy of residual
        log_probs = residual_dist.log_prob(residual).sum(dim=-1)
        entropy = residual_dist.entropy().sum(dim=-1)

        # Value network
        obs_for_value = obs
        if self.obs_normalizer is not None:
            obs_for_value = self.obs_normalizer(obs_for_value)
        values = self.value_network(obs_for_value).squeeze(-1)

        return log_probs, values, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimates only."""
        obs_for_value = obs
        if self.obs_normalizer is not None:
            obs_for_value = self.obs_normalizer(obs_for_value)
        return self.value_network(obs_for_value).squeeze(-1)

    def save(self, path: str):
        """Save policy to file."""
        torch.save(
            {
                "policy_state_dict": self.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )
        print(f"[ResidualRLPolicy] Saved model to: {path}")

    @classmethod
    def load(cls, path: str, device: str = "cuda"):
        """Load policy from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint.get("cfg", None)
        if cfg is None:
            raise ValueError(f"No config found in checkpoint: {path}")

        policy = cls(cfg)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.to(device)
        policy.eval()

        print(f"[ResidualRLPolicy] Loaded model from: {path}")
        return policy

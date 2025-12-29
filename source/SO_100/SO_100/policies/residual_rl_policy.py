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
    residual_scale: float = 0.1
    """Initial scale for residual actions (start small for stability)."""

    max_residual_scale: float = 0.5
    """Maximum scale for residual actions."""

    # RL training parameters
    init_noise_std: float = 0.3
    """Initial standard deviation for action noise (PPO)."""

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
            print(f"[ResidualRLPolicy] Loading pre-trained Graph DiT from: {cfg.pretrained_checkpoint}")
            self.graph_dit = GraphDiTPolicy.load(cfg.pretrained_checkpoint, device="cpu")
            # CRITICAL: Use the config from loaded checkpoint, not from cfg!
            self.graph_dit_cfg = self.graph_dit.cfg
            print(f"[ResidualRLPolicy] Loaded Graph DiT config: hidden_dim={self.graph_dit_cfg.hidden_dim}, "
                  f"action_dim={self.graph_dit_cfg.action_dim}, obs_dim={self.graph_dit_cfg.obs_dim}")
        else:
            # Fallback: use provided config
            print("[ResidualRLPolicy] Warning: No pretrained_checkpoint specified. Using provided config.")
            graph_dit_cfg = cfg.graph_dit_cfg
            if isinstance(graph_dit_cfg, dict):
                from .graph_dit_policy import GraphDiTPolicyCfg
                graph_dit_cfg = GraphDiTPolicyCfg(**graph_dit_cfg)
            self.graph_dit_cfg = graph_dit_cfg
            self.graph_dit = GraphDiTPolicy(graph_dit_cfg)

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
        print(f"[ResidualRLPolicy] PPO input dim: {ppo_input_dim} "
              f"(robot_state={cfg.robot_state_dim}, graph_embed={hidden_dim if cfg.use_graph_embedding else 0})")

        # Build activation function
        activation_fn = self._get_activation(cfg.residual_activation)

        # Residual Actor Network: [Robot_State, Graph_Embedding] → Residual Action
        # Use action_dim from loaded model config
        action_dim = self.graph_dit_cfg.action_dim
        actor_layers = []
        input_dim = ppo_input_dim
        for hidden_dim_layer in cfg.residual_hidden_dims:
            actor_layers.append(nn.Linear(input_dim, hidden_dim_layer))
            actor_layers.append(nn.LayerNorm(hidden_dim_layer))
            actor_layers.append(activation_fn())
            input_dim = hidden_dim_layer
        actor_layers.append(nn.Linear(input_dim, action_dim))  # Output: residual mean

        self.residual_actor = nn.Sequential(*actor_layers)

        # Log std for action noise (learnable)
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
        obs_dim = cfg.value_obs_dim if cfg.value_obs_dim is not None else self.graph_dit_cfg.obs_dim
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

        # Initialize weights
        self._init_weights()

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
                nn.init.orthogonal_(module.weight, gain=0.01)  # Small gain for stability
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize value network
        for module in self.value_network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _extract_robot_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract robot state (joint_pos, joint_vel) from observation.

        Args:
            obs: [batch, obs_dim] - Full observation

        Returns:
            robot_state: [batch, robot_state_dim] - Joint positions and velocities
        """
        # Default: first robot_state_dim elements are joint_pos + joint_vel
        return obs[:, :self.cfg.robot_state_dim]

    def _build_ppo_input(
        self,
        robot_state: torch.Tensor,
        features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build input for PPO residual network.

        Args:
            robot_state: [batch, robot_state_dim] - Joint positions and velocities
            features: Dict from graph_dit.extract_features()

        Returns:
            ppo_input: [batch, ppo_input_dim] - Concatenated features
        """
        components = [robot_state]

        if self.cfg.use_graph_embedding:
            graph_embed = features['graph_embedding']
            # Ensure graph_embedding has correct shape [batch, hidden_dim]
            if len(graph_embed.shape) == 3:
                # [batch, seq, hidden] -> [batch, hidden] (take last or mean)
                graph_embed = graph_embed[:, -1, :] if graph_embed.shape[1] > 0 else graph_embed.squeeze(1)
            elif len(graph_embed.shape) == 1:
                # [hidden] -> [1, hidden]
                graph_embed = graph_embed.unsqueeze(0)
            components.append(graph_embed)

        if self.cfg.use_node_features:
            # Flatten node features: [batch, 2, hidden_dim] → [batch, 2*hidden_dim]
            node_feat = features['node_features'].flatten(start_dim=1)
            components.append(node_feat)

        if self.cfg.use_edge_features:
            components.append(features['edge_features'])

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

        # 1. Get base action and features from Graph DiT (frozen)
        with torch.no_grad():
            # Extract features (scene understanding)
            features = self.graph_dit.extract_features(
                obs, action_history, ee_node_history,
                object_node_history, None, subtask_condition
            )

            # Get base action from diffusion
            base_action = self.graph_dit.get_base_action(
                obs, action_history, ee_node_history,
                object_node_history, None, subtask_condition
            )

        # 2. Build PPO input: [Robot_State, Graph_Embedding]
        robot_state = self._extract_robot_state(obs)
        ppo_input = self._build_ppo_input(robot_state, features)

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

        # 1. Get base action and features from Graph DiT
        with torch.no_grad():
            features = self.graph_dit.extract_features(
                obs, action_history, ee_node_history,
                object_node_history, None, subtask_condition
            )
            base_action = self.graph_dit.get_base_action(
                obs, action_history, ee_node_history,
                object_node_history, None, subtask_condition
            )

        # Cache for later (in case evaluate is called)
        self._cached_base_action = base_action.detach()
        self._cached_features = {k: v.detach() for k, v in features.items()}

        # 2. Build PPO input
        robot_state = self._extract_robot_state(obs)
        ppo_input = self._build_ppo_input(robot_state, features)

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

        This stores the distribution so that action_mean, action_std,
        and other properties can be accessed.
        """
        batch_size = obs.shape[0]

        # Get features from Graph DiT
        with torch.no_grad():
            features = self.graph_dit.extract_features(
                obs, action_history, ee_node_history,
                object_node_history, None, subtask_condition
            )
            base_action = self.graph_dit.get_base_action(
                obs, action_history, ee_node_history,
                object_node_history, None, subtask_condition
            )

        # Cache
        self._cached_base_action = base_action.detach()
        self._cached_features = {k: v.detach() for k, v in features.items()}

        # Build PPO input
        robot_state = self._extract_robot_state(obs)
        ppo_input = self._build_ppo_input(robot_state, features)

        # Debug: print dimensions on first call
        if not hasattr(self, '_debug_printed'):
            print(f"[ResidualRLPolicy] obs shape: {obs.shape}")
            print(f"[ResidualRLPolicy] robot_state shape: {robot_state.shape}")
            print(f"[ResidualRLPolicy] graph_embedding shape: {features['graph_embedding'].shape}")
            print(f"[ResidualRLPolicy] ppo_input shape: {ppo_input.shape}")
            print(f"[ResidualRLPolicy] Expected ppo_input_dim: {self.ppo_input_dim}")
            self._debug_printed = True

        # Get residual distribution
        residual_mean = self.residual_actor(ppo_input)
        
        # Ensure log_std is on the correct device and compute std
        log_std = self.log_std.to(ppo_input.device)
        # Clamp log_std for numerical stability
        log_std_clamped = torch.clamp(log_std, min=-20.0, max=2.0)
        residual_std = torch.exp(log_std_clamped).expand(batch_size, -1)
        
        # Ensure std is positive (add small epsilon for safety)
        residual_std = torch.clamp(residual_std, min=1e-6)

        # Store distribution
        from torch.distributions import Normal
        self.distribution = Normal(residual_mean, residual_std)

        # Store base action for combining later
        self._base_action_for_dist = base_action

    @property
    def action_mean(self) -> torch.Tensor:
        """Get mean of final action distribution."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        scale = torch.clamp(self.residual_scale, 0.01, self.cfg.max_residual_scale)
        return self._base_action_for_dist + scale * self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Get std of residual action distribution."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        scale = torch.clamp(self.residual_scale, 0.01, self.cfg.max_residual_scale)
        return scale * self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Get entropy of residual distribution."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities of actions (for PPO).

        Note: This computes log_prob of the RESIDUAL, not the final action.
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")

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

        # Get base action and features
        with torch.no_grad():
            features = self.graph_dit.extract_features(
                obs, action_history, ee_node_history,
                object_node_history, None, subtask_condition
            )
            base_action = self.graph_dit.get_base_action(
                obs, action_history, ee_node_history,
                object_node_history, None, subtask_condition
            )

        # Build PPO input
        robot_state = self._extract_robot_state(obs)
        ppo_input = self._build_ppo_input(robot_state, features)

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
        torch.save({
            "policy_state_dict": self.state_dict(),
            "cfg": self.cfg,
        }, path)
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

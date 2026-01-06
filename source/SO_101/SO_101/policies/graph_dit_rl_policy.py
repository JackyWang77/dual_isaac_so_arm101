# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Graph DiT RL Policy - Head-only fine-tuning for RL.

This module implements a RL policy based on pre-trained Graph DiT.
The Graph DiT backbone is frozen, and only a small RL head is trained.

Architecture:
    Frozen Graph DiT Backbone
        ↓ (extract features)
    Trainable RL Action Head (MLP)
        ↓
    Actions (with log probabilities for PPO)

    Independent Value Network (from obs)
        ↓
    Value estimates
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
class GraphDiTRLPolicyCfg:
    """Configuration for Graph DiT RL Policy (head-only fine-tuning)."""

    # Graph DiT backbone config (should match pre-trained model)
    graph_dit_cfg: GraphDiTPolicyCfg = MISSING
    """Configuration for the frozen Graph DiT backbone."""

    # RL Head configuration
    rl_head_hidden_dims: list[int] = [128, 64]
    """Hidden dimensions for RL action head."""

    rl_head_activation: str = "elu"
    """Activation function for RL head."""

    # Value network configuration
    value_hidden_dims: list[int] = [256, 128, 64]
    """Hidden dimensions for value network."""

    value_activation: str = "elu"
    """Activation function for value network."""

    value_obs_dim: int | None = None
    """Input dimension for value network. If None, uses graph_dit_cfg.obs_dim."""

    # RL training parameters
    init_noise_std: float = 0.5
    """Initial standard deviation for action noise (PPO)."""

    # Feature extraction mode
    feature_extraction_mode: str = "last_embedding"
    """
    How to extract features from Graph DiT:
    - "last_embedding": Use the final embedding before noise_head (recommended)
    - "noise_pred": Use the noise prediction output
    - "intermediate": Use intermediate layer features
    """

    # Graph DiT checkpoint path (for loading pre-trained weights)
    pretrained_checkpoint: str | None = None
    """Path to pre-trained Graph DiT checkpoint. If None, backbone is randomly initialized."""

    # Freeze settings
    freeze_backbone: bool = True
    """Whether to freeze the Graph DiT backbone."""


class GraphDiTRLPolicy(nn.Module):
    """Graph DiT RL Policy with head-only fine-tuning.

    This policy uses a frozen Graph DiT backbone as a feature extractor
    and trains only a small RL head on top for PPO training.

    Architecture:
        1. Frozen Graph DiT backbone (feature extraction)
        2. Trainable RL Action Head (maps features → actions)
        3. Independent Value Network (maps obs → values)

    For PPO training, it provides:
        - Actions with log probabilities
        - Value estimates
    """

    def __init__(self, cfg: GraphDiTRLPolicyCfg):
        super().__init__()
        self.cfg = cfg

        # Handle case where graph_dit_cfg might be a dict (from serialization)
        graph_dit_cfg = cfg.graph_dit_cfg
        if isinstance(graph_dit_cfg, dict):
            # Convert dict back to GraphDiTPolicyCfg
            from .graph_dit_policy import GraphDiTPolicyCfg

            graph_dit_cfg = GraphDiTPolicyCfg(**graph_dit_cfg)

        # Store the converted config for later use
        self.graph_dit_cfg = graph_dit_cfg

        # Load pre-trained Graph DiT backbone
        if cfg.pretrained_checkpoint is not None:
            print(
                f"[GraphDiTRLPolicy] Loading pre-trained Graph DiT from: {cfg.pretrained_checkpoint}"
            )
            self.graph_dit = GraphDiTPolicy.load(
                cfg.pretrained_checkpoint, device="cpu", weights_only=False
            )
            # Move to same device as this module later
        else:
            # When pretrained_checkpoint is None, weights will be loaded from RL checkpoint during runner.load()
            # This is normal during playback or when resuming RL training
            self.graph_dit = GraphDiTPolicy(graph_dit_cfg)

        # Freeze Graph DiT backbone
        if cfg.freeze_backbone:
            for param in self.graph_dit.parameters():
                param.requires_grad = False
            self.graph_dit.eval()  # Set to eval mode
            if cfg.pretrained_checkpoint is not None:
                print("[GraphDiTRLPolicy] Graph DiT backbone frozen")

        # Determine feature dimension
        hidden_dim = graph_dit_cfg.hidden_dim

        # Build activation function
        activation_fn = self._get_activation(cfg.rl_head_activation)

        # RL Action Head (maps Graph DiT features → actions)
        action_head_layers = []
        input_dim = hidden_dim  # Start with Graph DiT hidden dim
        for hidden_dim_head in cfg.rl_head_hidden_dims:
            action_head_layers.append(nn.Linear(input_dim, hidden_dim_head))
            action_head_layers.append(nn.LayerNorm(hidden_dim_head))
            action_head_layers.append(activation_fn())
            input_dim = hidden_dim_head

        # Output layer (mean and std for action distribution)
        action_dim = graph_dit_cfg.action_dim
        action_head_layers.append(nn.Linear(input_dim, action_dim))  # Mean
        action_head_layers.append(nn.Linear(input_dim, action_dim))  # Std (log_std)

        self.rl_action_head = nn.ModuleList(action_head_layers)
        self.action_mean_head_idx = len(action_head_layers) - 2
        self.action_std_head_idx = len(action_head_layers) - 1

        # Initialize action std
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * math.log(cfg.init_noise_std), requires_grad=True
        )

        # Value Network (independent, maps obs → value)
        value_activation_fn = self._get_activation(cfg.value_activation)
        value_layers = []
        # Use value_obs_dim if provided, otherwise use graph_dit_cfg.obs_dim
        obs_dim = (
            cfg.value_obs_dim
            if cfg.value_obs_dim is not None
            else graph_dit_cfg.obs_dim
        )
        input_dim = obs_dim
        for hidden_dim_value in cfg.value_hidden_dims:
            value_layers.append(nn.Linear(input_dim, hidden_dim_value))
            value_layers.append(nn.LayerNorm(hidden_dim_value))
            value_layers.append(value_activation_fn())
            input_dim = hidden_dim_value
        value_layers.append(nn.Linear(input_dim, 1))  # Single value output

        self.value_network = nn.Sequential(*value_layers)

        # Observation normalizer (optional, for value network)
        self.obs_normalizer = None  # Can be set externally

        # Distribution (for RSL-RL compatibility, set by update_distribution)
        self.distribution = None

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
        # Initialize RL head
        for module in self.rl_action_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize value network
        for module in self.value_network:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _extract_features(
        self, obs: torch.Tensor, subtask_condition: torch.Tensor | None = None
    ):
        """
        Extract features from Graph DiT backbone.

        Args:
            obs: Observations [batch, obs_dim] - actual obs from environment (may be 39-dim)
            subtask_condition: Subtask condition [batch, num_subtasks] (optional)

        Returns:
            features: [batch, hidden_dim] - Features extracted from Graph DiT
        """
        # Set Graph DiT to eval mode if frozen (for consistent behavior)
        if self.cfg.freeze_backbone:
            self.graph_dit.eval()

        # IMPORTANT: Graph DiT was trained on 32-dim obs (without target_object_position command)
        # After removing commands, environment now returns 30-dim obs (5 joints instead of 6)
        # Obs structure (with commands, 39-dim): [joint_pos(6), joint_vel(6), object_pos(3), object_ori(4),
        #                                          ee_pos(3), ee_ori(4), target_object_position(7), actions(6)]
        # Obs structure (without commands, 30-dim): [joint_pos(5), joint_vel(5), object_pos(3), object_ori(4),
        #                                            ee_pos(3), ee_ori(4), actions(6)]
        # Target (32-dim): [joint_pos(6), joint_vel(6), object_pos(3), object_ori(4),
        #                    ee_pos(3), ee_ori(4), actions(6)]
        obs_dim = obs.shape[-1]
        expected_graph_dit_obs_dim = self.graph_dit_cfg.obs_dim

        if obs_dim != expected_graph_dit_obs_dim:
            if obs_dim == 39 and expected_graph_dit_obs_dim == 32:
                # Remove target_object_position (7 dims at indices 26:33)
                obs_for_graph_dit = torch.cat([obs[:, :26], obs[:, 33:]], dim=-1)
            elif obs_dim == 30 and expected_graph_dit_obs_dim == 32:
                # Current obs: [joint_pos(5), joint_vel(5), object_pos(3), object_ori(4),
                #               ee_pos(3), ee_ori(4), actions(6)] = 30 dims
                # Need to pad with zeros to match 32 dims (add 1 dim to joint_pos and joint_vel)
                # Insert zeros after joint_pos (index 5) and joint_vel (index 10)
                joint_pos = obs[:, :5]  # 5 dims
                joint_vel = obs[:, 5:10]  # 5 dims
                rest = obs[:, 10:]  # 20 dims (3+4+3+4+6)
                # Pad joint_pos and joint_vel to 6 dims each
                joint_pos_padded = torch.cat(
                    [joint_pos, torch.zeros_like(joint_pos[:, :1])], dim=-1
                )  # 6 dims
                joint_vel_padded = torch.cat(
                    [joint_vel, torch.zeros_like(joint_vel[:, :1])], dim=-1
                )  # 6 dims
                obs_for_graph_dit = torch.cat(
                    [joint_pos_padded, joint_vel_padded, rest], dim=-1
                )  # 32 dims
            else:
                raise ValueError(
                    f"Observation dimension mismatch: got {obs_dim}, "
                    f"expected {expected_graph_dit_obs_dim} for Graph DiT. "
                    f"Please check observation structure. "
                    f"Supported conversions: 39->32 (remove target_object_position), 30->32 (pad joints)."
                )
        else:
            obs_for_graph_dit = obs

        # Extract node features using filtered observations
        ee_node, object_node = self.graph_dit._extract_node_features(obs_for_graph_dit)

        # Compute edge features
        edge_features_raw = self.graph_dit._compute_edge_features(ee_node, object_node)

        # Embed nodes and edges
        ee_node_embed = self.graph_dit.node_embedding(ee_node)
        object_node_embed = self.graph_dit.node_embedding(object_node)
        node_features = torch.stack([ee_node_embed, object_node_embed], dim=1)

        edge_features_embed = self.graph_dit.edge_embedding(edge_features_raw)

        # Extract last action from obs_for_graph_dit for action embedding
        action_input = obs_for_graph_dit[:, -self.graph_dit_cfg.action_dim :]
        action_embed = self.graph_dit.action_embedding(action_input)
        action_embed = action_embed.unsqueeze(1)

        # Process through Graph DiT units (frozen if specified)
        context_manager = (
            torch.no_grad() if self.cfg.freeze_backbone else torch.enable_grad()
        )
        with context_manager:
            noise_embed = action_embed.squeeze(1)  # Start with action embed
            for unit in self.graph_dit.graph_dit_units:
                noise_embed = unit(
                    action_embed, node_features, edge_features_embed, None
                )
                action_embed = noise_embed.unsqueeze(1)

        # Return the final embedding before noise_head (this is what RL head uses)
        features = noise_embed  # [batch, hidden_dim]

        return features

    def forward(
        self,
        obs: torch.Tensor,
        subtask_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for PPO training.

        Args:
            obs: Observations [batch, obs_dim] - may be 39-dim (with target_object_position)
            subtask_condition: Subtask condition [batch, num_subtasks] (optional)

        Returns:
            actions: [batch, action_dim] - Sampled actions
            log_probs: [batch] - Log probabilities of actions
            values: [batch] - Value estimates
        """
        # Extract features from Graph DiT (handles dimension filtering internally)
        features = self._extract_features(obs, subtask_condition)  # [batch, hidden_dim]

        # RL Action Head: features → action mean and std
        x = features
        for i, layer in enumerate(self.rl_action_head):
            if i == self.action_mean_head_idx:
                action_mean = layer(x)
            elif i == self.action_std_head_idx:
                action_log_std = layer(x)
            else:
                x = layer(x)

        # Clamp log_std for numerical stability
        action_log_std = torch.clamp(action_log_std, min=-20, max=2)
        action_std = torch.exp(action_log_std)

        # Sample actions
        action_dist = torch.distributions.Normal(action_mean, action_std)
        actions = action_dist.sample()

        # Compute log probabilities
        log_probs = action_dist.log_prob(actions).sum(dim=-1)

        # Value network: obs → value
        obs_for_value = obs
        if self.obs_normalizer is not None:
            obs_for_value = self.obs_normalizer.normalize(obs_for_value)

        values = self.value_network(obs_for_value).squeeze(-1)  # [batch]

        return actions, log_probs, values

    def act(
        self,
        obs: torch.Tensor,
        subtask_condition: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get actions for rollout (PPO).

        Args:
            obs: Observations [batch, obs_dim] - may be 39-dim (with target_object_position)
            subtask_condition: Subtask condition [batch, num_subtasks] (optional)
            deterministic: If True, return mean actions (no sampling)

        Returns:
            actions: [batch, action_dim] - Actions
            log_probs: [batch] - Log probabilities
        """
        # Extract features from Graph DiT (handles dimension filtering internally)
        features = self._extract_features(obs, subtask_condition)

        # RL Action Head
        x = features
        for i, layer in enumerate(self.rl_action_head):
            if i == self.action_mean_head_idx:
                action_mean = layer(x)
            elif i == self.action_std_head_idx:
                action_log_std = layer(x)
            else:
                x = layer(x)

        action_log_std = torch.clamp(action_log_std, min=-20, max=2)
        action_std = torch.exp(action_log_std)

        if deterministic:
            actions = action_mean
            # For deterministic, we still need log_prob for PPO
            action_dist = torch.distributions.Normal(action_mean, action_std)
            log_probs = action_dist.log_prob(actions).sum(dim=-1)
        else:
            action_dist = torch.distributions.Normal(action_mean, action_std)
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions).sum(dim=-1)

        return actions, log_probs

    def update_distribution(
        self, obs: torch.Tensor, subtask_condition: torch.Tensor | None = None
    ):
        """
        Update action distribution based on observations.

        This method computes and stores the action distribution for RSL-RL compatibility.
        The distribution is stored in self.distribution and can be accessed via
        action_mean, action_std, etc.

        Args:
            obs: Observations [batch, obs_dim] - may be 39-dim (with target_object_position)
            subtask_condition: Subtask condition [batch, num_subtasks] (optional)
        """
        # Extract features from Graph DiT (handles dimension filtering internally)
        features = self._extract_features(obs, subtask_condition)  # [batch, hidden_dim]

        # RL Action Head: features → action mean and std
        x = features
        action_mean = None
        action_log_std = None
        for i, layer in enumerate(self.rl_action_head):
            if i == self.action_mean_head_idx:
                action_mean = layer(x)
            elif i == self.action_std_head_idx:
                action_log_std = layer(x)
            else:
                x = layer(x)

        # Clamp log_std for numerical stability
        action_log_std = torch.clamp(action_log_std, min=-20, max=2)
        action_std = torch.exp(action_log_std)

        # Create and store distribution
        from torch.distributions import Normal

        self.distribution = Normal(action_mean, action_std)

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        subtask_condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for PPO training).

        Args:
            obs: Observations [batch, obs_dim] - may be 39-dim (with target_object_position)
            actions: Actions to evaluate [batch, action_dim]
            subtask_condition: Subtask condition [batch, num_subtasks] (optional)

        Returns:
            log_probs: [batch] - Log probabilities of given actions
            values: [batch] - Value estimates
            entropy: [batch] - Entropy of action distribution
        """
        # Extract features from Graph DiT (handles dimension filtering internally)
        features = self._extract_features(obs, subtask_condition)

        # RL Action Head
        x = features
        for i, layer in enumerate(self.rl_action_head):
            if i == self.action_mean_head_idx:
                action_mean = layer(x)
            elif i == self.action_std_head_idx:
                action_log_std = layer(x)
            else:
                x = layer(x)

        action_log_std = torch.clamp(action_log_std, min=-20, max=2)
        action_std = torch.exp(action_log_std)

        # Evaluate given actions
        action_dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = action_dist.log_prob(actions).sum(dim=-1)
        entropy = action_dist.entropy().sum(dim=-1)

        # Value network
        obs_for_value = obs
        if self.obs_normalizer is not None:
            obs_for_value = self.obs_normalizer.normalize(obs_for_value)

        values = self.value_network(obs_for_value).squeeze(-1)

        return log_probs, values, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimates only.

        Args:
            obs: Observations [batch, obs_dim]

        Returns:
            values: [batch] - Value estimates
        """
        # Ensure obs has correct shape
        if len(obs.shape) != 2:
            raise ValueError(f"Expected obs shape [batch, obs_dim], got {obs.shape}")

        obs_for_value = obs
        if self.obs_normalizer is not None:
            obs_for_value = self.obs_normalizer.normalize(obs_for_value)

        values = self.value_network(obs_for_value)  # [batch, 1]

        # Ensure values have shape [batch]
        values = values.squeeze(-1)  # [batch]

        # Final check: ensure it's 1D
        if len(values.shape) != 1:
            raise ValueError(
                f"Expected values shape [batch], got {values.shape} after squeeze"
            )

        return values

    def save(self, path: str):
        """Save policy to file.

        Args:
            path: Path to save the model.
        """
        torch.save(
            {
                "policy_state_dict": self.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )
        print(f"[GraphDiTRLPolicy] Saved model to: {path}")

    @classmethod
    def load(cls, path: str, device: str = "cuda", weights_only: bool = False):
        """Load policy from file.

        Args:
            path: Path to load the model from.
            device: Device to load the model on.
            weights_only: If True, only load weights (PyTorch 2.6+ compatibility).

        Returns:
            GraphDiTRLPolicy: Loaded policy.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=weights_only)

        cfg = checkpoint.get("cfg", None)
        if cfg is None:
            raise ValueError(f"No config found in checkpoint: {path}")

        policy = cls(cfg)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.to(device)
        policy.eval()  # Set to eval mode for inference (can change to train() if needed)

        print(f"[GraphDiTRLPolicy] Loaded model from: {path}")
        return policy

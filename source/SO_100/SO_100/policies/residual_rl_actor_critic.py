# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL ActorCritic wrapper for Residual RL Policy.

This module provides a RSL-RL compatible ActorCritic class that wraps
ResidualRLPolicy, allowing it to be used with RSL-RL's OnPolicyRunner
for efficient multi-environment training.

Key features:
- Inherits from RSL-RL's ActorCritic for full compatibility
- Uses ResidualRLPolicy internally for action computation
- Supports multi-environment parallel training
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

# RSL-RL imports
try:
    from rsl_rl.modules import ActorCritic
except (ImportError, ModuleNotFoundError) as e:
    ActorCritic = None
    _actor_critic_error = e
else:
    _actor_critic_error = None

# Config classes from isaaclab_rl wrapper
try:
    from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
except (ImportError, ModuleNotFoundError) as e:
    RslRlPpoActorCriticCfg = None
    _config_error = e
else:
    _config_error = None

from .residual_rl_policy import ResidualRLPolicy, ResidualRLPolicyCfg

# Validate imports
if ActorCritic is None:
    raise ImportError(
        f"RSL-RL ActorCritic not available. Error: {_actor_critic_error}"
    ) from _actor_critic_error

if RslRlPpoActorCriticCfg is None:
    raise ImportError(
        f"RSL-RL config classes not available. Error: {_config_error}"
    ) from _config_error


class ResidualActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for Residual ActorCritic (RSL-RL compatible).

    This config extends RslRlPpoActorCriticCfg and adds Residual RL specific settings.
    """

    def __init__(self, residual_rl_cfg: ResidualRLPolicyCfg | None = None, **kwargs):
        """Initialize Residual ActorCritic config.

        Args:
            residual_rl_cfg: Configuration for Residual RL Policy.
            **kwargs: Other arguments passed to RslRlPpoActorCriticCfg.
        """
        if residual_rl_cfg is None and 'residual_rl_cfg' in kwargs:
            residual_rl_cfg = kwargs.pop('residual_rl_cfg')

        if 'class_name' not in kwargs:
            kwargs['class_name'] = "SO_100.policies.residual_rl_actor_critic.ResidualActorCritic"

        super().__init__(**kwargs)

        if residual_rl_cfg is not None:
            self.residual_rl_cfg = residual_rl_cfg
        else:
            self.residual_rl_cfg = ResidualRLPolicyCfg()


class ResidualActorCritic(ActorCritic):
    """RSL-RL compatible ActorCritic wrapper for Residual RL Policy.

    This class wraps ResidualRLPolicy to make it compatible with RSL-RL's
    OnPolicyRunner, enabling efficient multi-environment training.
    """

    def __init__(
        self,
        obs: dict[str, Any],
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = None,
        critic_hidden_dims: list[int] = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs
    ):
        """Initialize Residual ActorCritic.

        Args:
            obs: Observation space dictionary from environment.
            obs_groups: Observation groups dictionary.
            num_actions: Number of actions (action dimension).
            **kwargs: Must include 'residual_rl_cfg' (ResidualRLPolicyCfg).
        """
        # Extract residual_rl_cfg from kwargs BEFORE calling super().__init__()
        if "residual_rl_cfg" not in kwargs:
            raise ValueError(
                "ResidualActorCritic requires 'residual_rl_cfg' in kwargs."
            )
        residual_rl_cfg_dict = kwargs.pop("residual_rl_cfg")

        # Convert dict back to ResidualRLPolicyCfg if needed
        if isinstance(residual_rl_cfg_dict, dict):
            residual_rl_cfg = ResidualRLPolicyCfg(**residual_rl_cfg_dict)
        else:
            residual_rl_cfg = residual_rl_cfg_dict

        # Sanitize parameters for base class
        if not isinstance(activation, str):
            activation = "elu"
        if not isinstance(noise_std_type, str):
            noise_std_type = "scalar"
        if not isinstance(init_noise_std, (int, float)):
            init_noise_std = 1.0
        if actor_hidden_dims is None or not isinstance(actor_hidden_dims, list):
            actor_hidden_dims = [256, 256, 256]
        if critic_hidden_dims is None or not isinstance(critic_hidden_dims, list):
            critic_hidden_dims = [256, 256, 256]

        # Initialize base class
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        # Store config
        self.residual_rl_cfg = residual_rl_cfg
        self.num_actions = num_actions
        self.obs_groups = obs_groups

        # Get actual observation dimension
        actual_obs_dim = None
        if 'policy' in obs:
            obs_sample = obs['policy']
            if isinstance(obs_sample, dict):
                actual_obs_dim = sum(
                    v.flatten().shape[0] if hasattr(v, 'shape') else len(v)
                    for v in obs_sample.values()
                )
            else:
                actual_obs_dim = obs_sample.shape[-1] if len(obs_sample.shape) > 1 else obs_sample.shape[0]
        else:
            total_dim = 0
            for v in obs.values():
                if isinstance(v, dict):
                    total_dim += sum(
                        vi.flatten().shape[0] if hasattr(vi, 'shape') else len(vi)
                        for vi in v.values()
                    )
                else:
                    total_dim += v.shape[-1] if len(v.shape) > 1 else v.shape[0]
            actual_obs_dim = total_dim

        # Update value_obs_dim if needed
        if actual_obs_dim is not None:
            residual_rl_cfg.value_obs_dim = actual_obs_dim
            print(f"[ResidualActorCritic] Set value_obs_dim to {actual_obs_dim}")

        # Create underlying Residual RL Policy
        self.policy = ResidualRLPolicy(residual_rl_cfg)

        # Initialize normalizers
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()
        self.policy.obs_normalizer = self.actor_obs_normalizer

        self.action_dim = num_actions

    def to(self, device):
        """Move to device."""
        result = super().to(device)
        if hasattr(self, 'policy') and self.policy is not None:
            self.policy.to(device)
        return result

    def get_actor_obs(self, obs):
        """Extract actor observations from observation dictionary."""
        if hasattr(obs, 'keys') and hasattr(obs, 'get'):
            if hasattr(self, 'obs_groups') and 'policy' in self.obs_groups:
                obs_list = []
                for obs_group in self.obs_groups["policy"]:
                    if obs_group in obs:
                        val = obs[obs_group]
                        if hasattr(val, 'to') or isinstance(val, torch.Tensor):
                            obs_list.append(val)
                if obs_list:
                    return torch.cat(obs_list, dim=-1)
            if 'policy' in obs:
                val = obs['policy']
                return val if isinstance(val, torch.Tensor) else torch.tensor(val)
            obs_list = []
            for k in obs.keys():
                val = obs[k]
                if isinstance(val, torch.Tensor):
                    if len(val.shape) > 2:
                        obs_list.append(val.flatten(start_dim=1))
                    else:
                        obs_list.append(val)
            if obs_list:
                return torch.cat(obs_list, dim=-1)
        elif isinstance(obs, torch.Tensor):
            return obs
        elif isinstance(obs, dict):
            if 'policy' in obs:
                return obs['policy']
            obs_list = []
            for v in obs.values():
                if isinstance(v, torch.Tensor):
                    obs_list.append(v.flatten(start_dim=1) if len(v.shape) > 2 else v)
            if obs_list:
                return torch.cat(obs_list, dim=-1)
        return obs

    def _extract_obs_tensor(self, obs) -> torch.Tensor:
        """Extract observation tensor from various input formats."""
        if hasattr(obs, 'keys') and hasattr(obs, 'get'):
            if 'policy' in obs:
                obs_tensor = obs['policy']
            else:
                keys = list(obs.keys())
                if keys:
                    obs_tensor = obs[keys[0]]
                    if len(keys) > 1:
                        obs_list = []
                        for k in keys:
                            val = obs[k]
                            if isinstance(val, torch.Tensor):
                                obs_list.append(val.flatten(start_dim=1) if len(val.shape) > 2 else val)
                        if obs_list:
                            obs_tensor = torch.cat(obs_list, dim=-1)
                else:
                    raise ValueError("Empty observation dict")
        elif isinstance(obs, torch.Tensor):
            obs_tensor = obs
        elif isinstance(obs, dict):
            if 'policy' in obs:
                obs_tensor = obs['policy']
            else:
                obs_list = []
                for v in obs.values():
                    if isinstance(v, torch.Tensor):
                        obs_list.append(v.flatten(start_dim=1) if len(v.shape) > 2 else v)
                if obs_list:
                    obs_tensor = torch.cat(obs_list, dim=-1)
                else:
                    raise ValueError("No valid observations in dict")
        else:
            raise TypeError(f"Unexpected obs type: {type(obs)}")
        return obs_tensor

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Actor forward pass - returns deterministic action (mean)."""
        self.policy.update_distribution(obs)
        return self.action_mean

    def update_distribution(self, obs):
        """Update action distribution based on observations."""
        obs_tensor = self._extract_obs_tensor(obs)
        obs_tensor = self.actor_obs_normalizer(obs_tensor)
        self.policy.update_distribution(obs_tensor)

    def act(self, obs, **kwargs) -> torch.Tensor:
        """Get actions from policy.
        
        Returns final action = base_action + scale * residual_action
        """
        self.update_distribution(obs)
        # Sample residual
        residual = self.policy.distribution.sample()
        # Combine with base action
        scale = torch.clamp(self.policy.residual_scale, 0.01, self.policy.cfg.max_residual_scale)
        final_action = self.policy._base_action_for_dist + scale * residual
        return final_action

    @property
    def action_mean(self) -> torch.Tensor:
        """Get mean of action distribution."""
        return self.policy.action_mean

    @property
    def action_std(self) -> torch.Tensor:
        """Get std of action distribution."""
        return self.policy.action_std

    @property
    def entropy(self) -> torch.Tensor:
        """Get entropy of distribution."""
        return self.policy.entropy

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities of actions."""
        return self.policy.get_actions_log_prob(actions)

    def evaluate(self, obs, **kwargs):
        """Evaluate observations to get value estimates."""
        obs_tensor = self._extract_obs_tensor(obs)
        obs_tensor = self.critic_obs_normalizer(obs_tensor)
        values = self.policy.get_value(obs_tensor)
        if len(values.shape) == 1:
            values = values.unsqueeze(-1)
        return values

    def forward(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training."""
        obs_tensor = self._extract_obs_tensor(obs)
        obs_tensor = self.actor_obs_normalizer(obs_tensor)
        actions, log_probs, values = self.policy.forward(obs_tensor)
        return actions, log_probs, values

    def get_value(self, obs) -> torch.Tensor:
        """Get value estimates only."""
        obs_tensor = self._extract_obs_tensor(obs)
        obs_tensor = self.critic_obs_normalizer(obs_tensor)
        return self.policy.get_value(obs_tensor)

    def __setattr__(self, name, value):
        """Override to sync normalizers."""
        if name == "actor_obs_normalizer":
            super().__setattr__(name, value)
            if hasattr(self, 'policy') and self.policy is not None:
                self.policy.obs_normalizer = value
        elif name == "critic_obs_normalizer":
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

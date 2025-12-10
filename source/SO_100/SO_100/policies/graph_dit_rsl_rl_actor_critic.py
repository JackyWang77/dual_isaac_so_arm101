# Copyright (c) 2024-2025, SO-ARM100 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL ActorCritic wrapper for Graph DiT RL Policy.

This module provides a RSL-RL compatible ActorCritic class that wraps
GraphDiTRLPolicy, allowing it to be used with RSL-RL's OnPolicyRunner
for efficient multi-environment training.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

# RSL-RL imports (required for RSL-RL training)
try:
    from rsl_rl.modules import ActorCritic
except (ImportError, ModuleNotFoundError) as e:
    ActorCritic = None
    _actor_critic_error = e
else:
    _actor_critic_error = None

# Config classes come from isaaclab_rl wrapper
try:
    from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
except (ImportError, ModuleNotFoundError) as e:
    RslRlPpoActorCriticCfg = None
    _config_error = e
else:
    _config_error = None

from .graph_dit_rl_policy import GraphDiTRLPolicy, GraphDiTRLPolicyCfg

# Validate that RSL-RL is available
if ActorCritic is None:
    error_msg = (
        "RSL-RL ActorCritic not available. "
        "This module requires RSL-RL for Graph DiT RL training. "
        f"\nOriginal import error: {_actor_critic_error}"
    )
    raise ImportError(error_msg) from _actor_critic_error

if RslRlPpoActorCriticCfg is None:
    error_msg = (
        "RSL-RL config classes not available. "
        "This module requires isaaclab_rl for Graph DiT RL training. "
        f"\nOriginal import error: {_config_error}"
    )
    raise ImportError(error_msg) from _config_error


class GraphDiTActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for Graph DiT ActorCritic (RSL-RL compatible).
    
    This config extends RslRlPpoActorCriticCfg and adds Graph DiT specific settings.
    It sets class_name to tell RSL-RL to use our custom GraphDiTActorCritic class.
    
    Note: We override __init__ to handle the custom 'graph_dit_rl_cfg' field since
    RslRlPpoActorCriticCfg doesn't accept it by default.
    """

    def __init__(self, graph_dit_rl_cfg: GraphDiTRLPolicyCfg | None = None, **kwargs):
        """Initialize Graph DiT ActorCritic config.
        
        Args:
            graph_dit_rl_cfg: Configuration for Graph DiT RL Policy. If None, uses default.
            **kwargs: Other arguments passed to RslRlPpoActorCriticCfg (e.g., class_name, etc.).
        """
        # Extract graph_dit_rl_cfg from kwargs if provided as positional
        # (it might be passed via keyword or positional)
        if graph_dit_rl_cfg is None and 'graph_dit_rl_cfg' in kwargs:
            graph_dit_rl_cfg = kwargs.pop('graph_dit_rl_cfg')
        
        # Set class_name in kwargs if not provided
        if 'class_name' not in kwargs:
            kwargs['class_name'] = "SO_100.policies.graph_dit_rsl_rl_actor_critic.GraphDiTActorCritic"
        
        # Initialize parent class with remaining kwargs
        super().__init__(**kwargs)
        
        # Set custom field (after parent init, so it's not passed to parent)
        if graph_dit_rl_cfg is not None:
            self.graph_dit_rl_cfg = graph_dit_rl_cfg
        else:
            self.graph_dit_rl_cfg = GraphDiTRLPolicyCfg()


class GraphDiTActorCritic(ActorCritic):
    """RSL-RL compatible ActorCritic wrapper for Graph DiT RL Policy.

    This class wraps GraphDiTRLPolicy to make it compatible with RSL-RL's
    OnPolicyRunner, allowing efficient multi-environment training.

    The wrapper delegates actor/critic operations to the underlying
    GraphDiTRLPolicy, maintaining the same interface as standard RSL-RL
    ActorCritic classes.
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
        """Initialize Graph DiT ActorCritic.

        This signature matches RSL-RL's standard ActorCritic.__init__ interface.

        Args:
            obs: Observation space dictionary from environment.
            obs_groups: Observation groups dictionary.
            num_actions: Number of actions (action dimension).
            actor_obs_normalization: Whether to normalize actor observations.
            critic_obs_normalization: Whether to normalize critic observations.
            actor_hidden_dims: Actor hidden dimensions (not used, kept for compatibility).
            critic_hidden_dims: Critic hidden dimensions (not used, kept for compatibility).
            activation: Activation function (not used, kept for compatibility).
            init_noise_std: Initial noise std (not used, kept for compatibility).
            noise_std_type: Noise std type (not used, kept for compatibility).
            **kwargs: Additional arguments, must include 'graph_dit_rl_cfg' (GraphDiTRLPolicyCfg).
        """
        # Extract graph_dit_rl_cfg from kwargs BEFORE calling super().__init__()
        # This prevents it from being passed to the base class, which might misinterpret it
        if "graph_dit_rl_cfg" not in kwargs:
            raise ValueError(
                "GraphDiTActorCritic requires 'graph_dit_rl_cfg' in kwargs. "
                "Make sure GraphDiTActorCriticCfg is properly configured with graph_dit_rl_cfg."
            )
        graph_dit_rl_cfg_dict = kwargs.pop("graph_dit_rl_cfg")
        
        # Convert dict back to GraphDiTRLPolicyCfg if needed
        if isinstance(graph_dit_rl_cfg_dict, dict):
            # Reconstruct from dict (this happens when config is serialized via to_dict())
            graph_dit_rl_cfg = GraphDiTRLPolicyCfg(**graph_dit_rl_cfg_dict)
        else:
            graph_dit_rl_cfg = graph_dit_rl_cfg_dict
        
        # Validate and sanitize parameters before passing to base class
        # Ensure activation is a string
        if not isinstance(activation, str):
            activation = "elu"
        
        # Ensure noise_std_type is a string
        if not isinstance(noise_std_type, str):
            noise_std_type = "scalar"
        
        # Ensure init_noise_std is a number (not a dict)
        if not isinstance(init_noise_std, (int, float)):
            init_noise_std = 1.0
        
        # Ensure actor_hidden_dims and critic_hidden_dims are lists
        if actor_hidden_dims is None or not isinstance(actor_hidden_dims, list):
            actor_hidden_dims = [256, 256, 256]
        if critic_hidden_dims is None or not isinstance(critic_hidden_dims, list):
            critic_hidden_dims = [256, 256, 256]
        
        # Initialize base class with the standard RSL-RL ActorCritic signature
        # Base class expects: (obs, obs_groups, num_actions, actor_obs_normalization, ...)
        # Note: obs_groups should be a dict like {"policy": ["policy"], "critic": ["policy"]}
        # where the values are lists of observation keys
        # We explicitly pass all parameters to avoid kwargs conflicts
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
            # Don't pass **kwargs here to avoid any conflicts with base class parameters
        )

        # Store our actual config
        self.graph_dit_rl_cfg = graph_dit_rl_cfg
        self.num_actions = num_actions
        self.obs_groups = obs_groups

        # Extract actual observation dimension from obs dict
        # This is needed because config might have wrong obs_dim
        actual_obs_dim = None
        if 'policy' in obs:
            # Get observation from 'policy' key
            obs_sample = obs['policy']
            if isinstance(obs_sample, dict):
                # If dict, flatten and get total dimension
                actual_obs_dim = sum(v.flatten().shape[0] if hasattr(v, 'shape') else len(v) for v in obs_sample.values())
            else:
                actual_obs_dim = obs_sample.shape[-1] if len(obs_sample.shape) > 1 else obs_sample.shape[0]
        else:
            # Concatenate all observations to get dimension
            total_dim = 0
            for v in obs.values():
                if isinstance(v, dict):
                    total_dim += sum(vi.flatten().shape[0] if hasattr(vi, 'shape') else len(vi) for vi in v.values())
                else:
                    total_dim += v.shape[-1] if len(v.shape) > 1 else v.shape[0]
            actual_obs_dim = total_dim
        
        # Update value_obs_dim in config if actual observation dimension differs
        # Note: We don't modify graph_dit_cfg.obs_dim because that should match the pre-trained model
        # Handle case where graph_dit_cfg might be a dict (from serialization)
        graph_dit_cfg_for_dim = graph_dit_rl_cfg.graph_dit_cfg
        if isinstance(graph_dit_cfg_for_dim, dict):
            # Convert dict to get obs_dim
            from .graph_dit_policy import GraphDiTPolicyCfg
            graph_dit_cfg_for_dim = GraphDiTPolicyCfg(**graph_dit_cfg_for_dim)
            # Also update graph_dit_rl_cfg.graph_dit_cfg if it's still a dict
            if isinstance(graph_dit_rl_cfg.graph_dit_cfg, dict):
                graph_dit_rl_cfg.graph_dit_cfg = graph_dit_cfg_for_dim
        
        graph_dit_obs_dim = graph_dit_cfg_for_dim.obs_dim if hasattr(graph_dit_cfg_for_dim, 'obs_dim') else graph_dit_cfg_for_dim.get('obs_dim', 32)
        
        if actual_obs_dim is not None and actual_obs_dim != graph_dit_obs_dim:
            print(f"[GraphDiTActorCritic] Actual obs_dim ({actual_obs_dim}) differs from Graph DiT config obs_dim ({graph_dit_obs_dim}).")
            print(f"[GraphDiTActorCritic] Setting value_obs_dim to {actual_obs_dim} for value network.")
            graph_dit_rl_cfg.value_obs_dim = actual_obs_dim

        # Create underlying Graph DiT RL Policy
        # If pretrained_checkpoint is set, it will be loaded in GraphDiTRLPolicy.__init__
        self.policy = GraphDiTRLPolicy(graph_dit_rl_cfg)
        # Device will be set via .to(device) later by RSL-RL

        # Initialize observation normalizer as Identity (will be set by RSL-RL if needed)
        # RSL-RL expects actor_obs_normalizer to be a direct attribute (not property), callable
        import torch.nn as nn
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()
        # Also set in policy for internal use (synchronized via __setattr__ override)
        self.policy.obs_normalizer = self.actor_obs_normalizer
        
        # Initialize distribution attribute (will be set by update_distribution)
        self.policy.distribution = None

        # Store action dimension
        self.action_dim = num_actions
    
    def to(self, device):
        """Move the ActorCritic and its policy to the specified device.
        
        This ensures that when RSL-RL calls .to(device), our Graph DiT policy
        is also moved to the correct device.
        """
        # Call parent to move base class components
        result = super().to(device)
        # Move our custom policy
        if hasattr(self, 'policy') and self.policy is not None:
            self.policy.to(device)
        return result

    def get_actor_obs(self, obs):
        """Extract actor observations from observation dictionary.
        
        This matches RSL-RL's standard interface.
        Handles dict, TensorDict, and tensor inputs.
        """
        # Handle TensorDict (from Isaac Lab's RslRlVecEnvWrapper)
        if hasattr(obs, 'keys') and hasattr(obs, 'get'):
            # It's a dict-like object (could be dict or TensorDict)
            # Extract observations based on obs_groups
            if hasattr(self, 'obs_groups') and 'policy' in self.obs_groups:
                obs_list = []
                for obs_group in self.obs_groups["policy"]:
                    if obs_group in obs:
                        val = obs[obs_group]
                        # Handle TensorDict values
                        if hasattr(val, 'to') or isinstance(val, torch.Tensor):
                            obs_list.append(val)
                if obs_list:
                    return torch.cat(obs_list, dim=-1)
            # Fallback: use 'policy' key if available
            if 'policy' in obs:
                val = obs['policy']
                return val if isinstance(val, torch.Tensor) or hasattr(val, 'to') else torch.tensor(val)
            # Fallback: concatenate all observations
            obs_list = []
            for k in obs.keys():
                val = obs[k]
                if isinstance(val, torch.Tensor) or hasattr(val, 'to'):
                    if isinstance(val, torch.Tensor) and len(val.shape) > 2:
                        obs_list.append(val.flatten(start_dim=1))
                    else:
                        obs_list.append(val)
            if obs_list:
                return torch.cat(obs_list, dim=-1)
        # If obs is already a tensor, return as-is
        elif isinstance(obs, torch.Tensor):
            return obs
        # Handle regular dict
        elif isinstance(obs, dict):
            if 'policy' in obs:
                return obs['policy']
            obs_list = []
            for v in obs.values():
                if isinstance(v, torch.Tensor):
                    if len(v.shape) > 2:
                        obs_list.append(v.flatten(start_dim=1))
                    else:
                        obs_list.append(v)
            if obs_list:
                return torch.cat(obs_list, dim=-1)
        
        # If all else fails, return as-is
        return obs

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Actor network forward pass - returns deterministic action (mean).
        
        This is used by act_inference for inference mode.
        """
        # Update distribution to get action mean
        # obs is already normalized tensor from act_inference
        self.policy.update_distribution(obs)
        # Return deterministic action (mean)
        return self.action_mean

    def update_distribution(self, obs: dict[str, torch.Tensor] | torch.Tensor | Any):
        """Update action distribution based on observations.
        
        This is called by PPO before accessing action_mean/action_std.
        We need to compute and store the distribution for RSL-RL compatibility.
        
        Args:
            obs: Observation dictionary from environment, tensor, or TensorDict.
        """
        # Handle TensorDict (from Isaac Lab's RslRlVecEnvWrapper)
        if hasattr(obs, 'keys') and hasattr(obs, 'get'):
            # It's a dict-like object (could be dict or TensorDict)
            if 'policy' in obs:
                obs_tensor = obs['policy'] if not hasattr(obs['policy'], 'to') else obs['policy']
            else:
                # Try to get first key or concatenate all values
                if hasattr(obs, 'keys'):
                    keys = list(obs.keys())
                    if keys:
                        obs_tensor = obs[keys[0]]
                        # If multiple keys, try to concatenate
                        if len(keys) > 1:
                            obs_list = []
                            for k in keys:
                                val = obs[k]
                                if isinstance(val, torch.Tensor):
                                    if len(val.shape) > 2:
                                        obs_list.append(val.flatten(start_dim=1))
                                    else:
                                        obs_list.append(val)
                            if obs_list:
                                obs_tensor = torch.cat(obs_list, dim=-1)
                    else:
                        raise ValueError("Empty observation dict/TensorDict")
                else:
                    raise TypeError(f"Cannot extract obs from type: {type(obs)}")
        elif isinstance(obs, torch.Tensor):
            obs_tensor = obs
        elif isinstance(obs, dict):
            # Extract observation vector (handle Isaac Lab's 'policy' key)
            if 'policy' in obs:
                obs_tensor = obs['policy']
            else:
                # Concatenate all observations
                obs_list = []
                for v in obs.values():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) > 2:
                            obs_list.append(v.flatten(start_dim=1))
                        else:
                            obs_list.append(v)
                if obs_list:
                    obs_tensor = torch.cat(obs_list, dim=-1)
                else:
                    raise ValueError("No valid observations in dict")
        else:
            raise TypeError(f"Unexpected obs type: {type(obs)}")

        # Normalize observation using actor_obs_normalizer (set by RSL-RL)
        obs_tensor = self.actor_obs_normalizer(obs_tensor)

        # Update distribution in the policy (stores mean and std internally)
        self.policy.update_distribution(obs_tensor)

    def act(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Get actions from policy.

        Args:
            obs: Observation dictionary from environment.
            **kwargs: Additional arguments (e.g., masks, hidden_states) - ignored for non-recurrent policies.

        Returns:
            actions: [batch, action_dim] - Actions (RSL-RL expects only actions, not tuple)
        """
        # Update distribution first (required by RSL-RL)
        self.update_distribution(obs)
        
        # Sample from distribution (stored in policy)
        return self.policy.distribution.sample()

    @property
    def action_mean(self) -> torch.Tensor:
        """Get mean of action distribution."""
        if not hasattr(self.policy, 'distribution') or self.policy.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        return self.policy.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Get standard deviation of action distribution."""
        if not hasattr(self.policy, 'distribution') or self.policy.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        return self.policy.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Get entropy of action distribution.
        
        Returns:
            entropy: [batch] - Entropy values for each sample in the batch.
        """
        if not hasattr(self.policy, 'distribution') or self.policy.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        return self.policy.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities of actions.
        
        Args:
            actions: Actions to evaluate [batch, action_dim].
            
        Returns:
            log_probs: [batch] - Log probabilities of actions.
        """
        if not hasattr(self.policy, 'distribution') or self.policy.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        return self.policy.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, obs: dict[str, torch.Tensor], **kwargs):
        """Evaluate observations to get value estimates.
        
        This is called by PPO during rollout to get values.
        Note: This matches RSL-RL's ActorCritic interface which only returns values.

        Args:
            obs: Observation dictionary from environment.
            **kwargs: Additional arguments (e.g., masks, hidden_states for recurrent).

        Returns:
            values: [batch] - Value estimates
        """
        # Extract observation vector
        if 'policy' in obs:
            obs_tensor = obs['policy']
        else:
            # Concatenate all observation components
            obs_components = []
            for v in obs.values():
                if len(v.shape) > 2:
                    # Flatten spatial dimensions, keep batch dimension
                    obs_components.append(v.flatten(start_dim=1))
                else:
                    obs_components.append(v)
            obs_tensor = torch.cat(obs_components, dim=-1)

        # Ensure obs_tensor has shape [batch, obs_dim]
        if len(obs_tensor.shape) != 2:
            raise ValueError(f"Expected obs_tensor shape [batch, obs_dim], got {obs_tensor.shape}")

        # Normalize observation using critic_obs_normalizer
        obs_tensor = self.critic_obs_normalizer(obs_tensor)

        # Get values only (for rollout)
        values = self.policy.get_value(obs_tensor)

        # Ensure values have shape [batch, 1] (matching RSL-RL's ActorCritic interface)
        if len(values.shape) == 1:
            values = values.unsqueeze(-1)  # [batch] -> [batch, 1]
        elif len(values.shape) == 0:
            values = values.unsqueeze(0).unsqueeze(-1)  # scalar -> [1, 1]
        
        # Final check: should be [batch, 1]
        if len(values.shape) != 2 or values.shape[-1] != 1:
            raise ValueError(f"Expected values shape [batch, 1], got {values.shape}")

        return values

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            obs: Observation dictionary from environment.

        Returns:
            actions: [batch, action_dim] - Sampled actions
            log_probs: [batch] - Log probabilities
            values: [batch] - Value estimates
        """
        # Extract observation vector
        if 'policy' in obs:
            obs_tensor = obs['policy']
        else:
            obs_tensor = torch.cat([v.flatten(start_dim=1) if len(v.shape) > 2 else v for v in obs.values()], dim=-1)

        # Normalize observation using actor_obs_normalizer
        obs_tensor = self.actor_obs_normalizer(obs_tensor)

        # Forward pass
        actions, log_probs, values = self.policy.forward(obs_tensor)

        return actions, log_probs, values

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get value estimates only.

        Args:
            obs: Observation dictionary from environment.

        Returns:
            values: [batch] - Value estimates
        """
        # Extract observation vector
        if 'policy' in obs:
            obs_tensor = obs['policy']
        else:
            obs_tensor = torch.cat([v.flatten(start_dim=1) if len(v.shape) > 2 else v for v in obs.values()], dim=-1)

        # Normalize observation using critic_obs_normalizer
        obs_tensor = self.critic_obs_normalizer(obs_tensor)

        # Get values
        values = self.policy.get_value(obs_tensor)

        return values

    def __setattr__(self, name, value):
        """Override __setattr__ to sync normalizers with policy."""
        # If setting actor_obs_normalizer, also sync to policy
        if name == "actor_obs_normalizer":
            super().__setattr__(name, value)
            # Sync to policy if it exists (may not exist during __init__)
            if hasattr(self, 'policy') and self.policy is not None:
                self.policy.obs_normalizer = value
        elif name == "critic_obs_normalizer":
            super().__setattr__(name, value)
            # Optionally sync to policy if needed
        else:
            super().__setattr__(name, value)

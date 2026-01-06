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
- Handles normalization correctly (DiT vs PPO branches)
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
            kwargs['class_name'] = "SO_101.policies.residual_rl_actor_critic.ResidualActorCritic"

        super().__init__(**kwargs)

        if residual_rl_cfg is not None:
            self.residual_rl_cfg = residual_rl_cfg
        else:
            self.residual_rl_cfg = ResidualRLPolicyCfg()


class ResidualActorCritic(ActorCritic):
    """RSL-RL compatible ActorCritic wrapper for Residual RL Policy.

    This class wraps ResidualRLPolicy to make it compatible with RSL-RL's
    OnPolicyRunner, enabling efficient multi-environment training.

    CRITICAL: Normalization handling
    ================================
    - DiT branch: Uses checkpoint normalization stats (norm_obs_mean/std)
    - PPO branch: Can use RSL-RL's running normalizer for value network
    - We pass RAW obs to policy, let it handle normalization internally
    """

    def __init__(
        self,
        obs: dict[str, Any],
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,  # We handle this manually!
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
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
        # ============================================================
        # CRITICAL: Force disable RSL-RL's automatic normalization!
        # We handle normalization internally (DiT uses checkpoint stats)
        # Double normalization will completely break DiT's predictions!
        # ============================================================
        if actor_obs_normalization:
            print("[ResidualActorCritic] !!! WARNING: actor_obs_normalization was True, forcing to False !!!")
            print("[ResidualActorCritic] !!! We handle normalization internally for DiT !!!")
        actor_obs_normalization = False  # FORCE DISABLE!
        
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
        # NOTE: We let RSL-RL create normalizer modules, but we control WHEN they are applied
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

        # Infer observation dimension for Value Network
        actual_obs_dim = self._infer_obs_dim(obs)
        if actual_obs_dim is not None:
            residual_rl_cfg.value_obs_dim = actual_obs_dim
            print(f"[ResidualActorCritic] Set value_obs_dim to {actual_obs_dim}")

        # Create underlying Residual RL Policy
        self.policy = ResidualRLPolicy(residual_rl_cfg)

        # [CRITICAL] Sync Normalizers
        # PPO (Student) can use RSL-RL's running normalizer for value network
        # DiT (Teacher) uses its own fixed checkpoint stats
        # We pass the RSL-RL normalizer so policy can use it for PPO branch if needed
        self.policy.obs_normalizer = self.actor_obs_normalizer

        self.action_dim = num_actions

    def _infer_obs_dim(self, obs) -> int | None:
        """Helper to figure out flat observation dimension."""
        if 'policy' in obs:
            sample = obs['policy']
            if isinstance(sample, torch.Tensor):
                return sample.shape[-1] if len(sample.shape) > 1 else sample.shape[0]
            elif isinstance(sample, dict):
                return sum(
                    v.flatten().shape[0] if hasattr(v, 'shape') else len(v)
                    for v in sample.values()
                )
        # Fallback for flattened dict
        total = 0
        for v in obs.values():
            if isinstance(v, torch.Tensor):
                total += v.shape[-1] if len(v.shape) > 1 else v.shape[0]
            elif isinstance(v, dict):
                total += sum(
                    vi.flatten().shape[0] if hasattr(vi, 'shape') else len(vi)
                    for vi in v.values()
                )
        return total if total > 0 else None

    def _extract_obs_tensor(self, obs) -> torch.Tensor:
        """Extract observation tensor from various input formats.
        
        Handles:
        - torch.Tensor: Return directly
        - dict: Extract 'policy' key or concat all tensors
        - TensorDict: Extract 'policy' key or convert to tensor
        """
        # Handle torch.Tensor
        if isinstance(obs, torch.Tensor):
            return obs
        
        # Handle TensorDict (from RSL-RL)
        # TensorDict has .get() and .keys() methods like dict
        if hasattr(obs, 'get') and hasattr(obs, 'keys'):
            if 'policy' in obs.keys():
                policy_obs = obs.get('policy')
                if isinstance(policy_obs, torch.Tensor):
                    return policy_obs
                # TensorDict might nest another TensorDict
                if hasattr(policy_obs, 'to_dict'):
                    policy_obs = policy_obs.to_dict()
                if isinstance(policy_obs, dict):
                    obs_list = []
                    for v in policy_obs.values():
                        if isinstance(v, torch.Tensor):
                            obs_list.append(v.flatten(start_dim=1) if len(v.shape) > 2 else v)
                    if obs_list:
                        return torch.cat(obs_list, dim=-1)
            
            # Try to convert TensorDict to regular dict
            if hasattr(obs, 'to_dict'):
                obs = obs.to_dict()
            
            # Concat all tensor values
            obs_list = []
            for k in obs.keys():
                v = obs.get(k) if hasattr(obs, 'get') else obs[k]
                if isinstance(v, torch.Tensor):
                    obs_list.append(v.flatten(start_dim=1) if len(v.shape) > 2 else v)
            if obs_list:
                return torch.cat(obs_list, dim=-1)
            raise ValueError("No valid observations in dict/TensorDict")
        
        # Handle regular dict
        if isinstance(obs, dict):
            if 'policy' in obs:
                return obs['policy']
            obs_list = []
            for v in obs.values():
                if isinstance(v, torch.Tensor):
                    obs_list.append(v.flatten(start_dim=1) if len(v.shape) > 2 else v)
            if obs_list:
                return torch.cat(obs_list, dim=-1)
            raise ValueError("No valid observations in dict")
        
        raise TypeError(f"Unexpected obs type: {type(obs)}")

    def to(self, device):
        """Move to device."""
        result = super().to(device)
        if hasattr(self, 'policy') and self.policy is not None:
            self.policy.to(device)
        return result

    def reset(self, dones: torch.Tensor | None = None):
        """
        [CRITICAL] Reset history buffers when environments reset.
        Called by RSL-RL runner.

        Without this, when environment resets:
        - action_history still contains old episode's actions
        - DiT's chunking buffer still has stale actions
        - Robot would execute leftover actions from previous episode
        """
        if dones is None:
            return

        # Get indices of envs that are done
        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.policy.reset_history(env_ids)

    def update_distribution(self, obs):
        """Update action distribution based on observations.

        [CRITICAL FIX] We pass RAW obs to policy.update_distribution.
        Inside policy:
          - DiT branch uses internal _normalize_obs (Checkpoint stats)
          - PPO branch uses raw obs for robot_state extraction

        DO NOT apply actor_obs_normalizer here! That would cause double-normalization
        and break DiT's predictions completely.
        """
        obs_tensor = self._extract_obs_tensor(obs)
        # DO NOT normalize here! Pass raw data to policy.
        # Policy handles normalization internally (DiT uses checkpoint stats)
        self.policy.update_distribution(obs_tensor)

    def act(self, obs, **kwargs) -> torch.Tensor:
        """Get actions from policy.

        Design: DiT predicts in chunks + PPO corrects step-by-step
        ================================================
        DiT: Predicts base_trajectory every exec_horizon steps, stores in buffer
        PPO: Computes residual at every step based on current obs (closed-loop control!)

        Final = Base_Action_t + scale * Residual_t

        Args:
            obs: Observation dictionary or tensor
            **kwargs: May contain 'inference=True' for deterministic actions
        """
        self.update_distribution(obs)

        # Check inference mode (for play/evaluation without noise)
        inference = kwargs.get('inference', False)

        if inference:
            # Deterministic: use mean (no sampling noise)
            residual = self.policy.distribution.mean
        else:
            # Training: sample from distribution
            residual = self.policy.distribution.sample()

        # Get base_action_t from buffer
        base_action_t = self.policy._base_action_for_dist

        # Combine: final = base + scale * residual
        scale = torch.clamp(self.policy.residual_scale, 0.01, self.policy.cfg.max_residual_scale)
        
        # ============================================================
        # DEBUG: Zero Residual Check
        # Set ZERO_RESIDUAL_CHECK = True to test if DiT alone works
        # If DiT works alone but fails with PPO, reduce init_noise_std
        # ============================================================
        ZERO_RESIDUAL_CHECK = True  # Set to True to diagnose
        if ZERO_RESIDUAL_CHECK:
            final_action = base_action_t  # PPO completely disabled
        else:
            final_action = base_action_t + scale * residual

        # Update history buffers with the action we just took
        obs_tensor = self._extract_obs_tensor(obs)
        self.policy._update_history_buffers(obs_tensor, final_action.detach())

        # INCREMENT BUFFER INDEX for next step
        if self.policy._buffer_indices is not None:
            self.policy._buffer_indices += 1

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
        """Evaluate observations to get value estimates (Critic).

        Note: Critic typically sees normalized obs (standard RL practice)
        """
        obs_tensor = self._extract_obs_tensor(obs)
        # Critic uses RSL-RL normalizer (running mean/std)
        obs_tensor = self.critic_obs_normalizer(obs_tensor)
        values = self.policy.get_value(obs_tensor)
        if len(values.shape) == 1:
            values = values.unsqueeze(-1)
        return values

    def forward(self, obs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for PPO Update (Gradient Calculation).

        Note: policy.forward handles normalization internally
        - DiT branch -> uses checkpoint norm stats
        - PPO branch -> uses raw robot state
        """
        obs_tensor = self._extract_obs_tensor(obs)
        # Pass raw obs, let policy handle normalization
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

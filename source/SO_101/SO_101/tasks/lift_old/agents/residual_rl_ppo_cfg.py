# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for Residual RL Policy.

This configuration sets up PPO training for the ResidualRLPolicy,
which uses a frozen Graph-DiT backbone for scene understanding
and trains only the residual action head.
"""

import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from SO_101.policies import GraphDiTPolicyCfg
from SO_101.policies.residual_rl_actor_critic import ResidualActorCriticCfg
from SO_101.policies.residual_rl_policy import ResidualRLPolicyCfg


def _get_pretrained_checkpoint() -> str:
    """Get pretrained checkpoint path from environment variable at runtime."""
    return os.environ.get(
        "RESIDUAL_RL_PRETRAINED_CHECKPOINT",
        os.environ.get(
            "GRAPH_DIT_PRETRAINED_CHECKPOINT",
            "./logs/graph_dit/lift_joint_flow_matching/best_model.pt",
        ),
    )


def _create_residual_rl_cfg() -> ResidualRLPolicyCfg:
    """Create ResidualRLPolicyCfg with runtime checkpoint path."""
    # Graph-DiT backbone configuration (will be overridden by loaded checkpoint)
    graph_dit_cfg = GraphDiTPolicyCfg(
        obs_dim=32,
        action_dim=6,
        hidden_dim=128,  # Will be overridden by loaded model
        num_layers=4,
        num_heads=4,
        node_dim=7,
        edge_dim=4,
        action_history_length=4,
        diffusion_steps=100,
        mode="flow_matching",
    )

    return ResidualRLPolicyCfg(
        graph_dit_cfg=graph_dit_cfg,
        pretrained_checkpoint=_get_pretrained_checkpoint(),
        # PPO residual network
        residual_hidden_dims=[256, 128, 64],
        residual_activation="elu",
        # Value network
        value_hidden_dims=[256, 128, 64],
        value_activation="elu",
        # Input dimensions
        robot_state_dim=12,  # joint_pos(6) + joint_vel(6)
        # Residual scaling (start VERY small for Residual RL!)
        residual_scale=0.05,  # Reduced from 0.1
        max_residual_scale=0.3,  # Reduced from 0.5
        # Noise (CRITICAL: keep small for Residual RL!)
        init_noise_std=0.1,  # Reduced from 0.3
        # Feature usage
        use_graph_embedding=True,
        use_node_features=False,
        use_edge_features=False,
        # Freeze backbone
        freeze_backbone=True,
    )


@configclass
class LiftResidualRLRunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner configuration for Residual RL training.

    This configures PPO to train only the residual head while keeping
    the Graph-DiT backbone frozen.
    """

    num_steps_per_env = 24
    max_iterations = 300  # Residual learning converges faster
    save_interval = 50
    experiment_name = "lift_residual_rl"
    empirical_normalization = False

    def __post_init__(self):
        """Initialize config after dataclass creation - reads env var at runtime."""
        super().__post_init__()

        # Create residual_rl_cfg at runtime (reads env var NOW, not at import time)
        residual_rl_cfg = _create_residual_rl_cfg()

        # Policy configuration using ResidualActorCriticCfg
        self.policy = ResidualActorCriticCfg(
            residual_rl_cfg=residual_rl_cfg,
            class_name="SO_101.policies.residual_rl_actor_critic.ResidualActorCritic",
            init_noise_std=0.1,  # Reduced from 0.3 for Residual RL
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        )

    # PPO algorithm configuration (tuned for residual learning)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,  # Standard PPO clip
        entropy_coef=0.01,  # Encourage exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,  # Lower LR for residual (small corrections)
        schedule="fixed",
        gamma=0.99,  # High discount for long-horizon tasks
        lam=0.95,  # GAE lambda
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

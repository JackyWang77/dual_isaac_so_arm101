# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for Residual RL Policy.

DEPRECATED: This file is kept for backward compatibility but the old
ResidualRLPolicy and ResidualActorCritic have been replaced by
GraphDiTResidualRLPolicy (independent training framework).

This configuration is no longer functional and should not be used.
Please use the new GraphDiTResidualRLPolicy instead.
"""

import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from SO_101.policies import GraphDiTPolicyCfg

# DEPRECATED: Old ResidualRLPolicy and ResidualActorCritic have been removed
# from SO_101.policies.residual_rl_actor_critic import ResidualActorCriticCfg
# from SO_101.policies.residual_rl_policy import ResidualRLPolicyCfg


def _get_pretrained_checkpoint() -> str:
    """Get pretrained checkpoint path from environment variable at runtime."""
    return os.environ.get(
        "RESIDUAL_RL_PRETRAINED_CHECKPOINT",
        os.environ.get(
            "GRAPH_DIT_PRETRAINED_CHECKPOINT",
            "./logs/graph_dit/lift_joint_flow_matching/best_model.pt",
        ),
    )


def _create_residual_rl_cfg():
    """DEPRECATED: This function is no longer functional."""
    raise NotImplementedError(
        "ResidualRLPolicyCfg has been removed. "
        "Please use GraphDiTResidualRLPolicy instead. "
        "See: SO_101.policies.graph_dit_residual_rl_policy"
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
    num_envs: int = 1

    def __post_init__(self):
        """DEPRECATED: This configuration is no longer functional."""
        raise NotImplementedError(
            "LiftResidualRLRunnerCfg is deprecated. "
            "The old ResidualRLPolicy and ResidualActorCritic have been removed. "
            "Please use GraphDiTResidualRLPolicy instead. "
            "See: SO_101.policies.graph_dit_residual_rl_policy"
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

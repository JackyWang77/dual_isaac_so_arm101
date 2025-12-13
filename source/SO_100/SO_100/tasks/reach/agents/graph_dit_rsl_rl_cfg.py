# Copyright (c) 2024-2025, SO-ARM100 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL configuration for Graph DiT RL Policy fine-tuning.

This config follows the same pattern as standard RSL-RL configs (e.g., ReachCubePPORunnerCfg).
RSL-RL will automatically load this config when using task SO-ARM101-Reach-Cube-v0.

Usage:
    # Set pretrained checkpoint via environment variable
    export GRAPH_DIT_PRETRAINED_CHECKPOINT=./logs/graph_dit/.../best_model.pt
    ./isaaclab.sh -p scripts/rsl_rl/train.py \
        --task SO-ARM101-Reach-Cube-v0 \
        --num_envs 64 \
        --max_iterations 1500
"""

import os

from isaaclab.utils import configclass

# Check if RSL-RL is available (required for this config)
try:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
    _HAS_RSL_RL = True
except (ImportError, ModuleNotFoundError) as e:
    raise ImportError(
        f"RSL-RL modules not available. This config is only needed for RSL-RL training. "
        f"Original error: {e}"
    ) from e

from SO_100.policies.graph_dit_rsl_rl_actor_critic import GraphDiTActorCriticCfg, GraphDiTActorCritic
from SO_100.policies.graph_dit_policy import GraphDiTPolicyCfg
from SO_100.policies.graph_dit_rl_policy import GraphDiTRLPolicyCfg

# Import the ActorCritic class so RSL-RL can find it
# RSL-RL uses class_name to look up the class, so we need to ensure it's imported
# When RSL-RL sees class_name="GraphDiTActorCritic", it will look for it in the Python namespace


@configclass
class ReachCubeGraphDiTRLRunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL configuration for Reach task with Graph DiT RL Policy.

    This configuration uses GraphDiTRLPolicy as the actor-critic,
    allowing efficient multi-environment training with RSL-RL.

    The pretrained_checkpoint can be set via environment variable:
        export GRAPH_DIT_PRETRAINED_CHECKPOINT=./path/to/best_model.pt
    """

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "reach_graph_dit_rl"
    empirical_normalization = False

    # Use Graph DiT ActorCritic wrapper
    # Note: class_name can be either:
    #   1. Simple name "GraphDiTActorCritic" (if class is in Python namespace via import)
    #   2. Full module path "SO_100.policies.graph_dit_rsl_rl_actor_critic.GraphDiTActorCritic"
    # RSL-RL will dynamically import and instantiate the class
    policy = GraphDiTActorCriticCfg(
        # Use full module path to ensure RSL-RL can find it
        class_name="SO_100.policies.graph_dit_rsl_rl_actor_critic.GraphDiTActorCritic",
        graph_dit_rl_cfg=GraphDiTRLPolicyCfg(
            # Graph DiT backbone config (should match pre-trained model)
            # These will be overridden by the actual model config when loading checkpoint
            graph_dit_cfg=GraphDiTPolicyCfg(
                obs_dim=32,  # Will be overridden based on environment
                action_dim=6,
                hidden_dim=512,
                num_layers=8,
                num_heads=16,
                num_subtasks=1,  # reach_object
                action_history_length=4,
                mode="flow_matching",
            ),
            # RL Head configuration
            rl_head_hidden_dims=[128, 64],
            rl_head_activation="elu",
            value_hidden_dims=[256, 128, 64],
            value_activation="elu",
            init_noise_std=0.5,
            feature_extraction_mode="last_embedding",
            freeze_backbone=True,  # Freeze Graph DiT backbone
            # Get pretrained checkpoint from environment variable
            # Can be set via: export GRAPH_DIT_PRETRAINED_CHECKPOINT=./path/to/model.pt
            pretrained_checkpoint=os.environ.get("GRAPH_DIT_PRETRAINED_CHECKPOINT", None),
        ),
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

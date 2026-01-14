# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom policy networks for SO-ARM101 tasks."""

from .graph_dit_policy import (ActionHistoryBuffer, GraphDiTPolicy,
                               GraphDiTPolicyCfg, JointStateHistoryBuffer,
                               NodeHistoryBuffer)
from .residual_rl_policy import ResidualRLPolicy, ResidualRLPolicyCfg

# Conditionally import RSL-RL ActorCritic (only needed for RSL-RL training)
# Don't import if isaaclab_rl is not available (e.g., during simple playback)
try:
    from .residual_rl_actor_critic import (ResidualActorCritic,
                                           ResidualActorCriticCfg)

    _HAS_RSL_RL = True
except (ImportError, ModuleNotFoundError):
    # RSL-RL modules not available, skip import (only needed for training)
    _HAS_RSL_RL = False
    ResidualActorCritic = None
    ResidualActorCriticCfg = None

__all__ = [
    "ActionHistoryBuffer",
    "JointStateHistoryBuffer",
    "NodeHistoryBuffer",
    "GraphDiTPolicy",
    "GraphDiTPolicyCfg",
    "ResidualRLPolicy",
    "ResidualRLPolicyCfg",
]

# Add RSL-RL exports only if available
if _HAS_RSL_RL:
    __all__.extend(
        [
            "ResidualActorCritic",
            "ResidualActorCriticCfg",
        ]
    )

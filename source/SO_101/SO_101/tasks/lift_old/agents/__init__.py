# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configurations for lift_old task."""

from .rsl_rl_ppo_cfg import LiftCubePPORunnerCfg

# Conditionally import Residual RL config
try:
    from .residual_rl_ppo_cfg import LiftResidualRLRunnerCfg
except ImportError:
    LiftResidualRLRunnerCfg = None

__all__ = [
    "LiftCubePPORunnerCfg",
]

if LiftResidualRLRunnerCfg is not None:
    __all__.append("LiftResidualRLRunnerCfg")

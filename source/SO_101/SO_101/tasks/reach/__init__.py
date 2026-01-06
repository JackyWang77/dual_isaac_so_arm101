# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import reach_ik_abs_env  # noqa: F401
from . import reach_old_joint_states_mimic_env  # noqa: F401
from . import (agents, reach_ik_abs_env_cfg, reach_ik_abs_mimic_env_cfg,
               reach_old_joint_states_mimic_env_cfg)

# Conditionally import Graph DiT RL config (only needed for RSL-RL training)
# Don't import if isaaclab_rl is not available (e.g., during simple playback)
try:
    from .agents import \
        graph_dit_rsl_rl_cfg  # noqa: F401  # Register Graph DiT RL config (used by SO-ARM101-Reach-Cube-v0)
except (ImportError, ModuleNotFoundError):
    # RSL-RL modules not available, skip registration (only needed for training)
    pass

##
# Register Gym environments.
##

# Register the SO-100 Cube Reach environment
# Use Graph DiT RL Policy for RSL-RL training
gym.register(
    id="SO-ARM101-Reach-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100ReachJointCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.graph_dit_rsl_rl_cfg:ReachCubeGraphDiTRLRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM101-Reach-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100ReachJointCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.graph_dit_rsl_rl_cfg:ReachCubeGraphDiTRLRunnerCfg",
    },
    disable_env_checker=True,
)

##
# IK Absolute Environment
##
_ENV_CFG_IK_ABS = reach_ik_abs_env_cfg.SoArm100ReachIKAbsEnvCfg
_ENTRY_POINT_IK_ABS = f"{__name__}.reach_ik_abs_env:SoArm100ReachIKAbsEnv"

##
# IK Absolute Mimic Environment (for data generation)
##
_ENV_CFG_IK_ABS_MIMIC = reach_ik_abs_mimic_env_cfg.SoArm100ReachIKAbsMimicEnvCfg
_ENTRY_POINT_IK_ABS_MIMIC = f"{__name__}.reach_ik_abs_env:SoArm100ReachIKAbsEnv"

##
# Joint States Recording with MimicEnvCfg (records joint states directly, no conversion)
##
_ENV_CFG_JOINT_STATES_MIMIC = (
    reach_old_joint_states_mimic_env_cfg.SoArm100ReachJointStatesMimicEnvCfg
)
_ENTRY_POINT_JOINT_STATES_MIMIC = (
    f"{__name__}.reach_old_joint_states_mimic_env:SoArm100ReachJointStatesMimicEnv"
)

##
# Register Mimic environments
##

# IK Absolute Mimic (for data generation with subtask configs)
gym.register(
    id="SO-ARM101-Reach-IK-Abs-Mimic-v0",
    entry_point=_ENTRY_POINT_IK_ABS_MIMIC,
    kwargs={"env_cfg_entry_point": _ENV_CFG_IK_ABS_MIMIC},
    disable_env_checker=True,
)

# Joint States Recording with MimicEnvCfg (records joint states directly, no conversion)
gym.register(
    id="SO-ARM101-Reach-Joint-States-Mimic-v0",
    entry_point=_ENTRY_POINT_JOINT_STATES_MIMIC,
    kwargs={"env_cfg_entry_point": _ENV_CFG_JOINT_STATES_MIMIC},
    disable_env_checker=True,
)

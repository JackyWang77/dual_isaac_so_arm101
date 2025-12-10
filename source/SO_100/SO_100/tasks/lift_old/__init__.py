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

from . import agents
from . import lift_ik_abs_env  # noqa: F401
from . import lift_ik_abs_env_cfg
from . import lift_ik_abs_mimic_env_cfg
from . import lift_old_joint_states_mimic_env  # noqa: F401
from . import lift_old_joint_states_mimic_env_cfg

##
# Register Gym environments.
##

# Register the SO-100 Cube Lift environment
gym.register(
    id="SO-ARM101-Lift-Cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100LiftJointCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM101-Lift-Cube-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm100LiftJointCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

##
# IK Absolute Environment
##
_ENV_CFG_IK_ABS = lift_ik_abs_env_cfg.SoArm100LiftIKAbsEnvCfg
_ENTRY_POINT_IK_ABS = f"{__name__}.lift_ik_abs_env:SoArm100LiftIKAbsEnv"

##
# IK Absolute Mimic Environment (for data generation)
##
_ENV_CFG_IK_ABS_MIMIC = lift_ik_abs_mimic_env_cfg.SoArm100LiftIKAbsMimicEnvCfg
_ENTRY_POINT_IK_ABS_MIMIC = f"{__name__}.lift_ik_abs_env:SoArm100LiftIKAbsEnv"

##
# Joint States Recording with MimicEnvCfg (records joint states directly, no conversion)
##
_ENV_CFG_JOINT_STATES_MIMIC = (
    lift_old_joint_states_mimic_env_cfg.SoArm100LiftJointStatesMimicEnvCfg
)
_ENTRY_POINT_JOINT_STATES_MIMIC = (
    f"{__name__}.lift_old_joint_states_mimic_env:SoArm100LiftJointStatesMimicEnv"
)

##
# Register Mimic environments
##

# IK Absolute Mimic (for data generation with subtask configs)
gym.register(
    id="SO-ARM101-Lift-IK-Abs-Mimic-v0",
    entry_point=_ENTRY_POINT_IK_ABS_MIMIC,
    kwargs={"env_cfg_entry_point": _ENV_CFG_IK_ABS_MIMIC},
    disable_env_checker=True,
)

# Joint States Recording with MimicEnvCfg (records joint states directly, no conversion)
gym.register(
    id="SO-ARM101-Lift-Joint-States-Mimic-v0",
    entry_point=_ENTRY_POINT_JOINT_STATES_MIMIC,
    kwargs={"env_cfg_entry_point": _ENV_CFG_JOINT_STATES_MIMIC},
    disable_env_checker=True,
)
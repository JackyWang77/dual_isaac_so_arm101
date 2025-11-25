# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym  # type: ignore[import]

from . import dual_pick_place_ik_rel_mimic_env_cfg
from . import dual_pick_place_ik_rel_mimic_env  # noqa: F401
from . import pick_place_ik_abs_env_cfg
from . import pick_place_ik_abs_env  # noqa: F401
from . import pick_place_joint_pos_env_cfg
from . import pick_place_joint_for_ik_abs_env  # noqa: F401
from . import pick_place_joint_for_ik_abs_mimic_env_cfg
from . import pick_place_joint_states_mimic_env  # noqa: F401
from . import pick_place_joint_states_mimic_env_cfg

##
# IK Relative Mimic Environment
##
_ENV_CFG_REL = (
    dual_pick_place_ik_rel_mimic_env_cfg.DualArmPickPlaceIKRelMimicEnvCfg
)
_ENTRY_POINT_REL = (
    f"{__name__}.dual_pick_place_ik_rel_mimic_env:"
    "DualArmPickPlaceIKRelMimicEnv"
)

##
# IK Absolute Environment
##
_ENV_CFG_ABS = pick_place_ik_abs_env_cfg.DualArmPickPlaceIKAbsEnvCfg
_ENTRY_POINT_ABS = f"{__name__}.pick_place_ik_abs_env:DualArmPickPlaceIKAbsEnv"

##
# Joint Control for IK Absolute Data Collection
##
_ENV_CFG_JOINT_FOR_IK_ABS = pick_place_joint_pos_env_cfg.DualArmPickPlaceJointPosEnvCfg
_ENTRY_POINT_JOINT_FOR_IK_ABS = f"{__name__}.pick_place_joint_for_ik_abs_env:DualArmPickPlaceJointForIKAbsEnv"

##
# Joint Control for IK Absolute Data Collection with MimicEnvCfg (subtask configs)
##
_ENV_CFG_JOINT_FOR_IK_ABS_MIMIC = (
    pick_place_joint_for_ik_abs_mimic_env_cfg.DualArmPickPlaceJointForIKAbsMimicEnvCfg
)
_ENTRY_POINT_JOINT_FOR_IK_ABS_MIMIC = (
    f"{__name__}.pick_place_joint_for_ik_abs_env:DualArmPickPlaceJointForIKAbsEnv"
)

##
# Joint States Recording with MimicEnvCfg (records joint states directly, no conversion)
##
_ENV_CFG_JOINT_STATES_MIMIC = (
    pick_place_joint_states_mimic_env_cfg.DualArmPickPlaceJointStatesMimicEnvCfg
)
_ENTRY_POINT_JOINT_STATES_MIMIC = (
    f"{__name__}.pick_place_joint_states_mimic_env:DualArmPickPlaceJointStatesMimicEnv"
)

##
# Register Gym environments.
##

# IK Relative Mimic
gym.register(
    id="SO-ARM100-Pick-Place-DualArm-IK-Rel-Mimic-v0",
    entry_point=_ENTRY_POINT_REL,
    kwargs={"env_cfg_entry_point": _ENV_CFG_REL},
    disable_env_checker=True,
)

# IK Absolute
gym.register(
    id="SO-ARM100-Pick-Place-DualArm-IK-Abs-v0",
    entry_point=_ENTRY_POINT_ABS,
    kwargs={"env_cfg_entry_point": _ENV_CFG_ABS},
    disable_env_checker=True,
)

# Joint Control for IK Absolute Data Collection
gym.register(
    id="SO-ARM100-Pick-Place-Joint-For-IK-Abs-v0",
    entry_point=_ENTRY_POINT_JOINT_FOR_IK_ABS,
    kwargs={"env_cfg_entry_point": _ENV_CFG_JOINT_FOR_IK_ABS},
    disable_env_checker=True,
)

# Joint Control for IK Absolute Data Collection with MimicEnvCfg (has subtask configs)
gym.register(
    id="SO-ARM100-Pick-Place-Joint-For-IK-Abs-Mimic-v0",
    entry_point=_ENTRY_POINT_JOINT_FOR_IK_ABS_MIMIC,
    kwargs={"env_cfg_entry_point": _ENV_CFG_JOINT_FOR_IK_ABS_MIMIC},
    disable_env_checker=True,
)

# Joint States Recording with MimicEnvCfg (records joint states directly, no conversion)
gym.register(
    id="SO-ARM100-Pick-Place-Joint-States-Mimic-v0",
    entry_point=_ENTRY_POINT_JOINT_STATES_MIMIC,
    kwargs={"env_cfg_entry_point": _ENV_CFG_JOINT_STATES_MIMIC},
    disable_env_checker=True,
)
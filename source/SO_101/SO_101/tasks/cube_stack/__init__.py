# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# Dual-arm cube stack: two SO-ARM101, two cubes, stack at fixed target (center).
# Pipeline: collect data (mimic) -> flow matching -> RL (same as single-arm).
#

import gymnasium as gym

from . import agents
from . import dual_cube_stack_ik_rel_mimic_env  # noqa: F401
from . import dual_cube_stack_ik_rel_mimic_env_cfg
from . import dual_cube_stack_joint_states_mimic_env  # noqa: F401
from . import dual_cube_stack_joint_states_mimic_env_cfg

_MIMIC_CFG = dual_cube_stack_ik_rel_mimic_env_cfg.DualCubeStackIKRelMimicEnvCfg
_MIMIC_ENTRY = f"{__name__}.dual_cube_stack_ik_rel_mimic_env:DualCubeStackIKRelMimicEnv"
_JOINT_STATES_MIMIC_CFG = dual_cube_stack_joint_states_mimic_env_cfg.DualCubeStackJointStatesMimicEnvCfg
_JOINT_STATES_MIMIC_ENTRY = f"{__name__}.dual_cube_stack_joint_states_mimic_env:DualCubeStackJointStatesMimicEnv"

gym.register(
    id="SO-ARM101-Dual-Cube-Stack-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_env_cfg:DualSoArm101CubeStackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CubeStackPPORecurrentRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM101-Dual-Cube-Stack-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_env_cfg:DualSoArm101CubeStackEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CubeStackPPORecurrentRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM101-Dual-Cube-Stack-IK-Rel-Mimic-v0",
    entry_point=_MIMIC_ENTRY,
    kwargs={"env_cfg_entry_point": _MIMIC_CFG},
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-v0",
    entry_point=_JOINT_STATES_MIMIC_ENTRY,
    kwargs={"env_cfg_entry_point": _JOINT_STATES_MIMIC_CFG},
    disable_env_checker=True,
)

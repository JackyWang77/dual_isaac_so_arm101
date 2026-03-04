# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# Dual-arm table setting: two SO-ARM101, fork + knife placed onto tray with plate.
# Elderly care use case: autonomous table setting.
#

import gymnasium as gym

from . import agents
from . import dual_table_setting_joint_states_mimic_env  # noqa: F401
from . import dual_table_setting_joint_states_mimic_env_cfg
from . import joint_pos_env_cfg

_JOINT_STATES_MIMIC_CFG = dual_table_setting_joint_states_mimic_env_cfg.DualTableSettingJointStatesMimicEnvCfg
_JOINT_STATES_MIMIC_ENTRY = f"{__name__}.dual_table_setting_joint_states_mimic_env:DualTableSettingJointStatesMimicEnv"
_JOINT_STATES_MIMIC_PLAY_CFG = dual_table_setting_joint_states_mimic_env_cfg.DualTableSettingJointStatesMimicEnvCfg_PLAY

gym.register(
    id="SO-ARM101-Dual-Table-Setting-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualSoArm101TableSettingJointPosEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TableSettingPPORecurrentRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM101-Dual-Table-Setting-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:DualSoArm101TableSettingJointPosEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TableSettingPPORecurrentRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM101-Dual-Table-Setting-Joint-States-Mimic-v0",
    entry_point=_JOINT_STATES_MIMIC_ENTRY,
    kwargs={"env_cfg_entry_point": _JOINT_STATES_MIMIC_CFG},
    disable_env_checker=True,
)

gym.register(
    id="SO-ARM101-Dual-Table-Setting-Joint-States-Mimic-Play-v0",
    entry_point=_JOINT_STATES_MIMIC_ENTRY,
    kwargs={"env_cfg_entry_point": _JOINT_STATES_MIMIC_PLAY_CFG},
    disable_env_checker=True,
)

# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# Joint-states Mimic env config for dual-arm table setting (record demos from real robot).
#

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .table_setting_env_cfg import TableSettingSubtaskCfg
from .joint_pos_env_cfg import DualSoArm101TableSettingJointPosEnvCfg


@configclass
class DualTableSettingJointStatesMimicEnvCfg(
    DualSoArm101TableSettingJointPosEnvCfg, MimicEnvCfg
):
    """
    Record joint states for dual-arm table setting.
    Action = right_arm(6) + left_arm(6) = 12.
    """

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.concatenate_terms = False
        self.observations.policy.enable_corruption = False

        self.observations.subtask_terms = TableSettingSubtaskCfg()

        self.datagen_config.name = "table_setting"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_relative = False
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = [
            SubTaskConfig(
                object_ref="fork",
                subtask_term_signal="place_fork",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place fork",
                next_subtask_description="Place knife",
            ),
            SubTaskConfig(
                object_ref="knife",
                subtask_term_signal="place_knife",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place knife",
            ),
        ]
        self.subtask_configs["dual_arm"] = subtask_configs


@configclass
class DualTableSettingJointStatesMimicEnvCfg_PLAY(DualTableSettingJointStatesMimicEnvCfg):
    """Play config: gripper = BinaryJoint (model outputs 1/-1)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        from . import mdp
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["jaw_joint"],
            open_command_expr={"jaw_joint": 0.4},
            close_command_expr={"jaw_joint": 0.0002},
        )
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["jaw_joint"],
            open_command_expr={"jaw_joint": 0.4},
            close_command_expr={"jaw_joint": 0.0002},
        )

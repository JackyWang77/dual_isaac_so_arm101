# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .pick_place_joint_pos_env_cfg import DualArmPickPlaceJointPosEnvCfg


@configclass
class DualArmPickPlaceJointStatesMimicEnvCfg(DualArmPickPlaceJointPosEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for recording joint states directly.
    
    This environment:
    - Accepts joint_states control (can be controlled by real robot)
    - Records joint states directly (joint positions + gripper) - no conversion
    - Has subtask configurations for data generation
    - Joint states can be converted to EE pose later using forward kinematics
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "demo_src_pick_place_joint_states_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = False  # Joint states are absolute
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # ✅ Only 2 subtasks for testing: push_plate and pick_fork
        subtask_configs = []
        # Push plate
        subtask_configs.append(
            SubTaskConfig(
                object_ref="plate",
                subtask_term_signal="push_plate",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Push plate",
                next_subtask_description="Pick fork",
            )
        )
        # Pick fork (final subtask)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="fork",
                subtask_term_signal="pick_fork",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Pick fork",
            )
        )
        self.subtask_configs["end_effector"] = subtask_configs


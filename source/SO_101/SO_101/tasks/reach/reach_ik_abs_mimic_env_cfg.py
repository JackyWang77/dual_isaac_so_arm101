# Copyright (c) 2024-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""IK Absolute Mimic environment configuration for reach task data generation.

This environment:
- Uses IK Absolute control (action = EEF pose)
- Has subtask configurations for Mimic data generation
- Generates data suitable for training IK Abs policies
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .reach_ik_abs_env_cfg import SoArm100ReachIKAbsEnvCfg


@configclass
class SoArm100ReachIKAbsMimicEnvCfg(SoArm100ReachIKAbsEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for IK Absolute env.

    This environment:
    - Uses IK Absolute control (action = target EEF pose)
    - Has subtask configurations for data generation
    - Generates EEF pose actions for training IK Abs policies
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Disable randomize_object_positions event for Mimic data generation
        # Mimic will set initial states from annotated demos, so we don't want randomization
        if hasattr(self.events, "randomize_object_positions"):
            delattr(self.events, "randomize_object_positions")

        # Override the existing values
        self.datagen_config.name = "demo_src_reach_ik_abs"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = False  # IK-Abs uses absolute poses
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Subtask configuration for the REACH task
        subtask_configs = []

        # Subtask: Reach object (EE close to object and gripper closed)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",  # Reference to the cube object
                subtask_term_signal="reach_object",  # Termination signal name
                subtask_term_offset_range=(0, 0),  # Final subtask, no offset
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Reach object with EE and close gripper",
            )
        )

        self.subtask_configs["end_effector"] = subtask_configs

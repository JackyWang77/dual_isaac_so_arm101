# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .joint_pos_env_cfg import SoArm100LiftJointCubeEnvCfg


@configclass
class SoArm100LiftJointStatesMimicEnvCfg(SoArm100LiftJointCubeEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for recording joint states directly for lift task.

    This environment:
    - Accepts joint_states control (can be controlled by real robot)
    - Records joint states directly (joint positions + gripper) - no conversion
    - Has subtask configurations for data generation
    - Joint states can be converted to EE pose later using forward kinematics
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Disable concatenation for Mimic recording (recorder needs dict format)
        self.observations.policy.concatenate_terms = False
        self.observations.policy.enable_corruption = False

        # Override the existing values
        self.datagen_config.name = "lift"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = False  # Joint states are absolute
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Subtask configuration for the LIFT task
        subtask_configs = []

        # Subtask: Lift object to target height
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",  # Reference to the cube object
                subtask_term_signal="lift_object",  # Termination signal name
                subtask_term_offset_range=(0, 0),  # Final subtask, no offset
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Lift object to target height",
            )
        )

        self.subtask_configs["end_effector"] = subtask_configs

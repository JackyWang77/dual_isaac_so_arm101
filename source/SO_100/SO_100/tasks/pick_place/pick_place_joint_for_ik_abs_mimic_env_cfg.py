# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .pick_place_joint_pos_env_cfg import DualArmPickPlaceJointPosEnvCfg


@configclass
class DualArmPickPlaceJointForIKAbsMimicEnvCfg(DualArmPickPlaceJointPosEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Joint-For-IK-Abs env.
    
    This environment:
    - Accepts joint_states control (can be controlled by real robot)
    - Records EE absolute poses (for training IK Absolute policies)
    - Has subtask configurations for data generation
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()
        # # TODO: Figure out how we can move this to the MimicEnvCfg class
        # # The __post_init__() above only calls the init for DualArmPickPlaceJointPosEnvCfg and not MimicEnvCfg
        # # https://stackoverflow.com/questions/59986413/achieving-multiple-inheritance-using-python-dataclasses

        # Override the existing values
        self.datagen_config.name = "demo_src_pick_place_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = False  # IK-Abs uses absolute poses
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # ✅ Keep only two subtasks for testing: push_plate and pick_fork
        # The following are the subtask configurations for the pick and place task.
        subtask_configs = []
        # Push plate (instead of pick & place)
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="plate",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="push_plate",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(10, 20),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,  # Set to zero - no randomization
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
                description="Push plate",
                next_subtask_description="Pick fork",
            )
        )
        # Pick fork
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="fork",
                # Corresponding key for the binary indicator in "datagen_info" for completion
                subtask_term_signal="pick_fork",
                # Time offsets for data generation when splitting a trajectory
                subtask_term_offset_range=(10, 20),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,  # Set to zero - no randomization
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
                # No next subtask (this is the last one for testing)
            )
        )
        # ❌ Commented out for testing - will add back later
        # # Place fork
        # subtask_configs.append(
        #     SubTaskConfig(
        #         # Each subtask involves manipulation with respect to a single object frame.
        #         object_ref="fork",
        #         # Corresponding key for the binary indicator in "datagen_info" for completion
        #         subtask_term_signal="place_fork",
        #         # Time offsets for data generation when splitting a trajectory
        #         subtask_term_offset_range=(10, 20),
        #         # Selection strategy for source subtask segment
        #         selection_strategy="nearest_neighbor_object",
        #         # Optional parameters for the selection strategy function
        #         selection_strategy_kwargs={"nn_k": 3},
        #         # Amount of action noise to apply during this subtask
        #         action_noise=0.03,
        #         # Number of interpolation steps to bridge to this subtask segment
        #         num_interpolation_steps=5,
        #         # Additional fixed steps for the robot to reach the necessary pose
        #         num_fixed_steps=0,
        #         # If True, apply action noise during the interpolation phase and execution
        #         apply_noise_during_interpolation=False,
        #     )
        # )
        # # Pick knife
        # subtask_configs.append(
        #     SubTaskConfig(
        #         # Each subtask involves manipulation with respect to a single object frame.
        #         object_ref="knife",
        #         # Corresponding key for the binary indicator in "datagen_info" for completion
        #         subtask_term_signal="pick_knife",
        #         # Time offsets for data generation when splitting a trajectory
        #         subtask_term_offset_range=(10, 20),
        #         # Selection strategy for source subtask segment
        #         selection_strategy="nearest_neighbor_object",
        #         # Optional parameters for the selection strategy function
        #         selection_strategy_kwargs={"nn_k": 3},
        #         # Amount of action noise to apply during this subtask
        #         action_noise=0.03,
        #         # Number of interpolation steps to bridge to this subtask segment
        #         num_interpolation_steps=5,
        #         # Additional fixed steps for the robot to reach the necessary pose
        #         num_fixed_steps=0,
        #         # If True, apply action noise during the interpolation phase and execution
        #         apply_noise_during_interpolation=False,
        #         next_subtask_description="Place knife",
        #     )
        # )
        # # Place knife
        # subtask_configs.append(
        #     SubTaskConfig(
        #         # Each subtask involves manipulation with respect to a single object frame.
        #         object_ref="knife",
        #         # End of final subtask does not need to be detected
        #         subtask_term_signal="place_knife",
        #         # No time offsets for the final subtask
        #         subtask_term_offset_range=(0, 0),
        #         # Selection strategy for source subtask segment
        #         selection_strategy="nearest_neighbor_object",
        #         # Optional parameters for the selection strategy function
        #         selection_strategy_kwargs={"nn_k": 3},
        #         # Amount of action noise to apply during this subtask
        #         action_noise=0.03,
        #         # Number of interpolation steps to bridge to this subtask segment
        #         num_interpolation_steps=5,
        #         # Additional fixed steps for the robot to reach the necessary pose
        #         num_fixed_steps=0,
        #         # If True, apply action noise during the interpolation phase and execution
        #         apply_noise_during_interpolation=False,
        #     )
        # )
        self.subtask_configs["dual_arm"] = subtask_configs


# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# Joint-states Mimic env for dual-arm cube stack (record demos from real robot / joint_states).
# Same flow as lift_old: SO-ARM101-Lift-Joint-States-Mimic-v0.
#

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .joint_pos_env_cfg import DualSoArm101CubeStackJointPosEnvCfg


@configclass
class DualCubeStackJointStatesMimicEnvCfg(
    DualSoArm101CubeStackJointPosEnvCfg, MimicEnvCfg
):
    """
    Record joint states directly for dual-arm cube stack.
    - Action = right_arm(6) + left_arm(6) = 12.
    - Teleop: joint_states (e.g. ROS2 /joint_states with 12 dof).
    """

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.concatenate_terms = False
        self.observations.policy.enable_corruption = False

        # Subtask signals for data generation (must match get_subtask_term_signals)
        self.observations.subtask_terms = ObsGroup(
            pick_cube_top=ObsTerm(
                func=mdp.object_is_lifted,
                params={
                    "minimal_height": 0.04,
                    "object_cfg": SceneEntityCfg("object"),
                },
            ),
            stack_cube=ObsTerm(
                func=mdp.cube_stacked,
                params={
                    "xy_threshold": 0.025,
                    "z_tolerance": 0.01,
                    "cube_top_cfg": SceneEntityCfg("object"),
                    "cube_base_cfg": SceneEntityCfg("cube_base"),
                },
            ),
        )
        self.observations.subtask_terms.enable_corruption = False
        self.observations.subtask_terms.concatenate_terms = False

        self.datagen_config.name = "cube_stack"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_relative = False
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = [
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="pick_cube_top",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Pick cube top",
                next_subtask_description="Stack on base",
            ),
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="stack_cube",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Stack on base",
            ),
        ]
        self.subtask_configs["dual_arm"] = subtask_configs

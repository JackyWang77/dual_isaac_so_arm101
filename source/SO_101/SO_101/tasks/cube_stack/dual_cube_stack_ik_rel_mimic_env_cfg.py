# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .cube_stack_env_cfg import CubeStackSubtaskCfg
from .joint_env_cfg import DualSoArm101CubeStackEnvCfg


@configclass
class DualCubeStackIKRelMimicEnvCfg(DualSoArm101CubeStackEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Subtask signals for get_subtask_term_signals (annotated datagen)
        self.observations.subtask_terms = CubeStackSubtaskCfg()
        self.datagen_config.name = "demo_src_cube_stack_dual_arm"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = [
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="pick_cube",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Pick cube",
                next_subtask_description="Stack cube",
            ),
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="stack_cube",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Stack cube",
            ),
        ]
        self.subtask_configs["dual_arm"] = subtask_configs

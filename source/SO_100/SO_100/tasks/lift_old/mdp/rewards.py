# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    initial_height: float = 0.015,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object. Reward scales with lift distance.

    Args:
        env: The environment.
        minimal_height: The target height to reach (reward = 1.0 at this height).
        initial_height: The initial height of the object (reward = 0.0 at this height).
        object_cfg: The object configuration.

    Returns:
        Reward in range [0.0, 2.0] based on how high the object is lifted.
        Reward increases linearly from 0.0 (at initial_height) to 1.0 (at minimal_height),
        and continues to increase beyond minimal_height (capped at 2.0 for heights >= minimal_height * 2).
    """
    object: RigidObject = env.scene[object_cfg.name]
    current_height = object.data.root_pos_w[:, 2]

    # Calculate lift distance from initial height
    lift_distance = current_height - initial_height

    # Normalize reward: 0.0 at initial_height, 1.0 at minimal_height
    # Allow reward to continue increasing beyond minimal_height (capped at 2.0)
    target_lift_distance = minimal_height - initial_height
    if target_lift_distance > 0:
        # Linear scaling: reward = lift_distance / target_lift_distance
        # Cap at 2.0 to prevent unbounded rewards
        reward = torch.clamp(lift_distance / target_lift_distance, 0.0, 2.0)
    else:
        # Fallback if target <= initial (shouldn't happen normally)
        reward = torch.where(current_height > minimal_height, 1.0, 0.0)

    return reward


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it."""
    # Get reaching reward
    reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
    # Get lifting reward
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)
    # Combine rewards multiplicatively
    return reach_reward * lift_reward

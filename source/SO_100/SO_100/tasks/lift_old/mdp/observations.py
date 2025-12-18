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
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position of the end-effector in the robot's root frame.

    Returns:
        torch.Tensor: EE position in robot base frame [num_envs, 3]
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # [num_envs, 3]
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w
    )
    return ee_pos_b


def ee_orientation(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The orientation (quaternion) of the end-effector in world frame.

    Returns:
        torch.Tensor: EE orientation quaternion [num_envs, 4] (w, x, y, z)
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # [num_envs, 4]
    return ee_quat_w


def object_orientation(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation (quaternion) of the object in world frame.

    Returns:
        torch.Tensor: Object orientation quaternion [num_envs, 4] (w, x, y, z)
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_quat_w = object.data.root_quat_w  # [num_envs, 4]
    return object_quat_w


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.04,
    initial_height: float = 0.015,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if the object is lifted above a minimum height.
    
    Used as a subtask signal for "lift object" task completion.
    Checks if object is lifted to at least minimal_height above its initial position.
    
    Args:
        env: The environment
        minimal_height: Minimum height (in meters) above initial position for object to be considered "lifted"
        initial_height: Initial height of the object (default 0.015m for cube)
        object_cfg: Configuration for the object
        
    Returns:
        Boolean tensor indicating if object is above minimal_height (0.0 or 1.0)
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object height (z position in world frame)
    object_height = object.data.root_pos_w[:, 2]  # [num_envs]
    
    # Check if object is lifted to at least minimal_height above initial position
    lifted = object_height >= (initial_height + minimal_height)

    return lifted.float()  # Return float tensor (0.0 or 1.0) for observations
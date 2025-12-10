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


def object_reached(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.01,
    gripper_closed_threshold: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Check if the end-effector has reached the object and gripper is closed.
    
    Used as a subtask signal for "reach object" task completion.
    Checks if:
    1. Distance between EE and object is less than distance_threshold (default 0.01m)
    2. Gripper is closed (jaw_joint position < gripper_closed_threshold)
    
    Args:
        env: The environment
        distance_threshold: Maximum distance (in meters) for EE to be considered "reached" (default 0.01m)
        gripper_closed_threshold: Maximum jaw_joint position for gripper to be considered closed (default 0.15)
        object_cfg: Configuration for the object
        ee_frame_cfg: Configuration for the end-effector frame
        robot_cfg: Configuration for the robot
        
    Returns:
        Boolean tensor indicating if both conditions are met (0.0 or 1.0)
    """
    # Get objects
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get EE and object positions
    object_pos_w = object.data.root_pos_w[:, :3]  # [num_envs, 3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # [num_envs, 3]
    
    # Compute distance
    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)  # [num_envs]
    
    # Check distance condition
    distance_reached = distance < distance_threshold
    
    # Check gripper closed condition
    # Find jaw_joint index
    gripper_joint_names = ["jaw_joint"]
    all_joint_names = robot.joint_names
    gripper_joint_indices = [all_joint_names.index(name) for name in gripper_joint_names if name in all_joint_names]
    
    if len(gripper_joint_indices) > 0:
        gripper_pos = robot.data.joint_pos[:, gripper_joint_indices[0]]  # [num_envs]
        gripper_closed = gripper_pos < gripper_closed_threshold
    else:
        # If gripper joint not found, assume it's closed
        gripper_closed = torch.ones_like(distance, dtype=torch.bool)
    
    # Both conditions must be met
    reached = torch.logical_and(distance_reached, gripper_closed)
    
    return reached.float()  # Convert to float (0.0 or 1.0)

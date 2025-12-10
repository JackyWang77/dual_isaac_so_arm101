# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the reach task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reach_success(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.01,
    gripper_closed_threshold: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Termination condition for successfully reaching the object.
    
    Success condition:
    1. End-effector distance to object < distance_threshold (default 0.01m)
    2. Gripper is closed (jaw_joint position < gripper_closed_threshold, default 0.15)
    
    This is used as the success termination for recording demos.
    Named "success" so that record_demos.py can automatically detect task completion.
    
    Args:
        env: The environment
        distance_threshold: Maximum distance (in meters) for EE to be considered "reached" (default 0.01m)
        gripper_closed_threshold: Maximum jaw_joint position for gripper to be considered closed (default 0.15)
        object_cfg: Configuration for the object
        ee_frame_cfg: Configuration for the end-effector frame
        robot_cfg: Configuration for the robot
        
    Returns:
        Boolean tensor indicating success (True when both conditions are met)
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
    success = torch.logical_and(distance_reached, gripper_closed)
    
    return success

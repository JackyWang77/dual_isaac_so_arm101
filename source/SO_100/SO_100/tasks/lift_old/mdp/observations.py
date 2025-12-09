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
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
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


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if the object is lifted above a minimum height.
    
    Used as a subtask signal for "lift object" task completion.
    
    Args:
        env: The environment
        minimal_height: Minimum height (in meters) for object to be considered "lifted"
        object_cfg: Configuration for the object
        
    Returns:
        Boolean tensor indicating if object is above minimal_height
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object height (z position in world frame)
    object_height = object.data.root_pos_w[:, 2]  # [num_envs]
    
    lifted = object_height > minimal_height
    
    return lifted.float()  # Convert to float (0.0 or 1.0)
# Copyright (c) 2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Import robot-relative observation functions
from .observations import (
    object_pos_in_robot_frame,
    ee_pos_in_robot_frame,
    ee_quat_in_robot_frame,
)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height (relative to robot base)."""
    obj_pos_b = object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    return torch.where(obj_pos_b[:, 2] > minimal_height, 1.0, 0.0)

def graps_intent(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.04,
    max_distance: float = 0.05,  # Relaxed slightly for easier grasping
    gripper_closed_threshold: float = 0.35,  # Must be LESS than 'Open' (0.4)
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for grasping the object.
    
    Logic:
    - Gripper close_command = 0.0 (closed)
    - Gripper open_command = 0.4 (open)
    - So gripper_pos < threshold means gripper is CLOSING (good!)
    """
    robot = env.scene[robot_cfg.name]
    obj_pos_b = object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    ee_pos_b = ee_pos_in_robot_frame(env, robot_cfg, ee_frame_cfg)
    
    # 1. Check distance
    dist = torch.norm(obj_pos_b - ee_pos_b, dim=1)
    
    # 2. Check gripper - FIX: use LESS THAN (<) because 0.0 is closed!
    gripper_pos = robot.data.joint_pos[:, -1]
    
    # Logic: If distance is small AND gripper is closing (pos < threshold), give reward
    is_near = dist < max_distance
    is_closing = gripper_pos < gripper_closed_threshold
    
    return torch.where(torch.logical_and(is_near, is_closing), 1.0, 0.0)



def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel (robot-relative positions)."""
    obj_pos_b = object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    ee_pos_b = ee_pos_in_robot_frame(env, robot_cfg, ee_frame_cfg)
    
    object_ee_distance = torch.norm(obj_pos_b - ee_pos_b, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def gripper_penalty_when_far(
    env: ManagerBasedRLEnv,
    threshold_dist: float = 0.05,  # 5cm
    threshold_gripper: float = 0.35,  # Below this means "Trying to close"
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize the agent if it closes the gripper while being too far from the object.
    This prevents 'premature grasping' - closing the hand before the object is inside.
    
    Logic:
    - If distance > threshold_dist (too far) AND gripper < threshold_gripper (closing)
    - Return 1.0 (to be penalized with negative weight)
    """
    # 1. Calculate Distance
    obj_pos_b = object_pos_in_robot_frame(env, robot_cfg, object_cfg)
    ee_pos_b = ee_pos_in_robot_frame(env, robot_cfg, ee_frame_cfg)
    dist = torch.norm(obj_pos_b - ee_pos_b, dim=1)
    
    # 2. Get Gripper State (last joint is gripper)
    robot = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -1]
    
    # 3. Logic: If (Dist > threshold) AND (Gripper is closing), return 1.0 (to be penalized)
    is_far = dist > threshold_dist
    is_closing = gripper_pos < threshold_gripper
    
    # Return 1.0 where the bad behavior happens, 0.0 otherwise
    # Multiply this by a negative weight in config to penalize
    return torch.where(torch.logical_and(is_far, is_closing), 1.0, 0.0)

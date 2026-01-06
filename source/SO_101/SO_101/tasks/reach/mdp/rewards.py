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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
    cube_pos_w = object.data.root_pos_w[:, :3]
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def gripper_closing_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for closing gripper when close to object.

    Args:
        env: The environment
        std: Standard deviation for distance threshold (when close to object)
        object_cfg: Configuration for the object
        ee_frame_cfg: Configuration for the end-effector frame
        robot_cfg: Configuration for the robot

    Returns:
        Reward tensor encouraging gripper closing when near object
    """
    # Get objects
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Compute distance to object
    object_pos_w = object.data.root_pos_w[:, :3]  # [num_envs, 3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # [num_envs, 3]
    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)  # [num_envs]

    # Check if close to object (within std threshold)
    close_to_object = distance < std

    # Get gripper position
    gripper_joint_names = ["jaw_joint"]
    all_joint_names = robot.joint_names
    gripper_joint_indices = [
        all_joint_names.index(name)
        for name in gripper_joint_names
        if name in all_joint_names
    ]

    if len(gripper_joint_indices) > 0:
        gripper_pos = robot.data.joint_pos[:, gripper_joint_indices[0]]  # [num_envs]
        # Reward closing: closer to 0.0 is better (jaw_joint: 0.3 = open, 0.0 = closed)
        # Normalize: (0.3 - gripper_pos) / 0.3 gives 1.0 when closed, 0.0 when open
        gripper_closing = (0.3 - gripper_pos) / 0.3
        gripper_closing = torch.clamp(gripper_closing, 0.0, 1.0)
    else:
        gripper_closing = torch.zeros_like(distance)

    # Only reward gripper closing when close to object
    reward = close_to_object.float() * gripper_closing

    return reward

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


def grasp(
    env: ManagerBasedRLEnv,
    std: float,
    distance_threshold: float = 0.002,  # 0.2cm in meters
    open_joint_pos: float = 0.3,  # jaw_joint position when open
    close_joint_pos: float = 0.01,  # jaw_joint position when closed
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for appropriate gripper action based on distance to object.
    
    - When distance < threshold (0.2cm): encourage closing gripper
    - When distance >= threshold: encourage opening gripper
    
    Args:
        env: The environment.
        std: Standard deviation for reward scaling (unused, kept for compatibility).
        distance_threshold: Distance threshold in meters (default 0.002m = 0.2cm).
        open_joint_pos: Joint position when gripper is open (default 0.3).
        close_joint_pos: Joint position when gripper is closed (default 0.01).
        robot_cfg: Robot configuration.
        object_cfg: Object configuration.
        ee_frame_cfg: End-effector frame configuration.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w[:, :3]
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    
    # Get gripper joint position (jaw_joint is the last joint)
    gripper_joint_pos = robot.data.joint_pos[:, -1]  # (num_envs,)
    
    # Determine if gripper is closed (closer to close_joint_pos than open_joint_pos)
    # Use a stricter threshold: must be closer to close position than open position
    # Also add a tolerance: must be within 20% of the range from close position
    gripper_range = open_joint_pos - close_joint_pos
    close_tolerance = close_joint_pos + 0.2 * gripper_range  # 20% of range from close position
    
    # Gripper is considered closed if:
    # 1. It's closer to close position than open position, AND
    # 2. It's within the tolerance range (not too far from close position)
    distance_to_close = torch.abs(gripper_joint_pos - close_joint_pos)
    distance_to_open = torch.abs(gripper_joint_pos - open_joint_pos)
    is_closer_to_close = (distance_to_close < distance_to_open).float()
    is_within_tolerance = (gripper_joint_pos <= close_tolerance).float()
    is_closed = (is_closer_to_close * is_within_tolerance)  # 1.0 if closed, 0.0 if open
    
    # Reward logic:
    # - Distance < threshold AND closed: positive reward (encourage closing when close)
    # - Distance < threshold AND open: negative reward (discourage opening when close)
    # - Distance >= threshold AND open: small positive reward (slightly encourage opening when far)
    # - Distance >= threshold AND closed: negative reward (discourage closing when far)
    
    is_close = (object_ee_distance < distance_threshold).float()  # 1.0 if close, 0.0 if far
    
    # Reward when behavior matches distance:
    # - Close + Closed = good (reward = 1.0) - strong reward for closing when close
    # - Far + Open = okay (reward = 0.2) - small reward for opening when far
    # - Close + Open = bad (reward = -1.0) - strong penalty for opening when close
    # - Far + Closed = bad (reward = -1.0) - strong penalty for closing when far
    
    # Strong reward for closing when close, small reward for opening when far
    reward = is_close * is_closed * 1.0 + (1 - is_close) * (1 - is_closed) * 0.1 - \
             (is_close * (1 - is_closed) + (1 - is_close) * is_closed) * 1.0
    
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

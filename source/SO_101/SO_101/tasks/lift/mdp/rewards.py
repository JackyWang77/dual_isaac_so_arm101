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
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


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
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (
        1 - torch.tanh(distance / std)
    )


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


def closer_arm_reaches_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    ee_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
) -> torch.Tensor:
    """Reward only the closer arm for reaching the object.

    This encourages the closer arm to reach while the farther arm stays still.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_right: FrameTransformer = env.scene[ee_right_cfg.name]
    ee_left: FrameTransformer = env.scene[ee_left_cfg.name]

    # Object position
    obj_pos = object.data.root_pos_w

    # End-effector positions
    ee_right_pos = ee_right.data.target_pos_w[..., 0, :]
    ee_left_pos = ee_left.data.target_pos_w[..., 0, :]

    # Calculate distances
    dist_right = torch.norm(obj_pos - ee_right_pos, dim=1)
    dist_left = torch.norm(obj_pos - ee_left_pos, dim=1)

    # Determine which arm is closer (1.0 for closer, 0.0 for farther)
    right_is_closer = (dist_right < dist_left).float()
    left_is_closer = (dist_left < dist_right).float()

    # Reward only the closer arm for reaching
    right_reward = right_is_closer * (1 - torch.tanh(dist_right / std))
    left_reward = left_is_closer * (1 - torch.tanh(dist_left / std))

    # Return combined reward (only one arm gets rewarded at a time)
    return right_reward + left_reward


def farther_arm_stays_still(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    ee_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
) -> torch.Tensor:
    """Reward the farther arm for staying still (low joint velocities).

    This encourages the farther arm to not move while the closer arm reaches.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_right: FrameTransformer = env.scene[ee_right_cfg.name]
    ee_left: FrameTransformer = env.scene[ee_left_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]
    left_arm = env.scene[left_arm_cfg.name]

    # Object position
    obj_pos = object.data.root_pos_w

    # End-effector positions
    ee_right_pos = ee_right.data.target_pos_w[..., 0, :]
    ee_left_pos = ee_left.data.target_pos_w[..., 0, :]

    # Calculate distances
    dist_right = torch.norm(obj_pos - ee_right_pos, dim=1)
    dist_left = torch.norm(obj_pos - ee_left_pos, dim=1)

    # Determine which arm is farther
    right_is_farther = (dist_right > dist_left).float()
    left_is_farther = (dist_left > dist_right).float()

    # Calculate joint velocity magnitude for each arm
    right_vel_mag = torch.norm(right_arm.data.joint_vel, dim=1)
    left_vel_mag = torch.norm(left_arm.data.joint_vel, dim=1)

    # Reward farther arm for staying still (lower velocity = higher reward)
    right_stillness = right_is_farther * torch.exp(-right_vel_mag)
    left_stillness = left_is_farther * torch.exp(-left_vel_mag)

    return right_stillness + left_stillness


def grasp_intent(
    env: ManagerBasedRLEnv,
    proximity_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    ee_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
) -> torch.Tensor:
    """Reward the closer arm for closing gripper when near the object.

    This solves the "hovering" problem where the agent learns to approach
    but never actually grasps. When ee is close enough, reward gripper closing.

    Args:
        proximity_threshold: Distance threshold to start rewarding gripper closing.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_right: FrameTransformer = env.scene[ee_right_cfg.name]
    ee_left: FrameTransformer = env.scene[ee_left_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]
    left_arm = env.scene[left_arm_cfg.name]

    # Object position
    obj_pos = object.data.root_pos_w

    # End-effector positions
    ee_right_pos = ee_right.data.target_pos_w[..., 0, :]
    ee_left_pos = ee_left.data.target_pos_w[..., 0, :]

    # Calculate distances
    dist_right = torch.norm(obj_pos - ee_right_pos, dim=1)
    dist_left = torch.norm(obj_pos - ee_left_pos, dim=1)

    # Determine which arm is closer
    right_is_closer = (dist_right < dist_left).float()
    left_is_closer = (dist_left < dist_right).float()

    # Get gripper joint positions (jaw_joint: 0.698 = open, 0.0 = closed)
    # jaw_joint is the last joint
    right_gripper_pos = right_arm.data.joint_pos[:, -1]  # jaw_joint
    left_gripper_pos = left_arm.data.joint_pos[:, -1]  # jaw_joint

    # Gripper closing reward: higher when gripper is more closed (lower pos)
    # Normalized: (0.698 - pos) / 0.698 gives 0 when open, 1 when closed
    right_gripper_closed = (0.698 - right_gripper_pos) / 0.698
    left_gripper_closed = (0.698 - left_gripper_pos) / 0.698

    # Only reward gripper closing when close to object
    right_near = (dist_right < proximity_threshold).float()
    left_near = (dist_left < proximity_threshold).float()

    # Reward = closer_arm * near_object * gripper_closing
    right_reward = right_is_closer * right_near * right_gripper_closed
    left_reward = left_is_closer * left_near * left_gripper_closed

    return right_reward + left_reward

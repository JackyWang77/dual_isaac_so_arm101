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
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    ee_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
) -> torch.Tensor:
    """Reward both arms for reaching the object using tanh-kernel (dual-arm version)."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_right: FrameTransformer = env.scene[ee_right_cfg.name]
    ee_left: FrameTransformer = env.scene[ee_left_cfg.name]

    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w

    # End-effector positions: (num_envs, 3)
    ee_right_w = ee_right.data.target_pos_w[..., 0, :]
    ee_left_w = ee_left.data.target_pos_w[..., 0, :]

    # Distance of each end-effector to the object: (num_envs,)
    distance_right = torch.norm(cube_pos_w - ee_right_w, dim=1)
    distance_left = torch.norm(cube_pos_w - ee_left_w, dim=1)

    # Reward for each arm using tanh-kernel
    reward_right = 1 - torch.tanh(distance_right / std)
    reward_left = 1 - torch.tanh(distance_left / std)

    # Average reward across both arms (keep same structure as single-arm)
    return 0.5 * (reward_right + reward_left)


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    ee_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it (dual-arm version)."""
    # Get reaching reward
    reach_reward = object_ee_distance(env, std, object_cfg, ee_right_cfg, ee_left_cfg)

    # Get lifting reward
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)

    # Combine rewards multiplicatively
    return reach_reward * lift_reward


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel (dual-arm version).

    This function works with dual-arm setup by tracking a single command (e.g., object_pose_right).
    For dual-arm scenarios, you can use this with either right_arm or left_arm robot_cfg.
    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    # distance of the goal position to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def gripper_close_reward(
    env: ManagerBasedRLEnv,
    std: float,
    close_threshold: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
    ee_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    ee_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
    gripper_joint_name: str = "jaw_joint",
) -> torch.Tensor:
    """Reward both grippers for closing near the object; averaged across arms."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w[:, :3]

    ee_r: FrameTransformer = env.scene[ee_right_cfg.name]
    ee_l: FrameTransformer = env.scene[ee_left_cfg.name]
    ee_right = ee_r.data.target_pos_w[:, 0, :] if ee_r.data.target_pos_w.ndim >= 3 else ee_r.data.target_pos_w
    ee_left = ee_l.data.target_pos_w[:, 0, :] if ee_l.data.target_pos_w.ndim >= 3 else ee_l.data.target_pos_w

    arm_r: RigidObject = env.scene[right_arm_cfg.name]
    arm_l: RigidObject = env.scene[left_arm_cfg.name]

    # find gripper indices
    idx_r = next((i for i, n in enumerate(arm_r.joint_names) if n == gripper_joint_name), None)
    idx_l = next((i for i, n in enumerate(arm_l.joint_names) if n == gripper_joint_name), None)
    if idx_r is None or idx_l is None:
        return torch.zeros(env.num_envs, device=env.device)

    # joint states
    grip_r = arm_r.data.joint_pos[:, idx_r] < close_threshold
    grip_l = arm_l.data.joint_pos[:, idx_l] < close_threshold

    # proximity shaping
    dist_r = torch.norm(obj_pos - ee_right, dim=1)
    dist_l = torch.norm(obj_pos - ee_left, dim=1)
    prox_r = 1 - torch.tanh(dist_r / std)
    prox_l = 1 - torch.tanh(dist_l / std)

    reward_r = grip_r.float() * prox_r
    reward_l = grip_l.float() * prox_l
    return 0.5 * (reward_r + reward_l)


def successful_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    close_threshold: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
    ee_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    ee_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
    gripper_joint_name: str = "jaw_joint",
) -> torch.Tensor:
    """Reward when either or both grippers successfully grasp and lift the object."""
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos = obj.data.root_pos_w[:, :3]
    is_lifted = obj.data.root_pos_w[:, 2] > minimal_height

    ee_r: FrameTransformer = env.scene[ee_right_cfg.name]
    ee_l: FrameTransformer = env.scene[ee_left_cfg.name]
    ee_right = ee_r.data.target_pos_w[:, 0, :] if ee_r.data.target_pos_w.ndim >= 3 else ee_r.data.target_pos_w
    ee_left = ee_l.data.target_pos_w[:, 0, :] if ee_l.data.target_pos_w.ndim >= 3 else ee_l.data.target_pos_w

    arm_r: RigidObject = env.scene[right_arm_cfg.name]
    arm_l: RigidObject = env.scene[left_arm_cfg.name]

    idx_r = next((i for i, n in enumerate(arm_r.joint_names) if n == gripper_joint_name), None)
    idx_l = next((i for i, n in enumerate(arm_l.joint_names) if n == gripper_joint_name), None)
    if idx_r is None or idx_l is None:
        return torch.zeros(env.num_envs, device=env.device)

    grip_r_closed = arm_r.data.joint_pos[:, idx_r] < close_threshold
    grip_l_closed = arm_l.data.joint_pos[:, idx_l] < close_threshold

    dist_r = torch.norm(obj_pos - ee_right, dim=1)
    dist_l = torch.norm(obj_pos - ee_left, dim=1)

    prox_r = 1 - torch.tanh(dist_r / std)
    prox_l = 1 - torch.tanh(dist_l / std)

    grasp_r = grip_r_closed.float() * prox_r
    grasp_l = grip_l_closed.float() * prox_l

    # Both arms contribute equally; reward if either grasped and object is lifted
    grasp_mean = 0.5 * (grasp_r + grasp_l)
    return grasp_mean * is_lifted.float()


# Single-arm IK versions for comparison with single-arm training
def object_ee_distance_single_arm(
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


def object_goal_distance_single_arm(
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


def object_ee_distance_and_lifted_single_arm(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it."""
    # Get reaching reward
    reach_reward = object_ee_distance_single_arm(env, std, object_cfg, ee_frame_cfg)

    # Get lifting reward
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)

    # Combine rewards multiplicatively
    return reach_reward * lift_reward

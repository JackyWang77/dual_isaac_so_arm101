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
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
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


def closest_arm_reach_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_arm_ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    left_arm_ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
    target_index: int = 0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reward ONLY the arm that is currently closest to the object (using EE positions).

    Steps:
    - Read object world position.
    - Read right/left end-effector world positions from two FrameTransformers.
    - Compute distances (EE->object).
    - Shape distances into reach rewards: 1 - tanh(d / std).
    - Soft-select the closer arm via inverse-distance weights.
    """
    assert std > 0.0, "std must be positive"

    # -- object position (num_envs, 3)
    obj: RigidObject = env.scene[object_cfg.name]
    cube_pos_w = obj.data.root_pos_w[:, :3]

    # -- EE positions (each as FrameTransformer)
    ee_r: FrameTransformer = env.scene[right_arm_ee_cfg.name]
    ee_l: FrameTransformer = env.scene[left_arm_ee_cfg.name]

    # target_pos_w layout can be (..., T, 3) or already (..., 3) with an index dim.
    def _pick_target_pos(ft: FrameTransformer) -> torch.Tensor:
        pos = ft.data.target_pos_w
        # Common shapes:
        # - (num_envs, T, 3)
        # - (..., 0, 3) style (so [..., 0, :] works)
        if pos.ndim >= 3:
            # assume (..., T, 3)
            return pos[:, target_index, :]
        else:
            # fallback: last two dims are (..., 3)
            return pos

    ee_right_pos_w = _pick_target_pos(ee_r)  # (num_envs, 3)
    ee_left_pos_w  = _pick_target_pos(ee_l)  # (num_envs, 3)

    # -- distances (num_envs,)
    dist_right = torch.norm(cube_pos_w - ee_right_pos_w, dim=1)
    dist_left  = torch.norm(cube_pos_w - ee_left_pos_w,  dim=1)

    # -- reach shaping: higher if closer
    reach_right = 1.0 - torch.tanh(dist_right / std)
    reach_left  = 1.0 - torch.tanh(dist_left  / std)

    # -- soft selector: give credit to the closer arm
    inv_r = 1.0 / torch.clamp(dist_right, min=eps)
    inv_l = 1.0 / torch.clamp(dist_left,  min=eps)
    denom = inv_r + inv_l
    # avoid NaN if both zero (object exactly at both EEs, extremely unlikely)
    denom = torch.clamp(denom, min=eps)
    w_right = inv_r / denom
    w_left  = inv_l / denom

    reward = w_right * reach_right + w_left * reach_left
    return reward


def support_arm_stillness_penalty(
    env: ManagerBasedRLEnv,
    weight_farther: float,
    weight_closer: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
) -> torch.Tensor:
    """Penalize unnecessary motion of the support (farther) arm.

    Logic:
    - Compute which arm is closer to the object. That one is "primary".
    - The other arm is "support", it should stay calmer.
    - We penalize joint velocity^2 on both arms, BUT:
        * support arm gets higher weight (weight_farther)
        * primary arm gets smaller weight (weight_closer)
    - Return negative penalty (more movement = more negative reward).
    """

    # 1. positions
    obj: RigidObject = env.scene[object_cfg.name]
    cube_pos_w = obj.data.root_pos_w[:, :3]  # (num_envs, 3)

    right_arm: RigidObject = env.scene[right_arm_cfg.name]
    left_arm:  RigidObject = env.scene[left_arm_cfg.name]

    # we can estimate arm "location" by its root link pose (base link pose)
    right_pos_w = right_arm.data.root_pos_w[:, :3]  # (num_envs, 3)
    left_pos_w  = left_arm.data.root_pos_w[:, :3]   # (num_envs, 3)

    dist_right = torch.norm(cube_pos_w - right_pos_w, dim=1)  # (num_envs,)
    dist_left  = torch.norm(cube_pos_w - left_pos_w,  dim=1)  # (num_envs,)

    # 2. joint velocities
    # IsaacLab Articulation exposes joint_vel, shape (num_envs, n_dofs)
    qd_right = right_arm.data.joint_vel   # (num_envs, DoF_right)
    qd_left  = left_arm.data.joint_vel    # (num_envs, DoF_left)

    # squared L2 norm per env
    vel_cost_right = torch.sum(qd_right * qd_right, dim=1)  # (num_envs,)
    vel_cost_left  = torch.sum(qd_left  * qd_left,  dim=1)  # (num_envs,)

    # 3. decide which arm is primary (closer to object)
    closer_is_right = dist_right <= dist_left  # (num_envs,) bool
    closer_is_left  = ~closer_is_right

    # if right is closer: right arm allowed more motion, left arm should chill
    penalty_if_right_closer = (
        weight_closer * vel_cost_right + weight_farther * vel_cost_left
    )

    # if left is closer: left arm allowed more motion, right arm should chill
    penalty_if_left_closer = (
        weight_farther * vel_cost_right + weight_closer * vel_cost_left
    )

    penalty = torch.where(
        closer_is_right,
        penalty_if_right_closer,
        penalty_if_left_closer,
    )

    # negative reward (penalty)
    return -penalty

# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for dual-arm cube stack: stack two cubes at fixed target (center)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Reward for lifting object (cube_top) above minimal height."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def closer_arm_reaches_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    ee_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    ee_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
) -> torch.Tensor:
    """Reward the closer arm for reaching the object (cube_top)."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_right: FrameTransformer = env.scene[ee_right_cfg.name]
    ee_left: FrameTransformer = env.scene[ee_left_cfg.name]
    obj_pos = obj.data.root_pos_w
    ee_right_pos = ee_right.data.target_pos_w[..., 0, :]
    ee_left_pos = ee_left.data.target_pos_w[..., 0, :]
    dist_right = torch.norm(obj_pos - ee_right_pos, dim=1)
    dist_left = torch.norm(obj_pos - ee_left_pos, dim=1)
    right_is_closer = (dist_right < dist_left).float()
    left_is_closer = (dist_left < dist_right).float()
    right_reward = right_is_closer * (1 - torch.tanh(dist_right / std))
    left_reward = left_is_closer * (1 - torch.tanh(dist_left / std))
    return right_reward + left_reward


def farther_arm_stays_still(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    ee_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    ee_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
) -> torch.Tensor:
    """Reward the farther arm for staying still."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_right: FrameTransformer = env.scene[ee_right_cfg.name]
    ee_left: FrameTransformer = env.scene[ee_left_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]
    left_arm = env.scene[left_arm_cfg.name]
    obj_pos = obj.data.root_pos_w
    ee_right_pos = ee_right.data.target_pos_w[..., 0, :]
    ee_left_pos = ee_left.data.target_pos_w[..., 0, :]
    dist_right = torch.norm(obj_pos - ee_right_pos, dim=1)
    dist_left = torch.norm(obj_pos - ee_left_pos, dim=1)
    right_is_farther = (dist_right > dist_left).float()
    left_is_farther = (dist_left > dist_right).float()
    right_vel_mag = torch.norm(right_arm.data.joint_vel, dim=1)
    left_vel_mag = torch.norm(left_arm.data.joint_vel, dim=1)
    return right_is_farther * torch.exp(-right_vel_mag) + left_is_farther * torch.exp(-left_vel_mag)


def grasp_intent(
    env: ManagerBasedRLEnv,
    proximity_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    ee_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    ee_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
) -> torch.Tensor:
    """Reward closing gripper when near cube_top."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_right: FrameTransformer = env.scene[ee_right_cfg.name]
    ee_left: FrameTransformer = env.scene[ee_left_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]
    left_arm = env.scene[left_arm_cfg.name]
    obj_pos = obj.data.root_pos_w
    ee_right_pos = ee_right.data.target_pos_w[..., 0, :]
    ee_left_pos = ee_left.data.target_pos_w[..., 0, :]
    dist_right = torch.norm(obj_pos - ee_right_pos, dim=1)
    dist_left = torch.norm(obj_pos - ee_left_pos, dim=1)
    right_is_closer = (dist_right < dist_left).float()
    left_is_closer = (dist_left < dist_right).float()
    # JAW_OPEN=0.4 (与 lift_old 一致); 0 = closed
    jaw_open = 0.4
    right_gripper_closed = (jaw_open - right_arm.data.joint_pos[:, -1]) / jaw_open
    left_gripper_closed = (jaw_open - left_arm.data.joint_pos[:, -1]) / jaw_open
    right_near = (dist_right < proximity_threshold).float()
    left_near = (dist_left < proximity_threshold).float()
    return right_is_closer * right_near * right_gripper_closed + left_is_closer * left_near * left_gripper_closed


def cube_stack_alignment(
    env: ManagerBasedRLEnv,
    xy_std: float = 0.03,
    z_tolerance: float = 0.015,
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_base_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Reward cube_top being above cube_base (xy aligned, z just above base)."""
    top: RigidObject = env.scene[cube_top_cfg.name]
    base: RigidObject = env.scene[cube_base_cfg.name]
    top_pos = top.data.root_pos_w[:, :3]
    base_pos = base.data.root_pos_w[:, :3]
    xy_dist = torch.norm(top_pos[:, :2] - base_pos[:, :2], dim=1)
    z_above = (top_pos[:, 2] > base_pos[:, 2] + z_tolerance).float()
    return z_above * (1 - torch.tanh(xy_dist / xy_std))


def cube_near_target_xy(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.117, -0.011),
    xy_std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Reward object xy position near fixed target (center)."""
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w[:, :2]
    target = torch.tensor(
        target_xy, dtype=pos.dtype, device=env.device
    ).unsqueeze(0).expand(pos.shape[0], -1)
    xy_dist = torch.norm(pos - target, dim=1)
    return (1 - torch.tanh(xy_dist / xy_std))


# =========================================================
# RL fine-tuning rewards (for residual RL on top of BC)
# =========================================================

def gripper_release_when_stacked(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.015,
    z_tolerance: float = 0.01,
    jaw_open: float = 0.4,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
) -> torch.Tensor:
    """Reward opening grippers when cubes are stacked (either order).

    BC sometimes "doesn't dare to release" - this encourages letting go
    once the stack is aligned.
    """
    c1: RigidObject = env.scene[cube_1_cfg.name]
    c2: RigidObject = env.scene[cube_2_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]
    left_arm = env.scene[left_arm_cfg.name]

    p1 = c1.data.root_pos_w[:, :3]
    p2 = c2.data.root_pos_w[:, :3]

    # Check if cubes are stacked (either order)
    xy_dist = torch.norm(p1[:, :2] - p2[:, :2], dim=1)
    z_diff = torch.abs(p1[:, 2] - p2[:, 2])
    is_stacked = ((xy_dist < xy_threshold) & (z_diff > z_tolerance)).float()

    # Gripper openness: 0=closed, 1=fully open
    right_open = (right_arm.data.joint_pos[:, -1] / jaw_open).clamp(0, 1)
    left_open = (left_arm.data.joint_pos[:, -1] / jaw_open).clamp(0, 1)

    return is_stacked * (right_open + left_open) * 0.5


def stack_success_bonus(
    env: ManagerBasedRLEnv,
    expected_height: float = 0.018,
    eps_z: float = 0.005,
    eps_xy: float = 0.012,
    gripper_open_threshold: float = 0.1,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
) -> torch.Tensor:
    """Large one-time bonus when stack success conditions are met.

    Looser than termination criterion (no velocity/stability check)
    so the agent gets rewarded more frequently during learning.
    """
    c1: RigidObject = env.scene[cube_1_cfg.name]
    c2: RigidObject = env.scene[cube_2_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]
    left_arm = env.scene[left_arm_cfg.name]

    p1 = c1.data.root_pos_w[:, :3]
    p2 = c2.data.root_pos_w[:, :3]

    # Check both stacking orders: 1-on-2 or 2-on-1
    z_diff_1on2 = p1[:, 2] - p2[:, 2]
    z_diff_2on1 = p2[:, 2] - p1[:, 2]
    xy_dist = torch.norm(p1[:, :2] - p2[:, :2], dim=1)

    ok_1on2 = (torch.abs(z_diff_1on2 - expected_height) < eps_z) & (xy_dist < eps_xy)
    ok_2on1 = (torch.abs(z_diff_2on1 - expected_height) < eps_z) & (xy_dist < eps_xy)
    stacked = (ok_1on2 | ok_2on1).float()

    # Both grippers open
    both_open = (
        (right_arm.data.joint_pos[:, -1] > gripper_open_threshold)
        & (left_arm.data.joint_pos[:, -1] > gripper_open_threshold)
    ).float()

    return stacked * both_open

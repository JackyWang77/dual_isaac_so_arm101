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
    std: float = 0.01,
    target_height: float = 0.018,
    gripper_open_thresh: float = 0.1,
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_base_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
) -> torch.Tensor:
    """Alignment reward: right gripper open + cube2 above cube1 (z>0.018) + xy close.

    Not one-shot. Gated by conditions so no hack possible:
    - Right hand must be open (released cube)
    - Cube2 must be above cube1 by at least target_height
    - Reward = 1 - tanh(xy_dist / std), closer = higher
    Gripper joint: 0=open, -0.36=closed.
    """
    top: RigidObject = env.scene[cube_top_cfg.name]
    base: RigidObject = env.scene[cube_base_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]

    top_pos = top.data.root_pos_w[:, :3]
    base_pos = base.data.root_pos_w[:, :3]

    # Height gate: cube2 center must be above cube1 center by >= target_height
    z_diff = top_pos[:, 2] - base_pos[:, 2]
    height_ok = (z_diff >= target_height).float()

    # XY distance: closer = better
    xy_dist = torch.norm(top_pos[:, :2] - base_pos[:, :2], dim=1)
    alignment_quality = 1 - torch.tanh(xy_dist / std)

    # Right gripper must be open
    right_open = (right_arm.data.joint_pos[:, -1] > gripper_open_thresh).float()

    return alignment_quality * height_ok * right_open


def cube_stack_alignment_dense(
    env: ManagerBasedRLEnv,
    std: float = 0.01,
    target_height: float = 0.018,
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_base_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """Dense alignment reward without gripper gate. Continuously guides cube_top above cube_base."""
    top: RigidObject = env.scene[cube_top_cfg.name]
    base: RigidObject = env.scene[cube_base_cfg.name]

    top_pos = top.data.root_pos_w[:, :3]
    base_pos = base.data.root_pos_w[:, :3]

    z_diff = top_pos[:, 2] - base_pos[:, 2]
    height_ok = (z_diff >= target_height).float()

    xy_dist = torch.norm(top_pos[:, :2] - base_pos[:, :2], dim=1)
    alignment_quality = 1 - torch.tanh(xy_dist / std)

    return alignment_quality * height_ok


def cube_near_target_xy(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.117, -0.011),
    xy_std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Reward object xy position near fixed target (center). Uses env-local coords."""
    obj: RigidObject = env.scene[object_cfg.name]
    pos = (obj.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3])[:, :2]
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
    gripper_open_thresh: float = 0.1,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
) -> torch.Tensor:
    """One-time reward for RIGHT gripper release when cubes are stacked.

    Only checks right arm (which places cube_1).
    """
    c1: RigidObject = env.scene[cube_1_cfg.name]
    c2: RigidObject = env.scene[cube_2_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]

    num_envs = env.num_envs
    device = c1.data.root_pos_w.device

    if not hasattr(env, '_release_fired'):
        env._release_fired = torch.zeros(num_envs, dtype=torch.bool, device=device)

    p1 = c1.data.root_pos_w[:, :3]
    p2 = c2.data.root_pos_w[:, :3]

    xy_dist = torch.norm(p1[:, :2] - p2[:, :2], dim=1)
    z_diff = torch.abs(p1[:, 2] - p2[:, 2])
    is_stacked = (xy_dist < xy_threshold) & (z_diff > z_tolerance)

    right_open = (right_arm.data.joint_pos[:, -1] > gripper_open_thresh)

    release_now = is_stacked & right_open & (~env._release_fired)
    env._release_fired = env._release_fired | (is_stacked & right_open)

    new_episode = (env.episode_length_buf <= 1)
    env._release_fired[new_episode] = False

    return release_now.float()


def stack_success_bonus(
    env: ManagerBasedRLEnv,
    expected_height: float = 0.018,
    eps_z: float = 0.005,
    eps_xy: float = 0.012,
    gripper_open_thresh: float = 0.1,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
) -> torch.Tensor:
    """Large one-time bonus when stack success + right gripper open.

    Only checks right arm (which places cube_1).
    """
    c1: RigidObject = env.scene[cube_1_cfg.name]
    c2: RigidObject = env.scene[cube_2_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]

    num_envs = env.num_envs
    device = c1.data.root_pos_w.device

    if not hasattr(env, '_success_fired'):
        env._success_fired = torch.zeros(num_envs, dtype=torch.bool, device=device)

    p1 = c1.data.root_pos_w[:, :3]
    p2 = c2.data.root_pos_w[:, :3]

    z_diff_1on2 = p1[:, 2] - p2[:, 2]
    z_diff_2on1 = p2[:, 2] - p1[:, 2]
    xy_dist = torch.norm(p1[:, :2] - p2[:, :2], dim=1)

    ok_1on2 = (torch.abs(z_diff_1on2 - expected_height) < eps_z) & (xy_dist < eps_xy)
    ok_2on1 = (torch.abs(z_diff_2on1 - expected_height) < eps_z) & (xy_dist < eps_xy)
    stacked = ok_1on2 | ok_2on1

    right_open = (right_arm.data.joint_pos[:, -1] > gripper_open_thresh)

    success_now = stacked & right_open & (~env._success_fired)
    env._success_fired = env._success_fired | (stacked & right_open)

    new_episode = (env.episode_length_buf <= 1)
    env._success_fired[new_episode] = False

    return success_now.float()


def gripper_open_reward(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.02,
    z_min: float = 0.01,
    jaw_max: float = 0.4,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
) -> torch.Tensor:
    """Dense reward for RIGHT gripper opening when cubes are aligned.

    Gate: cube1 above cube2 (z_diff > z_min) AND xy close (< xy_threshold).
    Reward: normalized gripper opening degree [0, 1].
    Every step the gripper is more open while aligned → more reward.
    """
    c1: RigidObject = env.scene[cube_1_cfg.name]
    c2: RigidObject = env.scene[cube_2_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]

    p1 = c1.data.root_pos_w[:, :3]
    p2 = c2.data.root_pos_w[:, :3]

    # Alignment gate
    xy_dist = torch.norm(p1[:, :2] - p2[:, :2], dim=1)
    z_diff = p1[:, 2] - p2[:, 2]
    aligned = ((xy_dist < xy_threshold) & (z_diff > z_min)).float()

    # Gripper opening: joint_pos[-1] range [0.0002, 0.4], higher = more open
    gripper_pos = right_arm.data.joint_pos[:, -1]
    gripper_open_norm = torch.clamp(gripper_pos / jaw_max, 0.0, 1.0)

    return aligned * gripper_open_norm


def stack_stability(
    env: ManagerBasedRLEnv,
    std: float = 0.01,
    gripper_open_thresh: float = 0.1,
    vel_threshold: float = 0.1,
    z_min: float = 0.01,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
) -> torch.Tensor:
    """Dense reward for cube1 staying on cube2 after gripper release.

    Gate: right gripper open (> gripper_open_thresh).
    Reward: alignment quality * low-velocity bonus.
    Continuously rewards maintaining the stack after release.
    """
    c1: RigidObject = env.scene[cube_1_cfg.name]
    c2: RigidObject = env.scene[cube_2_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]

    p1 = c1.data.root_pos_w[:, :3]
    p2 = c2.data.root_pos_w[:, :3]

    # Gripper open gate
    gripper_open = (right_arm.data.joint_pos[:, -1] > gripper_open_thresh).float()

    # Height gate: cube1 must be above cube2
    z_diff = p1[:, 2] - p2[:, 2]
    height_ok = (z_diff > z_min).float()

    # XY alignment quality
    xy_dist = torch.norm(p1[:, :2] - p2[:, :2], dim=1)
    alignment = 1 - torch.tanh(xy_dist / std)

    # Low velocity bonus: both cubes should be still
    v1 = torch.norm(c1.data.root_lin_vel_w, dim=1)
    v2 = torch.norm(c2.data.root_lin_vel_w, dim=1)
    velocity_ok = ((v1 < vel_threshold) & (v2 < vel_threshold)).float()

    return gripper_open * height_ok * alignment * velocity_ok

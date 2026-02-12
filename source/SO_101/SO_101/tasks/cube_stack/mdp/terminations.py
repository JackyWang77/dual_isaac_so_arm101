# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause

"""Terminations for dual-arm cube stack."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _stacked_on(
    top_pos: torch.Tensor,
    base_pos: torch.Tensor,
    xy_threshold: float,
    z_tolerance: float,
) -> torch.Tensor:
    """True when top is stacked on base (xy aligned, z above)."""
    xy_ok = torch.norm(top_pos[:, :2] - base_pos[:, :2], dim=1) < xy_threshold
    z_ok = top_pos[:, 2] > base_pos[:, 2] + z_tolerance
    return xy_ok & z_ok


def cube_stacked(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.025,
    z_tolerance: float = 0.01,
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_base_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """True when cube_top is stacked on cube_base (xy aligned, z above)."""
    top: RigidObject = env.scene[cube_top_cfg.name]
    base: RigidObject = env.scene[cube_base_cfg.name]
    return _stacked_on(
        top.data.root_pos_w[:, :3],
        base.data.root_pos_w[:, :3],
        xy_threshold,
        z_tolerance,
    )


def object_placed_at_target(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.117, -0.011),
    target_z: float = 0.006,
    target_eps_xy: float = 0.2,
    target_eps_z: float = 0.005,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """True when object is placed at target: xy in range, z ≈ target_z (e.g. on table)."""
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w[:, :3]
    target_xy_t = torch.tensor(
        [target_xy[0], target_xy[1]],
        device=env.device,
        dtype=pos.dtype,
    ).unsqueeze(0).expand(pos.shape[0], -1)
    at_target_xy = torch.norm(pos[:, :2] - target_xy_t, dim=1) < target_eps_xy
    at_target_z = torch.abs(pos[:, 2] - target_z) < target_eps_z
    return at_target_xy & at_target_z


def either_cube_placed_at_target(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.117, -0.011),
    target_z: float = 0.006,
    target_eps_xy: float = 0.2,
    target_eps_z: float = 0.005,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """True when cube_1 OR cube_2 is placed at target (either order)."""
    c1 = object_placed_at_target(
        env, target_xy, target_z, target_eps_xy, target_eps_z, object_cfg=cube_1_cfg
    )
    c2 = object_placed_at_target(
        env, target_xy, target_z, target_eps_xy, target_eps_z, object_cfg=cube_2_cfg
    )
    return c1 | c2


def two_cubes_stacked_aligned(
    env: ManagerBasedRLEnv,
    expected_height: float = 0.018,
    eps_z: float = 0.003,
    eps_xy: float = 0.009,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """True when two cubes are stacked: only Z and XY alignment (any order)."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    p1 = cube_1.data.root_pos_w[:, :3]
    p2 = cube_2.data.root_pos_w[:, :3]
    # Option A: cube_1 on cube_2
    z_ok_a = torch.abs((p1[:, 2] - p2[:, 2]) - expected_height) < eps_z
    xy_ok_a = torch.norm(p1[:, :2] - p2[:, :2], dim=1) < eps_xy
    a = z_ok_a & xy_ok_a
    # Option B: cube_2 on cube_1
    z_ok_b = torch.abs((p2[:, 2] - p1[:, 2]) - expected_height) < eps_z
    xy_ok_b = torch.norm(p2[:, :2] - p1[:, :2], dim=1) < eps_xy
    b = z_ok_b & xy_ok_b
    return a | b


def two_cubes_stacked_aligned_gripper_released(
    env: ManagerBasedRLEnv,
    expected_height: float = 0.018,
    eps_z: float = 0.003,
    eps_xy: float = 0.009,
    gripper_open_threshold: float = 0.1,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
) -> torch.Tensor:
    """Stacked + both grippers open (must release before success)."""
    stacked = two_cubes_stacked_aligned(
        env, expected_height, eps_z, eps_xy, cube_1_cfg, cube_2_cfg
    )
    right_arm = env.scene[right_arm_cfg.name]
    left_arm = env.scene[left_arm_cfg.name]
    both_open = (right_arm.data.joint_pos[:, -1] > gripper_open_threshold) & (
        left_arm.data.joint_pos[:, -1] > gripper_open_threshold
    )
    return stacked & both_open


def two_cubes_stacked_at_target(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.117, -0.011),
    xy_threshold: float = 0.025,
    z_tolerance: float = 0.01,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
) -> torch.Tensor:
    """True when cube_1 and cube_2 are stacked (any order) at the fixed target position (center)."""
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_1_pos = cube_1.data.root_pos_w[:, :3]
    cube_2_pos = cube_2.data.root_pos_w[:, :3]
    target = torch.tensor(
        [target_xy[0], target_xy[1], 0.0],
        device=env.device,
        dtype=cube_1_pos.dtype,
    ).unsqueeze(0)
    target = target.expand(cube_1_pos.shape[0], -1)

    # Bottom cube must be at target xy; top cube stacked on bottom
    # Option A: cube_2 at target (bottom), cube_1 on cube_2
    bottom_at_target_a = torch.norm(cube_2_pos[:, :2] - target[:, :2], dim=1) < xy_threshold
    stacked_a = _stacked_on(cube_1_pos, cube_2_pos, xy_threshold, z_tolerance)
    a = bottom_at_target_a & stacked_a
    # Option B: cube_1 at target (bottom), cube_2 on cube_1
    bottom_at_target_b = torch.norm(cube_1_pos[:, :2] - target[:, :2], dim=1) < xy_threshold
    stacked_b = _stacked_on(cube_2_pos, cube_1_pos, xy_threshold, z_tolerance)
    b = bottom_at_target_b & stacked_b
    return a | b


def two_cubes_stacked_at_target_released(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.117, -0.011),
    expected_height: float = 0.018,
    eps_z: float = 0.003,
    eps_xy: float = 0.009,
    target_eps_xy: float = 0.2,
    gripper_open_threshold: float = 0.1,
    vel_threshold: float = 0.001,
    stable_steps_required: int = 50,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
    ee_right_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
    ee_left_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
) -> torch.Tensor:
    """Stack success with tolerance cylinder + released + stable for 1s.

    Spatial alignment (tolerance cylinder):
    - Z: |(z_top - z_base) - h| < eps_z  (B exactly one body height above A)
    - XY: ||(x_B,y_B) - (x_A,y_A)||_2 < eps_xy (stack alignment); bottom at target: < target_eps_xy
    - Both cube velocities < vel_threshold (0 = fully still)
    - Both grippers open
    - Stable for stable_steps_required (50 steps ≈ 1s at 50Hz)
    """
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    right_arm = env.scene[right_arm_cfg.name]
    left_arm = env.scene[left_arm_cfg.name]
    ee_right = env.scene[ee_right_cfg.name]
    ee_left = env.scene[ee_left_cfg.name]

    cube_1_pos = cube_1.data.root_pos_w[:, :3]
    cube_2_pos = cube_2.data.root_pos_w[:, :3]
    cube_1_vel = cube_1.data.root_lin_vel_w[:, :3]
    cube_2_vel = cube_2.data.root_lin_vel_w[:, :3]
    jaw_right = right_arm.data.joint_pos[:, -1]
    jaw_left = left_arm.data.joint_pos[:, -1]

    target = torch.tensor(
        [target_xy[0], target_xy[1], 0.0],
        device=env.device,
        dtype=cube_1_pos.dtype,
    ).unsqueeze(0).expand(cube_1_pos.shape[0], -1)

    # Both grippers open
    both_open = (jaw_right > gripper_open_threshold) & (jaw_left > gripper_open_threshold)

    # Both cube velocities near zero
    vel_1_ok = torch.norm(cube_1_vel, dim=1) <= vel_threshold
    vel_2_ok = torch.norm(cube_2_vel, dim=1) <= vel_threshold
    both_still = vel_1_ok & vel_2_ok

    # Option A: cube_2 at target (bottom), cube_1 on top
    bottom_a = torch.norm(cube_2_pos[:, :2] - target[:, :2], dim=1) < target_eps_xy
    xy_align_a = torch.norm(cube_1_pos[:, :2] - cube_2_pos[:, :2], dim=1) < eps_xy
    z_align_a = torch.abs((cube_1_pos[:, 2] - cube_2_pos[:, 2]) - expected_height) < eps_z
    a = bottom_a & xy_align_a & z_align_a & both_open & both_still

    # Option B: cube_1 at target (bottom), cube_2 on top
    bottom_b = torch.norm(cube_1_pos[:, :2] - target[:, :2], dim=1) < target_eps_xy
    xy_align_b = torch.norm(cube_2_pos[:, :2] - cube_1_pos[:, :2], dim=1) < eps_xy
    z_align_b = torch.abs((cube_2_pos[:, 2] - cube_1_pos[:, 2]) - expected_height) < eps_z
    b = bottom_b & xy_align_b & z_align_b & both_open & both_still

    basic_ok = a | b

    # Stability duration: must hold for stable_steps_required (e.g. 50 steps = 1s)
    buf_name = "_stack_stable_steps"
    if not hasattr(env, buf_name):
        setattr(
            env,
            buf_name,
            torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
        )
    stable_buf = getattr(env, buf_name)
    stable_buf[:] = torch.where(basic_ok, stable_buf + 1, 0)
    return stable_buf >= stable_steps_required

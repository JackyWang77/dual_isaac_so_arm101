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


def two_cubes_stacked_at_target(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float] = (0.2, 0.0),
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

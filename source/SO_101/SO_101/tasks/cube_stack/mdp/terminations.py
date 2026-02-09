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


def cube_stacked(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.025,
    z_tolerance: float = 0.01,
    cube_top_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cube_base_cfg: SceneEntityCfg = SceneEntityCfg("cube_base"),
) -> torch.Tensor:
    """True when cube_top is stacked on cube_base (xy aligned, z above)."""
    top: RigidObject = env.scene[cube_top_cfg.name]
    base: RigidObject = env.scene[cube_base_cfg.name]
    top_pos = top.data.root_pos_w[:, :3]
    base_pos = base.data.root_pos_w[:, :3]
    xy_ok = torch.norm(top_pos[:, :2] - base_pos[:, :2], dim=1) < xy_threshold
    z_ok = top_pos[:, 2] > base_pos[:, 2] + z_tolerance
    return xy_ok & z_ok

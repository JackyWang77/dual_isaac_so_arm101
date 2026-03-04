# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards for dual-arm table setting: place fork (left) and knife (right) on tray."""

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
    object_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
) -> torch.Tensor:
    """Reward for lifting object above minimal height."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def arm_reaches_object(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
) -> torch.Tensor:
    """Reward a specific arm for reaching its assigned object."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_cfg.name]
    obj_pos = obj.data.root_pos_w
    ee_pos = ee.data.target_pos_w[..., 0, :]
    dist = torch.norm(obj_pos - ee_pos, dim=1)
    return 1 - torch.tanh(dist / std)


def object_near_target_xy(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float],
    xy_std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
) -> torch.Tensor:
    """Reward object xy position near fixed target."""
    obj: RigidObject = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w[:, :2]
    target = torch.tensor(
        target_xy, dtype=pos.dtype, device=env.device
    ).unsqueeze(0).expand(pos.shape[0], -1)
    xy_dist = torch.norm(pos - target, dim=1)
    return 1 - torch.tanh(xy_dist / xy_std)


def grasp_intent_single(
    env: ManagerBasedRLEnv,
    proximity_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_left"),
    arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
) -> torch.Tensor:
    """Reward closing gripper when near assigned object."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_cfg.name]
    arm = env.scene[arm_cfg.name]
    obj_pos = obj.data.root_pos_w
    ee_pos = ee.data.target_pos_w[..., 0, :]
    dist = torch.norm(obj_pos - ee_pos, dim=1)
    jaw_open = 0.4
    gripper_closed = (jaw_open - arm.data.joint_pos[:, -1]) / jaw_open
    near = (dist < proximity_threshold).float()
    return near * gripper_closed

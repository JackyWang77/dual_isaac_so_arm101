# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for dual-arm cube stack: two cubes (base + top to stack)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Object position in world frame. Shape: (num_envs, 3)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w[:, :3]


def object_orientation_w(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Object orientation (quat) in world frame. Shape: (num_envs, 4)."""
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_quat_w


def ee_position_in_arm_root_frame(
    env: ManagerBasedRLEnv,
    arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
) -> torch.Tensor:
    """EE position in the given arm's root frame. Shape: (num_envs, 3)."""
    arm: Articulation = env.scene[arm_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(
        arm.data.root_state_w[:, :3], arm.data.root_state_w[:, 3:7], ee_pos_w
    )
    return ee_pos_b


def ee_position_w(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
) -> torch.Tensor:
    """EE position in world frame. Shape: (num_envs, 3)."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[..., 0, :]


def ee_orientation_w(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_right"),
) -> torch.Tensor:
    """EE orientation (quaternion) in world frame. Shape: (num_envs, 4)."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[..., 0, :]


def object_pos_in_arm_frame(
    env: ManagerBasedRLEnv,
    arm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """Object position in the given arm's root frame. Shape: (num_envs, 3)."""
    arm = env.scene[arm_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_in_arm, _ = subtract_frame_transforms(
        arm.data.root_state_w[:, :3],
        arm.data.root_state_w[:, 3:7],
        obj_pos_w,
    )
    return obj_pos_in_arm

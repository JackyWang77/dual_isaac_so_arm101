# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause

"""Terminations for dual-arm table setting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_placed_at_target(
    env: ManagerBasedRLEnv,
    target_xy: tuple[float, float],
    target_z: float,
    target_eps_xy: float = 0.03,
    target_eps_z: float = 0.01,
    object_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
) -> torch.Tensor:
    """True when object is placed near target (xy + z)."""
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


def both_objects_placed(
    env: ManagerBasedRLEnv,
    fork_target_xy: tuple[float, float],
    knife_target_xy: tuple[float, float],
    fork_target_z: float,
    knife_target_z: float,
    eps_xy: float = 0.03,
    eps_z: float = 0.01,
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
) -> torch.Tensor:
    """True when both fork and knife are at their target positions."""
    fork_ok = object_placed_at_target(
        env, fork_target_xy, fork_target_z, eps_xy, eps_z, object_cfg=fork_cfg
    )
    knife_ok = object_placed_at_target(
        env, knife_target_xy, knife_target_z, eps_xy, eps_z, object_cfg=knife_cfg
    )
    return fork_ok & knife_ok


def both_placed_and_released(
    env: ManagerBasedRLEnv,
    fork_target_xy: tuple[float, float],
    knife_target_xy: tuple[float, float],
    fork_target_z: float,
    knife_target_z: float,
    eps_xy: float = 0.03,
    eps_z: float = 0.01,
    gripper_open_threshold: float = 0.1,
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
) -> torch.Tensor:
    """Both placed + both grippers open."""
    placed = both_objects_placed(
        env, fork_target_xy, knife_target_xy, fork_target_z, knife_target_z,
        eps_xy, eps_z, fork_cfg, knife_cfg
    )
    right_arm = env.scene[right_arm_cfg.name]
    left_arm = env.scene[left_arm_cfg.name]
    both_open = (right_arm.data.joint_pos[:, -1] > gripper_open_threshold) & (
        left_arm.data.joint_pos[:, -1] > gripper_open_threshold
    )
    return placed & both_open


def both_placed_stable(
    env: ManagerBasedRLEnv,
    fork_target_xy: tuple[float, float],
    knife_target_xy: tuple[float, float],
    fork_target_z: float,
    knife_target_z: float,
    eps_xy: float = 0.03,
    eps_z: float = 0.01,
    stable_steps_required: int = 5,
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
) -> torch.Tensor:
    """Both placed + stable for N steps (no gripper check)."""
    placed = both_objects_placed(
        env, fork_target_xy, knife_target_xy, fork_target_z, knife_target_z,
        eps_xy, eps_z, fork_cfg, knife_cfg,
    )

    buf_name = "_table_setting_placed_stable_steps"
    if not hasattr(env, buf_name):
        setattr(env, buf_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
    stable_buf = getattr(env, buf_name)
    stable_buf[:] = torch.where(placed, stable_buf + 1, 0)
    return stable_buf >= stable_steps_required


def both_placed_released_stable(
    env: ManagerBasedRLEnv,
    fork_target_xy: tuple[float, float],
    knife_target_xy: tuple[float, float],
    fork_target_z: float,
    knife_target_z: float,
    eps_xy: float = 0.03,
    eps_z: float = 0.01,
    gripper_open_threshold: float = 0.1,
    vel_threshold: float = 0.001,
    stable_steps_required: int = 50,
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
    right_arm_cfg: SceneEntityCfg = SceneEntityCfg("right_arm"),
    left_arm_cfg: SceneEntityCfg = SceneEntityCfg("left_arm"),
) -> torch.Tensor:
    """Both placed + released + stable for N steps."""
    placed_released = both_placed_and_released(
        env, fork_target_xy, knife_target_xy, fork_target_z, knife_target_z,
        eps_xy, eps_z, gripper_open_threshold, fork_cfg, knife_cfg,
        right_arm_cfg, left_arm_cfg,
    )
    fork: RigidObject = env.scene[fork_cfg.name]
    knife: RigidObject = env.scene[knife_cfg.name]
    fork_still = torch.norm(fork.data.root_lin_vel_w[:, :3], dim=1) <= vel_threshold
    knife_still = torch.norm(knife.data.root_lin_vel_w[:, :3], dim=1) <= vel_threshold

    basic_ok = placed_released & fork_still & knife_still

    buf_name = "_table_setting_stable_steps"
    if not hasattr(env, buf_name):
        setattr(
            env,
            buf_name,
            torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
        )
    stable_buf = getattr(env, buf_name)
    stable_buf[:] = torch.where(basic_ok, stable_buf + 1, 0)
    return stable_buf >= stable_steps_required

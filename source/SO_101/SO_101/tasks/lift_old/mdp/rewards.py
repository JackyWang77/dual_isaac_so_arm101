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
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    initial_height: float = 0.015,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object. Reward scales with lift distance.

    Args:
        env: The environment.
        minimal_height: The target height to reach (reward = 1.0 at this height).
        initial_height: The initial height of the object (reward = 0.0 at this height).
        object_cfg: The object configuration.

    Returns:
        Reward in range [0.0, 2.0] based on how high the object is lifted.
        Reward increases linearly from 0.0 (at initial_height) to 1.0 (at minimal_height),
        and continues to increase beyond minimal_height (capped at 2.0 for heights >= minimal_height * 2).
    """
    object: RigidObject = env.scene[object_cfg.name]
    current_height = object.data.root_pos_w[:, 2]

    # Calculate lift distance from initial height
    lift_distance = current_height - initial_height

    # Normalize reward: 0.0 at initial_height, 1.0 at minimal_height
    # Allow reward to continue increasing beyond minimal_height (capped at 2.0)
    target_lift_distance = minimal_height - initial_height
    if target_lift_distance > 0:
        # Linear scaling: reward = lift_distance / target_lift_distance
        # Cap at 2.0 to prevent unbounded rewards
        reward = torch.clamp(lift_distance / target_lift_distance, 0.0, 2.0)
    else:
        # Fallback if target <= initial (shouldn't happen normally)
        reward = torch.where(current_height > minimal_height, 1.0, 0.0)

    return reward


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


def time_penalty(
    env: ManagerBasedRLEnv,
    value: float = -1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Constant negative reward per step. Encourages faster completion.
    Base can succeed; RL's value = succeed more efficiently (shorter time).
    Use weight=0.01 for -0.01 per step."""
    object: RigidObject = env.scene[object_cfg.name]
    num_envs = object.data.root_pos_w.shape[0]
    device = object.data.root_pos_w.device
    return torch.full((num_envs,), value, device=device, dtype=object.data.root_pos_w.dtype)


def success_termination_bonus(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.1,
    bonus: float = 10.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """One-time bonus when object reaches success height (episode terminates).
    Creates clear distinguishability: success=+bonus, fail=0.
    Reward = Success_Bonus (additive to per-step rewards)."""
    object: RigidObject = env.scene[object_cfg.name]
    height = object.data.root_pos_w[:, 2]
    return torch.where(height >= minimal_height, bonus, 0.0).float()


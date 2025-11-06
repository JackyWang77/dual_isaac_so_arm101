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
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pos_in_arm_frame(
    env: ManagerBasedRLEnv,
    arm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the object's position expressed in the given arm's root frame.
    
    arm_cfg: which arm to use as reference, e.g. SceneEntityCfg("right_arm") or ("left_arm")
    Output shape: (num_envs, 3)
    """
    arm: RigidObject = env.scene[arm_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    obj_pos_w = obj.data.root_pos_w[:, :3]  # (num_envs, 3)

    # subtract_frame_transforms(world_arm_pos, world_arm_quat, world_point)
    # -> point expressed in arm local frame
    obj_pos_in_arm, _ = subtract_frame_transforms(
        arm.data.root_state_w[:, :3],
        arm.data.root_state_w[:, 3:7],
        obj_pos_w,
    )
    return obj_pos_in_arm  # (num_envs, 3)
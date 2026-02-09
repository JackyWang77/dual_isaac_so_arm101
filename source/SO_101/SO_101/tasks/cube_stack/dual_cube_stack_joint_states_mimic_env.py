# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# Mimic env for recording dual-arm cube stack joint states (same flow as lift_old).
# Action = [right_arm 6, left_arm 6] = 12. Teleop: joint_states device supplies 12 dof.
#

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils.math import make_pose, matrix_from_quat

from .joint_pos_env_cfg import DualSoArm101CubeStackJointPosEnvCfg


class DualCubeStackJointStatesMimicEnv(ManagerBasedRLMimicEnv):
    """Record joint states for dual-arm cube stack. Action dim = 12 (right 6 + left 6)."""

    cfg: DualSoArm101CubeStackJointPosEnvCfg

    def get_robot_eef_pose(
        self, eef_name: str = "dual_arm", env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return right-arm EE pose as 4x4 (primary for compatibility)."""
        if env_ids is None:
            env_ids = slice(None)
        ee_right = self.scene["ee_right"]
        pos = ee_right.data.target_pos_w[env_ids, 0, :]
        quat = ee_right.data.target_quat_w[env_ids, 0, :]
        rot = matrix_from_quat(quat)
        return make_pose(pos, rot)

    def action_to_target_eef_pose(
        self, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Joint-state recording: return current EE pose (actions are joint positions)."""
        return {"dual_arm": self.get_robot_eef_pose()}

    def get_object_poses(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return cube_1 and cube_2 poses as 4x4 (no physical base; target at center)."""
        if env_ids is None:
            env_ids = slice(None)
        out = {}
        for name in ("cube_1", "cube_2"):
            obj = self.scene[name]
            pos = obj.data.root_pos_w[env_ids]
            quat = obj.data.root_quat_w[env_ids]
            out[name] = make_pose(pos, matrix_from_quat(quat))
        return out

    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Extract gripper from action: index 5 (right), index 11 (left)."""
        if actions.dim() == 1:
            return {
                "right": actions[5:6],
                "left": actions[11:12],
            }
        return {
            "right": actions[:, 5:6],
            "left": actions[:, 11:12],
        }

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        signals = {}
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["pick_cube_top"] = subtask_terms["pick_cube_top"][env_ids]
        signals["stack_cube"] = subtask_terms["stack_cube"][env_ids]
        return signals

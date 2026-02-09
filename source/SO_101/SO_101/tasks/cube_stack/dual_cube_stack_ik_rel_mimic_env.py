# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedRLMimicEnv


class DualCubeStackIKRelMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic env for dual-arm cube stack (datagen). Uses right arm as primary EE for teleop."""

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        ee_right = self.scene["ee_right"]
        pos = ee_right.data.target_pos_w[env_ids, 0, :]
        quat = ee_right.data.target_quat_w[env_ids, 0, :]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)
        delta_position = target_pos - curr_pos
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)
        (gripper_action,) = gripper_action_dict.values()
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict.get(eef_name, 0) * torch.randn_like(pose_action)
            if isinstance(noise, torch.Tensor):
                pose_action = pose_action + noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)
        right_action = torch.cat([pose_action, gripper_action], dim=0)
        left_zeros = torch.zeros(7, device=right_action.device, dtype=right_action.dtype)
        left_zeros[-1] = gripper_action.squeeze()
        return torch.cat([right_action, left_zeros], dim=0)

    def action_to_target_eef_pose(
        self, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)
        target_pos = curr_pos + delta_position
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / (delta_rotation_angle + 1e-8)
        delta_quat = PoseUtils.quat_from_angle_axis(
            delta_rotation_angle.squeeze(1), delta_rotation_axis
        )
        if delta_quat.dim() == 1:
            delta_quat = delta_quat.unsqueeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)
        target_poses = PoseUtils.make_pose(target_pos, target_rot)
        return {eef_name: target_poses}

    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        return {eef_name: actions[:, 6:7]}

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        signals = {}
        if "subtask_terms" in self.obs_buf:
            terms = self.obs_buf["subtask_terms"]
            signals["pick_cube_top"] = terms.get("pick_cube_top", torch.zeros(1))[env_ids]
            signals["stack_cube"] = terms.get("stack_cube", torch.zeros(1))[env_ids]
        return signals

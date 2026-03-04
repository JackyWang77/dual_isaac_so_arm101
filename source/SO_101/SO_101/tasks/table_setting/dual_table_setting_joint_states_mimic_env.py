# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# Mimic env for recording dual-arm table setting joint states.
# Action = [right_arm 6, left_arm 6] = 12.
#

from collections.abc import Sequence

import torch
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils.math import make_pose, matrix_from_quat, quat_from_matrix

from .joint_pos_env_cfg import DualSoArm101TableSettingJointPosEnvCfg


class DualTableSettingJointStatesMimicEnv(ManagerBasedRLMimicEnv):
    """Record joint states for dual-arm table setting. Action dim = 12 (right 6 + left 6)."""

    cfg: DualSoArm101TableSettingJointPosEnvCfg
    _ik_controller: DifferentialIKController | None = None
    _right_arm_joint_indices: list | None = None
    _right_jacobi_body_idx: int | None = None
    _left_arm_joint_indices: list | None = None

    def _setup_ik_controller(self):
        """Lazy initialization of IK controller and joint indices for right arm."""
        if self._ik_controller is not None:
            return

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        )
        self._ik_controller = DifferentialIKController(
            cfg=ik_cfg,
            num_envs=1,
            device=self.device,
        )

        right_arm = self.scene["right_arm"]
        left_arm = self.scene["left_arm"]
        arm_joint_names = self.cfg.actions.right_arm_action.joint_names

        self._right_arm_joint_indices = [
            right_arm.joint_names.index(name) for name in arm_joint_names
        ]
        self._left_arm_joint_indices = [
            left_arm.joint_names.index(name) for name in arm_joint_names
        ]

        body_ids, _ = right_arm.find_bodies("wrist_2_link")
        self._right_jacobi_body_idx = body_ids[0] - 1

    def _compute_right_arm_ik(
        self, target_pos: torch.Tensor, target_quat: torch.Tensor, env_id: int
    ) -> torch.Tensor:
        """Compute right arm joint positions via differential IK."""
        right_arm = self.scene["right_arm"]
        ee_right = self.scene["ee_right"]

        current_joint_pos = right_arm.data.joint_pos[
            env_id, self._right_arm_joint_indices
        ]
        ee_pos = ee_right.data.target_pos_w[env_id, 0, :]
        ee_quat = ee_right.data.target_quat_w[env_id, 0, :]

        self._ik_controller.ee_pos_des = target_pos.unsqueeze(0)
        self._ik_controller.ee_quat_des = target_quat.unsqueeze(0)

        jacobians = right_arm.root_physx_view.get_jacobians()
        jacobian_ee = jacobians[env_id, self._right_jacobi_body_idx, :, :]
        jacobian_arm = jacobian_ee[:, self._right_arm_joint_indices]

        target_joint_pos = self._ik_controller.compute(
            ee_pos=ee_pos.unsqueeze(0),
            ee_quat=ee_quat.unsqueeze(0),
            jacobian=jacobian_arm.unsqueeze(0),
            joint_pos=current_joint_pos.unsqueeze(0),
        )
        return target_joint_pos.squeeze(0)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF pose to joint action."""
        self._setup_ik_controller()

        target_eef_pose = target_eef_pose_dict["dual_arm"]
        if target_eef_pose.dim() == 3:
            target_eef_pose = target_eef_pose[0]

        target_pos = target_eef_pose[:3, 3]
        target_rot = target_eef_pose[:3, :3]
        target_quat = quat_from_matrix(target_rot.unsqueeze(0))[0]

        right_arm_joints = self._compute_right_arm_ik(
            target_pos, target_quat, env_id
        )

        gripper_action = gripper_action_dict["dual_arm"]
        if gripper_action.dim() == 2:
            gripper_action = gripper_action.squeeze(0)
        right_gripper = gripper_action[0:1]
        left_gripper = gripper_action[1:2]

        left_arm = self.scene["left_arm"]
        left_arm_joints = left_arm.data.joint_pos[
            env_id, self._left_arm_joint_indices
        ]

        action = torch.cat(
            [
                right_arm_joints,
                right_gripper,
                left_arm_joints,
                left_gripper,
            ],
            dim=0,
        )

        if action_noise_dict is not None and "dual_arm" in action_noise_dict:
            noise = action_noise_dict["dual_arm"] * torch.randn_like(action)
            action += noise

        return action.unsqueeze(0)

    def get_robot_eef_pose(
        self, eef_name: str = "dual_arm", env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return right-arm EE pose as 4x4."""
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
        """Joint-state recording: return current EE pose."""
        return {"dual_arm": self.get_robot_eef_pose()}

    def get_object_poses(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return fork and knife poses as 4x4."""
        if env_ids is None:
            env_ids = slice(None)
        out = {}
        for name in ("fork", "knife"):
            obj = self.scene[name]
            pos = obj.data.root_pos_w[env_ids]
            quat = obj.data.root_quat_w[env_ids]
            out[name] = make_pose(pos, matrix_from_quat(quat))
        return out

    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Extract gripper from action [right_arm(5)+gripper(1), left_arm(5)+gripper(1)]."""
        if actions.dim() == 1:
            return {"dual_arm": torch.cat([actions[5:6], actions[11:12]], dim=0)}
        return {"dual_arm": torch.cat([actions[:, 5:6], actions[:, 11:12]], dim=1)}

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        signals = {}
        subtask_terms = self.obs_buf["subtask_terms"]
        signals["place_fork"] = subtask_terms["place_fork"][env_ids]
        signals["place_knife"] = subtask_terms["place_knife"][env_ids]
        return signals

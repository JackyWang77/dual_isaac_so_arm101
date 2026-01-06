# Copyright (c) 2024-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Environment wrapper for recording joint states directly for reach task.

This environment records joint states (joint positions + gripper) as actions,
without converting to EE pose. The conversion can be done later when reading
the hdf5 file using forward kinematics.
"""

from collections.abc import Sequence

import torch
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import \
    DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils.math import make_pose, matrix_from_quat, quat_from_matrix

from .joint_pos_env_cfg import SoArm100ReachJointCubeEnvCfg


class SoArm100ReachJointStatesMimicEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic environment wrapper for recording joint states directly for reach task.

    This environment:
    - Accepts Joint Control actions for teleoperation (joint positions + gripper)
    - Records joint states directly as actions (no conversion to EE pose)
    - Uses iterative IK to convert target EEF poses back to joint angles for data generation

    Teleoperation action space: [joint_pos_1, ..., joint_pos_5, gripper] - Joint Control
    Recorded action space: [joint_pos_1, ..., joint_pos_5, gripper] - Joint States (same as input)
    """

    cfg: SoArm100ReachJointCubeEnvCfg
    _ik_controller: DifferentialIKController | None = None
    _arm_joint_indices: list | None = None
    _jacobi_body_idx: int | None = None

    def _setup_ik_controller(self):
        """Lazy initialization of IK controller and caching."""
        if self._ik_controller is not None:
            return

        # Setup IK controller with damped least squares
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",  # Damped least squares for stability
            ik_params={"lambda_val": 0.05},
        )
        self._ik_controller = DifferentialIKController(
            cfg=ik_cfg,
            num_envs=1,  # We compute IK one env at a time
            device=self.device,
        )

        # Cache arm joint indices
        robot = self.scene["robot"]
        arm_joint_names = self.cfg.actions.arm_action.joint_names
        all_joint_names = robot.joint_names
        self._arm_joint_indices = [
            all_joint_names.index(name) for name in arm_joint_names
        ]

        # Get body index for EE (fixed base robot, so -1)
        body_ids, _ = robot.find_bodies("end_effector")
        self._jacobi_body_idx = body_ids[0] - 1

    def _compute_ik_step(
        self, target_pos: torch.Tensor, target_quat: torch.Tensor, env_id: int
    ) -> torch.Tensor:
        """Compute one step of differential IK to move towards target.

        For data generation, we use single-step IK since the target poses
        from mimic are sequential and close to current pose.

        Args:
            target_pos: Target EEF position [3]
            target_quat: Target EEF quaternion [4] (w, x, y, z)
            env_id: Environment ID

        Returns:
            torch.Tensor: Computed joint positions [num_arm_joints]
        """
        robot = self.scene["robot"]
        ee_frame = self.scene["ee_frame"]

        # Get current state
        current_joint_pos = robot.data.joint_pos[
            env_id, self._arm_joint_indices
        ]  # [num_arm_joints]
        ee_pos = ee_frame.data.target_pos_w[env_id, 0, :]  # [3]
        ee_quat = ee_frame.data.target_quat_w[env_id, 0, :]  # [4]

        # Set IK target
        self._ik_controller.ee_pos_des = target_pos.unsqueeze(0)
        self._ik_controller.ee_quat_des = target_quat.unsqueeze(0)

        # Get Jacobian for the EE body
        # Shape: [num_envs, num_bodies-1, 6, num_joints] for fixed base
        jacobians = robot.root_physx_view.get_jacobians()
        jacobian_ee = jacobians[env_id, self._jacobi_body_idx, :, :]  # [6, num_joints]
        jacobian_arm = jacobian_ee[:, self._arm_joint_indices]  # [6, num_arm_joints]

        # Compute IK
        target_joint_pos = self._ik_controller.compute(
            ee_pos=ee_pos.unsqueeze(0),
            ee_quat=ee_quat.unsqueeze(0),
            jacobian=jacobian_arm.unsqueeze(0),
            joint_pos=current_joint_pos.unsqueeze(0),
        )

        return target_joint_pos.squeeze(0)  # [num_arm_joints]

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target EEF pose to joint action using differential IK.

        This method uses differential IK to compute joint angles
        that will move the robot towards the target EEF pose.

        Args:
            target_eef_pose_dict: Dictionary with target EEF pose (4x4 matrix)
            gripper_action_dict: Dictionary containing gripper action(s)
            action_noise_dict: Optional noise to apply
            env_id: Environment ID

        Returns:
            torch.Tensor: Joint states [joint_1, ..., joint_5, gripper]
        """
        # Initialize IK controller if needed
        self._setup_ik_controller()

        # Get target pose from dictionary
        (target_eef_pose,) = target_eef_pose_dict.values()  # [4, 4] or [1, 4, 4]
        if target_eef_pose.dim() == 3:
            target_eef_pose = target_eef_pose[0]  # [4, 4]

        # Extract position and rotation from 4x4 matrix
        target_pos = target_eef_pose[:3, 3]  # [3]
        target_rot = target_eef_pose[:3, :3]  # [3, 3]
        target_quat = quat_from_matrix(target_rot.unsqueeze(0))[0]  # [4]

        # Compute IK to get target joint positions
        arm_joint_pos = self._compute_ik_step(target_pos, target_quat, env_id)

        # Get gripper action
        (gripper_action,) = gripper_action_dict.values()

        # Combine arm joints and gripper
        action = torch.cat([arm_joint_pos, gripper_action], dim=0)

        # Apply noise if specified
        if action_noise_dict is not None:
            eef_name = list(self.cfg.subtask_configs.keys())[0]
            if eef_name in action_noise_dict:
                noise = action_noise_dict[eef_name] * torch.randn_like(action)
                action += noise

        return action.unsqueeze(0)  # Add batch dimension

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Get subtask termination signals from observation buffer.

        These signals indicate when the reach subtask is completed,
        and are used by the data generation pipeline.

        Args:
            env_ids: Environment IDs to query, defaults to all environments

        Returns:
            dict: Dictionary mapping subtask names to completion signals
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]

        # Map subtask termination signal from observation buffer
        # Subtask: reach_object - EE reached object and gripper closed
        signals["reach_object"] = subtask_terms["reach_object"][env_ids]

        return signals

    def get_robot_eef_pose(
        self, eef_name: str = "end_effector", env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Get the end-effector pose as 4x4 transformation matrix.

        This method is required by ManagerBasedRLMimicEnv for recording EEF poses
        during annotation and data generation.

        Args:
            eef_name: Name of the end-effector (not used, we only have one arm)
            env_ids: Environment IDs to query, defaults to all environments

        Returns:
            torch.Tensor: 4x4 pose matrix [num_envs, 4, 4]
        """
        if env_ids is None:
            env_ids = slice(None)

        # Get EEF pose from the ee_frame FrameTransformer
        ee_frame = self.scene["ee_frame"]

        # ee_frame.data.target_pos_w shape: [num_envs, num_targets, 3]
        # ee_frame.data.target_quat_w shape: [num_envs, num_targets, 4]
        # We have 1 target (end_effector), so index 0
        eef_pos = ee_frame.data.target_pos_w[env_ids, 0, :]  # [num_envs, 3]
        eef_quat = ee_frame.data.target_quat_w[env_ids, 0, :]  # [num_envs, 4]

        # Convert to 4x4 matrix
        eef_rot = matrix_from_quat(eef_quat)
        return make_pose(eef_pos, eef_rot)

    def action_to_target_eef_pose(
        self, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Convert action to target EEF pose as 4x4 transformation matrix.

        For joint states control, we return the current EEF pose since actions
        are joint positions, not EEF poses. The actual EEF pose after executing
        the action can be computed via forward kinematics.

        Args:
            action: The action tensor [num_envs, action_dim]

        Returns:
            dict: Dictionary mapping arm name to 4x4 pose matrix
        """
        # Get current EEF pose as 4x4 matrix (since we're using joint control, not IK)
        eef_pose = self.get_robot_eef_pose()

        # Return as dictionary with arm name
        return {"end_effector": eef_pose}

    def get_object_poses(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Get the poses of all manipulated objects as 4x4 transformation matrices.

        This method is required by ManagerBasedRLMimicEnv for recording object poses
        during annotation and data generation.

        Args:
            env_ids: Environment IDs to query, defaults to all environments

        Returns:
            dict: Dictionary mapping object name to 4x4 pose matrix
        """
        if env_ids is None:
            env_ids = slice(None)

        object_poses = {}

        # Get object (cube) pose as 4x4 matrix
        obj = self.scene["object"]
        obj_pos = obj.data.root_pos_w[env_ids]
        obj_quat = obj.data.root_quat_w[env_ids]
        obj_rot = matrix_from_quat(obj_quat)
        object_poses["object"] = make_pose(obj_pos, obj_rot)

        return object_poses

    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Extract gripper actions from the full action tensor.

        For joint states control, the action format is:
        [joint_1, joint_2, joint_3, joint_4, joint_5, gripper]

        Args:
            actions: Full action tensor [num_steps, action_dim] or [action_dim]

        Returns:
            dict: Dictionary mapping arm name to gripper action tensor
        """
        # Gripper is the last element of the action
        if actions.dim() == 1:
            gripper_action = actions[-1:]
        else:
            gripper_action = actions[:, -1:]

        return {"end_effector": gripper_action}

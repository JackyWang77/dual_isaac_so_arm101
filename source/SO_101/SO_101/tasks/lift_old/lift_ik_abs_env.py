# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.envs import ManagerBasedRLMimicEnv

from .lift_ik_abs_env_cfg import SoArm100LiftIKAbsEnvCfg


class SoArm100LiftIKAbsEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic environment wrapper for SO-ARM101 Lift task with IK Absolute control.

    Action space: [x, y, z, qw, qx, qy, qz, gripper] (8 dimensions)
    - Position (x, y, z): Target end-effector position in robot base frame
    - Quaternion (qw, qx, qy, qz): Target end-effector orientation
    - Gripper: Binary control (-1.0 = close, 1.0 = open)
    """

    cfg: SoArm100LiftIKAbsEnvCfg

    def __init__(self, cfg: SoArm100LiftIKAbsEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Get current robot end effector pose relative to robot base frame.

        IMPORTANT: For IK Absolute mode, the pose must be in the robot base frame,
        NOT world frame. The FrameTransformer provides target_pos_source/target_quat_source
        which are relative to the source frame (robot base).

        Args:
            eef_name: Name of the end effector (not used in single-arm setup)
            env_ids: Environment IDs to get poses for (None = all environments)

        Returns:
            torch.Tensor: End-effector poses as 4x4 transformation matrices in robot base frame
        """
        if env_ids is None:
            env_ids = slice(None)

        # Get EEF pose relative to robot base frame (source frame)
        # target_pos_source/target_quat_source are in the source frame (robot base)
        # target_pos_w/target_quat_w are in world frame - DO NOT USE for IK Absolute!
        ee_frame = self.scene["ee_frame"]
        eef_pos_b = ee_frame.data.target_pos_source[
            env_ids, 0, :
        ]  # [num_envs, 3] in base frame
        eef_quat_b = ee_frame.data.target_quat_source[
            env_ids, 0, :
        ]  # [num_envs, 4] (w, x, y, z) in base frame

        return PoseUtils.make_pose(eef_pos_b, PoseUtils.matrix_from_quat(eef_quat_b))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert target end-effector pose and gripper action to environment action.

        This method transforms a dictionary of target end-effector poses and gripper actions
        into a single action tensor that can be used by the IK Absolute controller.

        Args:
            target_eef_pose_dict: Dictionary containing target end-effector pose(s),
                with keys as eef names and values as 4x4 pose tensors.
            gripper_action_dict: Dictionary containing gripper action(s),
                with keys as eef names and values as action tensors.
            action_noise_dict: Optional dictionary with noise magnitudes per eef name.
                If provided, random noise is generated and added to the pose action.
            env_id: Environment ID for multi-environment setups, defaults to 0.

        Returns:
            torch.Tensor: Action tensor [pos(3), quat(4), gripper(1)] = 8 dimensions
        """
        # Extract target position and rotation from the first (and only) eef
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # Get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # Construct pose action: [x, y, z, qw, qx, qy, qz]
        # PoseUtils.quat_from_matrix returns quaternion in [w, x, y, z] format
        pose_action = torch.cat(
            [target_pos, PoseUtils.quat_from_matrix(target_rot)], dim=0
        )

        # Add exploration noise to action if specified
        if action_noise_dict is not None:
            eef_name = list(target_eef_pose_dict.keys())[0]
            if eef_name in action_noise_dict:
                noise = action_noise_dict[eef_name]
                noise_vec = noise * torch.randn_like(pose_action)
                pose_action += noise_vec

        # Combine pose action with gripper action: [pos(3), quat(4), gripper(1)]
        return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(
        self, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Convert action tensor back to target end-effector pose dictionary.

        Args:
            action: Action tensor [pos(3), quat(4), gripper(1)]

        Returns:
            dict: Dictionary with eef name as key and 4x4 pose tensor as value
        """
        eef_name = "end_effector"  # Single arm configuration

        # Extract position and quaternion from action
        target_pos = action[:, :3]
        target_quat = action[:, 3:7]
        target_rot = PoseUtils.matrix_from_quat(target_quat)

        # Construct 4x4 transformation matrix
        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Extract gripper actions from full action tensor.

        Args:
            actions: Full action tensor [pos(3), quat(4), gripper(1)]

        Returns:
            dict: Dictionary with eef name as key and gripper action as value
        """
        # Last dimension is gripper action
        return {"end_effector": actions[:, -1:]}

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Get subtask termination signals from observation buffer.

        These signals indicate when each subtask is completed,
        used by the Mimic data generation system to segment demonstrations.

        Args:
            env_ids: Environment IDs to get signals for (None = all environments)

        Returns:
            dict: Dictionary mapping subtask names to termination signal tensors
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]

        # Lift task: lift_object
        signals["lift_object"] = subtask_terms["lift_object"][env_ids]

        return signals

    def get_object_poses(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Get the poses of all manipulated objects as 4x4 transformation matrices.

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
        obj_rot = PoseUtils.matrix_from_quat(obj_quat)
        object_poses["object"] = PoseUtils.make_pose(obj_pos, obj_rot)

        return object_poses

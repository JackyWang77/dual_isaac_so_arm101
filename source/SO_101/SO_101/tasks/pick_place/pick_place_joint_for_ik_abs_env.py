# Copyright (c) 2024-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Environment wrapper for collecting IK Absolute demos using Joint Control teleoperation.

This environment allows teleoperation via Joint Control (more stable and intuitive)
but records end-effector absolute poses as actions for training IK Absolute policies.
"""

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.sensors import FrameTransformer

from .pick_place_joint_pos_env_cfg import DualArmPickPlaceJointPosEnvCfg


class DualArmPickPlaceJointForIKAbsEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic environment wrapper for collecting IK Absolute data using Joint Control.
    
    This environment:
    - Accepts Joint Control actions for teleoperation (stable and intuitive)
    - Records end-effector absolute poses as actions (for training IK Absolute policies)
    - Converts between joint space and Cartesian space transparently
    
    Teleoperation action space: [joint_pos_1, ..., joint_pos_5, gripper] - Joint Control
    Recorded action space: [x, y, z, qw, qx, qy, qz, gripper] - EE Absolute Pose
    """

    cfg: DualArmPickPlaceJointPosEnvCfg

    def __init__(self, cfg: DualArmPickPlaceJointPosEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Get the end-effector frame transformer from the scene
        self.ee_frame: FrameTransformer = self.scene["ee_frame"]

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose (relative to robot base frame).
        
        Args:
            eef_name: Name of the end-effector (not used for single arm)
            env_ids: Environment IDs to query, defaults to all environments
            
        Returns:
            torch.Tensor: 4x4 transformation matrix [num_envs, 4, 4]
        """
        if env_ids is None:
            env_ids = slice(None)

        # Get end-effector position and orientation from FrameTransformer
        # ee_frame provides the pose in the robot base frame
        ee_pos_b = self.ee_frame.data.target_pos_source[env_ids, 0, :]  # [num_envs, 3]
        ee_quat_b = self.ee_frame.data.target_quat_source[env_ids, 0, :]  # [num_envs, 4] (w, x, y, z)
        
        # Convert to rotation matrix and construct 4x4 pose
        ee_rot_mat = PoseUtils.matrix_from_quat(ee_quat_b)
        return PoseUtils.make_pose(ee_pos_b, ee_rot_mat)

    def target_eef_pose_to_action(
        self, target_eef_pose_dict: dict, gripper_action_dict: dict, noise: float | None = None, env_id: int = 0
    ) -> torch.Tensor:
        """Convert target end-effector pose to IK Absolute action format.
        
        This method is called when recording demonstrations to convert the current
        end-effector pose into an action tensor suitable for IK Absolute training.
        
        Args:
            target_eef_pose_dict: Dictionary containing target end-effector pose(s),
                with keys as eef names and values as 4x4 pose tensors.
            gripper_action_dict: Dictionary containing gripper action(s),
                with keys as eef names and values as action tensors.
            noise: Optional noise magnitude to apply to the pose action for exploration.
            env_id: Environment ID for multi-environment setups, defaults to 0.
            
        Returns:
            torch.Tensor: Action tensor [1, 8] with format [x, y, z, qw, qx, qy, qz, gripper]
        """
        # Extract target pose and gripper action
        (target_eef_pose,) = target_eef_pose_dict.values()
        (gripper_action,) = gripper_action_dict.values()

        # Decompose 4x4 pose matrix
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)
        
        # Convert rotation matrix to quaternion [w, x, y, z]
        target_quat = PoseUtils.quat_from_matrix(target_rot)
        
        # Construct pose action: [x, y, z, qw, qx, qy, qz]
        pose_action = torch.cat([target_pos, target_quat], dim=0)
        
        # Add exploration noise if specified
        if noise is not None:
            noise_vec = noise * torch.randn_like(pose_action)
            pose_action += noise_vec

        # Combine pose with gripper: [x, y, z, qw, qx, qy, qz, gripper]
        return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert IK Absolute action tensor back to target end-effector pose dictionary.
        
        Args:
            action: Action tensor [batch, 8] with format [x, y, z, qw, qx, qy, qz, gripper]
            
        Returns:
            dict: Dictionary with eef name as key and 4x4 pose tensor as value
        """
        eef_name = "end_effector"

        # Extract position and quaternion from action
        target_pos = action[:, :3]
        target_quat = action[:, 3:7]  # [w, x, y, z]
        
        # Convert quaternion to rotation matrix
        target_rot = PoseUtils.matrix_from_quat(target_quat)
        
        # Construct 4x4 transformation matrix
        target_poses = PoseUtils.make_pose(target_pos, target_rot)

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions from full action tensor.
        
        Args:
            actions: Full action tensor [batch, 8] with format [x, y, z, qw, qx, qy, qz, gripper]
            
        Returns:
            dict: Dictionary with eef name as key and gripper action as value
        """
        return {"end_effector": actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals from observation buffer.
        
        These signals indicate when each subtask is completed,
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
        
        # Map subtask termination signals from observation buffer
        # Simplified task: push_cube and lift_ee
        signals["push_cube"] = subtask_terms["push_cube"][env_ids]
        signals["lift_ee"] = subtask_terms["lift_ee"][env_ids]
        
        return signals

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get the poses of all manipulated objects as 4x4 transformation matrices.
        
        Args:
            env_ids: Environment IDs to query, defaults to all environments
        
        Returns:
            dict: Dictionary mapping object name to 4x4 pose matrix
        """
        if env_ids is None:
            env_ids = slice(None)
        
        object_poses = {}
        
        # Get cube pose as 4x4 matrix
        cube = self.scene["cube"]
        cube_pos = cube.data.root_pos_w[env_ids]
        cube_quat = cube.data.root_quat_w[env_ids]
        cube_rot = PoseUtils.matrix_from_quat(cube_quat)
        object_poses["cube"] = PoseUtils.make_pose(cube_pos, cube_rot)
        
        # Get tray pose as 4x4 matrix (target object)
        tray = self.scene["object"]
        tray_pos = tray.data.root_pos_w[env_ids]
        tray_quat = tray.data.root_quat_w[env_ids]
        tray_rot = PoseUtils.matrix_from_quat(tray_quat)
        object_poses["object"] = PoseUtils.make_pose(tray_pos, tray_rot)
        
        return object_poses



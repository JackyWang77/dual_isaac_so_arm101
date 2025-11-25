# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from .pick_place_ik_abs_env_cfg import DualArmPickPlaceIKAbsEnvCfg


class DualArmPickPlaceIKAbsEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic environment wrapper for SO-ARM100 Pick & Place with IK Absolute control.
    
    Action space: [x, y, z, qw, qx, qy, qz, gripper] (8 dimensions)
    - Position (x, y, z): Target end-effector position in robot base frame
    - Quaternion (qw, qx, qy, qz): Target end-effector orientation
    - Gripper: Binary control (-1.0 = close, 1.0 = open)
    """

    cfg: DualArmPickPlaceIKAbsEnvCfg

    def __init__(self, cfg: DualArmPickPlaceIKAbsEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose from observation buffer.
        
        Args:
            eef_name: Name of the end effector (not used in single-arm setup)
            env_ids: Environment IDs to get poses for (None = all environments)
            
        Returns:
            torch.Tensor: End-effector poses as 4x4 transformation matrices
        """
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is [w, x, y, z]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self, target_eef_pose_dict: dict, gripper_action_dict: dict, noise: float | None = None, env_id: int = 0
    ) -> torch.Tensor:
        """Convert target end-effector pose and gripper action to environment action.
        
        This method transforms a dictionary of target end-effector poses and gripper actions
        into a single action tensor that can be used by the IK Absolute controller.
        
        Args:
            target_eef_pose_dict: Dictionary containing target end-effector pose(s),
                with keys as eef names and values as 4x4 pose tensors.
            gripper_action_dict: Dictionary containing gripper action(s),
                with keys as eef names and values as action tensors.
            noise: Optional noise magnitude to apply to the pose action for exploration.
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
        pose_action = torch.cat([target_pos, PoseUtils.quat_from_matrix(target_rot)], dim=0)
        
        # Add exploration noise to action if specified
        if noise is not None:
            noise_vec = noise * torch.randn_like(pose_action)
            pose_action += noise_vec

        # Combine pose action with gripper action: [pos(3), quat(4), gripper(1)]
        return torch.cat([pose_action, gripper_action], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
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

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract gripper actions from full action tensor.
        
        Args:
            actions: Full action tensor [pos(3), quat(4), gripper(1)]
            
        Returns:
            dict: Dictionary with eef name as key and gripper action as value
        """
        # Last dimension is gripper action
        return {"end_effector": actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals from observation buffer.
        
        These signals indicate when each subtask (pick/place) is completed,
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
        
        # Map subtask termination signals from observation buffer
        # These correspond to the pick and place events for each object
        signals["push_plate"] = subtask_terms["push_plate"][env_ids]
        signals["pick_plate"] = subtask_terms["pick_plate"][env_ids]
        signals["place_plate"] = subtask_terms["place_plate"][env_ids]
        signals["pick_fork"] = subtask_terms["pick_fork"][env_ids]
        signals["place_fork"] = subtask_terms["place_fork"][env_ids]
        signals["pick_knife"] = subtask_terms["pick_knife"][env_ids]
        signals["place_knife"] = subtask_terms["place_knife"][env_ids]
        
        return signals

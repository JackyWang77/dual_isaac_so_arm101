# Copyright (c) 2024-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Environment wrapper for recording joint states directly.

This environment records joint states (joint positions + gripper) as actions,
without converting to EE pose. The conversion can be done later when reading
the hdf5 file using forward kinematics.
"""

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLMimicEnv

from .pick_place_joint_pos_env_cfg import DualArmPickPlaceJointPosEnvCfg


class DualArmPickPlaceJointStatesMimicEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic environment wrapper for recording joint states directly.
    
    This environment:
    - Accepts Joint Control actions for teleoperation (joint positions + gripper)
    - Records joint states directly as actions (no conversion to EE pose)
    - Can be converted to EE pose later using forward kinematics
    
    Teleoperation action space: [joint_pos_1, ..., joint_pos_5, gripper] - Joint Control
    Recorded action space: [joint_pos_1, ..., joint_pos_5, gripper] - Joint States (same as input)
    """

    cfg: DualArmPickPlaceJointPosEnvCfg

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Return current joint states as actions (no conversion to EE pose).
        
        This method is called during recording, but we directly return the current
        joint states instead of converting from EE pose. This allows us to record
        joint states directly, which can be converted to EE pose later using FK.
        
        Args:
            target_eef_pose_dict: Not used (kept for interface compatibility)
            gripper_action_dict: Dictionary containing gripper action(s)
            action_noise_dict: Optional noise to apply
            env_id: Environment ID
            
        Returns:
            torch.Tensor: Joint states [joint_1, ..., joint_5, gripper]
        """
        # Get current joint positions from the robot
        robot = self.scene["robot"]
        joint_pos = robot.data.joint_pos[env_id]  # [num_joints]
        
        # Get gripper action
        (gripper_action,) = gripper_action_dict.values()
        
        # Combine joint positions and gripper
        # Note: We need to select only the arm joints (excluding gripper)
        # The action space is [arm_joint_1, ..., arm_joint_5, gripper]
        arm_joint_names = self.cfg.actions.arm_action.joint_names
        gripper_joint_names = self.cfg.actions.gripper_action.joint_names
        
        # Get indices of arm joints
        all_joint_names = robot.joint_names
        arm_joint_indices = [all_joint_names.index(name) for name in arm_joint_names]
        arm_joint_pos = joint_pos[arm_joint_indices]
        
        # Combine arm joints and gripper
        action = torch.cat([arm_joint_pos, gripper_action], dim=0)
        
        # Apply noise if specified
        if action_noise_dict is not None:
            eef_name = list(self.cfg.subtask_configs.keys())[0]
            if eef_name in action_noise_dict:
                noise = action_noise_dict[eef_name] * torch.randn_like(action)
                action += noise
        
        return action.unsqueeze(0)  # Add batch dimension

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Get subtask termination signals from observation buffer.
        
        These signals indicate when each subtask (pick/place) is completed,
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
        signals["push_plate"] = subtask_terms["push_plate"][env_ids]
        signals["pick_plate"] = subtask_terms["pick_plate"][env_ids]
        signals["place_plate"] = subtask_terms["place_plate"][env_ids]
        signals["pick_fork"] = subtask_terms["pick_fork"][env_ids]
        signals["place_fork"] = subtask_terms["place_fork"][env_ids]
        signals["pick_knife"] = subtask_terms["pick_knife"][env_ids]
        signals["place_knife"] = subtask_terms["place_knife"][env_ids]
        
        return signals


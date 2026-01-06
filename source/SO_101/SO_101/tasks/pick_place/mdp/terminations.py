# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def cubes_stacked(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
#     cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
#     cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
#     xy_threshold: float = 0.04,
#     height_threshold: float = 0.005,
#     height_diff: float = 0.0468,
#     atol=0.0001,
#     rtol=0.0001,
# ):
#     robot: Articulation = env.scene[robot_cfg.name]
#     cube_1: RigidObject = env.scene[cube_1_cfg.name]
#     cube_2: RigidObject = env.scene[cube_2_cfg.name]
#     cube_3: RigidObject = env.scene[cube_3_cfg.name]

#     pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w
#     pos_diff_c23 = cube_2.data.root_pos_w - cube_3.data.root_pos_w

#     # Compute cube position difference in x-y plane
#     xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
#     xy_dist_c23 = torch.norm(pos_diff_c23[:, :2], dim=1)

#     # Compute cube height difference
#     h_dist_c12 = torch.norm(pos_diff_c12[:, 2:], dim=1)
#     h_dist_c23 = torch.norm(pos_diff_c23[:, 2:], dim=1)

#     # Check cube positions
#     stacked = torch.logical_and(xy_dist_c12 < xy_threshold, xy_dist_c23 < xy_threshold)
#     stacked = torch.logical_and(h_dist_c12 - height_diff < height_threshold, stacked)
#     stacked = torch.logical_and(pos_diff_c12[:, 2] < 0.0, stacked)
#     stacked = torch.logical_and(h_dist_c23 - height_diff < height_threshold, stacked)
#     stacked = torch.logical_and(pos_diff_c23[:, 2] < 0.0, stacked)

#     # Check gripper positions
#     if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
#         surface_gripper = env.scene.surface_grippers["surface_gripper"]
#         suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
#         suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
#         stacked = torch.logical_and(suction_cup_is_open, stacked)

#     else:
#         if hasattr(env.cfg, "gripper_joint_names"):
#             gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
#             assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

#             stacked = torch.logical_and(
#                 torch.isclose(
#                     robot.data.joint_pos[:, gripper_joint_ids[0]],
#                     torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
#                     atol=atol,
#                     rtol=rtol,
#                 ),
#                 stacked,
#             )
#             stacked = torch.logical_and(
#                 torch.isclose(
#                     robot.data.joint_pos[:, gripper_joint_ids[1]],
#                     torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
#                     atol=atol,
#                     rtol=rtol,
#                 ),
#                 stacked,
#             )
#         else:
#             raise ValueError("No gripper_joint_names found in environment config")

#     return stacked


def objects_picked_and_placed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_center_tolerance: float = 0.03,
    utensil_position_tolerance: float = 0.03,
    plate_height_target: float = 0.02,
    utensil_height_target: float = 0.02,
    height_tolerance: float = 0.02,
    fork_target_offset: tuple[float, float] = (0.0, 0.08),
    knife_target_offset: tuple[float, float] = (0.0, -0.08),
):
    """Check if plate/fork/knife follow the tray layout with the gripper released.

    Mirrors the original cube stacking logic: each object must be within its positional
    and height tolerances while the gripper returns to its open state.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]
    fork: RigidObject = env.scene[fork_cfg.name]
    knife: RigidObject = env.scene[knife_cfg.name]
    tray: RigidObject = env.scene[tray_cfg.name]

    tray_pos = tray.data.root_pos_w
    tray_quat = tray.data.root_quat_w

    plate_rel_pos, _ = math_utils.subtract_frame_transforms(
        tray_pos,
        tray_quat,
        plate.data.root_pos_w,
        plate.data.root_quat_w,
    )
    fork_rel_pos, _ = math_utils.subtract_frame_transforms(
        tray_pos,
        tray_quat,
        fork.data.root_pos_w,
        fork.data.root_quat_w,
    )
    knife_rel_pos, _ = math_utils.subtract_frame_transforms(
        tray_pos,
        tray_quat,
        knife.data.root_pos_w,
        knife.data.root_quat_w,
    )

    plate_centered = (
        torch.linalg.norm(plate_rel_pos[:, :2], dim=1) < plate_center_tolerance
    )
    plate_height_ok = (
        torch.abs(plate_rel_pos[:, 2] - plate_height_target) < height_tolerance
    )
    plate_ok = torch.logical_and(plate_centered, plate_height_ok)

    fork_target = torch.tensor(
        fork_target_offset, dtype=plate_rel_pos.dtype, device=env.device
    )
    knife_target = torch.tensor(
        knife_target_offset, dtype=plate_rel_pos.dtype, device=env.device
    )

    fork_planar_error = torch.linalg.norm(
        fork_rel_pos[:, :2] - fork_target, dim=1
    )
    fork_height_ok = (
        torch.abs(fork_rel_pos[:, 2] - utensil_height_target) < height_tolerance
    )
    fork_ok = torch.logical_and(
        fork_planar_error < utensil_position_tolerance,
        fork_height_ok,
    )

    knife_planar_error = torch.linalg.norm(
        knife_rel_pos[:, :2] - knife_target, dim=1
    )
    knife_height_ok = (
        torch.abs(knife_rel_pos[:, 2] - utensil_height_target) < height_tolerance
    )
    knife_ok = torch.logical_and(
        knife_planar_error < utensil_position_tolerance,
        knife_height_ok,
    )

    layout_ok = torch.logical_and(
        plate_ok,
        torch.logical_and(fork_ok, knife_ok),
    )

    if (
        hasattr(env.scene, "surface_grippers")
        and len(env.scene.surface_grippers) > 0
    ):
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        # 1: closed, 0: closing, -1: open
        suction_cup_status = surface_gripper.state.view(-1, 1)
        suction_cup_is_open = (suction_cup_status == -1).squeeze(1)
        layout_ok = torch.logical_and(suction_cup_is_open, layout_ok)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            open_val = torch.tensor(
                env.cfg.gripper_open_val, dtype=plate_rel_pos.dtype
            ).to(env.device)
            
            # Support both single-DOF and parallel grippers
            # Use larger tolerance (0.1) since gripper may not reach exact open position
            if len(gripper_joint_ids) == 1:
                # Single DOF gripper (like SO-ARM101)
                gripper_open = torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]] - open_val
                ) < 0.1
            elif len(gripper_joint_ids) == 2:
                # Parallel gripper with two joints
                finger_0_open = torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]] - open_val
                ) < 0.1
                finger_1_open = torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]] - open_val
                ) < 0.1
                gripper_open = torch.logical_and(finger_0_open, finger_1_open)
            else:
                raise ValueError(f"Unsupported gripper configuration with {len(gripper_joint_ids)} joints. Expected 1 or 2.")
            
            layout_ok = torch.logical_and(gripper_open, layout_ok)
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return layout_ok


def objects_picked_and_placed_no_knife(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),  # Not used, kept for compatibility
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    plate_center_tolerance: float = 0.03,  # Not used, kept for compatibility
    cube_position_tolerance: float = 0.05,  # 5cm tolerance (larger range)
    plate_height_target: float = 0.02,  # Not used, kept for compatibility
    cube_height_target: float = 0.02,
    height_tolerance: float = 0.03,  # 3cm height tolerance
    cube_target_offset: tuple[float, float] = (0.0, 0.0),  # Tray center
):
    """Check if cube is placed on tray with the gripper released.
    
    Simplified version - only checks cube placement (plate removed for testing).
    """

    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    tray: RigidObject = env.scene[tray_cfg.name]

    tray_pos = tray.data.root_pos_w
    tray_quat = tray.data.root_quat_w

    cube_rel_pos, _ = math_utils.subtract_frame_transforms(
        tray_pos,
        tray_quat,
        cube.data.root_pos_w,
        cube.data.root_quat_w,
    )

    cube_target = torch.tensor(
        cube_target_offset, dtype=cube_rel_pos.dtype, device=env.device
    )

    cube_planar_error = torch.linalg.norm(
        cube_rel_pos[:, :2] - cube_target, dim=1
    )
    cube_height_ok = (
        torch.abs(cube_rel_pos[:, 2] - cube_height_target) < height_tolerance
    )
    cube_ok = torch.logical_and(
        cube_planar_error < cube_position_tolerance,
        cube_height_ok,
    )

    # Only check cube (plate removed for testing)
    layout_ok = cube_ok

    if (
        hasattr(env.scene, "surface_grippers")
        and len(env.scene.surface_grippers) > 0
    ):
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        # 1: closed, 0: closing, -1: open
        suction_cup_status = surface_gripper.state.view(-1, 1)
        suction_cup_is_open = (suction_cup_status == -1).squeeze(1)
        layout_ok = torch.logical_and(suction_cup_is_open, layout_ok)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            open_val = torch.tensor(
                env.cfg.gripper_open_val, dtype=cube_rel_pos.dtype
            ).to(env.device)
            
            # Support both single-DOF and parallel grippers
            # Use larger tolerance (0.1) since gripper may not reach exact open position
            if len(gripper_joint_ids) == 1:
                # Single DOF gripper (like SO-ARM101)
                gripper_open = torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]] - open_val
                ) < 0.1
            elif len(gripper_joint_ids) == 2:
                # Parallel gripper with two joints
                finger_0_open = torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]] - open_val
                ) < 0.1
                finger_1_open = torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]] - open_val
                ) < 0.1
                gripper_open = torch.logical_and(finger_0_open, finger_1_open)
            else:
                raise ValueError(f"Unsupported gripper configuration with {len(gripper_joint_ids)} joints. Expected 1 or 2.")
            
            layout_ok = torch.logical_and(gripper_open, layout_ok)
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return layout_ok


def cube_pushed(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cube_position_tolerance: float = 0.05,  # 5cm tolerance
    cube_target_offset: tuple[float, float] = (0.0, 0.0),  # Tray center
):
    """Check if cube is pushed to target position (no gripper check).
    
    Simplified push task - only checks cube position on tray, no gripper state required.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    tray: RigidObject = env.scene[tray_cfg.name]

    tray_pos = tray.data.root_pos_w
    tray_quat = tray.data.root_quat_w

    cube_rel_pos, _ = math_utils.subtract_frame_transforms(
        tray_pos,
        tray_quat,
        cube.data.root_pos_w,
        cube.data.root_quat_w,
    )

    cube_target = torch.tensor(
        cube_target_offset, dtype=cube_rel_pos.dtype, device=env.device
    )

    cube_planar_error = torch.linalg.norm(
        cube_rel_pos[:, :2] - cube_target, dim=1
    )
    
    # Only check planar position, no height or gripper check
    pushed = cube_planar_error < cube_position_tolerance

    return pushed


def ee_lifted_termination(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    min_height: float = 0.07,  # 7cm above table
):
    """Check if the end-effector is lifted above a minimum height.
    
    Used as termination condition for "lift hand up" after completing a task.
    """
    from isaaclab.sensors import FrameTransformer
    
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Get EE height (z position in world frame)
    ee_height = ee_frame.data.target_pos_w[:, 0, 2]  # [num_envs]
    
    lifted = ee_height > min_height
    
    return lifted


def push_and_lift_complete(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    tray_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_position_tolerance: float = 0.05,
    cube_target_offset: tuple[float, float] = (0.0, 0.0),
    min_height: float = 0.07,
):
    """Check if both push and lift tasks are complete.
    
    Success condition: cube pushed to target AND EE lifted above 7cm.
    """
    # Check cube pushed
    pushed = cube_pushed(env, cube_cfg, tray_cfg, cube_position_tolerance, cube_target_offset)
    
    # Check EE lifted
    lifted = ee_lifted_termination(env, ee_frame_cfg, min_height)
    
    return torch.logical_and(pushed, lifted)

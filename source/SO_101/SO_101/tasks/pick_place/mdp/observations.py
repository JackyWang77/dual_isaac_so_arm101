# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),  # Not used, kept for compatibility
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),  # Not used, kept for compatibility
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),  # Not used, kept for compatibility
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """The position of the cube in the world frame."""
    cube: RigidObject = env.scene[cube_cfg.name]
    # Only cube for testing

    return cube.data.root_pos_w


def instance_randomize_cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),  # Not used, kept for compatibility
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),  # Not used, kept for compatibility
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),  # Not used, kept for compatibility
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """The position of the cube in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 3), fill_value=-1)  # Only cube (3 coords)

    cube: RigidObjectCollection = env.scene[cube_cfg.name]
    # Only cube for testing

    cube_pos_w = []
    for env_id in range(env.num_envs):
        cube_pos_w.append(cube.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
    cube_pos_w = torch.stack(cube_pos_w)

    return cube_pos_w


def cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),  # Not used, kept for compatibility
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),  # Not used, kept for compatibility
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),  # Not used, kept for compatibility
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
):
    """The orientation of the cube in the world frame."""
    cube: RigidObject = env.scene[cube_cfg.name]
    # Only cube for testing

    return cube.data.root_quat_w


def object_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),  # Not used, kept for compatibility
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),  # Not used, kept for compatibility
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
):
    """Alias for cube_orientations_in_world_frame to keep config compatibility."""

    return cube_orientations_in_world_frame(env, plate_cfg, fork_cfg, knife_cfg, cube_cfg)


def instance_randomize_object_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),  # Not used, kept for compatibility
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),  # Not used, kept for compatibility
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),  # Not used, kept for compatibility
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """The orientation of the cube in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 4), fill_value=-1)  # Only cube (4 quat)

    cube: RigidObjectCollection = env.scene[cube_cfg.name]
    # Only cube for testing

    cube_quat_w = []
    for env_id in range(env.num_envs):
        cube_quat_w.append(cube.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
    cube_quat_w = torch.stack(cube_quat_w)
    return cube_quat_w


def object_obs(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),  # Not used, kept for compatibility
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),  # Not used, kept for compatibility
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),  # Not used, kept for compatibility
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube pos,
        cube quat,
        gripper to cube,
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    # Only cube for testing
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = cube.data.root_pos_w
    cube_quat_w = cube.data.root_quat_w

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_cube = cube_pos_w - ee_pos_w

    return torch.cat(
        (
            cube_pos_w - env.scene.env_origins,
            cube_quat_w,
            gripper_to_cube,
        ),
        dim=1,
    )


def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),  # Not used, kept for compatibility
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),  # Not used, kept for compatibility
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),  # Not used, kept for compatibility
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        cube pos,
        cube quat,
        gripper to cube,
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 10), fill_value=-1)  # cube pos(3) + quat(4) + gripper_to_cube(3)

    cube: RigidObjectCollection = env.scene[cube_cfg.name]
    # Only cube for testing
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = []
    cube_quat_w = []
    for env_id in range(env.num_envs):
        cube_pos_w.append(cube.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        cube_quat_w.append(cube.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
    cube_pos_w = torch.stack(cube_pos_w)
    cube_quat_w = torch.stack(cube_quat_w)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_cube = cube_pos_w - ee_pos_w

    return torch.cat(
        (
            cube_pos_w - env.scene.env_origins,
            cube_quat_w,
            gripper_to_cube,
        ),
        dim=1,
    )


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Obtain the versatile gripper position of both Gripper and Suction Cup.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        # Handle multiple surface grippers by concatenating their states
        gripper_states = []
        for gripper_name, surface_gripper in env.scene.surface_grippers.items():
            gripper_states.append(surface_gripper.state.view(-1, 1))

        if len(gripper_states) == 1:
            return gripper_states[0]
        else:
            return torch.cat(gripper_states, dim=1)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            # Support both single-DOF grippers (e.g., SO-ARM101) and parallel grippers
            if len(gripper_joint_ids) == 1:
                # Single DOF gripper (like SO-ARM101 jaw_joint)
                return robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
            elif len(gripper_joint_ids) == 2:
                # Parallel gripper with two joints
                finger_joint_1 = robot.data.joint_pos[:, gripper_joint_ids[0]].clone().unsqueeze(1)
                finger_joint_2 = -1 * robot.data.joint_pos[:, gripper_joint_ids[1]].clone().unsqueeze(1)
                return torch.cat((finger_joint_1, finger_joint_2), dim=1)
            else:
                raise ValueError(f"Unsupported gripper configuration with {len(gripper_joint_ids)} joints. Expected 1 or 2.")
        else:
            raise NotImplementedError("[Error] Cannot find gripper_joint_names in the environment config")


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    diff_threshold: float = 0.06,
    min_lift_height: float = 0.01,
    table_height: float = 0.0,
) -> torch.Tensor:
    """Check if an object is picked by the specified robot.
    
    Requires:
    - EE close to object (within diff_threshold)
    - Gripper closed
    - Object lifted above table (by min_lift_height)
    
    Note: table_height is used because table is AssetBaseCfg (not RigidObject) and doesn't have data attribute.
    Default is 0.0 based on table initial position.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)
    
    # Check if object is lifted above table (using fixed table height since table is AssetBaseCfg)
    # Table initial position is [0.5, 0, 0], so table height is 0.0
    object_height_above_table = object_pos[:, 2] - table_height
    is_lifted = object_height_above_table > min_lift_height

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(
            torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold),
            is_lifted
        )

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            
            # Support both single-DOF and parallel grippers
            if len(gripper_joint_ids) == 1:
                # Single DOF gripper (like SO-ARM101)
                gripper_closed = torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                ) > env.cfg.gripper_threshold
                grasped = torch.logical_and(
                    torch.logical_and(pose_diff < diff_threshold, gripper_closed),
                    is_lifted
                )
            elif len(gripper_joint_ids) == 2:
                # Parallel gripper with two joints
                finger_0_closed = torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[0]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                ) > env.cfg.gripper_threshold
                finger_1_closed = torch.abs(
                    robot.data.joint_pos[:, gripper_joint_ids[1]]
                    - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                ) > env.cfg.gripper_threshold
                gripper_closed = torch.logical_and(finger_0_closed, finger_1_closed)
                grasped = torch.logical_and(
                    torch.logical_and(pose_diff < diff_threshold, gripper_closed),
                    is_lifted
                )
            else:
                raise ValueError(f"Unsupported gripper configuration with {len(gripper_joint_ids)} joints. Expected 1 or 2.")
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return grasped


def object_placed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    planar_offset: tuple[float, float] = (0.0, 0.0),
    planar_tolerance: float = 0.03,
    height_target: float = 0.02,
    height_tolerance: float = 0.02,
) -> torch.Tensor:
    """Check if an object is placed on the specified target with the correct layout."""

    del ee_frame_cfg  # not required for placement check, kept for backward compatibility

    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    target_pos = target.data.root_pos_w
    target_quat = target.data.root_quat_w
    obj_rel_pos, _ = math_utils.subtract_frame_transforms(
        target_pos, target_quat, obj.data.root_pos_w, obj.data.root_quat_w
    )

    planar_target = torch.tensor(planar_offset, dtype=obj_rel_pos.dtype, device=env.device)
    planar_error = torch.linalg.norm(obj_rel_pos[:, :2] - planar_target, dim=1)
    height_error = torch.abs(obj_rel_pos[:, 2] - height_target)

    placed = torch.logical_and(planar_error < planar_tolerance, height_error < height_tolerance)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).squeeze(1)
        placed = torch.logical_and(suction_cup_is_open, placed)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            open_val = torch.tensor(env.cfg.gripper_open_val, dtype=obj_rel_pos.dtype).to(env.device)
            
            # Support both single-DOF and parallel grippers
            # Use larger tolerance (0.1) since gripper may not reach exact open position
            if len(gripper_joint_ids) == 1:
                # Single DOF gripper (like SO-ARM101)
                # Check if gripper is close to open position (within 0.1 rad tolerance)
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
            placed = torch.logical_and(gripper_open, placed)
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return placed


def object_pushed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    planar_offset: tuple[float, float] = (0.0, 0.0),
    planar_tolerance: float = 0.03,
    height_target: float = 0.02,
    height_tolerance: float = 0.02,
) -> torch.Tensor:
    """Check if an object is pushed to the specified target position.
    
    Similar to object_placed, but doesn't require gripper to be open (since pushing doesn't involve grasping).
    """
    
    del ee_frame_cfg  # not required for push check, kept for backward compatibility
    del robot_cfg  # not required for push check

    obj: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    target_pos = target.data.root_pos_w
    target_quat = target.data.root_quat_w
    obj_rel_pos, _ = math_utils.subtract_frame_transforms(
        target_pos, target_quat, obj.data.root_pos_w, obj.data.root_quat_w
    )

    planar_target = torch.tensor(planar_offset, dtype=obj_rel_pos.dtype, device=env.device)
    planar_error = torch.linalg.norm(obj_rel_pos[:, :2] - planar_target, dim=1)

    # Push only checks planar position, height doesn't matter for pushing
    pushed = planar_error < planar_tolerance

    return pushed


def ee_lifted(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    min_height: float = 0.07,  # 7cm above table
) -> torch.Tensor:
    """Check if the end-effector is lifted above a minimum height.
    
    Used as a subtask signal for "lift hand up" after completing a task.
    
    Args:
        env: The environment
        ee_frame_cfg: Configuration for the EE frame
        min_height: Minimum height (in meters) for EE to be considered "lifted"
        
    Returns:
        Boolean tensor indicating if EE is above min_height
    """
    from isaaclab.sensors import FrameTransformer
    
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Get EE height (z position in world frame)
    ee_height = ee_frame.data.target_pos_w[:, 0, 2]  # [num_envs]
    
    lifted = ee_height > min_height
    
    return lifted


def object_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),  # Not used, kept for compatibility
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),  # Not used, kept for compatibility
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),  # Not used, kept for compatibility
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """The position and orientation of the cube in the robot base frame."""

    cube: RigidObject = env.scene[cube_cfg.name]
    # Only cube for testing

    pos_cube_world = cube.data.root_pos_w
    quat_cube_world = cube.data.root_quat_w

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    pos_cube_base, quat_cube_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_cube_world, quat_cube_world
    )

    if return_key == "pos":
        return pos_cube_base
    elif return_key == "quat":
        return quat_cube_base
    elif return_key is None:
        return torch.cat((pos_cube_base, quat_cube_base), dim=1)


def object_abs_obs_in_base_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),  # Not used, kept for compatibility
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),  # Not used, kept for compatibility
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),  # Not used, kept for compatibility
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Object Abs observations (in base frame): remove the relative observations, and add abs gripper pos and quat in robot base frame
        cube pos,
        cube quat,
        gripper pos,
        gripper quat,
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    # Only cube for testing
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    cube_pos_w = cube.data.root_pos_w
    cube_quat_w = cube.data.root_quat_w

    pos_cube_base, quat_cube_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, cube_pos_w, cube_quat_w
    )

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    ee_pos_base, ee_quat_base = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    return torch.cat(
        (
            pos_cube_base,
            quat_cube_base,
            ee_pos_base,
            ee_quat_base,
        ),
        dim=1,
    )


def ee_frame_pose_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """
    The end effector pose in the robot base frame.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_in_base, ee_quat_in_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_frame_pos_w, ee_frame_quat_w
    )

    if return_key == "pos":
        return ee_pos_in_base
    elif return_key == "quat":
        return ee_quat_in_base
    elif return_key is None:
        return torch.cat((ee_pos_in_base, ee_quat_in_base), dim=1)

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
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    plate: RigidObject = env.scene[plate_cfg.name]
    fork: RigidObject = env.scene[fork_cfg.name]
    knife: RigidObject = env.scene[knife_cfg.name]

    return torch.cat((plate.data.root_pos_w, fork.data.root_pos_w, knife.data.root_pos_w), dim=1)


def instance_randomize_cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
) -> torch.Tensor:
    """The position of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    plate: RigidObjectCollection = env.scene[plate_cfg.name]
    fork: RigidObjectCollection = env.scene[fork_cfg.name]
    knife: RigidObjectCollection = env.scene[knife_cfg.name]

    plate_pos_w = []
    fork_pos_w = []
    knife_pos_w = []
    for env_id in range(env.num_envs):
        plate_pos_w.append(plate.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        fork_pos_w.append(fork.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        knife_pos_w.append(knife.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
    plate_pos_w = torch.stack(plate_pos_w)
    fork_pos_w = torch.stack(fork_pos_w)
    knife_pos_w = torch.stack(knife_pos_w)

    return torch.cat((plate_pos_w, fork_pos_w, knife_pos_w), dim=1)


def cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
):
    """The orientation of the cubes in the world frame."""
    plate: RigidObject = env.scene[plate_cfg.name]
    fork: RigidObject = env.scene[fork_cfg.name]
    knife: RigidObject = env.scene[knife_cfg.name]

    return torch.cat((plate.data.root_quat_w, fork.data.root_quat_w, knife.data.root_quat_w), dim=1)


def object_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
):
    """Alias for cube_orientations_in_world_frame to keep config compatibility."""

    return cube_orientations_in_world_frame(env, plate_cfg, fork_cfg, knife_cfg)


def instance_randomize_object_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
) -> torch.Tensor:
    """The orientation of the cubes in the world frame."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    plate: RigidObjectCollection = env.scene[plate_cfg.name]
    fork: RigidObjectCollection = env.scene[fork_cfg.name]
    knife: RigidObjectCollection = env.scene[knife_cfg.name]

    plate_quat_w = []
    fork_quat_w = []
    knife_quat_w = []
    for env_id in range(env.num_envs):
        plate_quat_w.append(plate.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        fork_quat_w.append(fork.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        knife_quat_w.append(knife.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    plate_quat_w = torch.stack(plate_quat_w)
    fork_quat_w = torch.stack(fork_quat_w)
    knife_quat_w = torch.stack(knife_quat_w)
    return torch.cat((plate_quat_w, fork_quat_w, knife_quat_w), dim=1)


def object_obs(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        plate pos,
        plate quat,
        fork pos,
        fork quat,
        knife pos,
        knife quat,
        gripper to plate,
        gripper to fork,
        gripper to knife,
    """
    plate: RigidObject = env.scene[plate_cfg.name]
    fork: RigidObject = env.scene[fork_cfg.name]
    knife: RigidObject = env.scene[knife_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    plate_pos_w = plate.data.root_pos_w
    plate_quat_w = plate.data.root_quat_w

    fork_pos_w = fork.data.root_pos_w
    fork_quat_w = fork.data.root_quat_w

    knife_pos_w = knife.data.root_pos_w
    knife_quat_w = knife.data.root_quat_w

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_plate = plate_pos_w - ee_pos_w
    gripper_to_fork = fork_pos_w - ee_pos_w
    gripper_to_knife = knife_pos_w - ee_pos_w


    return torch.cat(
        (
            plate_pos_w - env.scene.env_origins,
            plate_quat_w,
            fork_pos_w - env.scene.env_origins,
            fork_quat_w,
            knife_pos_w - env.scene.env_origins,
            knife_quat_w,
            gripper_to_plate,
            gripper_to_fork,
            gripper_to_knife,
        ),
        dim=1,
    )


def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Object observations (in world frame):
        plate pos,
        plate quat,
        fork pos,
        fork quat,
        knife pos,
        knife quat,
        gripper to plate,
        gripper to fork,
        gripper to knife,
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    plate: RigidObjectCollection = env.scene[plate_cfg.name]
    fork: RigidObjectCollection = env.scene[fork_cfg.name]
    knife: RigidObjectCollection = env.scene[knife_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    plate_pos_w = []
    fork_pos_w = []
    knife_pos_w = []
    plate_quat_w = []
    fork_quat_w = []
    knife_quat_w = []
    for env_id in range(env.num_envs):
        plate_pos_w.append(plate.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        fork_pos_w.append(fork.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][1], :3])
        knife_pos_w.append(knife.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][2], :3])
        plate_quat_w.append(plate.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
        fork_quat_w.append(fork.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][1], :4])
        knife_quat_w.append(knife.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][2], :4])
    plate_pos_w = torch.stack(plate_pos_w)
    fork_pos_w = torch.stack(fork_pos_w)
    knife_pos_w = torch.stack(knife_pos_w)
    plate_quat_w = torch.stack(plate_quat_w)
    fork_quat_w = torch.stack(fork_quat_w)
    knife_quat_w = torch.stack(knife_quat_w)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_plate = plate_pos_w - ee_pos_w
    gripper_to_fork = fork_pos_w - ee_pos_w
    gripper_to_knife = knife_pos_w - ee_pos_w

    return torch.cat(
        (
            plate_pos_w - env.scene.env_origins,
            plate_quat_w,
            fork_pos_w - env.scene.env_origins,
            fork_quat_w,
            knife_pos_w - env.scene.env_origins,
            knife_quat_w,
            gripper_to_plate,
            gripper_to_fork,
            gripper_to_knife,
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
            # Support both single-DOF grippers (e.g., SO-ARM100) and parallel grippers
            if len(gripper_joint_ids) == 1:
                # Single DOF gripper (like SO-ARM100 jaw_joint)
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
    diff_threshold: float = 0.06,
) -> torch.Tensor:
    """Check if an object is picked by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_closed = (suction_cup_status == 1).to(torch.float32)
        grasped = torch.logical_and(suction_cup_is_closed, pose_diff < diff_threshold)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            
            # Support both single-DOF and parallel grippers
            if len(gripper_joint_ids) == 1:
                # Single DOF gripper (like SO-ARM100)
                grasped = torch.logical_and(
                    pose_diff < diff_threshold,
                    torch.abs(
                        robot.data.joint_pos[:, gripper_joint_ids[0]]
                        - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                    )
                    > env.cfg.gripper_threshold,
                )
            elif len(gripper_joint_ids) == 2:
                # Parallel gripper with two joints
                grasped = torch.logical_and(
                    pose_diff < diff_threshold,
                    torch.abs(
                        robot.data.joint_pos[:, gripper_joint_ids[0]]
                        - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                    )
                    > env.cfg.gripper_threshold,
                )
                grasped = torch.logical_and(
                    grasped,
                    torch.abs(
                        robot.data.joint_pos[:, gripper_joint_ids[1]]
                        - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
                    )
                    > env.cfg.gripper_threshold,
                )
            else:
                raise ValueError(f"Unsupported gripper configuration with {len(gripper_joint_ids)} joints. Expected 1 or 2.")

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
            if len(gripper_joint_ids) == 1:
                # Single DOF gripper (like SO-ARM100)
                gripper_open = torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    open_val,
                    atol=1e-4,
                    rtol=1e-4,
                )
            elif len(gripper_joint_ids) == 2:
                # Parallel gripper with two joints
                finger_0_open = torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    open_val,
                    atol=1e-4,
                    rtol=1e-4,
                )
                finger_1_open = torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    open_val,
                    atol=1e-4,
                    rtol=1e-4,
                )
                gripper_open = torch.logical_and(finger_0_open, finger_1_open)
            else:
                raise ValueError(f"Unsupported gripper configuration with {len(gripper_joint_ids)} joints. Expected 1 or 2.")
            placed = torch.logical_and(gripper_open, placed)
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return placed


def object_poses_in_base_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    return_key: Literal["pos", "quat", None] = None,
) -> torch.Tensor:
    """The position and orientation of the cubes in the robot base frame."""

    plate: RigidObject = env.scene[plate_cfg.name]
    fork: RigidObject = env.scene[fork_cfg.name]
    knife: RigidObject = env.scene[knife_cfg.name]

    pos_plate_world = plate.data.root_pos_w
    pos_fork_world = fork.data.root_pos_w
    pos_knife_world = knife.data.root_pos_w

    quat_plate_world = plate.data.root_quat_w
    quat_fork_world = fork.data.root_quat_w
    quat_knife_world = knife.data.root_quat_w

    robot: Articulation = env.scene[robot_cfg.name]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    pos_cube_1_base, quat_cube_1_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_plate_world, quat_plate_world
    )
    pos_fork_base, quat_fork_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_fork_world, quat_fork_world
    )
    pos_knife_base, quat_knife_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_knife_world, quat_knife_world
    )

    pos_objects_base = torch.cat((pos_plate_base, pos_fork_base, pos_knife_base), dim=1)
    quat_objects_base = torch.cat((quat_plate_base, quat_fork_base, quat_knife_base), dim=1)

    if return_key == "pos":
        return pos_objects_base
    elif return_key == "quat":
        return quat_objects_base
    elif return_key is None:
        return torch.cat((pos_objects_base, quat_objects_base), dim=1)


def object_abs_obs_in_base_frame(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    fork_cfg: SceneEntityCfg = SceneEntityCfg("fork"),
    knife_cfg: SceneEntityCfg = SceneEntityCfg("knife"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Object Abs observations (in base frame): remove the relative observations, and add abs gripper pos and quat in robot base frame
        plate pos,
        plate quat,
        fork pos,
        fork quat,
        knife pos,
        knife quat,
        gripper pos,
        gripper quat,
    """
    plate: RigidObject = env.scene[plate_cfg.name]
    fork: RigidObject = env.scene[fork_cfg.name]
    knife: RigidObject = env.scene[knife_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    plate_pos_w = plate.data.root_pos_w
    plate_quat_w = plate.data.root_quat_w

    fork_pos_w = fork.data.root_pos_w
    fork_quat_w = fork.data.root_quat_w

    knife_pos_w = knife.data.root_pos_w
    knife_quat_w = knife.data.root_quat_w

    pos_plate_base, quat_plate_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, plate_pos_w, plate_quat_w
    )
    pos_fork_base, quat_fork_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, fork_pos_w, fork_quat_w
    )
    pos_knife_base, quat_knife_base = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, knife_pos_w, knife_quat_w
    )

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    ee_pos_base, ee_quat_base = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    return torch.cat(
        (
            pos_plate_base,
            quat_plate_base,
            pos_fork_base,
            quat_fork_base,
            pos_knife_base,
            quat_knife_base,
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

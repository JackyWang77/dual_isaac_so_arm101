from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def ee_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return the End-Effector position expressed in the Robot Base frame."""
    # 1. Get Robot Base State (The Reference)
    robot = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    # 2. Get EE World State
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Note: [..., 0, :] assumes the first target frame is the EE
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    # 3. Compute Relative Position
    # We only need the position output (index 0)
    ee_pos_b, _ = subtract_frame_transforms(
        robot_pos_w, robot_quat_w, ee_pos_w
    )
    return ee_pos_b


def ee_quat_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return the End-Effector orientation expressed in the Robot Base frame."""
    robot = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Calculate relative rotation
    _, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, 
        robot.data.root_quat_w, 
        ee_frame.data.target_pos_w[..., 0, :],
        ee_frame.data.target_quat_w[..., 0, :]
    )
    return ee_quat_b


def object_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the Object position expressed in the Robot Base frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    # Calculate relative position
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, 
        robot.data.root_quat_w, 
        obj.data.root_pos_w
    )
    return obj_pos_b


def object_quat_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the Object orientation expressed in the Robot Base frame."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    # Calculate relative orientation
    _, obj_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, 
        robot.data.root_quat_w, 
        obj.data.root_pos_w, # Pos is needed for the math, even if we only want quat
        obj.data.root_quat_w
    )
    return obj_quat_b


def object_ee_distance_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Return the vector from EE to Object, rotated into the Robot Base frame.
    This tells the robot which direction to move relative to its own body.
    """
    # Get positions in Robot Frame (re-using logic from above for clarity)
    # Ideally, call the functions above, but implemented inline for speed:
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    # Get Relative Object Pos
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w
    )

    # Get Relative EE Pos
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_frame.data.target_pos_w[..., 0, :]
    )
    
    # Vector difference in Robot Frame
    return obj_pos_b - ee_pos_b
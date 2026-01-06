# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import \
    DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import \
    GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import \
    Se3RelRetargeterCfg
from isaaclab.envs.mdp.actions.actions_cfg import \
    DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import pick_place_joint_pos_env_cfg

##
# Pre-defined configs
##
from SO_101.robots.so_arm100_roscon import SO_ARM101_ROSCON_HIGH_PD_CFG  # isort: skip


@configclass
class DualArmPickPlaceIKRelEnvCfg(
    pick_place_joint_pos_env_cfg.DualArmPickPlaceJointPosEnvCfg
):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set DualArm as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = SO_ARM101_ROSCON_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=SO_ARM101_ROSCON_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.0),  # Lower robot by 0.05m
            ),
        )

        # Set actions for the specific robot type (dual_arm)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_pitch_joint",
                "wrist_roll_joint",
            ],
            body_name="wrist_2_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[-0.005, -0.1, -0.1]
            ),  # Real robot gripper offset
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

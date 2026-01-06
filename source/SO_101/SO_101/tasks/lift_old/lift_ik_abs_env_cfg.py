# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import \
    DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.envs.mdp.actions.actions_cfg import \
    DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg, mdp

from SO_101.robots.so_arm101_roscon import SO_ARM101_ROSCON_HIGH_PD_CFG  # isort: skip


@configclass
class SoArm100LiftIKAbsEnvCfg(joint_pos_env_cfg.SoArm100LiftJointCubeEnvCfg):
    """IK Absolute environment configuration for lift task.

    This environment uses IK Absolute control where actions are target end-effector poses.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set robot with high PD gains for better IK tracking
        self.scene.robot = SO_ARM101_ROSCON_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.spawn.articulation_props.fix_root_link = True

        # IK ABSOLUTE mode: action is target EE pose in robot base frame
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
                command_type="pose",
                use_relative_mode=False,  # ABSOLUTE mode - target pose in robot base frame
                ik_method="dls",
            ),
            scale=1.0,  # No scaling for absolute mode
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, -0.1]
            ),  # Gripper offset
        )

        # Gripper action (binary control)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["jaw_joint"],
            open_command_expr={"jaw_joint": 0.3},
            close_command_expr={"jaw_joint": 0.00023},
        )

        # Teleop devices for keyboard control
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

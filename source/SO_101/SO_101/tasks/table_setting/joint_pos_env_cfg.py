# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# Dual-arm table setting with JOINT POSITION control (for teleop joint_states recording).
# Action = right_arm(6) + left_arm(6) = 12.
#

import os

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from SO_101.robots.so_arm101_roscon import SO_ARM101_ROSCON_HIGH_PD_CFG
from SO_101.tasks.table_setting.table_setting_env_cfg import (
    TableSettingEnvCfg,
    TableSettingSubtaskCfg,
)

from . import mdp
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets"))

_rigid_props = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)


@configclass
class DualSoArm101TableSettingJointPosEnvCfg(TableSettingEnvCfg):
    """Dual SO-ARM101 table setting with joint position action (for joint_states teleop)."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.right_arm = SO_ARM101_ROSCON_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Right_Arm",
            init_state=SO_ARM101_ROSCON_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, -0.25, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    **SO_ARM101_ROSCON_HIGH_PD_CFG.init_state.joint_pos,
                    "jaw_joint": 0.4,
                },
            ),
        )
        self.scene.right_arm.spawn.articulation_props.fix_root_link = True

        self.scene.left_arm = SO_ARM101_ROSCON_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Left_Arm",
            init_state=SO_ARM101_ROSCON_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, 0.25, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    **SO_ARM101_ROSCON_HIGH_PD_CFG.init_state.joint_pos,
                    "jaw_joint": 0.4,
                },
            ),
        )
        self.scene.left_arm.spawn.articulation_props.fix_root_link = True

        self.sim.dt = 1.0 / 200
        self.decimation = 4
        self.sim.render_interval = self.decimation

        # Fork: dynamic rigid object, spawns left side OUTSIDE tray
        # Rotated 90° around Z so it spawns vertically (perpendicular to tray)
        self.scene.fork = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Fork",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.08, 0.12, 0.01], rot=[0.7071, 0, 0, -0.7071]
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(ASSETS_DIR, "fork.usd"),
                scale=(1.0, 1.0, 1.0),
                rigid_props=_rigid_props,
            ),
        )

        # Knife: dynamic rigid object, spawns right side OUTSIDE tray
        # Rotated 90° around Z so it spawns vertically (perpendicular to tray)
        self.scene.knife = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Knife",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.08, -0.12, 0.01], rot=[0.7071, 0, 0, -0.7071]
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(ASSETS_DIR, "knife.usd"),
                scale=(1.0, 1.0, 1.0),
                rigid_props=_rigid_props,
            ),
        )

        arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_pitch_joint",
            "wrist_roll_joint",
        ]
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=arm_joint_names,
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["jaw_joint"],
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=arm_joint_names,
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["jaw_joint"],
            scale=1.0,
            use_default_offset=False,
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_right = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Right_Arm/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Right_Arm/wrist_2_link",
                    name="ee_right",
                    offset=OffsetCfg(pos=[0.002, 0.0, -0.1]),
                ),
            ],
        )
        self.scene.ee_left = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Left_Arm/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Left_Arm/wrist_2_link",
                    name="ee_left",
                    offset=OffsetCfg(pos=[0.002, 0.0, -0.1]),
                ),
            ],
        )


@configclass
class DualSoArm101TableSettingJointPosEnvCfg_PLAY(DualSoArm101TableSettingJointPosEnvCfg):
    """Play config: fewer envs, no obs noise, subtask terms for phase signals."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.subtask_terms = TableSettingSubtaskCfg()
        self.actions.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["jaw_joint"],
            scale=1.0,
            use_default_offset=False,
        )
        self.actions.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["jaw_joint"],
            scale=1.0,
            use_default_offset=False,
        )

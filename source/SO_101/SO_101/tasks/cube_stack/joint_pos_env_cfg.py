# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# Dual-arm cube stack with JOINT POSITION control (for teleop joint_states recording).
# Action = right_arm(6) + left_arm(6) = 12.
#

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from SO_101.robots.so_arm101_roscon import SO_ARM101_ROSCON_HIGH_PD_CFG
from SO_101.tasks.cube_stack.cube_stack_env_cfg import CubeStackEnvCfg, CubeStackSubtaskCfg

from . import mdp
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402

_rigid_props = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)


@configclass
class DualSoArm101CubeStackJointPosEnvCfg(CubeStackEnvCfg):
    """Dual SO-ARM101 cube stack with joint position action (for joint_states teleop)."""

    def __post_init__(self):
        super().__post_init__()

        # Same as lift_old: SO_ARM101_ROSCON_HIGH_PD_CFG for responsive teleop
        # (SO_ARM101_ROSCON_CFG has lower stiffness -> laggy feel)
        self.scene.right_arm = SO_ARM101_ROSCON_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Right_Arm",
            init_state=SO_ARM101_ROSCON_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, -0.25, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    **SO_ARM101_ROSCON_HIGH_PD_CFG.init_state.joint_pos,
                    "jaw_joint": 0.698,
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
                    "jaw_joint": 0.698,
                },
            ),
        )
        self.scene.left_arm.spawn.articulation_props.fix_root_link = True

        # Same sim rate as lift_old for smooth teleop (200Hz physics, decimation=4 -> 50Hz control)
        self.sim.dt = 1.0 / 200
        self.decimation = 4
        self.sim.render_interval = self.decimation

        # (Target visual removed: FrameTransformer requires rigid bodies; Table is not.
        #  Target = TARGET_XY (0.117, -0.011), z=table.)

        # cube_1 & cube_2 only; target zone fixed at TARGET_XY (0.117, -0.011), no physical base
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube1",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.20, 0.0, 0.015], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.3, 0.3, 0.3),
                rigid_props=_rigid_props,
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube2",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.20, 0.0, 0.015], rot=[1, 0, 0, 0]
            ),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.3, 0.3, 0.3),
                rigid_props=_rigid_props,
            ),
        )

        # Joint position action (same as lift_old): 5 arm + 1 gripper per arm
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
        # EE frame offset: wrist_2_link + [0.002, 0, -0.1]
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
class DualSoArm101CubeStackJointPosEnvCfg_PLAY(DualSoArm101CubeStackJointPosEnvCfg):
    """Play config for joint position policy: fewer envs, no obs noise, obs=64, action=12.
    Adds subtask_terms (pick_cube, stack_cube) so play can use same phase signals as Mimic training."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # 与 Mimic 一致: 添加 subtask_terms，供 get_subtask_term_signals 使用
        self.observations.subtask_terms = CubeStackSubtaskCfg()

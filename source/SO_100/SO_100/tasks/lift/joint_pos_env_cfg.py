# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.assets import RigidObjectCfg

# from isaaclab.managers NotImplementedError
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from SO_100.robots import SO_ARM100_CFG, SO_ARM100_ROSCON_CFG  # noqa: F401
from SO_100.tasks.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

# ----------------------------------------------------------------
# --------------- LycheeAI live asset ----------------------------
# ----------------------------------------------------------------


@configclass
class DualSoArm100LiftCubeEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        # place right arm slightly to the -y side
        self.scene.right_arm = SO_ARM100_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Right_Arm",
            init_state=SO_ARM100_CFG.init_state.replace(
                # (x, y, z) in world for the robot base
                pos=(0.0, -0.25, 0.0),
                # keep same base rotation you had before
                rot=(0.7071068, 0.0, 0.0, 0.7071068),
            ),
        )

        # place left arm slightly to the +y side
        self.scene.left_arm = SO_ARM100_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Left_Arm",
            init_state=SO_ARM100_CFG.init_state.replace(
                pos=(0.0, 0.25, 0.0),
                rot=(0.7071068, 0.0, 0.0, 0.7071068),
            ),
        )
        # self.scene.robot = self.scene.right_arm

        # override actions
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["Gripper"],
            open_command_expr={"Gripper": 0.5},
            close_command_expr={"Gripper": 0.0},
        )
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["Shoulder_Rotation", "Shoulder_Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["Gripper"],
            open_command_expr={"Gripper": 0.5},
            close_command_expr={"Gripper": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose_right.body_name = ["Fixed_Gripper"]
        self.commands.object_pose_left.body_name = ["Fixed_Gripper"]
        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.3, 0.3, 0.3),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        # right end-effector frame tracker
        self.scene.ee_right = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Right_Arm/Base",  # base frame of right arm
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Right_Arm/Fixed_Gripper",
                    name="ee_right",
                    offset=OffsetCfg(pos=[0.01, 0.0, 0.1]),
                ),
            ],
        )

        # left end-effector frame tracker
        self.scene.ee_left = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Left_Arm/Base",   # base frame of left arm
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Left_Arm/Fixed_Gripper",
                    name="ee_left",
                    offset=OffsetCfg(pos=[0.01, 0.0, 0.1]),
                ),
            ],
        )




@configclass
class DualSoArm100LiftCubeEnvCfg_PLAY(DualSoArm100LiftCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

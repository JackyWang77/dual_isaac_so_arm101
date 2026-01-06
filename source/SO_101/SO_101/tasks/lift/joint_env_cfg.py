# ----------------------------------------------------------------
# --------------- Dual Arm IK Relative Lift Environment ----------
# ----------------------------------------------------------------

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import \
    DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import \
    DifferentialInverseKinematicsActionCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg, OffsetCfg)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from SO_101.robots.so_arm100_roscon import SO_ARM101_ROSCON_HIGH_PD_CFG
from SO_101.tasks.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip



@configclass
class DualSoArm100LiftCubeEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # initialize parent configuration
        super().__post_init__()

        # ----------------------------------------------------------
        # Replace both arms with SO_ARM101_ROSCON_HIGH_PD_CFG (stiffer PD for IK)
        # ----------------------------------------------------------
        self.scene.right_arm = SO_ARM101_ROSCON_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Right_Arm",
            init_state=SO_ARM101_ROSCON_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, -0.25, 0.0),
                rot=(0.7071068, 0.0, 0.0, 0.7071068),
                joint_pos={
                    **SO_ARM101_ROSCON_HIGH_PD_CFG.init_state.joint_pos,
                    "jaw_joint": 0.698,  # Start fully open
                },
            ),
        )
        self.scene.left_arm = SO_ARM101_ROSCON_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Left_Arm",
            init_state=SO_ARM101_ROSCON_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, 0.25, 0.0),
                rot=(0.7071068, 0.0, 0.0, 0.7071068),
                joint_pos={
                    **SO_ARM101_ROSCON_HIGH_PD_CFG.init_state.joint_pos,
                    "jaw_joint": 0.698,  # Start fully open
                },
            ),
        )

        # ----------------------------------------------------------
        # Define IK Relative control actions for both arms
        # Action is delta EE pose (translation + rotation)
        # ----------------------------------------------------------
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="right_arm",
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
                use_relative_mode=True,  # RELATIVE mode
                ik_method="dls",
            ),
            scale=0.5,  # Scale for relative actions (0.5 means action 1.0 -> 0.5m/s or rad/s)
            # IMPORTANT: Must match ee_right offset for reward/IK alignment!
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.004, -0.102, 0.0]
            ),
        )

        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="left_arm",
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
                use_relative_mode=True,  # RELATIVE mode
                ik_method="dls",
            ),
            scale=0.5,  # Scale for relative actions
            # IMPORTANT: Must match ee_left offset for reward/IK alignment!
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.004, -0.102, 0.0]
            ),
        )

        # Gripper actions (binary control)
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["jaw_joint"],
            open_command_expr={"jaw_joint": 0.698},
            close_command_expr={"jaw_joint": 0.0},
        )
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["jaw_joint"],
            open_command_expr={"jaw_joint": 0.698},
            close_command_expr={"jaw_joint": 0.0},
        )

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.2, 0.0, 0.015], rot=[1, 0, 0, 0]
            ),
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

        # End-effector frame trackers
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
                    offset=OffsetCfg(pos=[0.004, -0.102, 0]),
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
                    offset=OffsetCfg(pos=[0.004, -0.102, 0]),
                ),
            ],
        )


@configclass
class DualSoArm100LiftCubeEnvCfg_PLAY(DualSoArm100LiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for testing
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5

        # Disable observation corruption for stable visualization
        self.observations.policy.enable_corruption = False

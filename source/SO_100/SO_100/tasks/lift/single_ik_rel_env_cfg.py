# ----------------------------------------------------------------
# --------------- Dual Arm IK Lift Environment -------------------
# ----------------------------------------------------------------

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# from isaaclab.managers NotImplementedError
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from SO_100.robots import SO_ARM100_ROSCON_HIGH_PD_CFG  # high-stiffness PD controller
from SO_100.tasks.lift.single_lift_env_cfg import LiftEnvCfg  # base environment (joint-position controlled)


@configclass
class SoArm100LiftCube_IK_EnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # initialize parent configuration
        super().__post_init__()

        # ----------------------------------------------------------
        # Replace both arms with high-stiffness PD controllers
        # to improve tracking accuracy during IK control
        # ----------------------------------------------------------
        self.scene.robot = SO_ARM100_ROSCON_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=SO_ARM100_ROSCON_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.0),
                # keep same base rotation you had before
                rot=(0.7071068, 0.0, 0.0, 0.7071068),
                # Start with gripper open for easier approach
                joint_pos={
                    **SO_ARM100_ROSCON_HIGH_PD_CFG.init_state.joint_pos,
                    "jaw_joint": 0.698,  # Maximum joint limit: start fully open (matches open_command_expr)
                },
            ),
        )
        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/object",
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

        #end-effector frame tracker
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",  # base frame of robot
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_2_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[-0.005, -0.1, 0.0]),
                ),
            ],
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = ["wrist_2_link"]

        # Define IK-based control actions for arm
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
            scale=0.1,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            body_offset=OffsetCfg(pos=[-0.005, -0.1, 0.0])

        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["jaw_joint"],
            open_command_expr={"jaw_joint": 0.698},  # Maximum joint limit: fully open gripper
            close_command_expr={"jaw_joint": 0.0},
        )
        
        
@configclass
class SoArm100LiftCubeIK_EnvCfg_PLAY(SoArm100LiftCube_IK_EnvCfg):
    def __post_init__(self):
        # initialize parent configuration
        super().__post_init__()

        # ----------------------------------------------------------
        # Create a smaller scene for testing or demonstration
        # ----------------------------------------------------------
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5

        # Disable observation corruption for stable visualization
        self.observations.policy.enable_corruption = False

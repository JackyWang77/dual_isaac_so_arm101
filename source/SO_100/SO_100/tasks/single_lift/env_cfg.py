# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import copy
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers.config import FRAME_MARKER_CFG

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from SO_100.robots.so_arm100_roscon import SO_ARM100_ROSCON_HIGH_PD_CFG
from . import mdp


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object."""

    # robot: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    cube_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # arm joint control (using IK Absolute)
    arm_action: DifferentialInverseKinematicsActionCfg = MISSING
    # gripper binary control (open/close)
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # Proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # End-effector state (robot-relative frame)
        ee_pos = ObsTerm(
            func=mdp.ee_pos_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            }
        )
        ee_quat = ObsTerm(
            func=mdp.ee_quat_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            }
        )
        
        
        # Object position relative to robot base
        object_pos_rel = ObsTerm(
            func=mdp.object_pos_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            }
        )
        
        obeject_ori_rel = ObsTerm(
            func=mdp.object_quat_in_robot_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            }
        )
        # Previous action
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.14, 0.17), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Reaching reward
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.05,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=1.0,
    )

    # Lifting reward (minimal_height must be ABOVE object's initial height ~0.05m)
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={
            "minimal_height": 0.08,
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=20.0,
    )

    grasping_object = RewTerm(
        func=mdp.graps_intent,
        params={
            "minimal_height": 0.04,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=10.0,
    )

    # Penalize closing the gripper when far away (prevents "premature grasping")
    early_closing_penalty = RewTerm(
        func=mdp.gripper_penalty_when_far,
        params={
            "threshold_dist": 0.06,  # If further than 6cm...
            "threshold_gripper": 0.35,  # ...and trying to close...
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
        weight=-0.5,  # Negative reward (Penalty)
    )

    # Action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # Joint velocity penalty
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": -0.05,
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    action_rate_curriculum = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "action_rate",
            "weight": -1e-1,
            "num_steps": 10000,
        },
    )


@configclass
class SoArm100LiftCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the single arm lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        
        # general settings
        self.decimation = 1
        self.episode_length_s = 5.0
        self.viewer.eye = (2.0, 2.0, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.use_gpu = True
        
        # Increase GPU buffers for large env count
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_heap_capacity = 67108864  # 64MB
        self.sim.physx.gpu_temp_buffer_capacity = 33554432  # 32MB

        # ----------------------------------------------------------
        # Robot Definition
        # ----------------------------------------------------------
        robot_cfg = SO_ARM100_ROSCON_HIGH_PD_CFG.copy()
        # Fix the root link since root_joint was removed from USD
        robot_cfg.spawn.articulation_props.fix_root_link = True
        # Revert physics optimization as it caused instability with IK
        # robot_cfg.spawn.articulation_props.solver_position_iteration_count = 4
        # robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 0

        self.scene.robot = robot_cfg.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=SO_ARM100_ROSCON_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    **SO_ARM100_ROSCON_HIGH_PD_CFG.init_state.joint_pos,
                    # Adjust initial joint angles to make EE point forward (towards object)
                    "shoulder_pan_joint": 0.0,  # Point forward
                    "shoulder_lift_joint": -0.5,  # More horizontal (was -1.745, too downward)
                    "elbow_joint": 1.0,  # Less bent (was 1.560)
                    "wrist_pitch_joint": 0.0,  # Horizontal (was 0.1)
                    "wrist_roll_joint": 0.0,
                    "jaw_joint": 0.5,  # Start fully open
                },
            ),
        )

        # ----------------------------------------------------------
        # Actions Definition (IK Absolute)
        # ----------------------------------------------------------
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
                use_relative_mode=False,  # ABSOLUTE mode
                ik_method="dls",
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.006, 0.0, -0.102]
            ),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["jaw_joint"],
            open_command_expr={"jaw_joint": 0.5},  # Open position
            close_command_expr={"jaw_joint": 0.0},   # Closed position
        )
        # ----------------------------------------------------------
        # Object Definition (Cube)
        # ----------------------------------------------------------
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.15, 0.0, 0.05], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.3, 0.3, 0.3),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,  # Reverted to 16 for stability
                    solver_velocity_iteration_count=1,   # Reverted to 1
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # ----------------------------------------------------------
        # EE Frame Tracker
        # ----------------------------------------------------------
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=True,  # Enable visualization to see the EE point
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_2_link",
                    name="ee_frame",
                    offset=OffsetCfg(pos=[0.006, 0.0, -0.102]),
                ),
            ],
        )

        # ----------------------------------------------------------
        # Cube Frame Tracker (for visualization)
        # Use deepcopy() to create independent visualizer config to avoid PointInstancer prototype mismatch
        # ----------------------------------------------------------
        cube_marker_cfg = copy.deepcopy(FRAME_MARKER_CFG)
        cube_marker_cfg.markers["frame"].scale = (0.06, 0.06, 0.06)
        cube_marker_cfg.prim_path = "/Visuals/FrameTransformer/Cube"  # Unique prim path

        self.scene.cube_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,  # Disable visualization to avoid PointInstancer conflict
            visualizer_cfg=cube_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    name="cube_frame",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )


@configclass
class SoArm100LiftCubeEnvCfg_PLAY(SoArm100LiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        self.observations.policy.enable_corruption = False


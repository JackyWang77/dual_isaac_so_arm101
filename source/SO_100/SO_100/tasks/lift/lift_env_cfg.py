# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils

from . import mdp
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
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
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# from isaaclab.utils.offset import OffsetCfg
# from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# from isaaclab.utils.visualizer import FRAME_MARKER_CFG
# from isaaclab.utils.assets import RigidBodyPropertiesCfg


##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    left_arm: ArticulationCfg = MISSING
    right_arm: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_right: FrameTransformerCfg = MISSING
    ee_left: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

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

    # Cameras disabled for faster training (uncomment if you need camera observations)
    # camera_top = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/camera_top",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    # camera_left_wrist = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/camera_left_wrist",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

    # camera_right_wrist = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/camera_right_wrist",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose_right = mdp.UniformPoseCommandCfg(
        asset_name="right_arm",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.3, -0.1),
            pos_z=(0.2, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    object_pose_left = mdp.UniformPoseCommandCfg(
        asset_name="left_arm",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(0.1, 0.3),
            pos_z=(0.2, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the dual-arm MDP.

    We explicitly expose separate actions for the right and left arm.
    Each arm has:
    - a joint position (or IK) action head
    - a gripper (binary open/close) action head

    We intentionally REMOVE the legacy single-arm fields `arm_action` and
    `gripper_action`, because we want the agent to conceptually be
    a dual-arm controller, not "main arm + maybe second arm".
    """

    # right arm joint control (must be filled in env __post_init__)
    right_arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING

    # right gripper open/close
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

    # left arm joint control
    left_arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING

    # left gripper open/close
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # --------------------------
        # Left arm proprioception
        # --------------------------
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("left_arm")},
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("left_arm")},
        )

        # Object position in LEFT arm coordinates
        left_obj_rel = ObsTerm(
            func=mdp.object_pos_in_arm_frame,    # <<< changed name
            params={"arm_cfg": SceneEntityCfg("left_arm")},
        )

        # --------------------------
        # Right arm proprioception
        # --------------------------
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("right_arm")},
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("right_arm")},
        )

        # Object position in RIGHT arm coordinates
        right_obj_rel = ObsTerm(
            func=mdp.object_pos_in_arm_frame,    # <<< changed name
            params={"arm_cfg": SceneEntityCfg("right_arm")},
        )

        # --------------------------
        # Task commands / targets
        # --------------------------
        # Goal pose sampled for the right arm
        target_right = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose_right"},
        )

        # Goal pose sampled for the left arm
        target_left = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose_left"},
        )

        # --------------------------
        # Previous action (for smoothness / stabilization)
        # --------------------------
        last_action_all = ObsTerm(
            func=mdp.last_action,
        )

        def __post_init__(self):
            # add noise/corruption, and concat all ObsTerm into single vector
            self.enable_corruption = True
            self.concatenate_terms = True

    # top level group
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Increased Y range to ensure equal exposure to both arms
            # Y range: (-0.3, 0.3) covers both arms symmetrically
            # Right arm at y=-0.25, Left arm at y=0.25
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.3, 0.3), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class DualRewardsCfg:
    """Reward terms for the MDP (dual-arm IK version)."""

    # Smart reaching: only the closer arm reaches, the farther one stays still
    # Significantly increased weight to strongly encourage arm selection based on proximity
    closer_arm_reaches = RewTerm(
        func=mdp.closer_arm_reaches_object,
        params={
            "std": 0.05,
            "ee_right_cfg": SceneEntityCfg("ee_right"),
            "ee_left_cfg": SceneEntityCfg("ee_left"),
        },
        weight=10.0  # Much higher weight to enforce smart arm selection
    )
    
    farther_arm_stays_still = RewTerm(
        func=mdp.farther_arm_stays_still,
        params={
            "ee_right_cfg": SceneEntityCfg("ee_right"),
            "ee_left_cfg": SceneEntityCfg("ee_left"),
            "right_arm_cfg": SceneEntityCfg("right_arm"),
            "left_arm_cfg": SceneEntityCfg("left_arm"),
        },
        weight=2.0  # Increased to strongly penalize unnecessary motion
    )
    
    # Original separate rewards (commented out, can be enabled if needed)
    # reaching_object_right = RewTerm(
    #     func=mdp.object_ee_distance,
    #     params={
    #         "std": 0.05,
    #         "ee_frame_cfg": SceneEntityCfg("ee_right"),
    #     },
    #     weight=1.0
    # )
    # 
    # reaching_object_left = RewTerm(
    #     func=mdp.object_ee_distance,
    #     params={
    #         "std": 0.05,
    #         "ee_frame_cfg": SceneEntityCfg("ee_left"),
    #     },
    #     weight=1.0
    # )

    lifting_object_right = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=15.0
    )
    lifting_object_left = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=15.0
    )

    # object goal tracking reward for right arm
    object_goal_tracking_right = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "object_pose_right",
            "robot_cfg": SceneEntityCfg("right_arm"),
        },
        weight=16.0,
    )
    object_goal_tracking_left = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3,
                "minimal_height": 0.04,
                "command_name": "object_pose_left",
                "robot_cfg": SceneEntityCfg("left_arm")
        },
        weight=16.0,
    )
    # object goal tracking reward for right arm (fine-grained)
    object_goal_tracking_fine_grained_right = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "object_pose_right",
            "robot_cfg": SceneEntityCfg("right_arm"),
        },
        weight=5.0,
    )

    # object goal tracking reward for left arm (fine-grained)
    object_goal_tracking_fine_grained_left = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "object_pose_left",
            "robot_cfg": SceneEntityCfg("left_arm"),
        },
        weight=5.0,
    )

    # action penalty for right arm
    action_rate_right = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    action_rate_left = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # joint velocity penalty for right arm
    joint_vel_right = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("right_arm")},
    )
    joint_vel_left = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("left_arm")},
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the dual-arm MDP.

    We gradually increase penalties on unnecessary motion and jitter.
    This encourages both arms to move smoothly and stay stable,
    especially the support arm.
    """

    # Gradually ramp up the global "action smoothness" penalty.
    # This corresponds to the reward term "action_rate_all"
    # in DualRewardsCfg, which punishes large action deltas.
    action_rate_right = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            # must match the key in DualRewardsCfg
            "term_name": "action_rate_right",
            # final target weight after curriculum:
            # more negative => stronger penalty for jerky actions
            "weight": -1e-1,
            # number of steps over which we anneal from original weight
            "num_steps": 10000,
        },
    )
    action_rate_left = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "action_rate_left",
            "weight": -1e-1,
            "num_steps": 10000,
        },
    )
    # Gradually increase velocity penalty for the right arm.
    # This aligns with the reward term "joint_vel_right"
    # in DualRewardsCfg, which penalizes high joint speeds.
    joint_vel_right = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "joint_vel_right",
            "weight": -1e-1,
            "num_steps": 10000,
        },
    )

    # Same idea, but for the left arm.
    joint_vel_left = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "joint_vel_left",
            "weight": -1e-1,
            "num_steps": 10000,
        },
    )


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1024, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: DualRewardsCfg = DualRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.use_gpu = True
        self.sim.physx.gpu_max_rigid_contact_count = 65536
        self.sim.physx.gpu_max_rigid_patch_count = 32768

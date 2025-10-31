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
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class DualRewardsCfg:
    """
    Reward terms for a cooperative dual-arm manipulation task.

    High-level behavior we want:
    - Exactly one arm ("primary arm") should actively manipulate the object.
      The primary arm is chosen dynamically as the arm currently closer to the object.
    - The other arm ("support arm") should stay mostly stable / compliant,
      not flailing around or fighting for control.
    - Lifting the object is rewarded as a shared team success.
    - Moving / placing the object toward commanded poses is encouraged.
    - Rapid / jerky motion is discouraged for both arms, but especially for
      the support (non-primary) arm.

    NOTE:
    Some of these terms (e.g. `primary_reach`, `support_stability_penalty`)
    require custom reward functions that we will implement in rewards.py later.
    For now we only define the config interface here.
    """

    # ------------------------------------------------------------------
    # 1. Primary arm reaching reward
    # ------------------------------------------------------------------
    primary_reach = RewTerm(
        func=mdp.closest_arm_reach_reward,
        params={
            # std: shaping factor for distance -> reward curve.
            # smaller std = only very small distances give high reward.
            "std": 0.05,
            # We assume closest_arm_reach_reward will:
            #   - compute distance from EACH gripper (right/left) to the object,
            #   - pick / weight the closer one,
            #   - return a single scalar reward per env.
        },
        weight=1.0,
    )


    # ------------------------------------------------------------------
    # 2. Object lifting reward (team success)
    # ------------------------------------------------------------------
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={
            # minimal_height: how high above the table we consider "lifted"
            "minimal_height": 0.04,
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=15.0,
    )


    # ------------------------------------------------------------------
    # 3. Goal tracking rewards (per-arm goal following)
    # ------------------------------------------------------------------
    # These encourage moving the object toward commanded poses.
    # We keep them separate for right and left arms.
    # By tuning the weights, we can bias one arm to be the "main placer"
    # (e.g. right arm does the fine placement, left arm mostly stabilizes).
    object_goal_tracking_right = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "object_pose_right",           # from CommandsCfg
            "robot_cfg": SceneEntityCfg("right_arm"),      # frame for transform
            "object_cfg": SceneEntityCfg("object"),        # object to place
        },
        weight=12.0,
    )

    object_goal_tracking_right_fine = RewTerm(
        func=mdp.object_goal_distance,
        params={
            # smaller std means we care about very accurate placement
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "object_pose_right",
            "robot_cfg": SceneEntityCfg("right_arm"),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=4.0,
    )

    object_goal_tracking_left = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.04,
            "command_name": "object_pose_left",            # from CommandsCfg
            "robot_cfg": SceneEntityCfg("left_arm"),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=12.0,
    )

    object_goal_tracking_left_fine = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "object_pose_left",
            "robot_cfg": SceneEntityCfg("left_arm"),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=4.0,
    )
    # ------------------------------------------------------------------
    # 4. Stability penalty for the support arm
    # ------------------------------------------------------------------
    support_stability_penalty = RewTerm(
        func=mdp.support_arm_stillness_penalty,
        params={
            # "weight_farther" is how hard we punish the arm that is NOT primary
            # for moving / thrashing.
            # "weight_closer" is how much we punish the primary arm for moving.
            #
            # In practice, support_arm_stillness_penalty() should:
            #   - figure out which arm is farther from the object (support arm),
            #   - penalize that arm's joint velocity more heavily,
            #   - penalize the primary (closer) arm less.
            "weight_farther": 1e-4,
            "weight_closer": 1e-5,
        },
        weight=1.0,
    )
    # ------------------------------------------------------------------
    # 5. Global smoothness / anti-twitch term
    # ------------------------------------------------------------------
    # This penalizes very jerky changes in the overall action vector
    # (both arms together). It keeps the behavior smooth and coordinated.
    action_rate_all = RewTerm(
        func=mdp.action_rate_l2,
        params={
            # no extra params needed in current API; this term naturally
            # reads the last and current actions from the env manager.
        },
        weight=-1e-4,
    )

    # ------------------------------------------------------------------
    # 6. (optional) Per-arm velocity penalty
    # ------------------------------------------------------------------
    # Even though support_stability_penalty will already penalize the
    # support arm more, we can still include per-arm joint velocity
    # penalties. These encourage both arms to move with intention
    # instead of flailing.
    joint_vel_right = RewTerm(
        func=mdp.joint_vel_l2,
        params={
            "asset_cfg": SceneEntityCfg("right_arm"),
        },
        weight=-1e-4,
    )

    joint_vel_left = RewTerm(
        func=mdp.joint_vel_l2,
        params={
            "asset_cfg": SceneEntityCfg("left_arm"),
        },
        weight=-1e-4,
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
    action_rate_all = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            # must match the key in DualRewardsCfg
            "term_name": "action_rate_all",
            # final target weight after curriculum:
            # more negative => stronger penalty for jerky actions
            "weight": -1e-1,
            # number of steps over which we anneal from original weight
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
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
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

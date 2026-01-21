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
# import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp
from isaaclab.assets import (ArticulationCfg, AssetBaseCfg,
                             DeformableObjectCfg, RigidObjectCfg)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import \
    FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import (GroundPlaneCfg,
                                                             UsdFileCfg)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

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
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
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
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: (
        mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # CRITICAL: Use absolute joint positions, not relative!
        # Training data was collected with absolute positions, so RL must match.
        # joint_pos_rel would return 0 at start, but training data has non-zero mean.
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_orientation = ObsTerm(func=mdp.object_orientation)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        ee_orientation = ObsTerm(func=mdp.ee_orientation)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # Lift object - check if object is lifted to target height
        # Note: This uses the observation function from mdp.observations, not the reward function
        # CRITICAL: initial_height must match cube's actual initial height (0.015m from joint_pos_env_cfg.py)
        lift_object = ObsTerm(
            func=mdp.object_is_lifted,
            params={
                "minimal_height": 0.04,  # Target height: 0.04m (2.5cm lift from initial 0.015m)
                "initial_height": 0.015,  # Initial height of cube (matches joint_pos_env_cfg.py)
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


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
class RewardsCfg:
    """Reward terms for the MDP."""

    # ============================================================
    # 1. Reaching Reward (密集，引导接近物体)
    # ============================================================
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.05},
        weight=1.0,
    )

    # ============================================================
    # 2. Grasp Behavior Reward (密集，教gripper正确开合) - 关键新增！
    # ============================================================
    grasp_behavior = RewTerm(
        func=mdp.grasp_reward,
        params={"threshold_distance": 0.02},  # EE距离物体0.02m内算"接近"
        weight=2.0,  # 权重要高，直接教gripper行为
    )

    # ============================================================
    # 3. Object Grasped Reward (中等密集，检测是否真正抓住)
    # ============================================================
    object_grasped = RewTerm(
        func=mdp.object_grasped_reward,
        params={
            "lift_threshold": 0.02,  # 物体需要抬起0.02m
            "grasp_distance_threshold": 0.05,  # 抓住时EE和物体距离<0.05m
            "initial_height": 0.015,
        },
        weight=3.0,  # 权重要高
    )

    # ============================================================
    # 4. Lifting Reward (密集，鼓励抬起物体)
    # ============================================================
    # CRITICAL: initial_height must match cube's actual initial height (0.015m from joint_pos_env_cfg.py)
    # minimal_height=0.04 means target height is 0.04m (2.5cm lift from initial 0.015m)
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04, "initial_height": 0.015},
        weight=15.0,  # 保持高权重
    )

    # ============================================================
    # 5. Success Bonus (稀疏大奖励)
    # ============================================================
    task_complete = RewTerm(
        func=mdp.task_success_bonus,
        params={
            "bonus": 10.0,  # 成功时给予10.0奖励
            "target_height": 0.1,  # 目标高度0.1m
            "hold_time_steps": 10,  # 需要保持10步
        },
        weight=1.0,  # 内部已经是10.0，这里权重设为1.0
    )

    # ============================================================
    # 6. Gripper Smoothness Penalty (防止频繁开合)
    # ============================================================
    gripper_smooth = RewTerm(
        func=mdp.gripper_action_penalty,
        weight=1.0,
    )

    # ============================================================
    # 7. Regularization Terms (保持原有)
    # ============================================================
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    # Success condition: object lifted to target height
    # This is important for mimic recording to know when a demo is complete
    # Named "success" so that record_demos.py can automatically detect task completion
    # Check if object is lifted to at least 0.08m above initial position (0.015m)
    success = DoneTerm(
        func=mdp.object_lifted_termination,
        params={
            "minimal_height": 0.1,  # 8cm above initial position (higher threshold)
            "initial_height": 0.015,  # Initial height of cube
            "object_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-3, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-3, "num_steps": 10000},
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 1.0 / 200
        # Render interval should match decimation to avoid rendering intermediate physics steps
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.use_gpu = True
        self.sim.device = "cuda:0"

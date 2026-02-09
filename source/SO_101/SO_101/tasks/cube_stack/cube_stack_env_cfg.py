# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# Dual-arm cube stack: two SO-ARM101 arms, cube_base (bottom), cube_top (pick and stack).
# Same pipeline as single-arm lift: collect data -> flow matching -> RL.
#

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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

from . import mdp


@configclass
class CubeStackSceneCfg(InteractiveSceneCfg):
    """Scene: two SO-ARM101 arms + cube_base (bottom) + cube_top (object to stack)."""

    left_arm: ArticulationCfg = MISSING
    right_arm: ArticulationCfg = MISSING
    ee_right: FrameTransformerCfg = MISSING
    ee_left: FrameTransformerCfg = MISSING
    cube_base: RigidObjectCfg = MISSING
    object: RigidObjectCfg = MISSING  # cube_top: pick and stack on cube_base

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Dual-arm actions (same as lift): joint position or IK per arm + gripper."""

    right_arm_action: (
        mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    left_arm_action: (
        mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("left_arm")},
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("left_arm")},
        )
        left_obj_rel = ObsTerm(
            func=mdp.object_pos_in_arm_frame,
            params={
                "arm_cfg": SceneEntityCfg("left_arm"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("right_arm")},
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("right_arm")},
        )
        right_obj_rel = ObsTerm(
            func=mdp.object_pos_in_arm_frame,
            params={
                "arm_cfg": SceneEntityCfg("right_arm"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        cube_base_pos = ObsTerm(
            func=mdp.object_position_w,
            params={"object_cfg": SceneEntityCfg("cube_base")},
        )
        object_pos = ObsTerm(
            func=mdp.object_position_w,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        object_ori = ObsTerm(
            func=mdp.object_orientation_w,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        last_action_all = ObsTerm(
            func=mdp.last_action,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Randomize cube_base and cube_top (object) positions on reset."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cube_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.15, 0.22), "y": (-0.12, 0.12), "z": (0.015, 0.015)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_base", body_names="Object"),
        },
    )

    reset_cube_top = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.18, 0.25), "y": (-0.15, 0.15), "z": (0.015, 0.015)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class CubeStackRewardsCfg:
    """Reach cube_top -> grasp -> lift -> stack on cube_base."""

    closer_arm_reaches = RewTerm(
        func=mdp.closer_arm_reaches_object,
        params={
            "std": 0.05,
            "ee_right_cfg": SceneEntityCfg("ee_right"),
            "ee_left_cfg": SceneEntityCfg("ee_left"),
        },
        weight=10.0,
    )
    farther_arm_stays_still = RewTerm(
        func=mdp.farther_arm_stays_still,
        params={
            "ee_right_cfg": SceneEntityCfg("ee_right"),
            "ee_left_cfg": SceneEntityCfg("ee_left"),
            "right_arm_cfg": SceneEntityCfg("right_arm"),
            "left_arm_cfg": SceneEntityCfg("left_arm"),
        },
        weight=2.0,
    )
    grasp_intent = RewTerm(
        func=mdp.grasp_intent,
        params={
            "proximity_threshold": 0.05,
            "ee_right_cfg": SceneEntityCfg("ee_right"),
            "ee_left_cfg": SceneEntityCfg("ee_left"),
            "right_arm_cfg": SceneEntityCfg("right_arm"),
            "left_arm_cfg": SceneEntityCfg("left_arm"),
        },
        weight=8.0,
    )
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=15.0,
    )
    stack_alignment = RewTerm(
        func=mdp.cube_stack_alignment,
        params={
            "xy_std": 0.03,
            "z_tolerance": 0.015,
            "cube_top_cfg": SceneEntityCfg("object"),
            "cube_base_cfg": SceneEntityCfg("cube_base"),
        },
        weight=20.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
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
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    stack_success = DoneTerm(
        func=mdp.cube_stacked,
        params={
            "xy_threshold": 0.025,
            "z_tolerance": 0.01,
            "cube_top_cfg": SceneEntityCfg("object"),
            "cube_base_cfg": SceneEntityCfg("cube_base"),
        },
    )


@configclass
class CubeStackEnvCfg(ManagerBasedRLEnvCfg):
    """Dual-arm cube stack env config."""

    scene: CubeStackSceneCfg = CubeStackSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: CubeStackRewardsCfg = CubeStackRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0  # stack can take longer than lift
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**23
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_heap_capacity = 33554432 * 2
        self.sim.physx.gpu_temp_buffer_capacity = 16777216 * 2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.use_gpu = True

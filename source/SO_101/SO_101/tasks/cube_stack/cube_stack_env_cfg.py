# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# Dual-arm cube stack: two SO-ARM101 arms, two random cubes, fixed target at center.
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


# Fixed target position (stack goal; either order OK)
TARGET_XY = (0.117, -0.011)
TARGET_Z_TABLE = 0.015

# Gripper joint (jaw_joint): 与 joint_pos_env_cfg init_state / joint_env_cfg BinaryJoint 一致
JAW_OPEN = 0.4  # 与 lift_old 一致
JAW_CLOSED = 0.0

# Subtask params: analyze_cube_stack_subtask_params.py 统计
# pick + stack: 都用每个 demo 最后 N 步 (已叠放=目标位置)
# pick target_xy = 叠放位置, target_z = 底座 z (桌面)
PICK_TARGET_XY = (0.1623, -0.023)
PICK_TARGET_Z = 0.006
PICK_EPS_XY = 0.0603
PICK_EPS_Z = 0.002
STACK_EXPECTED_HEIGHT = 0.018
STACK_EPS_Z = 0.001
STACK_EPS_XY = 0.0073


@configclass
class CubeStackSubtaskCfg(ObsGroup):
    """Subtask signals for cube stack (pick_cube, stack_cube)."""

    # 参数来自 analyze_cube_stack_subtask_params.py 对 dual_cube_stack_annotated_dataset.hdf5 的统计
    pick_cube = ObsTerm(
        func=mdp.either_cube_placed_at_target,
        params={
            "target_xy": PICK_TARGET_XY,
            "target_z": PICK_TARGET_Z,
            "target_eps_xy": PICK_EPS_XY,
            "target_eps_z": PICK_EPS_Z,
            "cube_1_cfg": SceneEntityCfg("cube_1"),
            "cube_2_cfg": SceneEntityCfg("cube_2"),
        },
    )
    stack_cube = ObsTerm(
        func=mdp.two_cubes_stacked_aligned,
        params={
            "expected_height": STACK_EXPECTED_HEIGHT,
            "eps_z": STACK_EPS_Z,
            "eps_xy": STACK_EPS_XY,
            "cube_1_cfg": SceneEntityCfg("cube_1"),
            "cube_2_cfg": SceneEntityCfg("cube_2"),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class CubeStackSceneCfg(InteractiveSceneCfg):
    """Scene: two SO-ARM101 arms + cube_1 + cube_2 (both random). Target zone fixed at center."""

    left_arm: ArticulationCfg = MISSING
    right_arm: ArticulationCfg = MISSING
    ee_right: FrameTransformerCfg = MISSING
    ee_left: FrameTransformerCfg = MISSING
    cube_1: RigidObjectCfg = MISSING
    cube_2: RigidObjectCfg = MISSING
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
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("right_arm")},
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("right_arm")},
        )
        left_ee_position = ObsTerm(
            func=mdp.ee_position_w,
            params={"ee_frame_cfg": SceneEntityCfg("ee_left")},
        )
        left_ee_orientation = ObsTerm(
            func=mdp.ee_orientation_w,
            params={"ee_frame_cfg": SceneEntityCfg("ee_left")},
        )
        right_ee_position = ObsTerm(
            func=mdp.ee_position_w,
            params={"ee_frame_cfg": SceneEntityCfg("ee_right")},
        )
        right_ee_orientation = ObsTerm(
            func=mdp.ee_orientation_w,
            params={"ee_frame_cfg": SceneEntityCfg("ee_right")},
        )
        cube_1_pos = ObsTerm(
            func=mdp.object_position_w,
            params={"object_cfg": SceneEntityCfg("cube_1")},
        )
        cube_1_ori = ObsTerm(
            func=mdp.object_orientation_w,
            params={"object_cfg": SceneEntityCfg("cube_1")},
        )
        cube_2_pos = ObsTerm(
            func=mdp.object_position_w,
            params={"object_cfg": SceneEntityCfg("cube_2")},
        )
        cube_2_ori = ObsTerm(
            func=mdp.object_orientation_w,
            params={"object_cfg": SceneEntityCfg("cube_2")},
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
    """Two random cubes (cube_1, cube_2). Target zone fixed at center (no physical base)."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # cube_1: right side; cube_2: left side (original spawn zones)
    # Target for stacking = TARGET_XY (0.117, -0.011)
    reset_cube_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.01), "y": (-0.18, -0.14), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_1", body_names="Cube1"),
        },
    )

    reset_cube_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.01), "y": (0.14, 0.18), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_2", body_names="Cube2"),
        },
    )


@configclass
class CubeStackRewardsCfg:
    """Stack cube_1 and cube_2 at fixed target (center), any order."""

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
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("cube_1")},
        weight=15.0,
    )
    # Cubes near fixed target (center)
    cube_1_near_target = RewTerm(
        func=mdp.cube_near_target_xy,
        params={"target_xy": TARGET_XY, "object_cfg": SceneEntityCfg("cube_1")},
        weight=10.0,
    )
    cube_2_near_target = RewTerm(
        func=mdp.cube_near_target_xy,
        params={"target_xy": TARGET_XY, "object_cfg": SceneEntityCfg("cube_2")},
        weight=10.0,
    )
    # Stack cube_1 on cube_2 (either order)
    stack_1_on_2 = RewTerm(
        func=mdp.cube_stack_alignment,
        params={
            "xy_std": 0.03,
            "z_tolerance": 0.015,
            "cube_top_cfg": SceneEntityCfg("cube_1"),
            "cube_base_cfg": SceneEntityCfg("cube_2"),
        },
        weight=20.0,
    )
    stack_2_on_1 = RewTerm(
        func=mdp.cube_stack_alignment,
        params={
            "xy_std": 0.03,
            "z_tolerance": 0.015,
            "cube_top_cfg": SceneEntityCfg("cube_2"),
            "cube_base_cfg": SceneEntityCfg("cube_1"),
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
    cube_1_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_1")},
    )
    cube_2_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube_2")},
    )
    # Named "success" for record_demos.py auto-reset when all done (must release gripper)
    success = DoneTerm(
        func=mdp.two_cubes_stacked_aligned_gripper_released,
        params={
            "expected_height": 0.018,
            "eps_z": 0.003,
            "eps_xy": 0.009,
            "gripper_open_threshold": 0.1,
            "cube_1_cfg": SceneEntityCfg("cube_1"),
            "cube_2_cfg": SceneEntityCfg("cube_2"),
            "right_arm_cfg": SceneEntityCfg("right_arm"),
            "left_arm_cfg": SceneEntityCfg("left_arm"),
        },
    )
    stack_success = DoneTerm(
        func=mdp.two_cubes_stacked_at_target_released,
        params={
            "target_xy": TARGET_XY,
            "expected_height": 0.018,
            "eps_z": 0.003,
            "eps_xy": 0.009,
            "target_eps_xy": 0.2,
            "gripper_open_threshold": 0.1,
            "vel_threshold": 0.001,
            "stable_steps_required": 50,
            "cube_1_cfg": SceneEntityCfg("cube_1"),
            "cube_2_cfg": SceneEntityCfg("cube_2"),
            "right_arm_cfg": SceneEntityCfg("right_arm"),
            "left_arm_cfg": SceneEntityCfg("left_arm"),
            "ee_right_cfg": SceneEntityCfg("ee_right"),
            "ee_left_cfg": SceneEntityCfg("ee_left"),
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
        # Demos: mean~239 steps (4.8s), max~395 (7.9s) at 50Hz; 8s covers p95+
        self.episode_length_s = 8.0
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

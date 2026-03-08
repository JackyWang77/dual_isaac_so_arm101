# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
#
# Dual-arm table setting: two SO-ARM101 arms place fork (left) and knife (right) onto tray.
# Tray + plate are static (center). Fork spawns left, knife spawns right.
#

import os
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

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../assets"))

# Target positions: fork left of plate, knife right of plate (on tray)
# Closer to plate edge for realistic table setting
FORK_TARGET_XY = (0.17, 0.03)
KNIFE_TARGET_XY = (0.17, -0.03)
TARGET_Z = 0.015  # Should match tray/plate surface height

# Subtask thresholds (tight: must be precisely placed, not just nearby)
PLACE_EPS_XY = 0.015   # 1.5cm XY tolerance (was 4cm — too loose, triggered false success)
PLACE_EPS_Z = 0.005    # 5mm Z tolerance (was 1.5cm — too loose, must match plate height)

JAW_OPEN = 0.4
JAW_CLOSED = 0.0


@configclass
class TableSettingSubtaskCfg(ObsGroup):
    """Subtask signals for table setting (place_fork, place_knife)."""

    place_fork = ObsTerm(
        func=mdp.object_placed_at_target,
        params={
            "target_xy": FORK_TARGET_XY,
            "target_z": TARGET_Z,
            "target_eps_xy": PLACE_EPS_XY,
            "target_eps_z": PLACE_EPS_Z,
            "object_cfg": SceneEntityCfg("fork"),
        },
    )
    place_knife = ObsTerm(
        func=mdp.object_placed_at_target,
        params={
            "target_xy": KNIFE_TARGET_XY,
            "target_z": TARGET_Z,
            "target_eps_xy": PLACE_EPS_XY,
            "target_eps_z": PLACE_EPS_Z,
            "object_cfg": SceneEntityCfg("knife"),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class TableSettingSceneCfg(InteractiveSceneCfg):
    """Scene: two SO-ARM101 arms + tray (static) + plate (static) + fork + knife."""

    left_arm: ArticulationCfg = MISSING
    right_arm: ArticulationCfg = MISSING
    ee_right: FrameTransformerCfg = MISSING
    ee_left: FrameTransformerCfg = MISSING
    fork: RigidObjectCfg = MISSING
    knife: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tray",
        init_state=AssetBaseCfg.InitialStateCfg(
            # Rotate 90° around Z so long edge runs left-right (Y direction)
            pos=[0.17, 0.0, 0.005], rot=[0.7071, 0, 0, 0.7071]
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "tray.usd"),
            scale=(1.0, 1.0, 1.0),
        ),
    )

    plate = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Plate",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.17, 0.0, 0.01], rot=[1, 0, 0, 0]
        ),
        spawn=UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "plate.usd"),
            scale=(1.0, 1.0, 1.0),
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
    """Dual-arm actions: joint position or IK per arm + gripper."""

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
        fork_pos = ObsTerm(
            func=mdp.object_position_w,
            params={"object_cfg": SceneEntityCfg("fork")},
        )
        fork_ori = ObsTerm(
            func=mdp.object_orientation_w,
            params={"object_cfg": SceneEntityCfg("fork")},
        )
        knife_pos = ObsTerm(
            func=mdp.object_position_w,
            params={"object_cfg": SceneEntityCfg("knife")},
        )
        knife_ori = ObsTerm(
            func=mdp.object_orientation_w,
            params={"object_cfg": SceneEntityCfg("knife")},
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
    """Fork spawns left, knife spawns right. Tray+plate are static at center."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # pose_range is OFFSET from init_state. Fork init_state is [0.17, 0.15, 0.01].
    # Small jitter around spawn position, not absolute coordinates.
    reset_fork = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("fork"),
        },
    )

    reset_knife = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("knife"),
        },
    )


@configclass
class TableSettingRewardsCfg:
    """Place fork (left arm) and knife (right arm) onto tray beside plate."""

    # Left arm reaches fork
    left_arm_reaches_fork = RewTerm(
        func=mdp.arm_reaches_object,
        params={
            "std": 0.05,
            "object_cfg": SceneEntityCfg("fork"),
            "ee_cfg": SceneEntityCfg("ee_left"),
        },
        weight=10.0,
    )
    # Right arm reaches knife
    right_arm_reaches_knife = RewTerm(
        func=mdp.arm_reaches_object,
        params={
            "std": 0.05,
            "object_cfg": SceneEntityCfg("knife"),
            "ee_cfg": SceneEntityCfg("ee_right"),
        },
        weight=10.0,
    )
    # Grasp intent
    left_grasp_fork = RewTerm(
        func=mdp.grasp_intent_single,
        params={
            "proximity_threshold": 0.05,
            "object_cfg": SceneEntityCfg("fork"),
            "ee_cfg": SceneEntityCfg("ee_left"),
            "arm_cfg": SceneEntityCfg("left_arm"),
        },
        weight=8.0,
    )
    right_grasp_knife = RewTerm(
        func=mdp.grasp_intent_single,
        params={
            "proximity_threshold": 0.05,
            "object_cfg": SceneEntityCfg("knife"),
            "ee_cfg": SceneEntityCfg("ee_right"),
            "arm_cfg": SceneEntityCfg("right_arm"),
        },
        weight=8.0,
    )
    # Lifting
    fork_lifted = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("fork")},
        weight=15.0,
    )
    knife_lifted = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04, "object_cfg": SceneEntityCfg("knife")},
        weight=15.0,
    )
    # Objects near target
    fork_near_target = RewTerm(
        func=mdp.object_near_target_xy,
        params={
            "target_xy": FORK_TARGET_XY,
            "object_cfg": SceneEntityCfg("fork"),
        },
        weight=10.0,
    )
    knife_near_target = RewTerm(
        func=mdp.object_near_target_xy,
        params={
            "target_xy": KNIFE_TARGET_XY,
            "object_cfg": SceneEntityCfg("knife"),
        },
        weight=10.0,
    )
    # Regularization
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
    fork_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("fork")},
    )
    knife_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("knife")},
    )
    success = DoneTerm(
        func=mdp.both_placed_and_released,
        params={
            "fork_target_xy": FORK_TARGET_XY,
            "knife_target_xy": KNIFE_TARGET_XY,
            "target_z": TARGET_Z,
            "eps_xy": PLACE_EPS_XY,
            "eps_z": PLACE_EPS_Z,
            "gripper_open_threshold": 0.1,
            "fork_cfg": SceneEntityCfg("fork"),
            "knife_cfg": SceneEntityCfg("knife"),
            "right_arm_cfg": SceneEntityCfg("right_arm"),
            "left_arm_cfg": SceneEntityCfg("left_arm"),
        },
    )
    table_setting_success = DoneTerm(
        func=mdp.both_placed_released_stable,
        params={
            "fork_target_xy": FORK_TARGET_XY,
            "knife_target_xy": KNIFE_TARGET_XY,
            "target_z": TARGET_Z,
            "eps_xy": PLACE_EPS_XY,
            "eps_z": PLACE_EPS_Z,
            "gripper_open_threshold": 0.1,
            "vel_threshold": 0.001,
            "stable_steps_required": 50,
            "fork_cfg": SceneEntityCfg("fork"),
            "knife_cfg": SceneEntityCfg("knife"),
            "right_arm_cfg": SceneEntityCfg("right_arm"),
            "left_arm_cfg": SceneEntityCfg("left_arm"),
        },
    )


@configclass
class TableSettingEnvCfg(ManagerBasedRLEnvCfg):
    """Dual-arm table setting env config."""

    scene: TableSettingSceneCfg = TableSettingSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: TableSettingRewardsCfg = TableSettingRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
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

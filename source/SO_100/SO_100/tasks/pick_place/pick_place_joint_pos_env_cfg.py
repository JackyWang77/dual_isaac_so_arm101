# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp import dual_pick_place_events

from .pick_place_env_cfg import PickPlaceEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from SO_100.robots.so_arm100_roscon import SO_ARM100_ROSCON_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for events."""

    # Apply high friction material to objects at startup (once per environment initialization)
    apply_friction_to_plate = EventTerm(
        func=dual_pick_place_events.apply_high_friction_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("plate"),
            "static_friction": 2.0,
            "dynamic_friction": 2.0,
            "restitution": 0.0,
        },
    )
    apply_friction_to_fork = EventTerm(
        func=dual_pick_place_events.apply_high_friction_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fork"),
            "static_friction": 2.0,
            "dynamic_friction": 2.0,
            "restitution": 0.0,
        },
    )
    apply_friction_to_knife = EventTerm(
        func=dual_pick_place_events.apply_high_friction_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("knife"),
            "static_friction": 2.0,
            "dynamic_friction": 2.0,
            "restitution": 0.0,
        },
    )

    init_dual_arm_pose = EventTerm(
        func=dual_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            # SO-ARM100: 5 arm joints + 1 gripper joint (total 6 values)
            # [shoulder_pan, shoulder_lift, elbow, wrist_pitch, wrist_roll, jaw]
            "default_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.698],
        },
    )

    # Disabled: causes jittering during demo recording
    # randomize_dual_arm_joint_state = EventTerm(
    #     func=dual_pick_place_events.randomize_joint_by_gaussian_offset,
    #     mode="reset",
    #     params={
    #         "mean": 0.0,
    #         "std": 0.02,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    randomize_object_positions = EventTerm(
        func=dual_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            # SO-ARM100 comfortable workspace: X=0.15-0.25m (15-25cm), this is the golden workspace for small robot arm
            # Originally too far (0.35-0.44), unreachable for SO-100 with 30-40cm reach
            "pose_range": {
                "x": (0.18, 0.22),  # Reduce X axis range: 18cm to 22cm
                "y": (-0.10, 0.10),  # Reduce Y axis range: -10cm to 10cm
                "z": (0.02, 0.02),  # Slightly closer to table
                "yaw": (-0.5, 0.5)  # Reduce rotation range: -0.5 to 0.5 radians
            },
            "min_separation": 0.04,  # Reduce object spacing: 4cm
            "asset_cfgs": [SceneEntityCfg("plate"), SceneEntityCfg("fork"), SceneEntityCfg("knife")],
        },
    )


@configclass
class DualArmPickPlaceJointPosEnvCfg(PickPlaceEnvCfg):

    simplify_scene: bool = False

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        if not self.simplify_scene:
            self.events = EventCfg()
        else:
            self.events = None

        # Set DualArm as robot
        self.scene.robot = SO_ARM100_ROSCON_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=SO_ARM100_ROSCON_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.0),  # Lower robot by 0.1m
            ),
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (dual_arm)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_pitch_joint", "wrist_roll_joint"], scale=1.0, use_default_offset=False
        )
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["jaw_joint"],
            scale=1.0,
            use_default_offset=False,
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["jaw_joint"]
        self.gripper_open_val = 0.698
        self.gripper_threshold = 0.005

        # Rigid body properties of each cube
        # Increase physics iteration count to improve calculation accuracy
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,  # Increased from 16 to 32 for more accurate calculation
            solver_velocity_iteration_count=4,   # Increased from 1 to 4
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # Get assets directory path
        assets_dir = os.path.join(
            os.path.dirname(__file__), "../../../assets"
        )
        assets_dir = os.path.abspath(assets_dir)

        # Place objects on table surface
        self.scene.plate = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.55, 0.0, 0.01], rot=[1, 0, 0, 0]  # Further away (0.5 -> 0.55) to ensure no contact
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(assets_dir, "Plate/Plate.usd"),
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "plate")],
            ),
        )
        self.scene.fork = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Fork",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.20, 0.05, 0.005], rot=[1, 0, 0, 0]  # Changed from 0.45 to 0.20
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(assets_dir, "Fork/Fork.usd"),
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "fork")],
            ),
        )
        self.scene.knife = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Knife",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.20, -0.08, 0.005], rot=[1, 0, 0, 0]  # Changed from 0.45 to 0.20
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(assets_dir, "Knife/Knife.usd"),
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "knife")],
            ),
        )

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Tray",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.30, 0.0, 0.02], rot=[0.707, 0, 0, 0.707]
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(assets_dir, "ikea_tray.usd"),
                scale=(1.0, 1.0, 1.0),
                rigid_props=cube_properties,
                semantic_tags=[("class", "tray")],
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_2_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[-0.004, 0.0, -0.102],
                    ),
                ),
            ],
        )

        # Visual markers for objects (plate, fork, knife)
        # Plate marker (blue-ish)
        plate_marker_cfg = FRAME_MARKER_CFG.copy()
        plate_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        plate_marker_cfg.prim_path = "/Visuals/FrameTransformer/Plate"
        self.scene.plate_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=plate_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Plate",
                    name="plate",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

        # Fork marker (green-ish)
        fork_marker_cfg = FRAME_MARKER_CFG.copy()
        fork_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        fork_marker_cfg.prim_path = "/Visuals/FrameTransformer/Fork"
        self.scene.fork_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=fork_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Fork",
                    name="fork",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

        # Knife marker (red-ish)
        knife_marker_cfg = FRAME_MARKER_CFG.copy()
        knife_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        knife_marker_cfg.prim_path = "/Visuals/FrameTransformer/Knife"
        self.scene.knife_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=knife_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Knife",
                    name="knife",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

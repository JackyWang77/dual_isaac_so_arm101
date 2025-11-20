# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg, CollisionPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.shapes import shapes_cfg
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

    init_dual_arm_pose = EventTerm(
        func=dual_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            # SO-ARM100: 5 arm joints + 1 gripper joint (total 6 values)
            # [shoulder_pan, shoulder_lift, elbow, wrist_pitch, wrist_roll, jaw]
            "default_pose": [0.0, -0.5, 0.0, -1.5, 0.0, 0.698],
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
            # Keep objects away from tray (at x=0.5, y=0.0, z=0.058) to avoid triggering success on init
            # Success condition requires all 3 objects placed correctly:
            #   - Plate at tray center (0.5±0.03, 0.0±0.03) with relative height 0.02±0.02
            #   - Fork at (0.5±0.03, 0.08±0.03) with relative height 0.02±0.02
            #   - Knife at (0.5±0.03, -0.08±0.03) with relative height 0.02±0.02
            # Initialize far from x=0.5 on table at safe height (z=0.08, well above dropping threshold 0.03)
            "pose_range": {"x": (0.35, 0.44), "y": (-0.20, 0.20), "z": (0.08, 0.08), "yaw": (-1.0, 1.0)},
            "min_separation": 0.08,
            "asset_cfgs": [SceneEntityCfg("plate"), SceneEntityCfg("fork"), SceneEntityCfg("knife")],
        },
    )


@configclass
class DualArmPickPlaceJointPosEnvCfg(PickPlaceEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set DualArm as robot
        self.scene.robot = SO_ARM100_ROSCON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (dual_arm)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_pitch_joint", "wrist_roll_joint"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["jaw_joint"],
            open_command_expr={"jaw_joint": 0.698},
            close_command_expr={"jaw_joint": 0.0},
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["jaw_joint"]
        self.gripper_open_val = 0.698
        self.gripper_threshold = 0.005

        # Rigid body properties of each cube
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
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

        # Place objects on table surface (table top at z=0.055)
        self.scene.plate = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Plate",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.45, -0.01468, 0.01760], rot=[1, 0, 0, 0]
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
                pos=[0.45, -0.01468, 0.01760], rot=[1, 0, 0, 0]
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
                pos=[0.45, -0.15, 0.06], rot=[1, 0, 0, 0]
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
                pos=[0.5, 0.0, 0.058], rot=[1, 0, 0, 0]
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
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
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
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

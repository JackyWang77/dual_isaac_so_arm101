# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from pathlib import Path

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim import (
    MeshCuboidCfg,
    DeformableBodyMaterialCfg,
    DeformableBodyPropertiesCfg,
    PreviewSurfaceCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Get data directory path
TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"

from . import mdp


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

    # 🍞 软面包片 (Deformable Bread) - COMMENTED OUT for now (will be used later)
    # bread = DeformableObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Bread",
    #     # 1. 定义形状：使用 Mesh 生成器 (必须是 Mesh，不能是 Shape)
    #     spawn=MeshCuboidCfg(
    #         size=(0.10, 0.10, 0.01),  # 10cm x 10cm x 1cm (薄面包片)
    #         # 2. 视觉材质：看起来像面包
    #         visual_material=PreviewSurfaceCfg(
    #             diffuse_color=(0.8, 0.6, 0.3),  # 焦黄色
    #             roughness=0.9,
    #         ),
    #         # 3. 物理材质：决定软硬 (关键!)
    #         physics_material=DeformableBodyMaterialCfg(
    #             youngs_modulus=5e4,  # 50000 Pa (越小越软，太小会塌)
    #             poissons_ratio=0.4,  # 0.4 (像海绵一样)
    #             damping_scale=0.1,  # 阻尼 (防止像果冻一样乱晃)
    #             dynamic_friction=1.0,  # 摩擦力 (设大点，防止从盘子里滑出去)
    #         ),
    #         # 4. 物理属性：决定计算精度 (关键!)
    #         deformable_props=DeformableBodyPropertiesCfg(
    #             rest_offset=0.0,
    #             contact_offset=0.005,  # 接触厚度 (设为 5mm 左右防止穿模)
    #             # 🔥 关键：网格分辨率。
    #             # 这个数决定了把你的面包切成多少个小块来计算变形。
    #             # 设太小(如 2)就不软了，设太大(如 50)显卡会爆。
    #             # 10 左右对于这个尺寸是黄金值。
    #             simulation_hexahedral_resolution=10,
    #             solver_position_iteration_count=16,  # 计算迭代次数 (防穿模)
    #         ),
    #     ),
    #     # 5. 初始位置：放在盘子上方一点点（盘子位置约在 x=0.28, z=0.0）
    #     init_state=DeformableObjectCfg.InitialStateCfg(
    #         pos=(0.28, 0.0, 0.08),  # 在盘子中心上方 8cm，让它自然掉落
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    #     debug_vis=False,
    # )

    # ---------------------------------------------------------
    # 📷 1. Fixed Camera (Top Camera - Overhead view, looking down)
    # ---------------------------------------------------------
    camera_top = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraTop",  # Generate path
        update_period=0.1,  # 10Hz capture frequency (set to 0 for per-frame capture)
        height=224,  # Image height (ResNet typically uses 224x224)
        width=224,  # Image width
        data_types=["rgb"],  # Only RGB needed, add "depth" if needed
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        # Overhead angle: camera position and orientation
        # Target: x=0, y=0, z=-90 degrees (rotate -90 degrees around Z axis)
        offset=CameraCfg.OffsetCfg(
            pos=(0.2, 0.0, 1.3),  # x=0.2, y=0.0, z=1.3
            rot=(0.0, -0.7071, 0.7071, 0.0),  # Rotate -90 degrees around Z axis (x=0, y=0, z=-90)
            convention="ros",  # Use ROS coordinate system (Z forward, X right, Y down)
        ),
        debug_vis=False,  # Disable debug visualization
    )

    # ---------------------------------------------------------
    # 📷 2. Wrist Camera (Eye in Hand)
    # ---------------------------------------------------------
    # Use camera already in SO-ARM101-NEW-TF2.usd file
    # Camera is already in USD file, reference directly
    camera_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_1_link/Camera",  # Camera path in USD file
        spawn=None,  # Don't spawn new camera, use the one in USD file directly
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb"],
        # offset set to (0,0,0) and (1,0,0,0) means use original position and orientation from USD file
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),  # Use original position from USD file
            rot=(1.0, 0.0, 0.0, 0.0),  # Use original orientation from USD file (no rotation)
            convention="ros",
        ),
        debug_vis=False,  # Disable debug visualization
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        object_positions = ObsTerm(func=mdp.object_positions_in_world_frame)
        object_orientations = ObsTerm(func=mdp.object_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images.
        
        Note: Currently empty - kept for future use in sim2real distillation.
        Teacher policy (Graph-DiT) and student policy (RL fine-tuning) both use
        state-based observations for consistency.
        """

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # Plate tasks - COMMENTED OUT for testing (only cube)
        # push_plate = ObsTerm(
        #     func=mdp.object_pushed,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        #         "object_cfg": SceneEntityCfg("plate"),
        #         "target_cfg": SceneEntityCfg("object"),
        #         "planar_offset": (0.0, 0.0),
        #         "planar_tolerance": 0.03,
        #         "height_target": 0.02,
        #         "height_tolerance": 0.02,
        #     },
        # )
        # pick_plate = ObsTerm(
        #     func=mdp.object_grasped,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        #         "object_cfg": SceneEntityCfg("plate"),
        #     },
        # )
        # place_plate = ObsTerm(
        #     func=mdp.object_placed,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        #         "object_cfg": SceneEntityCfg("plate"),
        #         "target_cfg": SceneEntityCfg("object"),
        #         "planar_offset": (0.0, 0.0),
        #         "planar_tolerance": 0.03,
        #         "height_target": 0.02,
        #         "height_tolerance": 0.02,
        #     },
        # )
        # Fork - COMMENTED OUT (replaced with cube)
        # pick_fork = ObsTerm(
        #     func=mdp.object_grasped,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        #         "object_cfg": SceneEntityCfg("fork"),
        #         "table_height": 0.0,  # Table initial position z=0.0 (table is AssetBaseCfg, not RigidObject)
        #         "min_lift_height": 0.01,  # Fork must be lifted 1cm above table to be considered picked
        #     },
        # )
        # place_fork = ObsTerm(
        #     func=mdp.object_placed,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        #         "object_cfg": SceneEntityCfg("fork"),
        #         "target_cfg": SceneEntityCfg("object"),
        #         "planar_offset": (0.0, 0.08),
        #         "planar_tolerance": 0.03,
        #         "height_target": 0.02,
        #         "height_tolerance": 0.02,
        #     },
        # )
        # pick_cube - COMMENTED OUT for push task
        # pick_cube = ObsTerm(
        #     func=mdp.object_grasped,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        #         "object_cfg": SceneEntityCfg("cube"),
        #         "table_height": 0.0,
        #         "min_lift_height": 0.01,
        #     },
        # )
        
        # Push cube - only checks position, no gripper check
        push_cube = ObsTerm(
            func=mdp.object_pushed,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
                "target_cfg": SceneEntityCfg("object"),
                "planar_offset": (0.0, 0.0),  # Push to tray center
                "planar_tolerance": 0.05,     # 5cm tolerance
            },
        )
        
        # Lift EE - check if hand is raised above 7cm
        lift_ee = ObsTerm(
            func=mdp.ee_lifted,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "min_height": 0.07,  # 7cm above table
            },
        )
        
        # Knife - COMMENTED OUT (not generated)
        # pick_knife = ObsTerm(
        #     func=mdp.object_grasped,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        #         "object_cfg": SceneEntityCfg("knife"),
        #     },
        # )
        # place_knife = ObsTerm(
        #     func=mdp.object_placed,
        #     params={
        #         "robot_cfg": SceneEntityCfg("robot"),
        #         "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        #         "object_cfg": SceneEntityCfg("knife"),
        #         "target_cfg": SceneEntityCfg("object"),
        #         "planar_offset": (0.0, -0.08),
        #         "planar_tolerance": 0.03,
        #         "height_target": 0.02,
        #         "height_tolerance": 0.02,
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # DISABLED: Dropping detection causes issues with physics engine settling
    # Objects may temporarily dip below threshold during initialization/contact
    # Can be re-enabled later if needed, but not critical for demo recording
    # plate_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": 0.03, "asset_cfg": SceneEntityCfg("plate")}
    # )
    # fork_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": 0.03, "asset_cfg": SceneEntityCfg("fork")}
    # )
    # knife_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": 0.03, "asset_cfg": SceneEntityCfg("knife")}
    # )

    # Success condition: cube pushed to target AND EE lifted above 7cm
    # Two subtasks: push_cube -> lift_ee
    success = DoneTerm(func=mdp.push_and_lift_complete)


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick and place environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(-0.1, -0.5, -1.05),
        anchor_rot=(0.866, 0, 0, -0.5),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        # 60Hz golden setup (physics:render:display = 1:1:1)
        self.decimation = 1
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 1.0 / 90
        # Render interval should match decimation to avoid rendering intermediate physics steps
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # 🔥 必须开启！否则 Deformable Object 会报错或变成刚体
        self.sim.physx.use_gpu = True
        self.sim.device = "cuda:0"
        

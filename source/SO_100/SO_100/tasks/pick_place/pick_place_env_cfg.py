# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# 获取 data 目录路径
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

    # ---------------------------------------------------------
    # 📷 1. 固定相机 (Top Camera - 俯视相机，从上往下看)
    # ---------------------------------------------------------
    camera_top = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraTop",  # 生成路径
        update_period=0.1,  # 10Hz 采集频率 (设为 0 则每帧采集)
        height=224,  # 图像高度 (ResNet通常用 224x224)
        width=224,  # 图像宽度
        data_types=["rgb"],  # 只需要 RGB，如果需要深度加 "depth"
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        # 俯视角度：相机位置和朝向
        # 目标：x=0, y=0, z=-90度（绕Z轴旋转-90度）
        offset=CameraCfg.OffsetCfg(
            pos=(0.2, 0.0, 1.3),  # x=0.2, y=0.0, z=1.3
            rot=(0.0, -0.7071, 0.7071, 0.0),  # 绕 Z 轴旋转 -90 度 (x=0, y=0, z=-90)
            convention="ros",  # 使用 ROS 坐标系 (Z向前, X向右, Y向下)
        ),
        debug_vis=False,  # 关闭调试可视化
    )

    # ---------------------------------------------------------
    # 📷 2. 手腕相机 (Wrist Camera - Eye in Hand)
    # ---------------------------------------------------------
    # 🔥 使用 SO-ARM101-NEW-TF2.usd 文件中已有的相机
    #    相机已经在 USD 文件中，直接引用即可
    camera_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_1_link/Camera",  # USD 文件中的相机路径
        spawn=None,  # 不生成新相机，直接使用 USD 文件中的
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb"],
        # offset 设为 (0,0,0) 和 (1,0,0,0) 表示使用 USD 文件中相机的原始位置和朝向
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),  # 使用 USD 文件中的原始位置
            rot=(1.0, 0.0, 0.0, 0.0),  # 使用 USD 文件中的原始朝向（无旋转）
            convention="ros",
        ),
        debug_vis=False,  # 关闭调试可视化
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
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        push_plate = ObsTerm(
            func=mdp.object_pushed,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("plate"),
                "target_cfg": SceneEntityCfg("object"),
                "planar_offset": (0.0, 0.0),
                "planar_tolerance": 0.03,
                "height_target": 0.02,
                "height_tolerance": 0.02,
            },
        )
        # Keep pick_plate for backward compatibility if needed
        pick_plate = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("plate"),
            },
        )
        place_plate = ObsTerm(
            func=mdp.object_placed,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("plate"),
                "target_cfg": SceneEntityCfg("object"),
                "planar_offset": (0.0, 0.0),
                "planar_tolerance": 0.03,
                "height_target": 0.02,
                "height_tolerance": 0.02,
            },
        )
        pick_fork = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("fork"),
            },
        )
        place_fork = ObsTerm(
            func=mdp.object_placed,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("fork"),
                "target_cfg": SceneEntityCfg("object"),
                "planar_offset": (0.0, 0.08),
                "planar_tolerance": 0.03,
                "height_target": 0.02,
                "height_tolerance": 0.02,
            },
        )
        pick_knife = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("knife"),
            },
        )
        place_knife = ObsTerm(
            func=mdp.object_placed,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("knife"),
                "target_cfg": SceneEntityCfg("object"),
                "planar_offset": (0.0, -0.08),
                "planar_tolerance": 0.03,
                "height_target": 0.02,
                "height_tolerance": 0.02,
            },
        )

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

    # Success condition: all objects correctly placed on tray with gripper open
    success = DoneTerm(func=mdp.objects_picked_and_placed)


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
        self.sim.dt = 1.0 / 60.0
        # Render interval should match decimation to avoid rendering intermediate physics steps
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

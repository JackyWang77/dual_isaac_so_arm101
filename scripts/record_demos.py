# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import argparse
import contextlib
import re

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad, ros2. Not all tasks support all built-ins."
        " Use 'ros2' to teleoperate from real robots via ROS2 topics."
    ),
)
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--ros2_ee_pose_topic",
    type=str,
    default="/ee_pose",
    help="ROS2 topic for end-effector pose (only used with --teleop_device ros2).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


def _maybe_add_default_version(task_str: str) -> tuple[str, bool]:
    """Append '-v0' to Isaac task ids that omit the Gym version suffix."""
    segments = task_str.split(":")
    base_id = segments[0]
    if base_id.startswith("Isaac-") and not re.search(r"-v\d+$", base_id):
        segments[0] = f"{base_id}-v0"
        return ":".join(segments), True
    return task_str, False


# Validate required arguments
if args_cli.task is None:
    parser.error("--task is required")
else:
    normalized_task, task_was_modified = _maybe_add_default_version(args_cli.task)
    if task_was_modified:
        print(
            f"[record_demos] Task '{args_cli.task}' is missing a Gym version suffix. "
            f"Using '{normalized_task}' instead."
        )
        args_cli.task = normalized_task

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


# Third-party imports
import gymnasium as gym
import logging
import os
import time
import torch

import omni.ui as ui

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device

import isaaclab_mimic.envs  # noqa: F401
import SO_100.tasks # noqa: F401  # Registers custom SO-100 tasks
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions

# Import ROS2 device for teleoperation
from SO_100.devices import Se3ROS2, Se3ROS2Cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

from collections.abc import Callable

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# import logger
logger = logging.getLogger(__name__)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None]:
    """Create and configure the environment configuration.

    Parses the environment configuration and makes necessary adjustments for demo recording.
    Extracts the success termination function and configures the recorder manager.

    Args:
        output_dir: Directory where recorded demonstrations will be saved
        output_file_name: Name of the file to store the demonstrations

    Returns:
        tuple[isaaclab_tasks.utils.parse_cfg.EnvCfg, Optional[object]]: A tuple containing:
            - env_cfg: The configured environment configuration
            - success_term: The success termination object or None if not available

    Raises:
        Exception: If parsing the environment configuration fails
    """
    # parse configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task.split(":")[-1]
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    if args_cli.xr:
        # If cameras are not enabled and XR is enabled, remove camera configs
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """Create the environment from the configuration.

    Args:
        env_cfg: The environment configuration object that defines the environment properties.
            This should be an instance of EnvCfg created by parse_env_cfg().

    Returns:
        gym.Env: A Gymnasium environment instance for the specified task.

    Raises:
        Exception: If environment creation fails for any reason.
    """
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        exit(1)


def setup_teleop_device(callbacks: dict[str, Callable]) -> object:
    """Set up the teleoperation device based on configuration.

    Attempts to create a teleoperation device based on the environment configuration.
    Falls back to default devices if the specified device is not found in the configuration.

    Args:
        callbacks: Dictionary mapping callback keys to functions that will be
                   attached to the teleop device

    Returns:
        object: The configured teleoperation device interface

    Raises:
        Exception: If teleop device creation fails
    """
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            elif args_cli.teleop_device.lower() == "ros2":
                # Create ROS2 teleop device
                ros2_cfg = Se3ROS2Cfg(
                    ee_pose_topic=args_cli.ros2_ee_pose_topic,
                    num_dof=8,  # 3 pos + 4 quat + 1 gripper (auto-converts from euler)
                    input_format="euler",  # Expects [x,y,z,roll,pitch,yaw,gripper] input
                    pos_scale=1.0,
                )
                teleop_interface = Se3ROS2(ros2_cfg)
                print("[record_demos] 🤖 Using ROS2 teleoperation device")
                print("[record_demos] 📡 Waiting for messages from real robot:")
                print(f"               ✓ Topic: {ros2_cfg.ee_pose_topic}")
                print("[record_demos] 💡 Press 'R' to reset episode, Ctrl+C to stop")
            elif args_cli.teleop_device.lower() == "joint_states":
                # Create Joint States ROS2 device
                from SO_100.devices import JointStatesROS2, JointStatesROS2Cfg
                
                joint_states_cfg = JointStatesROS2Cfg(
                    joint_state_topic="/joint_states",  # 直接订阅原始话题（最低延迟）
                    # 如果需要使用 relay，改为: joint_state_topic="/isaac_joint_command"
                    num_dof=6,  # 5 arm joints + 1 gripper
                    joint_names=[
                        "shoulder_pan_joint",
                        "shoulder_lift_joint", 
                        "elbow_joint",
                        "wrist_pitch_joint",
                        "wrist_roll_joint",
                        "jaw_joint"
                    ],
                    scale=1.0,
                )
                teleop_interface = JointStatesROS2(joint_states_cfg)
                print("[record_demos] 🤖 Using ROS2 Joint States teleoperation device")
                print("[record_demos] 📡 Waiting for joint states from real robot:")
                print(f"               ✓ Topic: {joint_states_cfg.joint_state_topic}")
                print(f"               ✓ Joints: {joint_states_cfg.joint_names}")
                print("[record_demos] 💡 Real robot controls simulated robot, EE poses recorded")
                print("[record_demos] 💡 Press 'R' to reset episode, Ctrl+C to stop")
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Supported devices: keyboard, spacemouse, ros2, joint_states, handtracking")
                exit(1)

            # Add callbacks to fallback device
            for key, callback in callbacks.items():
                teleop_interface.add_callback(key, callback)
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        exit(1)

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        exit(1)

    return teleop_interface


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """Set up the user interface elements.

    Creates instruction display and UI window with labels for showing information
    to the user during demonstration recording.

    Args:
        label_text: Text to display showing current recording status
        env: The environment instance for which UI is being created

    Returns:
        InstructionDisplay: The configured instruction display object
    """
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def handle_reset(
    env: gym.Env, success_step_count: int, instruction_display: InstructionDisplay, label_text: str
) -> int:
    """Handle resetting the environment.

    Resets the environment, recorder manager, and related state variables.
    Updates the instruction display with current status.

    Args:
        env: The environment instance to reset
        success_step_count: Current count of consecutive successful steps
        instruction_display: The display object to update
        label_text: Text to display showing current recording status

    Returns:
        int: Reset success step count (0)
    """
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object | None,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Sets up callback functions for teleop device, initializes the UI,
    and runs the main loop that processes user inputs and environment steps.
    Records demonstrations when success conditions are met.

    Args:
        env: The environment instance
        teleop_interface: Optional teleop interface (will be created if None)
        success_term: The success termination object or None if not available
        rate_limiter: Optional rate limiter to control simulation speed

    Returns:
        int: Number of successful demonstrations recorded
    """
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    # ✅ 默认 False (暂停状态)。等你准备好了按 'L' 键开始。
    running_recording_instance = False
    print("🚀 Ready! Press 'L' to START/STOP recording, 'R' to RESET.")

    # Callback closures for the teleop device
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("🔄 Recording instance reset requested")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("▶️  Recording started")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("⏸️  Recording paused")

    # Set up teleoperation callbacks
    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    teleop_interface = setup_teleop_device(teleoperation_callbacks)
    teleop_interface.add_callback("R", reset_recording_instance)

    # === 🔥 新增：独立键盘监听 (专门用于 ROS 模式下的 UI 控制) ===
    extra_keyboard = None
    if args_cli.teleop_device.lower() in ["ros2", "joint_states"]:
        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
        # 创建一个灵敏度为0的键盘，只用来监听按键，不控制机器人
        extra_keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0))
        
        # 定义切换录制状态的函数
        def toggle_recording():
            nonlocal running_recording_instance
            if running_recording_instance:
                stop_recording_instance()
            else:
                start_recording_instance()
        
        # 绑定按键：L 键切换录制状态，R 键重置
        extra_keyboard.add_callback("L", toggle_recording)
        extra_keyboard.add_callback("R", reset_recording_instance)
        print("[record_demos] ⌨️  Keyboard control active: [L] = Start/Pause, [R] = Reset")
    # ============================================================

    # === 🔥 新增：从 ROS2 读取初始关节位置并同步到 Isaac Sim ===
    initial_joint_pos_saved = None  # 保存初始位置，用于 reset 后立即设置
    if args_cli.teleop_device.lower() == "joint_states" and hasattr(teleop_interface, "_latest_joint_positions"):
        print("[record_demos] 🔄 Waiting for initial joint states from real robot...")
        max_wait_time = 5.0  # 最多等待 5 秒
        start_time = time.time()
        initial_joint_pos = None
        
        # 等待收到第一个 ROS2 消息
        while time.time() - start_time < max_wait_time:
            # 尝试获取关节位置（这会触发 ROS2 spin）
            _ = teleop_interface.advance()
            if hasattr(teleop_interface, "_is_connected") and teleop_interface._is_connected():
                # 读取初始关节位置
                if hasattr(teleop_interface, "_latest_joint_positions"):
                    initial_joint_pos = teleop_interface._latest_joint_positions.clone()
                    if initial_joint_pos.numel() > 0:
                        # 转换为列表格式
                        joint_pos_list = initial_joint_pos[0].cpu().tolist()
                        print(f"[record_demos] ✅ Received initial joint positions: {joint_pos_list}")
                        break
            time.sleep(0.1)  # 等待 100ms 再试
        
        # 如果成功读取到初始位置，设置到环境中
        if initial_joint_pos is not None and initial_joint_pos.numel() > 0:
            robot = env.scene["robot"]
            joint_pos_list = initial_joint_pos[0].cpu().tolist()
            
            # 保存初始位置，用于 reset 后立即设置
            initial_joint_pos_saved = {
                "joint_pos_list": joint_pos_list,
                "joint_names": teleop_interface.cfg.joint_names,
            }
            
            # 设置所有环境的初始关节位置（用于 reset 时使用）
            for env_id in range(env.num_envs):
                for i, joint_name in enumerate(teleop_interface.cfg.joint_names):
                    if i < len(joint_pos_list):
                        # 设置初始关节位置
                        if joint_name in robot.joint_names:
                            joint_idx = robot.joint_names.index(joint_name)
                            robot.data.default_joint_pos[env_id, joint_idx] = joint_pos_list[i]
            
            print("[record_demos] ✅ Set Isaac Sim robot initial pose to match real robot")
            joint_pos_dict = dict(zip(teleop_interface.cfg.joint_names, joint_pos_list))
            print(f"[record_demos]    Joint positions: {joint_pos_dict}")
        else:
            print("[record_demos] ⚠️  Could not read initial joint positions from ROS2, using default pose")
    # ============================================================

    # Reset before starting
    env.sim.reset()
    env.reset()
    teleop_interface.reset()
    
    # === 🔥 优化：立即设置当前关节位置，确保第一帧画面就是同步的 ===
    if initial_joint_pos_saved is not None:
        robot = env.scene["robot"]
        joint_pos_list = initial_joint_pos_saved["joint_pos_list"]
        joint_names = initial_joint_pos_saved["joint_names"]
        
        # 构建完整的关节位置张量（包括所有关节，不仅仅是 ROS2 控制的关节）
        joint_pos_tensor = robot.data.default_joint_pos.clone()
        joint_vel_tensor = torch.zeros_like(robot.data.default_joint_vel)
        
        # 更新 ROS2 控制的关节位置
        for env_id in range(env.num_envs):
            for i, joint_name in enumerate(joint_names):
                if i < len(joint_pos_list):
                    if joint_name in robot.joint_names:
                        joint_idx = robot.joint_names.index(joint_name)
                        joint_pos_tensor[env_id, joint_idx] = joint_pos_list[i]
        
        # 立即写入物理引擎，确保第一帧画面就是同步的
        robot.write_joint_state_to_sim(joint_pos_tensor, joint_vel_tensor)
        print("[record_demos] ✅ Immediately synchronized robot pose in physics engine")
    # ============================================================

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)

    # Check if environment has subtask_configs
    if hasattr(env.cfg, "subtask_configs"):
        print(f"[DEBUG] Environment has subtask_configs: {list(env.cfg.subtask_configs.keys())}")
    else:
        print("[DEBUG] Warning: Environment does not have subtask_configs attribute")

    subtasks = {}
    # 初始化终端显示标志
    run_simulation_loop._last_task_desc = ""

    with torch.inference_mode():
        while simulation_app.is_running():
            # Get command from teleop device (ROS/Robot)
            action = teleop_interface.advance()

            # === 🔥 新增：刷新键盘状态 ===
            if extra_keyboard:
                extra_keyboard.advance()  # 这一步会检测你有没有按 L 或 R
            # ==========================

            # Expand to batch dimension
            actions = action.repeat(env.num_envs, 1)

            # Perform action on environment
            if running_recording_instance:
                # Compute actions based on environment
                # 每帧都获取观测，保证 UI 实时响应（逻辑检查开销通常可忽略不计）
                obv, _, _, _, _ = env.step(actions)
                
                # 更新 UI 提示（subtask instructions）
                if subtasks is not None:
                    if subtasks == {}:
                        # 第一次初始化 subtasks
                        # obv 已经是字典了，不需要 [0]
                        if "subtask_terms" in obv:
                            subtasks = obv.get("subtask_terms")
                            print(f"[DEBUG] Initialized subtasks: {list(subtasks.keys()) if subtasks else None}")
                        else:
                            print(f"[DEBUG] Warning: 'subtask_terms' not found in observation. Available keys: {list(obv.keys())}")
                            subtasks = None
                    
                    # 每一帧都刷新提示，确保 UI 实时更新（如 "Pick plate" -> "Place plate"）
                    if subtasks:
                        # 1. 尝试更新 UI (保留原逻辑)
                        try:
                            show_subtask_instructions(instruction_display, subtasks, [obv], env.cfg)
                        except Exception:
                            # UI 挂了就挂了，不管它，继续用终端显示
                            pass

                        # 🔥 2. 终端实时播报 (这是给你的定心丸)
                        try:
                            # 获取当前使用的末端名称 (通常是 "end_effector")
                            if hasattr(env.cfg, "subtask_configs") and env.cfg.subtask_configs:
                                eef_name = list(env.cfg.subtask_configs.keys())[0]
                                configs = env.cfg.subtask_configs[eef_name]

                                current_task_desc = "🎉 All Done!"  # 默认全部完成

                                # 遍历所有子任务，找到第一个"未完成"的任务
                                for i, cfg in enumerate(configs):
                                    sig_name = cfg.subtask_term_signal
                                    # 检查信号值 (1.0 = 完成, 0.0 = 未完成)
                                    if sig_name in subtasks:
                                        # subtasks[sig_name] 是一个 tensor，取它的值
                                        signal_value = subtasks[sig_name]
                                        # 处理 tensor，可能是标量或数组
                                        if isinstance(signal_value, torch.Tensor):
                                            val = signal_value.item() if signal_value.numel() == 1 else signal_value[0].item()
                                        else:
                                            val = float(signal_value)

                                        if val < 0.5:  # 还没完成
                                            # 获取任务描述，如果没有则使用信号名称
                                            task_desc = getattr(cfg, "description", None) or cfg.subtask_term_signal.replace("_", " ").title()
                                            current_task_desc = f"Step {i+1}/{len(configs)}: {task_desc}"
                                            break  # 找到了，停止遍历

                                # \r 让光标回到行首，实现原地刷新，不会刷屏
                                # 只在任务描述改变时打印，避免频繁刷新
                                if current_task_desc != getattr(run_simulation_loop, "_last_task_desc", ""):
                                    print(f"\r[Instructor] 🤖 当前任务: {current_task_desc} " + " " * 20, end="", flush=True)
                                    run_simulation_loop._last_task_desc = current_task_desc

                        except Exception as e:
                            # 如果出错，打印调试信息（只打印一次，避免刷屏）
                            if not hasattr(run_simulation_loop, "_debug_printed"):
                                print(f"\n[Instructor] Debug Error: {e}", flush=True)
                                if hasattr(env.cfg, "subtask_configs"):
                                    print(f"[Instructor] subtask_configs keys: {list(env.cfg.subtask_configs.keys())}", flush=True)
                                if isinstance(subtasks, dict):
                                    print(f"[Instructor] subtasks keys: {list(subtasks.keys())}", flush=True)
                                run_simulation_loop._debug_printed = True
            else:
                env.sim.render()

            # Check for success condition
            success_step_count, success_reset_needed = process_success_condition(env, success_term, success_step_count)
            if success_reset_needed:
                should_reset_recording_instance = True

            # Update demo count if it has changed
            if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                print(label_text)

            # Check if we've reached the desired number of demos
            if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                instruction_display.show_demo(label_text)
                print(label_text)
                target_time = time.time() + 0.8
                while time.time() < target_time:
                    if rate_limiter:
                        rate_limiter.sleep(env)
                    else:
                        env.sim.render()
                break

            # Handle reset if requested
            if should_reset_recording_instance:
                success_step_count = handle_reset(env, success_step_count, instruction_display, label_text)
                should_reset_recording_instance = False
                
                # 🔥 Reset 后必须清空任务缓存！
                subtasks = {}  # 清空引用，强迫下一帧重新获取新的 subtasks
                run_simulation_loop._last_task_desc = ""  # 清空打印记录，强迫终端重新显示第一步
                print("\r" + " " * 50 + "\r", end="")  # 清除旧的打印行

            # Check if simulation is stopped
            if env.sim.is_stopped():
                break

            # Rate limiting
            if rate_limiter:
                rate_limiter.sleep(env)

    return current_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using teleop interfaces.

    Main function that orchestrates the entire process:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Runs the simulation loop to collect demonstrations
    5. Cleans up resources when done

    Raises:
        Exception: Propagates exceptions from any of the called functions
    """
    # if handtracking is selected, rate limiting is achieved via OpenXR
    if args_cli.xr:
        rate_limiter = None
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        # Assign the teleop visualization manager to the visualization system
        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # Set up output directories
    output_dir, output_file_name = setup_output_directories()

    # Create and configure environment
    global env_cfg  # Make env_cfg available to setup_teleop_device
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)

    # Create environment
    env = create_environment(env_cfg)

    # Run simulation loop
    current_recorded_demo_count = run_simulation_loop(env, None, success_term, rate_limiter)

    # Clean up
    env.close()
    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

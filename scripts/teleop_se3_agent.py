# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

import carb.settings
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad, ros2, joint_states, dummy_joint. Not all tasks support all built-ins."
        " Use 'ros2' to teleoperate from a real robot via ROS2 topics."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--target_hz",
    type=float,
    default=30.0,
    help="Target loop rate for teleop steps in Hz.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.experience = "isaacsim.exp.full.kit"
app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

# Disable VSync / Present Mode via API to avoid waiting on display
settings = carb.settings.get_settings()
settings.set_int("/app/window/presentMode", 0)
# print("⚡ Present Mode set to IMMEDIATE (VSync OFF)")  # Reduced console output

"""Rest everything follows."""


import time

import gymnasium as gym
import logging
import torch

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

# Import ROS2 device
try:
    from SO_101.devices import Se3ROS2, Se3ROS2Cfg
    ROS2_DEVICE_AVAILABLE = True
except ImportError:
    ROS2_DEVICE_AVAILABLE = False

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

# import logger
logger = logging.getLogger(__name__)


class DummyJointTeleop:
    """Simple teleop device that always returns zero joint commands."""

    def __init__(self, action_dim: int) -> None:
        self._action = torch.zeros((1, action_dim), dtype=torch.float32)
        self._callbacks: dict[str, Callable[[], None]] = {}

    def add_callback(self, key: str, callback: Callable[[], None]) -> None:
        self._callbacks[key] = callback

    def reset(self) -> None:
        self._action.zero_()

    def advance(self) -> torch.Tensor:
        return self._action


def main() -> None:
    """
    Run keyboard teleoperation with Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        # External cameras are not supported with XR teleop
        # Check for any camera configs and disable them
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For hand tracking devices, add additional callbacks
    if args_cli.xr:
        # Default to inactive for hand tracking
        teleoperation_active = False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "ros2":
                if not ROS2_DEVICE_AVAILABLE:
                    logger.error("ROS2 device not available. Please install SO_101 package.")
                    env.close()
                    simulation_app.close()
                    return
                # Create ROS2 teleop device
                ros2_cfg = Se3ROS2Cfg(
                    ee_pose_topic="/ee_pose",
                    num_dof=8,  # 3 pos + 4 quat + 1 gripper (auto-converts from euler)
                    input_format="euler",  # Expects [x,y,z,roll,pitch,yaw,gripper] input
                    pos_scale=1.0,
                )
                teleop_interface = Se3ROS2(ros2_cfg)
                # Reduced console output
                # print("[teleop_se3_agent] 🤖 Using ROS2 teleoperation")
                # print(f"[teleop_se3_agent] 📡 Subscribed to: {ros2_cfg.ee_pose_topic}")
            elif args_cli.teleop_device.lower() == "joint_states":
                if not ROS2_DEVICE_AVAILABLE:
                    logger.error("ROS2 device not available. Please install SO_101 package.")
                    env.close()
                    simulation_app.close()
                    return
                # Create Joint States ROS2 device
                from SO_101.devices import JointStatesROS2, JointStatesROS2Cfg
                joint_states_cfg = JointStatesROS2Cfg(
                    joint_state_topic="/joint_states",
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
                # Reduced console output
                # print("[teleop_se3_agent] 🤖 Using ROS2 Joint States teleoperation")
                # print(f"[teleop_se3_agent] 📡 Subscribed to: {joint_states_cfg.joint_state_topic}")
                # print("[teleop_se3_agent] 💡 Real robot controls simulated robot")
            elif args_cli.teleop_device.lower() in ("dummy_joint", "dummy"):
                action_shape = env.action_space.shape
                if isinstance(action_shape, tuple):
                    action_dim = action_shape[-1]
                else:
                    action_dim = action_shape
                teleop_interface = DummyJointTeleop(action_dim)
                # print("[teleop_se3_agent] 🤖 Using dummy joint teleop device (zero actions)")  # Reduced console output
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Supported devices: keyboard, spacemouse, gamepad, ros2, joint_states, handtracking")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    # Device info removed to reduce console spam
    # print(f"Using teleop device: {teleop_interface}")

    # reset environment
    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")

    # simulate environment
    last_step_time = time.time()
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                loop_start = time.time()

                # get device command
                advance_start = time.time()
                action = teleop_interface.advance()
                advance_end = time.time()

                # Only apply teleop commands when active
                if teleoperation_active:
                    # process actions
                    actions = action.repeat(env.num_envs, 1)
                    # apply actions
                    step_start = time.time()
                    env.step(actions)
                    step_end = time.time()
                else:
                    env.sim.render()

                if should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
                loop_mid = time.time()
                total_loop = loop_mid - loop_start
                target_step_duration = 1.0 / args_cli.target_hz
                sleep_time = max(0.0, target_step_duration - total_loop)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                loop_end = time.time()
                step_duration = loop_end - last_step_time
                last_step_time = loop_end
                # Performance stats removed to reduce console spam
                # Uncomment below if needed for debugging:
                # advance_dt = advance_end - advance_start
                # step_dt = (step_end - step_start) if teleoperation_active else 0.0
                # print(f"[teleop_se3_agent] advance_dt={advance_dt:.4f}s step_dt={step_dt:.4f}s step_time={step_duration:.4f}s total_loop={total_loop:.4f}s")
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

# Copyright (c) 2024-2025, SO-ARM100 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 device for receiving joint states from real robot."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.devices.device_base import DeviceBase
from isaaclab.utils import configclass
import carb.settings
# ROS2 imports
import rclpy
from sensor_msgs.msg import JointState


@configclass
class JointStatesROS2Cfg:
    """Configuration for ROS2 Joint States device."""

    joint_state_topic: str = "/joint_states"
    """ROS2 topic for joint states. Default is "/joint_states"."""

    num_dof: int = MISSING
    """Number of degrees of freedom (joints + gripper)."""

    joint_names: list[str] = MISSING
    """List of joint names to extract from JointState message, in order."""

    timeout_sec: float = 2.0
    """Timeout in seconds for ROS2 connection check. Default is 2.0 seconds."""

    scale: float = 1.0
    """Scaling factor for joint positions. Default is 1.0 (no scaling)."""

    enable_interpolation: bool = False
    """Enable linear interpolation between messages for smoother control. Default is True.
    Recommended when robot publishes at low frequency (< 30Hz)."""

    interpolation_rate_hz: float = 30.0
    """Target interpolation rate in Hz. Used when enable_interpolation=True. Default is 30.0 Hz."""

    # ❌ 原来是 0.03 (30ms)，太极限了，刚好卡住
    # ✅ 改成 0.05 (50ms)，甚至是 0.1
    # 只要在这个时间内收到数据，就不算断连
    max_wait_no_msg_sec: float = 0.1
    """When no message arrives within this duration, return cached action instead of blocking."""


class JointStatesROS2(DeviceBase):
    """ROS2 device for receiving joint states from real robot.

    This device subscribes to a ROS2 JointState topic and returns the joint positions
    as actions for teleoperation. The real robot's joint positions are used to drive
    the simulated robot, and the recorded actions are automatically converted to
    end-effector poses by the environment.

    Workflow:
        1. Real robot publishes JointState to /joint_states
        2. This device receives and extracts joint positions
        3. Returns joint positions as actions
        4. Environment (Joint-For-IK-Abs) records EE Absolute Pose via FK
    """

    cfg: JointStatesROS2Cfg

    def __init__(self, cfg: JointStatesROS2Cfg):
        """Initialize ROS2 Joint States device.

        Args:
            cfg: Configuration for the ROS2 Joint States device.
        """
        # Store configuration
        self.cfg = cfg
        carb.settings.get_settings().set_int("/app/window/presentMode", 0)
        # Internal state - use torch tensor directly to avoid conversions
        self._latest_joint_positions = torch.zeros((1, cfg.num_dof), dtype=torch.float32)
        self._previous_joint_positions = torch.zeros((1, cfg.num_dof), dtype=torch.float32)  # For interpolation
        self._last_update_time = 0.0
        self._previous_update_time = 0.0  # For interpolation
        self._connection_warned = False
        self._callbacks = {}

        # Pre-allocate numpy array for callback (avoids allocation each time)
        self._joint_positions_array = np.zeros(cfg.num_dof, dtype=np.float32)

        # Interpolation state
        self._interpolation_interval = 1.0 / cfg.interpolation_rate_hz if cfg.enable_interpolation else 0.0
        self._last_interpolation_time = 0.0
        self._last_joint_msg_time = 0.0
        self._last_timing_log_time = 0.0
        self._timing_log_interval = 1.0
        self._last_wait_warning_time = 0.0
        self._wait_warning_interval = 0.2
        self._last_returned_positions = torch.zeros_like(self._latest_joint_positions)

        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()

        # Create ROS2 node with optimized QoS settings for low latency
        self.node = rclpy.create_node('joint_states_ros2_device')

        # Subscribe to joint states topic with more forgiving QoS to avoid missed bridge callbacks
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        qos_profile = QoSProfile(
            depth=2,  # Buffer a few messages to tolerate bursts/relay delays
            reliability=ReliabilityPolicy.RELIABLE,  # Ensure delivery even if bridge hops exist
            durability=DurabilityPolicy.VOLATILE  # No persistence
        )
        self.subscription = self.node.create_subscription(
            JointState,
            self.cfg.joint_state_topic,
            self._joint_state_callback,
            qos_profile  # Use optimized QoS
        )

        # Pre-compute joint name mapping for faster lookup
        self._joint_name_to_index = {name: i for i, name in enumerate(cfg.joint_names)}

        print("[JointStatesROS2] 🤖 Initialized ROS2 Joint States teleoperation device")
        print(f"[JointStatesROS2] 📡 Subscribed to: {cfg.joint_state_topic}")
        print(f"[JointStatesROS2] 🔧 Expected joints: {cfg.joint_names}")
        print(f"[JointStatesROS2] 📊 DOF: {cfg.num_dof}")
        print("[JointStatesROS2] 💡 Waiting for joint state data...")

    def __del__(self):
        """Destructor to clean up ROS2 resources."""
        self.close()

    def __str__(self) -> str:
        """Returns: A string containing the information of the device."""
        msg = "ROS2 Joint States Teleoperation Device:\n"
        msg += f"\tTopic: {self.cfg.joint_state_topic}\n"
        msg += f"\tDOF: {self.cfg.num_dof}\n"
        msg += f"\tJoints: {self.cfg.joint_names}\n"
        msg += f"\tConnection: {'🟢 Active' if self._is_connected() else '🔴 Lost'}"
        return msg

    """
    Operations
    """

    def reset(self):
        """Reset the ROS2 device state.

        This clears any cached joint position data and resets internal counters.
        """
        self._latest_joint_positions.zero_()
        self._previous_joint_positions.zero_()
        current_time = time.time()
        self._last_update_time = current_time
        self._previous_update_time = current_time
        self._connection_warned = False
        print("[JointStatesROS2] ♻️  Device reset")

    def add_callback(self, key: str, func: Callable):
        """Add a callback function for a specific key/event.

        Args:
            key: The key/event identifier (e.g., "R" for reset).
            func: The callback function to execute.
        """
        self._callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Advance the device by one step and return the current joint positions.

        This method returns the latest joint positions from ROS callbacks integrated into the main loop.
        If interpolation is enabled, it provides smooth interpolation between messages.

        Returns:
            torch.Tensor: Action array with joint positions.
                Shape: (1, num_dof) - joint positions for all DOF

        Note:
            If no messages have been received or connection is lost, returns zero action.
        """
        # Spin ROS2 in the main loop instead of a dedicated thread
        if rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.0)

        # Check connection status (only check, don't block)
        if not self._is_connected():
            if not self._connection_warned:
                print(f"[JointStatesROS2] ⚠️  No ROS2 messages received for {self.cfg.timeout_sec}s")
                print(f"[JointStatesROS2] Make sure real robot is publishing to: {self.cfg.joint_state_topic}")
                self._connection_warned = True
            return torch.zeros((1, self.cfg.num_dof), dtype=torch.float32)

        # Reset warning flag when connected
        if self._connection_warned and self._is_connected():
            print("[JointStatesROS2] ✅ Connection restored")
            self._connection_warned = False

        current_positions = self._latest_joint_positions.clone()
        current_time = self._last_update_time
        prev_time = self._previous_update_time

        # Apply interpolation if enabled and we have valid timing data
        if (
            self.cfg.enable_interpolation
            and current_time > 0
            and prev_time > 0
            and current_time > prev_time
        ):

            now = time.time()
            msg_interval = current_time - prev_time

            # If we have a valid message interval, interpolate between previous and current
            if msg_interval > 0.001:  # At least 1ms interval
                prev_positions = self._previous_joint_positions.clone()

                # Calculate interpolation factor:
                # - At time prev_time: factor = 0.0 (previous message)
                # - At time current_time: factor = 1.0 (current message)
                # - Between: linear interpolation
                time_since_prev = now - prev_time
                interp_factor = time_since_prev / msg_interval

                # Clamp to [0, 1] to handle edge cases
                interp_factor = min(1.0, max(0.0, interp_factor))

                # Linear interpolation: prev + (current - prev) * factor
                interpolated = prev_positions + (current_positions - prev_positions) * interp_factor
                return interpolated * self.cfg.scale

        # No interpolation: return latest positions (but avoid waiting too long)
        now = time.time()
        time_since_last_msg = now - self._last_joint_msg_time
        if (
            self._last_joint_msg_time > 0.0
            and time_since_last_msg > 0.02
            and (now - self._last_wait_warning_time) > self._wait_warning_interval
        ):
            # Reduced console output - waiting message removed
            # print(
            #     f"[JointStatesROS2] ⏳ Waiting for bridge/new message for {time_since_last_msg:.3f}s "
            #     f"(threshold {self._wait_warning_interval:.1f}s)"
            # )
            self._last_wait_warning_time = now
        if (
            self._last_joint_msg_time > 0.0
            and time_since_last_msg > self.cfg.max_wait_no_msg_sec
        ):
            print(
                f"[JointStatesROS2] ⚠️  No update for {time_since_last_msg:.3f}s > "
                f"{self.cfg.max_wait_no_msg_sec:.3f}s, reusing cached joint values"
            )
            return self._last_returned_positions.clone() * self.cfg.scale
        self._last_returned_positions.copy_(current_positions)
        if (
            self._last_joint_msg_time > 0.0
            and time_since_last_msg > 0.1
            and (now - self._last_timing_log_time) > self._timing_log_interval
        ):
            # Reduced console output - timing log removed
            # print(
            #     f"[JointStatesROS2] ⏱️ No new joint message for {time_since_last_msg:.3f}s "
            #     f"(threshold {self._timing_log_interval:.1f}s)"
            # )
            self._last_timing_log_time = now
        return current_positions * self.cfg.scale

    def close(self):
        """Close the ROS2 device and cleanup resources."""
        if hasattr(self, 'node') and self.node is not None:
            try:
                self.node.destroy_node()
                # Note: Don't call rclpy.shutdown() here as it might be used by other nodes
                print("[JointStatesROS2] Device closed")
            except Exception as e:
                print(f"[JointStatesROS2] Warning during cleanup: {e}")

    """
    Internal helpers
    """

    def _is_connected(self) -> bool:
        """Check if ROS2 connection is active.

        Returns:
            bool: True if messages have been received recently, False otherwise.
        """
        time_since_update = time.time() - self._last_update_time
        # Check if we've received at least one message (last_update_time > 0)
        has_received_data = self._last_update_time > 0.0
        return has_received_data and time_since_update < self.cfg.timeout_sec

    def _joint_state_callback(self, msg: JointState):
        """Callback for joint states from ROS2.

        Optimized for low latency:
        - Pre-allocated array to avoid memory allocation
        - Direct indexing instead of list operations
        - Thread-safe tensor update

        Args:
            msg: ROS2 JointState message containing joint names and positions.
        """
        try:
            # Create mapping from message joint names to indices (only once)
            if not hasattr(self, '_msg_name_to_idx'):
                self._msg_name_to_idx = {name: i for i, name in enumerate(msg.name)}
                print("[JointStatesROS2] ✅ Received first joint state message:")
                print(f"  Joint names in message: {msg.name}")

            # Extract joint positions in the order specified in cfg.joint_names
            # Use pre-allocated array and direct assignment (faster than list append)
            msg_name_to_idx = getattr(self, '_msg_name_to_idx', None)

            for i, joint_name in enumerate(self.cfg.joint_names):
                if msg_name_to_idx is not None and joint_name in msg_name_to_idx:
                    idx = msg_name_to_idx[joint_name]
                    self._joint_positions_array[i] = msg.position[idx]
                else:
                    # Joint not found - try direct lookup (fallback)
                    try:
                        idx = msg.name.index(joint_name)
                        self._joint_positions_array[i] = msg.position[idx]
                    except ValueError:
                        # Joint not found in message, use 0.0 as default
                        self._joint_positions_array[i] = 0.0
                        if not hasattr(self, '_missing_joint_warned'):
                            print(f"[JointStatesROS2] ⚠️  Joint '{joint_name}' not found in JointState message")
                            self._missing_joint_warned = True

            # Update tensor directly
            current_time = time.time()
            # Store previous positions for interpolation
            self._previous_joint_positions.copy_(self._latest_joint_positions)
            self._previous_update_time = self._last_update_time

            # Direct assignment from numpy array (no intermediate conversion)
            self._latest_joint_positions[0] = torch.from_numpy(self._joint_positions_array)

            self._last_update_time = current_time
            self._last_joint_msg_time = current_time

            # Print first successful message (debug)
            if not hasattr(self, '_first_msg_printed'):
                print(f"[JointStatesROS2] ✅ Processing joint states (positions: {self._joint_positions_array})")
                self._first_msg_printed = True

        except Exception as e:
            print(f"[JointStatesROS2] ❌ Error processing JointState message: {e}")
            import traceback
            traceback.print_exc()

# End of ROS2 Joint States device

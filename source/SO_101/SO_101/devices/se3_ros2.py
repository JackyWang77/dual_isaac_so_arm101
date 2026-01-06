# Copyright (c) 2024-2025, SO-ARM101 Dual Arm Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 device for SE(3) control using real robot teleoperation."""

from __future__ import annotations

import time
import weakref
from collections.abc import Callable
from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.devices.device_base import DeviceBase
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation


# Enable ROS2 bridge extension
def _enable_ros2_extension():
    """Enable Isaac Sim ROS2 bridge extension if available."""
    try:
        import omni.isaac.core.utils.extensions as extensions_utils

        # Enable the extension
        extensions_utils.enable_extension("isaacsim.ros2.bridge")
        print("[Se3ROS2] ✓ Enabled isaacsim.ros2.bridge extension")

        # Give the extension time to load (update simulation app)
        try:
            # Try to get simulation app for update
            from omni.isaac.kit import SimulationApp

            app = SimulationApp.instance()
            if app:
                app.update()
                app.update()  # Double update to ensure everything is loaded
                print("[Se3ROS2] ✓ Updated simulation app")
        except:
            # If we can't get the app, that's okay - extension is still enabled
            pass

        return True
    except Exception as e:
        print(f"[Se3ROS2] ⚠️  Could not enable ROS2 bridge extension: {e}")
        return False


# Enable extension on module import
_ros2_extension_enabled = _enable_ros2_extension()

# ROS2 imports (must be after SimulationApp initialization and extension enabled)
try:
    import rclpy
    from std_msgs.msg import Float64MultiArray

    ROS2_AVAILABLE = True
    print("[Se3ROS2] ✓ ROS2 Python bindings available")
except ImportError as e:
    ROS2_AVAILABLE = False
    print(f"[Se3ROS2] ⚠️  ROS2 not available: {e}")
    print("[Se3ROS2] Please ensure:")
    print("          1. ROS2 is sourced: source /opt/ros/humble/setup.bash")
    print("          2. Isaac Sim ROS2 bridge extension is installed")
    print("          3. Device module is imported AFTER AppLauncher/SimulationApp")


@configclass
class Se3ROS2Cfg:
    """Configuration for ROS2-based SE(3) teleoperation device.

    This device subscribes to ROS2 topics to receive joint states from real robots
    and converts them into actions for the Isaac Lab environment.
    """

    ee_pose_topic: str = "/ee_pose"
    """ROS2 topic for end-effector pose. Default is "/ee_pose"."""

    num_dof: int = 8
    """Number of degrees of freedom (3 pos + 4 quat + 1 gripper). Default is 8."""

    input_format: str = "euler"
    """Input format: 'euler' (roll,pitch,yaw) or 'quat' (w,x,y,z). Default is 'euler'."""

    timeout_sec: float = 2.0
    """Timeout in seconds for considering connection lost. Default is 2.0."""

    pos_scale: float = 1.0
    """Scaling factor for joint position commands. Default is 1.0."""


class Se3ROS2(DeviceBase):
    """A ROS2-based teleoperation device for SE(3) control.

    This class subscribes to ROS2 topics to receive joint states from a real robot
    and provides them as actions for Isaac Lab environments. It follows the same
    interface as Se3Keyboard and Se3SpaceMouse.
    """

    def __init__(self, cfg: Se3ROS2Cfg):
        """Initialize the ROS2 teleoperation device.

        Args:
            cfg: Configuration for the ROS2 device.

        Raises:
            RuntimeError: If ROS2 is not available or initialization fails.
        """
        if not ROS2_AVAILABLE:
            raise RuntimeError(
                "ROS2 is not available. Please ensure:\n"
                "1. Isaac Sim ROS2 bridge extension is enabled\n"
                "2. ROS2 is sourced before launching Isaac Sim"
            )

        # Store configuration
        self.cfg = cfg

        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node("isaac_lab_ros2_teleop")

        # Storage for latest pose data
        self._latest_pose_data = None
        self._last_update_time = time.time()
        self._connection_warned = False

        # Create subscriber
        self._pose_sub = self.node.create_subscription(
            Float64MultiArray, cfg.ee_pose_topic, self._pose_callback, 10
        )

        # Callback dictionary for key events (like reset)
        self._callbacks = {}

        print(f"[Se3ROS2] 🤖 Initialized ROS2 teleoperation device")
        print(f"[Se3ROS2] 📡 Subscribed to: {cfg.ee_pose_topic}")
        print(f"[Se3ROS2] 💡 Waiting for end-effector pose data...")

    def __del__(self):
        """Destructor to clean up ROS2 resources."""
        self.close()

    def __str__(self) -> str:
        """Returns: A string containing the information of the device."""
        msg = f"ROS2 Teleoperation Device:\n"
        msg += f"\tTopic: {self.cfg.ee_pose_topic}\n"
        msg += f"\tDOF: {self.cfg.num_dof}\n"
        msg += f"\tConnection: {'🟢 Active' if self._is_connected() else '🔴 Lost'}"
        return msg

    """
    Operations
    """

    def reset(self):
        """Reset the ROS2 device state.

        This clears any cached pose data and resets internal counters.
        """
        self._latest_pose_data = None
        self._last_update_time = time.time()
        self._connection_warned = False
        print("[Se3ROS2] ♻️  Device reset")

    def add_callback(self, key: str, func: Callable):
        """Add a callback function for a specific key/event.

        Args:
            key: The key/event identifier (e.g., "R" for reset).
            func: The callback function to execute.
        """
        self._callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Advance the device by one step and return the current action.

        This method spins the ROS2 node to process incoming messages and
        returns the latest end-effector pose as an action array.

        Returns:
            torch.Tensor: Action array with end-effector pose and gripper state.
                Shape: (1, 8) - [x, y, z, qw, qx, qy, qz, gripper]

        Note:
            If no messages have been received or connection is lost, returns zero action.
            Input euler angles (roll, pitch, yaw) are automatically converted to quaternion.
        """
        # Spin ROS2 node to process callbacks
        rclpy.spin_once(self.node, timeout_sec=0.0)

        # Check connection status
        if not self._is_connected():
            if not self._connection_warned:
                print(
                    f"[Se3ROS2] ⚠️  No ROS2 messages received for {self.cfg.timeout_sec}s"
                )
                print(
                    f"[Se3ROS2] Make sure real robot is publishing to: {self.cfg.ee_pose_topic}"
                )
                self._connection_warned = True
            return torch.zeros((1, self.cfg.num_dof), dtype=torch.float32)

        # Reset warning flag when connected
        if self._connection_warned and self._is_connected():
            print(f"[Se3ROS2] ✅ Connection restored")
            self._connection_warned = False

        # Return latest pose data with scaling
        # Shape: (1, num_dof) to match expected action format
        if self._latest_pose_data is not None:
            action = self._latest_pose_data * self.cfg.pos_scale
            action_2d = action.reshape(1, -1)  # Convert to 2D: (1, num_dof)
            return torch.from_numpy(action_2d).float()  # Convert to torch tensor
        else:
            return torch.zeros((1, self.cfg.num_dof), dtype=torch.float32)

    def close(self):
        """Close the ROS2 device and cleanup resources."""
        if hasattr(self, "node") and self.node is not None:
            try:
                self.node.destroy_node()
                # Note: Don't call rclpy.shutdown() here as it might be used by other nodes
                print("[Se3ROS2] Device closed")
            except Exception as e:
                print(f"[Se3ROS2] Warning during cleanup: {e}")

    """
    Internal helpers
    """

    def _is_connected(self) -> bool:
        """Check if ROS2 connection is active.

        Returns:
            bool: True if messages have been received recently, False otherwise.
        """
        time_since_update = time.time() - self._last_update_time
        has_received_data = self._latest_pose_data is not None
        return has_received_data and time_since_update < self.cfg.timeout_sec

    def _pose_callback(self, msg: Float64MultiArray):
        """Callback for end-effector pose from ROS2.

        Args:
            msg: ROS2 Float64MultiArray message.
                If input_format='euler': [x, y, z, roll, pitch, yaw, gripper] (7 elements)
                If input_format='quat': [x, y, z, qw, qx, qy, qz, gripper] (8 elements)
        """
        if self.cfg.input_format == "euler":
            # Expect 7 elements: [x, y, z, roll, pitch, yaw, gripper]
            if len(msg.data) >= 7:
                pos = np.array(msg.data[:3], dtype=np.float32)  # x, y, z
                euler = np.array(msg.data[3:6], dtype=np.float32)  # roll, pitch, yaw
                gripper = np.array([msg.data[6]], dtype=np.float32)  # gripper

                # Convert euler angles to quaternion (w, x, y, z)
                rot = Rotation.from_euler("xyz", euler, degrees=False)
                quat = rot.as_quat()  # Returns [x, y, z, w] format
                quat_wxyz = np.array(
                    [quat[3], quat[0], quat[1], quat[2]], dtype=np.float32
                )  # Convert to [w, x, y, z]

                # Assemble: [x, y, z, qw, qx, qy, qz, gripper]
                pose_data = np.concatenate([pos, quat_wxyz, gripper])
                self._latest_pose_data = pose_data
                self._last_update_time = time.time()

                # Debug output (print first message only)
                if not hasattr(self, "_first_msg_printed"):
                    print(f"[Se3ROS2] ✅ Received first message:")
                    print(f"  Input (euler):  {msg.data[:7]}")
                    print(f"  Output (quat): {pose_data}")
                    self._first_msg_printed = True
            else:
                print(
                    f"[Se3ROS2] ⚠️  Received {len(msg.data)} elements, expected 7 for euler format"
                )

        elif self.cfg.input_format == "quat":
            # Expect 8 elements: [x, y, z, qw, qx, qy, qz, gripper]
            if len(msg.data) >= 8:
                pose_data = np.array(msg.data[:8], dtype=np.float32)
                self._latest_pose_data = pose_data
                self._last_update_time = time.time()
            else:
                print(
                    f"[Se3ROS2] ⚠️  Received {len(msg.data)} elements, expected 8 for quat format"
                )

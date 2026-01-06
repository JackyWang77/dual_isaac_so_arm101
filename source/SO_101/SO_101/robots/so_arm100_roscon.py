# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO-ARM101 5-DOF robot arm for livestream.

The following configurations are available:

* :obj:`SO_ARM101_ROSCON_CFG`: SO-ARM101 robot arm configuration more adapted for sim2real.
        ->  converted from the xacro of this repository:
        https://github.com/JafarAbdi/ros2_so_arm100
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

##
# Configuration
##

SO_ARM101_ROSCON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Robots/SO-ARM101-NEW-TF2.usd",
        # usd_path=f"{TEMPLATE_ASSETS_DATA_DIR}/Robots/so_arm100_roscon/so_arm100.usd",
        activate_contact_sensors=False,  # Adjust based on need
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0),  # No rotation (w, x, y, z), keep robot upright
        pos=(0.0, 0.0, 0.0),  # Fixed comma error (0,0 -> 0.0)
        # joint_pos={
        #     "shoulder_pan_joint": 0,  # From real robot ROS2
        #     "shoulder_lift_joint": -1.745,  # From real robot ROS2, clamped to limit [-1.745, 1.745]
        #     "elbow_joint": 1.560,  # Changed to fit joint limits
        #     "wrist_pitch_joint": 0.1,  # Changed to fit joint limits
        #     "wrist_roll_joint": 0.0,  # From real robot ROS2
        #     "jaw_joint": 0.698,  # From real robot ROS2, clamped to limit [-0.175, 1.745]
        # },
        joint_pos={
            "shoulder_pan_joint": 0,  # From real robot ROS2
            "shoulder_lift_joint": 0,  # From real robot ROS2, clamped to limit [-1.745, 1.745]
            "elbow_joint": 0,  # Changed to fit joint limits
            "wrist_pitch_joint": 0,  # Changed to fit joint limits
            "wrist_roll_joint": 0,  # From real robot ROS2
            "jaw_joint": 0.4,  # From real robot ROS2, clamped to limit [-0.175, 1.745]
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Shoulder Pan      moves: ALL masses                   (~0.8kg total)
        # Shoulder Lift     moves: Everything except base       (~0.65kg)
        # Elbow             moves: Lower arm, wrist, gripper    (~0.38kg)
        # Wrist Pitch       moves: Wrist and gripper            (~0.24kg)
        # Wrist Roll        moves: Gripper assembly             (~0.14kg)
        # Jaw               moves: Only moving jaw              (~0.034kg)
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_joint", "wrist_.*"],
            effort_limit_sim=5.0,
            velocity_limit_sim=10.0,
            stiffness={
                "shoulder_pan_joint": 200.0,  # Highest - moves all mass
                "shoulder_lift_joint": 170.0,  # Slightly less than rotation
                "elbow_joint": 120.0,  # Reduced based on less mass
                "wrist_pitch_joint": 80.0,  # Reduced for less mass
                "wrist_roll_joint": 50.0,  # Low mass to move
            },
            damping={
                "shoulder_pan_joint": 80.0,
                "shoulder_lift_joint": 65.0,
                "elbow_joint": 45.0,
                "wrist_pitch_joint": 30.0,
                "wrist_roll_joint": 20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["jaw_joint"],
            effort_limit_sim=4,
            velocity_limit_sim=15,
            stiffness=40.0,  # Increased from 25.0 to 60.0 for more reliable closing
            damping=2,  # Increased from 10.0 to 20.0 for stability
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

"""Configuration of SO-ARM robot more adapted for sim2real."""

SO_ARM101_ROSCON_HIGH_PD_CFG = SO_ARM101_ROSCON_CFG.copy()
SO_ARM101_ROSCON_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = False
# [Effort] STS3215 stall torque is approximately 30kg.cm ≈ 3Nm
# Add some margin in simulation (4-5Nm) to prevent stalling due to friction
SO_ARM101_ROSCON_HIGH_PD_CFG.actuators["arm"].effort_limit_sim = 5.0
# [Velocity] Set slightly higher to avoid bottleneck, 10 rad/s is fast enough
SO_ARM101_ROSCON_HIGH_PD_CFG.actuators["arm"].velocity_limit_sim = 10.0
# [Stiffness] Calculated based on 5Hz frequency (Mass * 1000)
# With this setting, the arm has sufficient support force and won't explode
SO_ARM101_ROSCON_HIGH_PD_CFG.actuators["arm"].stiffness = {
    "shoulder_pan_joint": 800.0,  # Mass ~0.8kg  -> 0.8 * 1000
    "shoulder_lift_joint": 650.0,  # Mass ~0.65kg -> 0.65 * 1000
    "elbow_joint": 400.0,         # Mass ~0.38kg -> 0.38 * 1000
    "wrist_pitch_joint": 240.0,   # Mass ~0.24kg -> 0.24 * 1000
    "wrist_roll_joint": 140.0,    # Mass ~0.14kg -> 0.14 * 1000
}
# [Damping] Calculated based on formula D = 2 * sqrt(Mass * Stiffness)
# This is the perfect value to eliminate oscillations
SO_ARM101_ROSCON_HIGH_PD_CFG.actuators["arm"].damping = {
    "shoulder_pan_joint": 50.0,  # 2 * sqrt(0.8 * 800) = 2 * 25.2 = 50.4
    "shoulder_lift_joint": 41.0,  # 2 * sqrt(0.65 * 650) = 2 * 20.5 = 41
    "elbow_joint": 25.0,         # 2 * sqrt(0.38 * 400) = 2 * 12.3 = 24.6
    "wrist_pitch_joint": 15.0,   # 2 * sqrt(0.24 * 240) = 2 * 7.6 = 15.2
    "wrist_roll_joint": 10.0,    # 2 * sqrt(0.14 * 140) = 2 * 4.4 = 8.8
}
# Gripper configuration
SO_ARM101_ROSCON_HIGH_PD_CFG.actuators["gripper"].effort_limit_sim = 3.0  # Gripper doesn't need too much force, 3Nm is enough
SO_ARM101_ROSCON_HIGH_PD_CFG.actuators["gripper"].velocity_limit_sim = 10.0
# Gripper has very small mass (0.034kg), set frequency slightly higher (6-7Hz) for compliant grasping
# Or keep 5Hz for stability. Here we use 5Hz:
SO_ARM101_ROSCON_HIGH_PD_CFG.actuators["gripper"].stiffness = 40.0  # Mass 0.034 * 1000 = 34 -> rounded to 40
# Damping calculation: 2 * sqrt(0.034 * 40) = 2 * 1.16 = 2.3
SO_ARM101_ROSCON_HIGH_PD_CFG.actuators["gripper"].damping = 2  # Slightly higher (2.5) to prevent any tiny jitter
# Physical contact parameters (must add, otherwise can't grasp)
SO_ARM101_ROSCON_HIGH_PD_CFG.actuators["gripper"].friction = 0.1
SO_ARM101_ROSCON_HIGH_PD_CFG.actuators["gripper"].armature = 0.005  # Must add this virtual inertia for very small mass objects!
"""Configuration of SO-ARM robot with stiffer PD control."""

# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    initial_height: float = 0.015,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object. Reward scales with lift distance.

    Args:
        env: The environment.
        minimal_height: The target height to reach (reward = 1.0 at this height).
        initial_height: The initial height of the object (reward = 0.0 at this height).
        object_cfg: The object configuration.

    Returns:
        Reward in range [0.0, 2.0] based on how high the object is lifted.
        Reward increases linearly from 0.0 (at initial_height) to 1.0 (at minimal_height),
        and continues to increase beyond minimal_height (capped at 2.0 for heights >= minimal_height * 2).
    """
    object: RigidObject = env.scene[object_cfg.name]
    current_height = object.data.root_pos_w[:, 2]

    # Calculate lift distance from initial height
    lift_distance = current_height - initial_height

    # Normalize reward: 0.0 at initial_height, 1.0 at minimal_height
    # Allow reward to continue increasing beyond minimal_height (capped at 2.0)
    target_lift_distance = minimal_height - initial_height
    if target_lift_distance > 0:
        # Linear scaling: reward = lift_distance / target_lift_distance
        # Cap at 2.0 to prevent unbounded rewards
        reward = torch.clamp(lift_distance / target_lift_distance, 0.0, 2.0)
    else:
        # Fallback if target <= initial (shouldn't happen normally)
        reward = torch.where(current_height > minimal_height, 1.0, 0.0)

    return reward


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


# ============================================================
# Grasp Reward 
# ============================================================
def grasp_reward(
    env: ManagerBasedRLEnv,
    threshold_distance: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward for correct gripper behavior:
    - 当EE接近物体时，gripper关闭 -> 正奖励
    - 当EE远离物体时，gripper打开 -> 小正奖励（准备状态）
    - 不正确的gripper状态 -> 0或负奖励

    Args:
        threshold_distance: EE需要离物体多近才算"准备好抓" (默认0.02m)
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # 计算距离
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)

    # 获取gripper状态
    # 方法1: 从action获取（如果可用）
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
        gripper_action = env.action_manager.action[:, -1]  # 假设最后一维是gripper
        # gripper_action > 0 表示打开, < 0 表示关闭
        gripper_closed = gripper_action < 0.0
    else:
        # 方法2: 从joint_pos获取（gripper通常是最后一个joint）
        gripper_pos = robot.data.joint_pos[:, -1]
        # 假设gripper_pos < 0.5 表示关闭（根据实际机器人调整）
        gripper_closed = gripper_pos < 0.5

    is_close = distance < threshold_distance  # EE是否接近物体

    reward = torch.zeros_like(distance)

    # 情况1: 接近物体且gripper关闭 -> 大奖励 (正确的抓取行为)
    reward = torch.where(
        is_close & gripper_closed,
        torch.ones_like(reward) * 1.0,
        reward,
    )

    # 情况2: 远离物体且gripper打开 -> 小奖励 (准备接近)
    reward = torch.where(
        ~is_close & ~gripper_closed,
        torch.ones_like(reward) * 0.1,
        reward,
    )

    # 情况3: 接近物体但gripper打开 -> 0 (应该关闭但没关)
    # 情况4: 远离物体但gripper关闭 -> 0 (不必要的关闭)

    return reward


# ============================================================
# Object Grasped Reward - 检测是否真正抓住
# ============================================================
def object_grasped_reward(
    env: ManagerBasedRLEnv,
    lift_threshold: float = 0.02,
    grasp_distance_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    initial_height: float = 0.015,
) -> torch.Tensor:
    """
    Reward for successfully grasping and holding the object.
    物体被抬起 + EE和物体一起移动 = 成功抓取

    Args:
        lift_threshold: 物体需要抬起多高才算被抓住 (默认0.02m)
        grasp_distance_threshold: 抓住时EE和物体的距离阈值 (默认0.05m)
        initial_height: 物体的初始高度
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # 物体当前高度
    object_height = object.data.root_pos_w[:, 2]

    # 物体是否被抬起
    is_lifted = object_height > (initial_height + lift_threshold)

    # EE和物体的距离 (抓住时应该很近)
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)

    is_holding = distance < grasp_distance_threshold  # 抓住时距离应该很近

    # 只有物体被抬起且EE在附近，才算成功抓取
    grasped = is_lifted & is_holding

    return grasped.float()


# ============================================================
# Gripper Action Penalty - 防止频繁开合
# ============================================================
def gripper_action_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize rapid gripper changes to encourage smooth behavior.
    """
    # 获取当前action
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
        current_action = env.action_manager.action
        curr_gripper = current_action[:, -1]  # 假设最后一维是gripper
    else:
        # 如果没有action_manager，从joint_pos获取
        robot: Articulation = env.scene[robot_cfg.name]
        curr_gripper = robot.data.joint_pos[:, -1]

    # 如果有上一步action的记录
    if not hasattr(env, "_prev_gripper_action"):
        env._prev_gripper_action = curr_gripper.clone()
        return torch.zeros(env.num_envs, device=env.device)

    prev_gripper = env._prev_gripper_action

    # 计算gripper变化量
    gripper_change = torch.abs(curr_gripper - prev_gripper)

    # 更新记录
    env._prev_gripper_action = curr_gripper.clone()

    # 返回负奖励 (变化越大惩罚越大)
    return -gripper_change * 0.1


# ============================================================
# Success Bonus - 稀疏大奖励
# ============================================================
def task_success_bonus(
    env: ManagerBasedRLEnv,
    bonus: float = 10.0,
    target_height: float = 0.15,
    hold_time_steps: int = 10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Large sparse bonus for completing the task.
    物体保持在目标高度以上一定时间 = 成功

    Args:
        bonus: 成功时的奖励值 (默认10.0)
        target_height: 目标高度 (默认0.1m)
        hold_time_steps: 需要保持多少步 (默认10步)
    """
    object: RigidObject = env.scene[object_cfg.name]
    current_height = object.data.root_pos_w[:, 2]

    is_at_target = current_height > target_height

    # 追踪保持时间
    if not hasattr(env, "_hold_counter"):
        env._hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # 在目标高度以上时增加计数
    env._hold_counter = torch.where(
        is_at_target,
        env._hold_counter + 1,
        torch.zeros_like(env._hold_counter),  # 掉下来就重置
    )

    # 达到保持时间给予bonus
    success = env._hold_counter >= hold_time_steps

    # 给予bonus后重置计数 (避免重复奖励)
    env._hold_counter = torch.where(
        success,
        torch.zeros_like(env._hold_counter),
        env._hold_counter,
    )

    return success.float() * bonus

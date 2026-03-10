#!/usr/bin/env python3
"""
Play (inference) with RL-trained residual policy on dual arm cube stack.

Loads:
1. Pretrained BC backbone (DualArmDisentangledPolicyGated or GraphUnetPolicy)
2. RL residual policy checkpoint (actor, critic, z_adapter, alpha_net)

Usage:
    python scripts/graph_dit_rl/play_rl.py \
        --pretrained_checkpoint ./logs/gated_small/best_model.pt \
        --rl_checkpoint ./logs/dual_arm_rl/best_model.pt \
        --num_envs 64 --num_batches 2
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play RL-trained dual arm stack policy")
parser.add_argument("--task", type=str, default="SO-ARM101-Dual-Cube-Stack-Play-v0")
parser.add_argument("--pretrained_checkpoint", type=str, required=True, help="BC backbone checkpoint")
parser.add_argument("--rl_checkpoint", type=str, required=True, help="RL residual policy checkpoint")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_batches", type=int, default=2, help="Number of batches to evaluate (total episodes = num_envs * num_batches)")

AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import SO_101.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from SO_101.policies.graph_unet_policy import GraphUnetPolicy, UnetPolicy
from SO_101.policies.dual_arm_unet_policy import DualArmDisentangledPolicy
from SO_101.policies.dual_arm_unet_policy_gated import DualArmDisentangledPolicyGated
from SO_101.policies.graph_unet_residual_rl_policy import (
    GraphUnetBackboneAdapter,
    GraphUnetResidualRLPolicy,
)


def main():
    device = getattr(args, "device", "cuda")
    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================
    # 1. Load backbone (auto-detect type from checkpoint)
    # ============================================================
    ckpt = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", None)
    del ckpt

    if ckpt_cfg is not None and getattr(ckpt_cfg, "arm_action_dim", None) is not None:
        ckpt_preview = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
        if "graph_gate_logit" in ckpt_preview.get("model_state_dict", {}):
            BackboneClass = DualArmDisentangledPolicyGated
        else:
            BackboneClass = DualArmDisentangledPolicy
        del ckpt_preview
    else:
        BackboneClass = GraphUnetPolicy

    print(f"[Play RL] Loading backbone: {BackboneClass.__name__}")
    backbone_policy = BackboneClass.load(args.pretrained_checkpoint, device=device)
    backbone_policy.eval()
    for p in backbone_policy.parameters():
        p.requires_grad = False

    backbone = GraphUnetBackboneAdapter(backbone_policy)

    # ============================================================
    # 2. Load RL policy
    # ============================================================
    print(f"[Play RL] Loading RL checkpoint: {args.rl_checkpoint}")
    rl_policy = GraphUnetResidualRLPolicy.load(args.rl_checkpoint, backbone=backbone, device=device)
    rl_policy.eval()
    rl_policy.to(device)

    # Load normalization stats from BC checkpoint
    bc_ckpt = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
    obs_stats = bc_ckpt.get("obs_stats", None)
    node_stats = bc_ckpt.get("node_stats", None)
    action_stats = bc_ckpt.get("action_stats", None)

    obs_mean = obs_std = None
    if obs_stats is not None:
        obs_mean = torch.tensor(obs_stats["mean"], device=device, dtype=torch.float32)
        obs_std = torch.tensor(obs_stats["std"], device=device, dtype=torch.float32)

    ee_mean = ee_std = obj_mean = obj_std = None
    if node_stats is not None:
        ee_mean = torch.tensor(node_stats.get("ee_node_mean", node_stats.get("mean", None)), device=device, dtype=torch.float32) if "ee_node_mean" in node_stats or "mean" in node_stats else None
        ee_std = torch.tensor(node_stats.get("ee_node_std", node_stats.get("std", None)), device=device, dtype=torch.float32) if "ee_node_std" in node_stats or "std" in node_stats else None
        obj_mean = torch.tensor(node_stats.get("object_node_mean", node_stats.get("mean", None)), device=device, dtype=torch.float32) if "object_node_mean" in node_stats else None
        obj_std = torch.tensor(node_stats.get("object_node_std", node_stats.get("std", None)), device=device, dtype=torch.float32) if "object_node_std" in node_stats else None

    action_mean = action_std = None
    if action_stats is not None:
        action_mean = torch.tensor(action_stats["mean"], device=device, dtype=torch.float32)
        action_std = torch.tensor(action_stats["std"], device=device, dtype=torch.float32)

    rl_policy.set_normalization_stats(
        ee_node_mean=ee_mean, ee_node_std=ee_std,
        object_node_mean=obj_mean, object_node_std=obj_std,
        action_mean=action_mean, action_std=action_std,
    )
    del bc_ckpt

    # ============================================================
    # 3. Create environment
    # ============================================================
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    num_envs = args.num_envs

    cfg = rl_policy.cfg
    is_dual_arm = (cfg.action_dim == 12)
    action_dim = cfg.action_dim

    # env_origins for coordinate conversion
    _base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    _env_origins = _base_env.scene.env_origins.clone()

    # obs structure for position conversion
    obs_structure = cfg.obs_structure
    SUCCESS_HEIGHT = 0.10

    # Determine OBJ_HEIGHT_IDX
    if obs_structure is not None and "cube_1_pos" in obs_structure:
        OBJ_HEIGHT_IDX = obs_structure["cube_1_pos"][0] + 2
    elif obs_structure is not None and "object_position" in obs_structure:
        OBJ_HEIGHT_IDX = obs_structure["object_position"][0] + 2
    else:
        OBJ_HEIGHT_IDX = 14

    print(f"[Play RL] action_dim={action_dim}, dual_arm={is_dual_arm}, OBJ_HEIGHT_IDX={OBJ_HEIGHT_IDX}")

    # ============================================================
    # 4. Run episodes
    # ============================================================
    rl_policy.init_env_buffers(num_envs)

    all_successes = []

    for batch_idx in range(args.num_batches):
        obs, _ = env.reset()
        if isinstance(obs, dict):
            obs = torch.cat([v.float() for v in obs.values()], dim=-1) if isinstance(list(obs.values())[0], torch.Tensor) else torch.tensor(list(obs.values()), device=device).float()
        obs = obs.to(device).float()

        # Convert world positions to local
        if obs_structure is not None:
            pos_keys = ["left_ee_position", "right_ee_position", "cube_1_pos", "cube_2_pos"]
            for pk in pos_keys:
                if pk in obs_structure:
                    start = obs_structure[pk][0]
                    obs[:, start:start + 3] -= _env_origins

        ep_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        ep_successes = torch.zeros(num_envs, dtype=torch.float32, device=device)
        ema_joints = None

        for step in range(400):  # Max episode steps
            if ep_done.all():
                break

            # Normalize obs
            if obs_mean is not None:
                obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
            else:
                obs_norm = obs

            with torch.no_grad():
                action, info = rl_policy.act(
                    obs_raw=obs,
                    obs_norm=obs_norm,
                    deterministic=True,
                )

            # Process action for environment
            action_for_sim = action.clone()

            if is_dual_arm and action_dim == 12:
                # EMA for joints
                joint_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
                joints = action_for_sim[:, joint_indices]
                if ema_joints is None:
                    ema_joints = joints.clone()
                else:
                    ema_joints = 0.7 * joints + 0.3 * ema_joints
                for i, idx in enumerate(joint_indices):
                    action_for_sim[:, idx] = ema_joints[:, i]

                # Gripper mapping
                for g_idx in [5, 11]:
                    joint_val = action_for_sim[:, g_idx].float()
                    action_for_sim[:, g_idx] = 12.5 * joint_val + 0.5

                # Reorder [left_6, right_6] → [right_6, left_6]
                action_for_sim = torch.cat([action_for_sim[:, 6:12], action_for_sim[:, 0:6]], dim=1)
            else:
                # Single arm
                action_for_sim[:, 5] = torch.where(
                    action_for_sim[:, 5] > -0.2, 1.0, -1.0
                )

            next_obs, reward, terminated, truncated, env_info = env.step(action_for_sim)
            done = terminated | truncated

            # Check success for done envs
            for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                if not ep_done[i]:
                    is_truncated = bool(truncated[i])
                    if is_truncated:
                        ep_successes[i] = 0.0
                    else:
                        obj_h = obs[i, OBJ_HEIGHT_IDX].item()
                        ep_successes[i] = float(obj_h >= SUCCESS_HEIGHT)
                    ep_done[i] = True

            # Process next obs
            obs = next_obs
            if isinstance(obs, dict):
                obs = torch.cat([v.float() for v in obs.values()], dim=-1)
            obs = obs.to(device).float()

            # Convert positions to local
            if obs_structure is not None:
                pos_keys = ["left_ee_position", "right_ee_position", "cube_1_pos", "cube_2_pos"]
                for pk in pos_keys:
                    if pk in obs_structure:
                        start = obs_structure[pk][0]
                        obs[:, start:start + 3] -= _env_origins

            # Reset done envs' RL state
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_envs) > 0:
                rl_policy.reset_envs(done_envs)
                if ema_joints is not None:
                    ema_joints[done_envs] = 0

        # Mark timeout envs as failed
        for i in range(num_envs):
            if not ep_done[i]:
                ep_successes[i] = 0.0

        batch_sr = ep_successes.mean().item()
        all_successes.extend(ep_successes.tolist())
        print(f"[Batch {batch_idx + 1}/{args.num_batches}] SR: {batch_sr * 100:.1f}% ({int(ep_successes.sum())}/{num_envs})")

    total_sr = np.mean(all_successes)
    total_eps = len(all_successes)
    print(f"\n[TOTAL] SR: {total_sr * 100:.1f}% ({int(sum(all_successes))}/{total_eps})")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

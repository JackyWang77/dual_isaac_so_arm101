#!/usr/bin/env python3
"""
Playback script for trained Graph-Unet + Residual RL Policy (residual RL is Unet-only).

Usage:
    ./isaaclab.sh -p scripts/graph_dit_rl/play_graph_rl.py \
        --task SO-ARM101-Lift-Cube-v0 \
        --checkpoint ./logs/graph_dit_rl/policy_final.pt \
        --pretrained_checkpoint ./logs/graph_dit/best_model.pt \
        --num_envs 64
"""

from isaaclab.app import AppLauncher

# CLI args for AppLauncher (parse early to get headless flag)
import sys
import argparse
parser_launcher = argparse.ArgumentParser(description="Play Graph-Unet + Residual RL Policy")
AppLauncher.add_app_launcher_args(parser_launcher)
args_launcher, _ = parser_launcher.parse_known_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_launcher)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np

import gymnasium as gym
import SO_101.tasks  # noqa: F401  # Register environments
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from SO_101.policies.graph_unet_policy import UnetPolicy, GraphUnetPolicy
from SO_101.policies.graph_unet_residual_rl_policy import (
    GraphUnetBackboneAdapter,
    GraphUnetResidualRLPolicy,
)


def _detect_checkpoint_type(path: str) -> str:
    """Detect if checkpoint is IL (GraphUnetPolicy) or RL (GraphUnetResidualRLPolicy)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg")
    if cfg is None:
        return "unknown"
    if hasattr(cfg, "node_dim"):
        return "il"
    return "rl"


def play_graph_rl_policy(
    task_name: str,
    checkpoint_path: str,
    pretrained_checkpoint: str,
    num_envs: int = 64,
    num_episodes: int = 10,
    device: str = "cuda",
    deterministic: bool = True,
    policy_type: str = "unet",
):
    """Play trained Graph-Unet + Residual RL policy."""
    # Auto-detect and swap if user passed (IL, RL) instead of (RL, IL)
    t1, t2 = _detect_checkpoint_type(checkpoint_path), _detect_checkpoint_type(pretrained_checkpoint)
    if t1 == "il" and t2 == "rl":
        checkpoint_path, pretrained_checkpoint = pretrained_checkpoint, checkpoint_path
        print("[Play] Detected swapped args (IL first, RL second) - auto-corrected to (RL, IL)")

    print(f"[Play] ===== Residual RL Policy Playback (Graph-Unet) =====")
    print(f"[Play] Task: {task_name}")
    print(f"[Play] RL Checkpoint: {checkpoint_path}")
    print(f"[Play] Graph-Unet Checkpoint: {pretrained_checkpoint}")
    print(f"[Play] Num envs: {num_envs}")
    OBJ_HEIGHT_IDX = 14
    SUCCESS_HEIGHT = 0.10

    # Load pretrained backbone
    PolicyClass = GraphUnetPolicy if policy_type == "graph_unet" else UnetPolicy
    print(f"\n[Play] Loading pretrained backbone ({PolicyClass.__name__}): {pretrained_checkpoint}")
    backbone_policy = PolicyClass.load(pretrained_checkpoint, device=device)
    backbone_policy.eval()
    for p in backbone_policy.parameters():
        p.requires_grad = False
    print(f"[Play] Graph-Unet loaded and frozen")

    # Create backbone adapter
    backbone_adapter = GraphUnetBackboneAdapter(backbone_policy)

    # Load RL policy
    print(f"\n[Play] Loading RL policy: {checkpoint_path}")
    policy = GraphUnetResidualRLPolicy.load(checkpoint_path, backbone=backbone_adapter, device=device)
    policy.num_diffusion_steps = 10  # must match train
    policy.eval()
    print(f"[Play] RL policy loaded (diffusion_steps=10)")

    # Load normalization stats from Graph-Unet checkpoint (same as train script)
    checkpoint = torch.load(pretrained_checkpoint, map_location=device, weights_only=False)
    
    # Load obs stats
    obs_stats = checkpoint.get("obs_stats", None)
    obs_mean, obs_std = None, None
    if obs_stats is not None:
        if isinstance(obs_stats["mean"], np.ndarray):
            obs_mean = torch.from_numpy(obs_stats["mean"]).squeeze().to(device)
            obs_std = torch.from_numpy(obs_stats["std"]).squeeze().to(device)
        else:
            obs_mean = obs_stats["mean"].squeeze().to(device)
            obs_std = obs_stats["std"].squeeze().to(device)
        print(f"[Play] Loaded obs normalization stats")
    
    # Load node stats
    node_stats = checkpoint.get("node_stats", None)
    ee_node_mean, ee_node_std = None, None
    object_node_mean, object_node_std = None, None
    if node_stats is not None:
        if isinstance(node_stats["ee_mean"], np.ndarray):
            ee_node_mean = torch.from_numpy(node_stats["ee_mean"]).squeeze().to(device)
            ee_node_std = torch.from_numpy(node_stats["ee_std"]).squeeze().to(device)
            object_node_mean = torch.from_numpy(node_stats["object_mean"]).squeeze().to(device)
            object_node_std = torch.from_numpy(node_stats["object_std"]).squeeze().to(device)
        else:
            ee_node_mean = node_stats["ee_mean"].squeeze().to(device)
            ee_node_std = node_stats["ee_std"].squeeze().to(device)
            object_node_mean = node_stats["object_mean"].squeeze().to(device)
            object_node_std = node_stats["object_std"].squeeze().to(device)
        print(f"[Play] Loaded node normalization stats")
    
    # Load action stats
    action_stats = checkpoint.get("action_stats", None)
    action_mean, action_std = None, None
    if action_stats is not None:
        if isinstance(action_stats["mean"], np.ndarray):
            action_mean = torch.from_numpy(action_stats["mean"]).squeeze().to(device)
            action_std = torch.from_numpy(action_stats["std"]).squeeze().to(device)
        else:
            action_mean = action_stats["mean"].squeeze().to(device)
            action_std = action_stats["std"].squeeze().to(device)
        print(f"[Play] Loaded action normalization stats")
    
    # Load joint stats
    joint_stats = checkpoint.get("joint_stats", None)
    joint_mean, joint_std = None, None
    if joint_stats is not None:
        if isinstance(joint_stats["mean"], np.ndarray):
            joint_mean = torch.from_numpy(joint_stats["mean"]).squeeze().to(device)
            joint_std = torch.from_numpy(joint_stats["std"]).squeeze().to(device)
        else:
            joint_mean = joint_stats["mean"].squeeze().to(device)
            joint_std = joint_stats["std"].squeeze().to(device)
        print(f"[Play] Loaded joint normalization stats")
    
    # Set normalization stats in policy (same as train_graph_rl.py)
    policy.set_normalization_stats(
        ee_node_mean=ee_node_mean,
        ee_node_std=ee_node_std,
        object_node_mean=object_node_mean,
        object_node_std=object_node_std,
        action_mean=action_mean,
        action_std=action_std,
    )
    print(f"[Play] Set normalization stats in policy")

    # Create environment
    print(f"\n[Play] Creating environment...")
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    env = gym.make(task_name, cfg=env_cfg)

    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space

    print(f"[Play] Observation space: {obs_space}")
    print(f"[Play] Action space: {action_space}")

    # Initialize policy buffers
    policy.init_env_buffers(num_envs)

    # History buffers for Graph-DiT (optimized: single tensor per history type)
    action_history_length = getattr(backbone_policy.cfg, "action_history_length", 4)
    # NOTE: GraphDiT expects 6 dimensions for joint states (was trained with 6)
    # The residual RL config's joint_dim=5 is only for EMA smoothing, not for joint state extraction
    graph_dit_joint_dim = 6  # Backbone (Unet) expects 6 for joint states
    action_dim = policy.cfg.action_dim if hasattr(policy, 'cfg') and hasattr(policy.cfg, 'action_dim') else 6
    
    # Use single tensor [num_envs, history_len, dim] for efficiency
    action_history = torch.zeros(num_envs, action_history_length, action_dim, device=device)
    ee_node_history = torch.zeros(num_envs, action_history_length, 7, device=device)
    object_node_history = torch.zeros(num_envs, action_history_length, 7, device=device)
    joint_state_history = torch.zeros(num_envs, action_history_length, graph_dit_joint_dim, device=device)

    # Reset environment
    obs, info = env.reset()
    obs = _process_obs(obs, device)

    # Stats (tensor for per-env tracking; lists for final stats like play.py)
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    episode_success = []  # list of bool for SR(100ep)
    episode_rewards_list = []  # list of float for mean_reward

    print(f"\n[Play] Starting playback (deterministic={deterministic})...")
    print(f"[Play] Press Ctrl+C to stop\n")

    # EMA smoothing for joints (gripper excluded); 1.0=no smoothing (graph_unet prefers raw)
    ema_alpha = 1.0
    ema_smoothed_joints = None

    # Main loop
    step_count = 0
    try:
        while simulation_app.is_running() and len(episode_success) < num_episodes:
            # CRITICAL: Normalize observations (same as train_graph_rl.py)
            if obs_mean is not None and obs_std is not None:
                obs_norm = (obs - obs_mean) / obs_std
            else:
                obs_norm = obs
            
            # Extract current node features and joint states from obs
            ee_node_current, object_node_current = policy._extract_nodes_from_obs(obs)
            # NOTE: GraphDiT expects 6 dimensions (was trained with 6), regardless of residual RL config
            joint_states_current = obs[:, :6]  # [B, 6] - always 6 for GraphDiT compatibility
            
            # CRITICAL: Normalize node and joint histories (same as train_graph_rl.py)
            # History tensors are already in batch format [num_envs, history_len, dim]
            action_history_norm = action_history  # [B, H, action_dim] - already normalized (stored as normalized)
            
            # Normalize node histories
            ee_node_history_norm = ee_node_history.clone()  # [B, H, 7]
            object_node_history_norm = object_node_history.clone()  # [B, H, 7]
            if ee_node_mean is not None and ee_node_std is not None:
                ee_node_history_norm = (ee_node_history_norm - ee_node_mean) / ee_node_std
            if object_node_mean is not None and object_node_std is not None:
                object_node_history_norm = (object_node_history_norm - object_node_mean) / object_node_std
            
            # Normalize joint history
            joint_state_history_norm = joint_state_history.clone()  # [B, H, 6]
            if joint_mean is not None and joint_std is not None:
                joint_state_history_norm = (joint_state_history_norm - joint_mean) / joint_std
            
            # Get action from policy
            with torch.no_grad():
                action, policy_info = policy.act(
                    obs_raw=obs,
                    obs_norm=obs_norm,  # Use normalized obs
                    action_history=action_history_norm,  # [num_envs, history_len, action_dim] - normalized
                    ee_node_history=ee_node_history_norm,  # [num_envs, history_len, 7] - normalized
                    object_node_history=object_node_history_norm,  # [num_envs, history_len, 7] - normalized
                    joint_states_history=joint_state_history_norm,  # [num_envs, history_len, joint_dim] - normalized
                    deterministic=deterministic
                )

            # Clip action to action space bounds
            if hasattr(env, 'action_space'):
                if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                    action = torch.clamp(
                        action,
                        torch.tensor(env.action_space.low, device=action.device, dtype=action.dtype),
                        torch.tensor(env.action_space.high, device=action.device, dtype=action.dtype)
                    )

            # EMA smoothing for joints only (gripper excluded)
            if action.shape[-1] >= 6:
                joints = action[:, :5]
                gripper_continuous = action[:, 5:6]
                if ema_smoothed_joints is None:
                    ema_smoothed_joints = joints.clone()
                else:
                    ema_smoothed_joints = ema_alpha * joints + (1 - ema_alpha) * ema_smoothed_joints
                action = torch.cat([ema_smoothed_joints, gripper_continuous], dim=-1)

            # Binarize gripper action (same as play.py)
            if action.shape[-1] >= 6:
                action[:, 5] = torch.where(
                    action[:, 5] > -0.2,
                    torch.tensor(1.0, device=action.device, dtype=action.dtype),
                    torch.tensor(-1.0, device=action.device, dtype=action.dtype)
                )

            # Step environment
            next_obs, reward, terminated, truncated, env_info = env.step(action)
            done = terminated | truncated

            # Process
            next_obs = _process_obs(next_obs, device)
            reward = reward.to(device).float()
            done = done.to(device).float()

            # Update history buffers
            # NOTE: Graph-DiT's action is in normalized space for history
            # So we can directly use action for history
            action_for_history = action
            
            # Roll all histories: shift left by 1, new data goes to last position
            action_history = torch.roll(action_history, -1, dims=1)
            action_history[:, -1, :] = action_for_history  # [num_envs, action_dim]
            
            ee_node_history = torch.roll(ee_node_history, -1, dims=1)
            ee_node_history[:, -1, :] = ee_node_current  # [num_envs, 7]
            
            object_node_history = torch.roll(object_node_history, -1, dims=1)
            object_node_history[:, -1, :] = object_node_current  # [num_envs, 7]
            
            joint_state_history = torch.roll(joint_state_history, -1, dims=1)
            joint_state_history[:, -1, :] = joint_states_current  # [num_envs, joint_dim]

            # Track stats
            episode_rewards += reward
            episode_lengths += 1

            # Handle resets: Isaac Lab 在 reset 之后才计算 obs，所以 next_obs 对 done env 是 reset 后的新 episode 初始 obs（cube 在地面）
            # 必须用 obs（step 前 = 终止前最后一帧）判断 success
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_envs) > 0:
                policy.reset_envs(done_envs)
                for i in done_envs.tolist():
                    obj_height = obs[i, OBJ_HEIGHT_IDX].item()  # obs = 终止前最后一帧
                    is_truncated = bool(truncated[i].item() if truncated.dim() > 0 else truncated.item())
                    if is_truncated:
                        is_success = False
                    else:
                        is_success = obj_height >= SUCCESS_HEIGHT
                    episode_success.append(is_success)
                    episode_rewards_list.append(episode_rewards[i].item())
                    episode_count = len(episode_success)
                    status = "✅" if is_success else "❌"
                    sr = sum(episode_success) / len(episode_success) * 100.0
                    print(f"[Play] Ep {episode_count:3d} h={obj_height:.3f}m {status} | SR={sr:.1f}%")

                # Vectorized reset: zero out all done environments at once
                action_history[done_envs] = 0
                ee_node_history[done_envs] = 0
                object_node_history[done_envs] = 0
                joint_state_history[done_envs] = 0
                episode_rewards[done_envs] = 0
                episode_lengths[done_envs] = 0
                if ema_smoothed_joints is not None:
                    ema_smoothed_joints[done_envs] = 0

            obs = next_obs
            step_count += 1

            # Print progress (same as play graph unet)
            if step_count % 100 == 0 and episode_success:
                n = len(episode_success)
                sr = sum(episode_success) / n * 100.0
                last_100 = episode_success[-100:] if n >= 100 else episode_success
                sr_100 = sum(last_100) / len(last_100) * 100.0 if last_100 else 0.0
                print(f"[Play] Ep {n}/{num_episodes} | SR={sr:5.1f}% (100ep:{sr_100:5.1f}%) [{n}ep]")

    except KeyboardInterrupt:
        print("\n[Play] Interrupted by user")

    # Final statistics (same format as play graph unet)
    if episode_success:
        n = len(episode_success)
        sr_final = sum(episode_success) / n * 100.0
        last_100 = episode_success[-100:] if n >= 100 else episode_success
        sr_100_final = sum(last_100) / len(last_100) * 100.0 if last_100 else 0.0
        mean_reward = sum(episode_rewards_list) / len(episode_rewards_list)
        print(f"\n[Play] ===== Final Statistics =====")
        print(
            f"[Play] SR={sr_final:.1f}% (100ep:{sr_100_final:.1f}%) [{n}ep] | "
            f"Rew_mean={mean_reward:.1f} | {sum(episode_success)}/{n} success"
        )
    print(f"\n[Play] Playback completed!")

    # Cleanup
    env.close()


def _process_obs(obs, device: str) -> torch.Tensor:
    """处理 observation"""
    if isinstance(obs, dict):
        if "policy" in obs:
            obs = obs["policy"]
        else:
            obs = torch.cat([v.flatten(start_dim=1) for v in obs.values()], dim=-1)
    return obs.to(device).float()


def main():
    parser = argparse.ArgumentParser(description="Play Graph-Unet + Residual RL Policy")
    parser.add_argument("--headless", action="store_true", help="(AppLauncher) No display window")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--checkpoint", type=str, required=True, help="RL policy checkpoint path")
    parser.add_argument(
        "--pretrained_checkpoint", type=str, required=True,
        help="Pretrained Graph-Unet checkpoint (residual RL is Unet-only)",
    )
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic actions"
    )
    parser.add_argument(
        "--policy_type", type=str, default="unet",
        choices=["unet", "graph_unet"],
        help="Policy class for the pretrained backbone (default: unet)",
    )

    args = parser.parse_args()

    play_graph_rl_policy(
        task_name=args.task,
        checkpoint_path=args.checkpoint,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        device=args.device,
        deterministic=args.deterministic,
        policy_type=args.policy_type,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()

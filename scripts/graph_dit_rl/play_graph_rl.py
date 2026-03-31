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

import math
import numpy as np

import gymnasium as gym
import SO_101.tasks  # noqa: F401  # Register environments
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from SO_101.policies.graph_unet_policy import UnetPolicy, GraphUnetPolicy
from SO_101.policies.dual_arm_unet_policy import DualArmDisentangledPolicy
from SO_101.policies.dual_arm_unet_policy_gated import DualArmDisentangledPolicyGated
from SO_101.policies.graph_unet_residual_rl_policy import (
    GraphUnetBackboneAdapter,
    GraphUnetResidualRLPolicy,
)
from stacking_funnel import stacking_funnel_mask_from_cubes


def _detect_checkpoint_type(path: str) -> str:
    """Detect if checkpoint is IL (GraphUnetPolicy) or RL (GraphUnetResidualRLPolicy)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg")
    if cfg is None:
        return "unknown"
    if hasattr(cfg, "node_dim"):
        return "il"
    return "rl"


def _check_position_success(obs: torch.Tensor, env_idx: int, cfg) -> bool:
    """Position-based success check with gripper open requirement."""
    s = getattr(cfg, "obs_structure", None)
    if s is None:
        return False
    if "cube_1_pos" in s and "cube_2_pos" in s:
        c1 = obs[env_idx, s["cube_1_pos"][0]:s["cube_1_pos"][1]]
        c2 = obs[env_idx, s["cube_2_pos"][0]:s["cube_2_pos"][1]]
        z_diff_a = torch.abs((c1[2] - c2[2]) - 0.018)
        xy_dist_a = torch.norm(c1[:2] - c2[:2])
        z_diff_b = torch.abs((c2[2] - c1[2]) - 0.018)
        xy_dist_b = torch.norm(c2[:2] - c1[:2])
        stacked = bool(
            (z_diff_a < 0.003 and xy_dist_a < 0.009) or
            (z_diff_b < 0.003 and xy_dist_b < 0.009)
        )
        if not stacked:
            return False
        # Check right gripper is open (open ≈ 0, closed ≈ -0.36, threshold -0.1)
        if "right_joint_pos" in s:
            right_end = s["right_joint_pos"][1]
            right_gripper = obs[env_idx, right_end - 1].item()  # last joint = gripper
            if right_gripper < -0.1:  # gripper still closed
                return False
        return True
    return False


def _setup_success_detection(env):
    """Set up termination manager refs ONCE (matching play.py approach)."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    tm = base.termination_manager
    term_names = list(tm._term_names)
    success_idx = None
    for sname in ("success", "stack_success"):
        if sname in term_names:
            success_idx = term_names.index(sname)
            break
    drop_indices = []
    for dname in ("object_dropping", "cube_1_dropping", "cube_2_dropping",
                    "fork_dropping", "knife_dropping"):
        if dname in term_names:
            drop_indices.append(term_names.index(dname))
    print(f"[Play] Termination terms: {term_names}")
    print(f"[Play] Success term idx: {success_idx} ({term_names[success_idx] if success_idx is not None else 'NONE'})")
    print(f"[Play] Drop term indices: {[(i, term_names[i]) for i in drop_indices]}")
    return tm, term_names, success_idx, drop_indices


def _get_success_flags(tm, success_idx, drop_indices):
    """Per-env success flags from _last_episode_dones (matching play.py exactly)."""
    if success_idx is None:
        return None
    led = tm._last_episode_dones
    success_mask = led[:, success_idx].bool()
    for di in drop_indices:
        success_mask = success_mask & (~led[:, di].bool())
    return success_mask


def _check_success_from_info(env, env_info, env_idx: int, obs_before_step: torch.Tensor, cfg,
                              tm=None, success_idx=None, drop_indices=None) -> bool:
    """Check success from termination manager signal (matching play.py)."""
    # Primary: vectorized termination manager signal
    if tm is not None and success_idx is not None:
        success_flags = _get_success_flags(tm, success_idx, drop_indices or [])
        if success_flags is not None:
            return bool(success_flags[env_idx].item())

    # Fallback: position-based check from pre-step obs
    if obs_before_step is not None:
        s = getattr(cfg, "obs_structure", None)
        if s is not None and "cube_1_pos" in s and "cube_2_pos" in s:
            c1 = obs_before_step[env_idx, s["cube_1_pos"][0]:s["cube_1_pos"][1]]
            c2 = obs_before_step[env_idx, s["cube_2_pos"][0]:s["cube_2_pos"][1]]
            z_diff_a = torch.abs((c1[2] - c2[2]) - 0.018)
            xy_dist_a = torch.norm(c1[:2] - c2[:2])
            z_diff_b = torch.abs((c2[2] - c1[2]) - 0.018)
            xy_dist_b = torch.norm(c2[:2] - c1[:2])
            stack_ok = (z_diff_a < 0.003 and xy_dist_a < 0.009) or \
                       (z_diff_b < 0.003 and xy_dist_b < 0.009)
            if stack_ok:
                return True
    return False


def play_graph_rl_policy(
    task_name: str,
    checkpoint_path: str,
    pretrained_checkpoint: str,
    num_envs: int = 64,
    num_episodes: int = 10,
    device: str = "cuda",
    deterministic: bool = True,
    policy_type: str = "unet",
    episode_length_s: float = None,
    backbone_only: bool = False,
    no_gripper_head: bool = False,
):
    """Play trained Graph-Unet + Residual RL policy (aligned with train_graph_rl.py).

    If backbone_only=True, runs policy.act(..., zero_residual=True): Graph-Unet base action only
    (no residual δ, no RL gripper head), for fair comparison with full RL.

    If no_gripper_head=True (and not backbone_only): keep residual RL on arms, but use a_base for
    gripper dims (strip RL gripper head + play logits override). Use to eval alignment-only / no-gripper-BCE runs.
    """
    # Auto-detect and swap if user passed (IL, RL) instead of (RL, IL)
    t1, t2 = _detect_checkpoint_type(checkpoint_path), _detect_checkpoint_type(pretrained_checkpoint)
    if t1 == "il" and t2 == "rl":
        checkpoint_path, pretrained_checkpoint = pretrained_checkpoint, checkpoint_path
        print("[Play] Detected swapped args (IL first, RL second) - auto-corrected to (RL, IL)")

    print(f"[Play] ===== Residual RL Policy Playback (Graph-Unet) =====")
    print(f"[Play] Task: {task_name}")
    _mode = "backbone only (zero residual)" if backbone_only else (
        "residual RL, no gripper head (gripper=a_base)" if no_gripper_head else "full residual RL"
    )
    print(f"[Play] Mode: {_mode}")
    print(f"[Play] RL Checkpoint: {checkpoint_path}")
    print(f"[Play] Graph-Unet Checkpoint: {pretrained_checkpoint}")
    print(f"[Play] Num envs: {num_envs}")

    # Load pretrained backbone (matching train_graph_rl.py backbone class mapping + auto-detection)
    _backbone_classes = {
        "graph_unet": GraphUnetPolicy,
        "unet": UnetPolicy,
        "dual_arm_gated": DualArmDisentangledPolicyGated,
        "dual_arm": DualArmDisentangledPolicy,
    }
    # Auto-detect dual arm from checkpoint (matching train_graph_rl.py)
    _ckpt_preview = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False)
    _ckpt_cfg = _ckpt_preview.get("cfg", None)
    if _ckpt_cfg is not None and hasattr(_ckpt_cfg, "arm_action_dim"):
        _sd = _ckpt_preview.get("policy_state_dict", _ckpt_preview.get("model_state_dict", {}))
        if "graph_gate_logit" in _sd:
            policy_type = "dual_arm_gated"
        else:
            policy_type = "dual_arm"
        print(f"[Play] Auto-detected dual arm checkpoint: policy_type={policy_type}")
    del _ckpt_preview

    PolicyClass = _backbone_classes.get(policy_type, GraphUnetPolicy)
    print(f"\n[Play] Loading pretrained backbone ({PolicyClass.__name__}): {pretrained_checkpoint}")
    backbone_policy = PolicyClass.load(pretrained_checkpoint, device=device)
    backbone_policy.eval()
    for p in backbone_policy.parameters():
        p.requires_grad = False
    print(f"[Play] Backbone loaded and frozen")

    # Create backbone adapter
    backbone_adapter = GraphUnetBackboneAdapter(backbone_policy)

    # Load RL policy
    print(f"\n[Play] Loading RL policy: {checkpoint_path}")
    policy = GraphUnetResidualRLPolicy.load(checkpoint_path, backbone=backbone_adapter, device=device)
    policy.num_diffusion_steps = 15  # must match train
    policy.eval()
    print(f"[Play] RL policy loaded (diffusion_steps=15)")
    # Training-time expert settings (saved in checkpoint cfg; play itself does not run expert)
    _pcfg = policy.cfg
    _uei = getattr(_pcfg, "use_expert_intervention", False)
    _nego = getattr(_pcfg, "no_expert_gripper_override", False)
    _ratio = getattr(_pcfg, "expert_intervention_ratio", None)
    _decay = getattr(_pcfg, "expert_intervention_decay", None)
    print(
        f"[Play] Checkpoint (train) expert: use_expert_intervention={_uei}, "
        f"no_expert_gripper_override={_nego} "
        f"(if True: arms expert only; no env gripper override, no expert gripper BCE)"
    )
    if _ratio is not None and _decay is not None:
        print(f"[Play] Checkpoint (train) expert ratio/decay: {_ratio}, {_decay}")
    print(f"[Play] Eval never runs expert — only the policy; line above is for logging/comparison.")

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
    if episode_length_s is not None:
        env_cfg.episode_length_s = episode_length_s
        print(f"[Play] Override episode_length_s = {episode_length_s}")
    env = gym.make(task_name, cfg=env_cfg)

    # env_origins for world→local conversion (matching train_graph_rl.py)
    _base = env.unwrapped if hasattr(env, "unwrapped") else env
    _env_origins = _base.scene.env_origins.clone().to(device=device, dtype=torch.float32)
    # Success detection: set up termination manager refs ONCE (matching play.py)
    _tm, _term_names, _success_idx, _drop_indices = _setup_success_detection(env)
    cfg = policy.cfg
    is_dual_arm = (cfg.action_dim == 12)
    robot_state_dim = 12 if is_dual_arm else 6
    obs_struct = getattr(cfg, "obs_structure", None)

    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space

    print(f"[Play] Observation space: {obs_space}")
    print(f"[Play] Action space: {action_space}")

    # Initialize policy buffers
    policy.init_env_buffers(num_envs)

    # History buffers for Graph-DiT (matching train_graph_rl.py)
    action_history_length = getattr(backbone_policy.cfg, "action_history_length", 4)
    graph_dit_joint_dim = robot_state_dim  # 6 single arm, 12 dual arm
    action_dim = policy.cfg.action_dim if hasattr(policy, 'cfg') and hasattr(policy.cfg, 'action_dim') else 6
    _pos_keys = ["left_ee_position", "right_ee_position", "cube_1_pos", "cube_2_pos", "fork_pos", "knife_pos"]

    # Use single tensor [num_envs, history_len, dim] for efficiency
    action_history = torch.zeros(num_envs, action_history_length, action_dim, device=device)
    ee_node_history = torch.zeros(num_envs, action_history_length, 7, device=device)
    object_node_history = torch.zeros(num_envs, action_history_length, 7, device=device)
    joint_state_history = torch.zeros(num_envs, action_history_length, graph_dit_joint_dim, device=device)

    # Reset environment
    obs, info = env.reset()
    obs = _process_obs(obs, device)
    obs = _obs_to_local(obs, _env_origins, obs_struct, _pos_keys)

    # CRITICAL: Reset policy buffers for ALL envs (fresh start, matching train_graph_rl.py)
    policy.reset_envs(torch.arange(num_envs, device=device))

    # CRITICAL: Pre-fill histories with first observation (matches BC training padding).
    # BC training pads early-episode histories with copies of the first real frame, NOT zeros.
    # Zero-histories give out-of-distribution inputs → garbage backbone predictions.
    # (matching train_graph_rl.py: _prefill_histories_from_obs)
    _init_ee_node, _init_obj_node = policy._extract_nodes_from_obs(obs)
    if obs_struct is not None and "left_joint_pos" in obs_struct:
        _init_joint = torch.cat([
            obs[:, obs_struct["left_joint_pos"][0]:obs_struct["left_joint_pos"][1]],
            obs[:, obs_struct["right_joint_pos"][0]:obs_struct["right_joint_pos"][1]],
        ], dim=-1)
    else:
        _init_joint = obs[:, :robot_state_dim]

    for i in range(num_envs):
        for t in range(action_history_length):
            ee_node_history[i, t] = _init_ee_node[i]
            object_node_history[i, t] = _init_obj_node[i]
            joint_state_history[i, t] = _init_joint[i]
        if action_mean is not None and action_std is not None:
            _init_action_norm = (_init_joint[i] - action_mean) / (action_std + 1e-8)
            for t in range(action_history_length):
                action_history[i, t, :_init_action_norm.shape[0]] = _init_action_norm
        else:
            action_history[i] = 0

    # Pre-fill multi-node temporal history in policy (matching train_graph_rl.py)
    policy.prefill_node_history(obs)
    print(f"[Play] Pre-filled histories from initial obs (matching BC training padding)")

    # Save home EE positions for fixing stale body transforms after auto-reset.
    # Isaac Lab step() auto-reset calls _reset_idx() but NOT sim.forward(),
    # so EE/cube positions from FrameTransformer are STALE after reset.
    # We save home positions here and overwrite stale obs for reset envs.
    _home_ee_per_env = []
    _ee_keys = ["left_ee_position", "left_ee_orientation", "right_ee_position", "right_ee_orientation"]
    for _ei in range(num_envs):
        _slices = {}
        for _hk in _ee_keys:
            if obs_struct is not None and _hk in obs_struct:
                _s, _e = obs_struct[_hk]
                _slices[_hk] = obs[_ei, _s:_e].clone()
        _home_ee_per_env.append(_slices)
    if _home_ee_per_env and _home_ee_per_env[0]:
        print(f"[Play] Saved per-env home EE for stale obs fix: {list(_home_ee_per_env[0].keys())}")

    # Stats (tensor for per-env tracking; lists for final stats like play.py)
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    episode_success = []  # list of bool for SR(100ep)
    episode_rewards_list = []  # list of float for mean_reward
    episode_lengths_list = []  # list of int, steps per completed episode (aligned with episode_success)
    # Stacking-funnel alignment stats (dual-arm + cube obs only): 1-based step index when first in_alignment
    first_align_step_ep = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    succ_pre_align_steps_list = []  # steps before first in_alignment (success only)
    succ_align_phase_steps_list = []  # steps from first in_alignment to episode end, inclusive (success only)
    # Episode-end ||cube1_xy - cube2_xy|| (meters), one per completed episode (aligned with episode_success)
    episode_end_cube_xy_m_list = []

    # Per-term reward breakdown tracking
    _reward_term_names = None
    _reward_term_accum = None  # [num_envs, num_terms]
    _reward_term_episodes = []  # list of [num_terms] tensors
    try:
        _base = env.unwrapped if hasattr(env, "unwrapped") else env
        if hasattr(_base, "reward_manager"):
            _rm = _base.reward_manager
            _reward_term_names = list(_rm._term_names)
            _reward_term_accum = torch.zeros(num_envs, len(_reward_term_names), device=device)
            _reward_dt = getattr(_base, "step_dt", 0.02)
            print(f"[Play] Reward terms: {_reward_term_names}")
    except Exception as e:
        print(f"[Play] WARNING: reward breakdown init failed: {e}")

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
                obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
            else:
                obs_norm = obs
            
            # Extract current node features and joint states from obs (matching train_graph_rl.py)
            ee_node_current, object_node_current = policy._extract_nodes_from_obs(obs)
            if obs_struct is not None and "left_joint_pos" in obs_struct:
                joint_states_current = torch.cat([
                    obs[:, obs_struct["left_joint_pos"][0]:obs_struct["left_joint_pos"][1]],
                    obs[:, obs_struct["right_joint_pos"][0]:obs_struct["right_joint_pos"][1]],
                ], dim=-1)
            else:
                joint_states_current = obs[:, :robot_state_dim]
            
            # CRITICAL: Normalize node and joint histories (same as train_graph_rl.py)
            # History tensors are already in batch format [num_envs, history_len, dim]
            action_history_norm = action_history  # [B, H, action_dim] - already normalized (stored as normalized)
            
            # Normalize node histories
            ee_node_history_norm = ee_node_history.clone()  # [B, H, 7]
            object_node_history_norm = object_node_history.clone()  # [B, H, 7]
            if ee_node_mean is not None and ee_node_std is not None:
                ee_node_history_norm = (ee_node_history_norm - ee_node_mean) / (ee_node_std + 1e-8)
            if object_node_mean is not None and object_node_std is not None:
                object_node_history_norm = (object_node_history_norm - object_node_mean) / (object_node_std + 1e-8)
            
            # Normalize joint history
            joint_state_history_norm = joint_state_history.clone()  # [B, H, 6]
            if joint_mean is not None and joint_std is not None:
                joint_state_history_norm = (joint_state_history_norm - joint_mean) / (joint_std + 1e-8)
            
            # Get subtask condition (matching train_graph_rl.py)
            subtask_cond = _extract_subtask_condition(env, obs, policy, device)

            # Get action from policy
            with torch.no_grad():
                action, policy_info = policy.act(
                    obs_raw=obs,
                    obs_norm=obs_norm,  # Use normalized obs
                    action_history=action_history_norm,  # [num_envs, history_len, action_dim] - normalized
                    ee_node_history=ee_node_history_norm,  # [num_envs, history_len, 7] - normalized
                    object_node_history=object_node_history_norm,  # [num_envs, history_len, 7] - normalized
                    joint_states_history=joint_state_history_norm,  # [num_envs, history_len, joint_dim] - normalized
                    subtask_condition=subtask_cond,
                    deterministic=deterministic,
                    zero_residual=backbone_only,
                )

            # Strip RL gripper head: use backbone gripper (a_base); residual α=0 on grippers so this matches train IL
            if no_gripper_head and not backbone_only and "a_base" in policy_info:
                for _gi in policy.gripper_indices:
                    action[:, _gi] = policy_info["a_base"][:, _gi]

            # Stacking funnel: same mask as train_graph_rl (stacking_funnel.py)
            if is_dual_arm and obs_struct is not None and "cube_1_pos" in obs_struct and "cube_2_pos" in obs_struct:
                c1 = obs[:, obs_struct["cube_1_pos"][0]:obs_struct["cube_1_pos"][1]]
                c2 = obs[:, obs_struct["cube_2_pos"][0]:obs_struct["cube_2_pos"][1]]
                in_alignment = stacking_funnel_mask_from_cubes(c1, c2)
                # Record first step (1-based) entering alignment for this episode (before episode_lengths += 1)
                step_1based = episode_lengths.long() + 1
                _enter = (first_align_step_ep < 0) & in_alignment
                first_align_step_ep = torch.where(_enter, step_1based, first_align_step_ep)
                not_aligned = ~in_alignment
                if not_aligned.any() and "a_base" in policy_info:
                    action[not_aligned] = policy_info["a_base"][not_aligned]

            # Clip action to action space bounds
            if hasattr(env, 'action_space'):
                if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                    action = torch.clamp(
                        action,
                        torch.tensor(env.action_space.low, device=action.device, dtype=action.dtype),
                        torch.tensor(env.action_space.high, device=action.device, dtype=action.dtype)
                    )

            # Gripper head override: use gripper_logits from RL policy if available (skip if no_gripper_head)
            gripper_open_override = None
            if not no_gripper_head and "gripper_logits" in policy_info:
                gripper_logits = policy_info["gripper_logits"]  # [B, num_grippers]
                gripper_prob = torch.sigmoid(gripper_logits)  # [B, num_grippers]
                gripper_open_override = gripper_prob > 0.5  # True = open

            # Process action for env (matching train_graph_rl.py)
            action_for_sim = action.clone()
            if action_dim == 12:
                # CRITICAL: Backbone outputs [left_6, right_6], env expects [right_6, left_6]
                action_for_sim = torch.cat([action_for_sim[:, 6:12], action_for_sim[:, 0:6]], dim=1)
            arm_block = 6
            for arm_i in range(action_dim // arm_block):
                base = arm_i * arm_block
                joints_slice = slice(base, base + 5)
                if ema_smoothed_joints is None:
                    ema_smoothed_joints = action_for_sim[:, :action_dim].clone()
                ema_smoothed_joints[:, joints_slice] = (
                    ema_alpha * action_for_sim[:, joints_slice]
                    + (1 - ema_alpha) * ema_smoothed_joints[:, joints_slice]
                )
                action_for_sim[:, joints_slice] = ema_smoothed_joints[:, joints_slice]
                # Gripper: direct -1/1 mapping (policy outputs continuous, env expects +1/-1)
                gripper_idx = base + 5
                gripper_threshold = -0.25
                action_for_sim[:, gripper_idx] = torch.where(
                    action_for_sim[:, gripper_idx] > gripper_threshold,
                    torch.tensor(1.0, device=action.device, dtype=action.dtype),
                    torch.tensor(-1.0, device=action.device, dtype=action.dtype),
                )
                # Override with gripper head prediction if available
                # After swap: arm_i=0 is right arm (gripper_head idx=0), arm_i=1 is left arm (gripper_head idx=1)
                if gripper_open_override is not None and arm_i < gripper_open_override.shape[1]:
                    open_mask = gripper_open_override[:, arm_i]
                    action_for_sim[open_mask, gripper_idx] = 1.0  # force open

            # Step environment
            next_obs, reward, terminated, truncated, env_info = env.step(action_for_sim)
            done = terminated | truncated

            # Process next_obs (world→local)
            next_obs = _process_obs(next_obs, device)
            next_obs = _obs_to_local(next_obs, _env_origins, obs_struct, _pos_keys)
            reward = reward.to(device).float()
            done = done.to(device).float()

            # Update history buffers (matching train_graph_rl.py: normalized action for history)
            if action_mean is not None and action_std is not None:
                action_for_history = (action - action_mean) / (action_std + 1e-8)
            else:
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

            # Accumulate per-term rewards
            if _reward_term_accum is not None:
                try:
                    _reward_term_accum += _rm._step_reward * _reward_dt
                except Exception:
                    pass

            # Handle resets: Isaac Lab 在 reset 之后才计算 obs，所以 next_obs 对 done env 是 reset 后的新 episode 初始 obs（cube 在地面）
            # 必须用 obs（step 前 = 终止前最后一帧）判断 success
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_envs) > 0:
                # FIX stale EE positions after auto-reset (from BC play.py).
                # step() auto-reset calls _reset_idx() but NOT sim.forward(),
                # so EE positions from FrameTransformer are stale.
                # Overwrite stale EE in next_obs with saved home positions.
                if _home_ee_per_env and obs_struct is not None:
                    for env_id in done_envs.tolist():
                        if env_id < len(_home_ee_per_env):
                            for _hk, _hv in _home_ee_per_env[env_id].items():
                                if _hk in obs_struct:
                                    _s, _e = obs_struct[_hk]
                                    val = _hv.to(next_obs.device).clone()
                                    # Home EE is in local frame. next_obs was converted to local
                                    # by _obs_to_local which subtracts env_origins.
                                    # But next_obs for reset envs has stale WORLD-frame EE,
                                    # so after _obs_to_local it's wrong. Write correct local values.
                                    next_obs[env_id, _s:_e] = val

                policy.reset_envs(done_envs)
                # Vectorized success check (matching play.py)
                success_flags = _get_success_flags(_tm, _success_idx, _drop_indices)
                for i in done_envs.tolist():
                    is_truncated = bool(truncated[i].item() if truncated.dim() > 0 else truncated.item())
                    is_terminated = bool(terminated[i].item() if terminated.dim() > 0 else terminated.item())
                    # Matching play.py: truncated=failure, terminated=check signal
                    if is_truncated and not is_terminated:
                        is_success = False
                    elif success_flags is not None:
                        is_success = bool(success_flags[i].item())
                    else:
                        is_success = _check_success_from_info(
                            env, env_info, i, obs_before_step=obs, cfg=cfg,
                            tm=_tm, success_idx=_success_idx, drop_indices=_drop_indices)
                    # Debug: print fired terms
                    led = _tm._last_episode_dones
                    fired = [_term_names[j] for j in range(len(_term_names))
                             if bool(led[i, j].item())]
                    print(f"  [EP] env={i} T={is_terminated} Tr={is_truncated} "
                          f"S={is_success} fired={fired}")
                    episode_success.append(is_success)
                    episode_rewards_list.append(episode_rewards[i].item())
                    _T = int(episode_lengths[i].item())
                    episode_lengths_list.append(_T)
                    # Termination-frame obs: horizontal gap between cube centers (local frame, meters)
                    _end_xy_m = float("nan")
                    if obs_struct is not None and "cube_1_pos" in obs_struct and "cube_2_pos" in obs_struct:
                        _s = obs_struct
                        _c1 = obs[i, _s["cube_1_pos"][0]:_s["cube_1_pos"][1]]
                        _c2 = obs[i, _s["cube_2_pos"][0]:_s["cube_2_pos"][1]]
                        _end_xy_m = torch.norm(_c1[:2] - _c2[:2]).item()
                    episode_end_cube_xy_m_list.append(_end_xy_m)
                    if is_success and is_dual_arm and obs_struct is not None and "cube_1_pos" in obs_struct:
                        _fas = int(first_align_step_ep[i].item())
                        if _fas >= 1:
                            succ_pre_align_steps_list.append(_fas - 1)
                            succ_align_phase_steps_list.append(_T - _fas + 1)
                        else:
                            succ_pre_align_steps_list.append(_T)
                            succ_align_phase_steps_list.append(0)
                    if _reward_term_accum is not None:
                        _reward_term_episodes.append(_reward_term_accum[i].clone())
                    episode_count = len(episode_success)
                    status = "✅" if is_success else "❌"
                    sr = sum(episode_success) / len(episode_success) * 100.0 if episode_success else 0.0
                    print(f"[Play] Ep {episode_count:3d} {status} | SR={sr:.1f}%")
                # Reset per-term accumulators for done envs
                if _reward_term_accum is not None:
                    _reward_term_accum[done_envs] = 0

                # Vectorized reset: zero out and prefill histories (matching train_graph_rl.py)
                ee_node_reset, obj_node_reset = policy._extract_nodes_from_obs(next_obs)
                if obs_struct is not None and "left_joint_pos" in obs_struct:
                    joint_reset = torch.cat([
                        next_obs[:, obs_struct["left_joint_pos"][0]:obs_struct["left_joint_pos"][1]],
                        next_obs[:, obs_struct["right_joint_pos"][0]:obs_struct["right_joint_pos"][1]],
                    ], dim=-1)
                else:
                    joint_reset = next_obs[:, :robot_state_dim]
                for idx in done_envs.tolist():
                    for t in range(action_history_length):
                        ee_node_history[idx, t] = ee_node_reset[idx]
                        object_node_history[idx, t] = obj_node_reset[idx]
                        joint_state_history[idx, t] = joint_reset[idx]
                    if action_mean is not None and action_std is not None:
                        init_action_norm = (joint_reset[idx] - action_mean) / (action_std + 1e-8)
                        for t in range(action_history_length):
                            action_history[idx, t, :init_action_norm.shape[0]] = init_action_norm
                    else:
                        action_history[idx] = 0
                policy.prefill_node_history(next_obs, env_ids=done_envs)
                episode_rewards[done_envs] = 0
                episode_lengths[done_envs] = 0
                first_align_step_ep[done_envs] = -1
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
        succ_lens = [L for L, s in zip(episode_lengths_list, episode_success) if s]
        mean_succ_len = sum(succ_lens) / len(succ_lens) if succ_lens else None
        print(f"\n[Play] ===== Final Statistics =====")
        succ_len_str = (
            f"succ_len_mean={mean_succ_len:.1f} steps (n={len(succ_lens)})"
            if mean_succ_len is not None
            else "succ_len_mean=n/a (0 successes)"
        )
        print(
            f"[Play] SR={sr_final:.1f}% (100ep:{sr_100_final:.1f}%) [{n}ep] | "
            f"Rew_mean={mean_reward:.1f} | {sum(episode_success)}/{n} success | {succ_len_str}"
        )
        if succ_align_phase_steps_list:
            _n_ap = len(succ_align_phase_steps_list)
            _m_pre = sum(succ_pre_align_steps_list) / _n_ap
            _m_ap = sum(succ_align_phase_steps_list) / _n_ap
            _nz = sum(1 for x in succ_align_phase_steps_list if x > 0)
            print(
                f"[Play] Success alignment (stacking funnel): "
                f"pre_align_mean={_m_pre:.1f} steps | "
                f"align_phase_mean={_m_ap:.1f} steps (first in_alignment→end, incl.) | "
                f"n={_n_ap} ({_nz} with align_phase>0)"
            )
        elif succ_lens:
            print(
                "[Play] Success alignment: n/a (need dual-arm + cube_1/2 obs for stacking funnel mask)"
            )
        # Episode-end cube XY gap (same obs as success check; meters → mm)
        _xy_pairs = [
            (xy_m, s)
            for xy_m, s in zip(episode_end_cube_xy_m_list, episode_success)
            if not math.isnan(xy_m)
        ]
        if _xy_pairs:
            _succ_xy = [xy for xy, s in _xy_pairs if s]
            _fail_xy = [xy for xy, s in _xy_pairs if not s]
            _fmt = lambda xs: f"{1000.0 * sum(xs) / len(xs):.2f} mm (n={len(xs)})" if xs else "n/a"
            print(
                f"[Play] Episode-end cube XY gap (‖Δxy‖, local frame): "
                f"succ_mean={_fmt(_succ_xy)} | fail_mean={_fmt(_fail_xy)} | "
                f"all_mean={1000.0 * sum(x for x, _ in _xy_pairs) / len(_xy_pairs):.2f} mm (n={len(_xy_pairs)})"
            )
        # Print per-term reward breakdown
        if _reward_term_names and _reward_term_episodes:
            stacked = torch.stack(_reward_term_episodes)
            mean_per_term = stacked.mean(dim=0)
            parts = [f"{name}={val.item():.2f}" for name, val in zip(_reward_term_names, mean_per_term)]
            print(f"[Play] Rew breakdown: {' | '.join(parts)}")
    print(f"\n[Play] Playback completed!")

    # Cleanup
    env.close()


def _process_obs(obs, device: str) -> torch.Tensor:
    """Flatten dict obs to tensor (IsaacLab: obs['policy'] can be dict when concatenate_terms=False)."""
    if isinstance(obs, dict):
        if "policy" in obs:
            obs_val = obs["policy"]
            if isinstance(obs_val, torch.Tensor):
                obs = obs_val
            elif isinstance(obs_val, np.ndarray):
                obs = torch.from_numpy(obs_val).to(device)
            elif isinstance(obs_val, dict):
                _order = [
                    "left_joint_pos", "left_joint_vel", "right_joint_pos", "right_joint_vel",
                    "left_ee_position", "left_ee_orientation", "right_ee_position", "right_ee_orientation",
                    "cube_1_pos", "cube_1_ori", "cube_2_pos", "cube_2_ori",
                    "fork_pos", "fork_ori", "knife_pos", "knife_ori",
                    "last_action_all",
                ]
                keys_used = [k for k in _order if k in obs_val]
                parts = []
                for k in keys_used:
                    v = obs_val[k]
                    if isinstance(v, torch.Tensor):
                        parts.append(v.flatten(start_dim=1).to(device))
                    elif isinstance(v, np.ndarray):
                        parts.append(torch.from_numpy(v).flatten(start_dim=1).to(device))
                    else:
                        parts.append(torch.tensor(v).flatten(start_dim=1).to(device))
                if not parts:
                    raise RuntimeError("obs['policy'] is dict but no keys matched")
                obs = torch.cat(parts, dim=1)
            else:
                obs = torch.tensor(obs_val, device=device)
        else:
            obs = torch.cat([v.flatten(start_dim=1) for v in obs.values()], dim=-1)
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, device=device)
    return obs.to(device).float()


def _extract_subtask_condition(env, obs_local, policy, device) -> torch.Tensor:
    """Minimal subtask extraction for play (matching train_graph_rl.py)."""
    bb = policy.backbone
    actual_bb = bb.backbone if hasattr(bb, "backbone") else bb
    num_subtasks = getattr(getattr(actual_bb, "cfg", None), "num_subtasks", 0)
    if num_subtasks <= 0:
        return None
    B = obs_local.shape[0] if obs_local is not None else 1
    # Default: first subtask active
    cond = torch.zeros(B, num_subtasks, device=device)
    cond[:, 0] = 1.0
    try:
        base = env.unwrapped if hasattr(env, "unwrapped") else env
        if hasattr(base, "get_subtask_term_signals"):
            signals = base.get_subtask_term_signals()
            pick = signals.get("pick_cube", signals.get("place_fork", None))
            stack = signals.get("stack_cube", signals.get("place_knife", None))
            if pick is not None and stack is not None:
                pick = pick.to(device).float().bool()
                stack = stack.to(device).float().bool()
                cond = torch.zeros(B, num_subtasks, device=device)
                cond[~pick, 0] = 1.0
                cond[pick & ~stack, 1] = 1.0
                cond[pick & stack, 1] = 1.0
    except Exception:
        pass
    return cond


def _obs_to_local(obs: torch.Tensor, env_origins, obs_struct, pos_keys) -> torch.Tensor:
    """Subtract env_origins from position keys (matching train_graph_rl.py)."""
    if obs_struct is None:
        return obs
    origins = env_origins[:obs.shape[0], :3].to(obs.device)
    for pk in pos_keys:
        if pk in obs_struct:
            s, e = obs_struct[pk]
            obs[:, s:e] = obs[:, s:e] - origins
    return obs


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
        "--policy_type", type=str, default="dual_arm_gated",
        choices=["unet", "graph_unet", "dual_arm", "dual_arm_gated"],
        help="Policy class for the pretrained backbone (auto-detected from checkpoint)",
    )
    parser.add_argument(
        "--episode_length_s", type=float, default=None,
        help="Override episode length in seconds (default: use env config)",
    )
    parser.add_argument(
        "--backbone_only",
        action="store_true",
        help="Run Graph-Unet backbone only (policy.act zero_residual): no residual δ, no RL gripper head — compare against full RL",
    )
    parser.add_argument(
        "--no_gripper_head",
        action="store_true",
        help="Keep residual RL on arms; replace gripper dims with a_base (no RL gripper head, no logits override). Implied by --backbone_only.",
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
        episode_length_s=args.episode_length_s,
        backbone_only=args.backbone_only,
        no_gripper_head=args.no_gripper_head,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()

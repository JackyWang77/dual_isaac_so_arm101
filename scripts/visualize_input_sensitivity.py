# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
"""
Input Sensitivity Analysis: 证明 DiT 抖动源于对观测噪声敏感.

给同一观测加不同程度高斯噪声，看 DiT 与 U-Net 输出动作的偏离。
预期: U-Net 平缓上升(低通)，DiT 陡峭/跳变(对噪声放大)。
与 Loss Landscape 互补: 地形图看权重重敏感度，本实验直接看输入噪声→输出抖动。
Run from repo root:  python scripts/visualize_input_sensitivity.py
"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
POLICIES_DIR = os.path.join(REPO_ROOT, "source", "SO_101", "SO_101", "policies")
GRAPH_UNET_SCRIPT = os.path.join(SCRIPT_DIR, "graph_unet")
for p in (REPO_ROOT, POLICIES_DIR, GRAPH_UNET_SCRIPT):
    if p not in sys.path:
        sys.path.insert(0, p)

import graph_dit_policy as _graph_dit_policy_module


def _install_stubs():
    import types
    if "SO_101.policies.graph_dit_policy" in sys.modules:
        return
    so101 = types.ModuleType("SO_101")
    so101_policies = types.ModuleType("SO_101.policies")
    so101_policies.graph_dit_policy = _graph_dit_policy_module
    so101.policies = so101_policies
    sys.modules["SO_101"] = so101
    sys.modules["SO_101.policies"] = so101_policies
    sys.modules["SO_101.policies.graph_dit_policy"] = _graph_dit_policy_module


_install_stubs()

import graph_unet_policy as _graph_unet_policy_module

sys.modules["SO_101.policies.graph_unet_policy"] = _graph_unet_policy_module

from graph_dit_policy import GraphDiTPolicy
from graph_unet_policy import GraphUnetPolicy
from dataset import HDF5DemoDataset, demo_collate_fn


def get_probe_batch(dataset_path, obs_keys, cfg, device, batch_size=1):
    """取一个 batch，从中抽出单样本 probe (obs, action_history, ee_node_history, ...)."""
    dataset = HDF5DemoDataset(
        dataset_path,
        obs_keys=obs_keys,
        normalize_obs=True,
        normalize_actions=True,
        action_history_length=cfg.action_history_length,
        pred_horizon=cfg.pred_horizon,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=demo_collate_fn,
        shuffle=False,
    )
    batch = next(iter(loader))
    # 取第一个样本、第一个时间步
    obs = batch["obs_seq"][0, 0, :].unsqueeze(0).to(device)  # [1, obs_dim]
    action_history_6 = batch["action_history_seq"][0, 0, :, :].unsqueeze(0).to(device)  # [1, H, 6]
    action_history_5 = action_history_6[:, :, :5].contiguous()  # [1, H, 5] for DiT
    ee_node_history = batch["ee_node_history_seq"][0, 0, :, :].unsqueeze(0).to(device)  # [1, H, 7]
    object_node_history = batch["object_node_history_seq"][0, 0, :, :].unsqueeze(0).to(device)
    joint_states_history = batch["joint_states_history_seq"][0, 0, :, :].unsqueeze(0).to(device)
    subtask = batch.get("subtask_condition_seq")
    if subtask is not None:
        subtask = subtask[0, 0, :].unsqueeze(0).to(device)
    return {
        "obs": obs,
        "action_history_5": action_history_5,
        "action_history_6": action_history_6,
        "ee_node_history": ee_node_history,
        "object_node_history": object_node_history,
        "joint_states_history": joint_states_history,
        "subtask_condition": subtask,
    }


def compute_deviations(policy, probe, sigmas, device, num_diffusion_steps=15, num_trials=5):
    """对一组 sigma，在 obs 上加噪声，跑 predict，算输出相对 base（无噪声）的 L2 偏离；可多次 trial 取平均。"""
    policy.eval()
    obs = probe["obs"]  # [1, obs_dim]
    action_dim = getattr(policy.cfg, "action_dim", 6)
    if action_dim == 5:
        action_history = probe.get("action_history_5")
    else:
        action_history = probe.get("action_history_6")
    if action_history is None:
        raise ValueError(
            f"probe must have 'action_history_{action_dim}' for policy with action_dim={action_dim}. "
            "Check that get_probe_batch returns the expected keys."
        )

    with torch.no_grad():
        base_action = policy.predict(
            obs,
            action_history=action_history,
            ee_node_history=probe["ee_node_history"],
            object_node_history=probe["object_node_history"],
            joint_states_history=probe["joint_states_history"],
            subtask_condition=probe.get("subtask_condition"),
            num_diffusion_steps=num_diffusion_steps,
            deterministic=True,
        ).detach()

    diffs = []
    for sigma in sigmas:
        trial_diffs = []
        for _ in range(num_trials):
            with torch.no_grad():
                if sigma == 0:
                    noisy_obs = obs.clone()
                else:
                    noisy_obs = obs + sigma * torch.randn_like(obs, device=device)
                pred = policy.predict(
                    noisy_obs,
                    action_history=action_history,
                    ee_node_history=probe["ee_node_history"],
                    object_node_history=probe["object_node_history"],
                    joint_states_history=probe["joint_states_history"],
                    subtask_condition=probe.get("subtask_condition"),
                    num_diffusion_steps=num_diffusion_steps,
                    deterministic=True,
                )
                diff = torch.norm(pred - base_action).item()
                trial_diffs.append(diff)
        diffs.append(np.mean(trial_diffs))
    return np.array(diffs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.environ.get(
        "DATASET_PATH",
        os.path.join(REPO_ROOT, "datasets", "lift_annotated_dataset.hdf5"),
    )
    ckpt_dit = os.environ.get(
        "CKPT_DIT",
        os.path.join(REPO_ROOT, "logs", "graph_dit", "lift_joint_flow_matching", "latest", "best_model.pt"),
    )
    ckpt_unet = os.environ.get(
        "CKPT_UNET",
        os.path.join(REPO_ROOT, "logs", "graph_unet", "lift_joint_flow_matching", "latest", "best_model.pt"),
    )

    obs_keys = [
        "joint_pos", "joint_vel", "object_position", "object_orientation",
        "ee_position", "ee_orientation", "actions",
    ]

    def find_best(path, log_subdir):
        """log_subdir e.g. 'graph_dit' or 'graph_unet'."""
        if path and os.path.isfile(path):
            return path
        base = os.path.join(REPO_ROOT, "logs", log_subdir)
        if not os.path.isdir(base):
            return path
        found = None
        for root, _, files in os.walk(base):
            if "best_model.pt" in files:
                cand = os.path.join(root, "best_model.pt")
                if found is None or os.path.getmtime(cand) > os.path.getmtime(found):
                    found = cand
        return found or path

    ckpt_dit = find_best(ckpt_dit, "graph_dit")
    ckpt_unet = find_best(ckpt_unet, "graph_unet")
    if not ckpt_dit or not os.path.isfile(ckpt_dit):
        print(f"未找到 DiT checkpoint: {ckpt_dit}")
        return
    if not ckpt_unet or not os.path.isfile(ckpt_unet):
        print(f"未找到 Unet checkpoint: {ckpt_unet}")
        return

    print("[Input Sensitivity] 加载 DiT...")
    chk_dit = torch.load(ckpt_dit, map_location=device, weights_only=False)
    cfg_dit = chk_dit["cfg"]
    policy_dit = GraphDiTPolicy(cfg_dit).to(device)
    policy_dit.load_state_dict(chk_dit["policy_state_dict"])
    policy_dit.eval()

    print("[Input Sensitivity] 加载 Unet...")
    chk_unet = torch.load(ckpt_unet, map_location=device, weights_only=False)
    cfg_unet = chk_unet["cfg"]
    policy_unet = GraphUnetPolicy(cfg_unet).to(device)
    policy_unet.load_state_dict(chk_unet["policy_state_dict"])
    policy_unet.eval()

    if not os.path.isfile(dataset_path):
        print(f"数据集不存在: {dataset_path}")
        return

    print("[Input Sensitivity] 取 probe 样本 (6-dim dataset)...")
    probe = get_probe_batch(dataset_path, obs_keys, cfg_unet, device, batch_size=1)

    sigmas = np.linspace(0, 0.1, 50)
    num_steps = int(os.environ.get("NUM_INFERENCE_STEPS", "15"))
    num_trials = int(os.environ.get("NUM_TRIALS", "5"))

    print(f"[Input Sensitivity] 扫描噪声等级 0 ~ 0.1, {len(sigmas)} 点, 每点 {num_trials} 次取平均, 推理步数 {num_steps}...")
    print("  DiT ...")
    diffs_dit = compute_deviations(policy_dit, probe, sigmas, device, num_diffusion_steps=num_steps, num_trials=num_trials)
    print("  Unet ...")
    diffs_unet = compute_deviations(policy_unet, probe, sigmas, device, num_diffusion_steps=num_steps, num_trials=num_trials)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        np.savez(
            "input_sensitivity.npz",
            sigmas=sigmas,
            diffs_dit=diffs_dit,
            diffs_unet=diffs_unet,
        )
        print("未安装 matplotlib，已保存 input_sensitivity.npz")
        return

    # Light smoothing (window=5) to reduce jaggedness while preserving trend
    def _smooth(y, w=5):
        if len(y) < w:
            return y
        kernel = np.ones(w) / w
        return np.convolve(y, kernel, mode="same")

    diffs_dit_smooth = _smooth(diffs_dit)
    diffs_unet_smooth = _smooth(diffs_unet)

    # Refined color palette
    color_dit = "#6366f1"   # indigo
    color_unet = "#0d9488"  # teal

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#fafafa")
    ax.set_facecolor("#ffffff")
    ax.plot(
        sigmas, diffs_dit_smooth,
        label="Graph-DiT (Transformer)",
        color=color_dit,
        linewidth=2.5,
        alpha=0.9,
    )
    ax.plot(
        sigmas, diffs_unet_smooth,
        label="Graph-U-Net (CNN)",
        color=color_unet,
        linewidth=2.5,
        alpha=0.9,
    )
    # Subtle fill under curves
    ax.fill_between(sigmas, diffs_dit_smooth, alpha=0.08, color=color_dit)
    ax.fill_between(sigmas, diffs_unet_smooth, alpha=0.08, color=color_unet)

    ax.set_xlabel("Input Noise Level (Sensor Jitter)", fontsize=13, color="#374151")
    ax.set_ylabel("Output Action Deviation (Robot Jitter)", fontsize=13, color="#374151")
    ax.set_title("Why DiT Jitters: Input Sensitivity Analysis", fontsize=15, fontweight=600, color="#111827", pad=14)
    ax.legend(fontsize=12, framealpha=0.95, edgecolor="#e5e7eb", loc="upper right")
    ax.grid(True, alpha=0.25, color="#9ca3af", linestyle="-")
    ax.set_xlim(0, 0.1)
    ax.tick_params(colors="#6b7280", labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#e5e7eb")
    plt.tight_layout()
    out_path = os.path.join(REPO_ROOT, "input_sensitivity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"已保存: {out_path}")

    np.savez(
        os.path.join(REPO_ROOT, "input_sensitivity.npz"),
        sigmas=sigmas,
        diffs_dit=diffs_dit,
        diffs_unet=diffs_unet,
    )


if __name__ == "__main__":
    main()

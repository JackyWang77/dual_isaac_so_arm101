# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
"""
Visualize Graph-Unet Loss Landscape (Filter Normalized).

Same config as Graph-DiT script for fair comparison (RESOLUTION=50, RANGE=0.1, BATCH_SIZE=4).
Run from repo root in env_isaaclab:  python scripts/graph_unet/visualize_loss_landscape.py
"""

import copy
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
POLICIES_DIR = os.path.join(REPO_ROOT, "source", "SO_101", "SO_101", "policies")
for p in (REPO_ROOT, POLICIES_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Stub SO_101.policies so graph_unet_policy can import graph_dit_policy without loading Isaac
import graph_dit_policy as _graph_dit_policy_module


def _install_checkpoint_unpickle_stubs():
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


_install_checkpoint_unpickle_stubs()

import graph_unet_policy as _graph_unet_policy_module

sys.modules["SO_101.policies.graph_unet_policy"] = _graph_unet_policy_module

from graph_unet_policy import UnetPolicy
from dataset import HDF5DemoDataset, demo_collate_fn


def get_random_direction(model):
    """Filter Normalization: random direction with ||d|| = ||w|| per layer."""
    direction = {}
    for name, param in model.named_parameters():
        if param.dim() <= 1:
            direction[name] = torch.zeros_like(param)
            continue
        noise = torch.randn_like(param)
        w_norm = param.norm()
        d_norm = noise.norm()
        scale = w_norm / (d_norm + 1e-10) if d_norm != 0 else 1.0
        direction[name] = noise * scale
    return direction


def compute_loss(policy, batch, device):
    """Compute Flow Matching loss for one batch (same flatten logic as train.py)."""
    policy.eval()
    with torch.no_grad():
        obs_seq = batch["obs_seq"].to(device)
        action_trajectory_seq = batch["action_trajectory_seq"].to(device)
        action_history_seq = batch["action_history_seq"].to(device)
        ee_node_history_seq = batch["ee_node_history_seq"].to(device)
        object_node_history_seq = batch["object_node_history_seq"].to(device)
        joint_states_history_seq = batch["joint_states_history_seq"].to(device)
        trajectory_mask = batch["trajectory_mask"].to(device)

        subtask_condition_seq = batch.get("subtask_condition_seq")
        if subtask_condition_seq is not None:
            subtask_condition_seq = subtask_condition_seq.to(device)

        B, max_T = obs_seq.shape[:2]
        pred_horizon = action_trajectory_seq.shape[2]
        action_history_length = action_history_seq.shape[2]

        obs_flat = obs_seq.reshape(B * max_T, -1)
        actions_flat = action_trajectory_seq.reshape(B * max_T, pred_horizon, -1)
        action_history_flat = action_history_seq.reshape(B * max_T, action_history_length, -1)
        ee_node_history_flat = ee_node_history_seq.reshape(B * max_T, action_history_length, -1)
        object_node_history_flat = object_node_history_seq.reshape(B * max_T, action_history_length, -1)
        joint_states_history_flat = joint_states_history_seq.reshape(B * max_T, action_history_length, -1)
        trajectory_mask_flat = trajectory_mask.reshape(B * max_T, pred_horizon)
        subtask_flat = None
        if subtask_condition_seq is not None:
            subtask_flat = subtask_condition_seq.reshape(B * max_T, -1)

        loss_dict = policy.loss(
            obs_flat,
            actions_flat,
            action_history=action_history_flat,
            ee_node_history=ee_node_history_flat,
            object_node_history=object_node_history_flat,
            joint_states_history=joint_states_history_flat,
            subtask_condition=subtask_flat,
            mask=trajectory_mask_flat,
        )
        return loss_dict["total_loss"].item()


def main():
    # ================= 配置（与 Graph-DiT 脚本一致，便于对比）=================
    CKPT_PATH = os.environ.get(
        "CKPT_PATH",
        "./logs/graph_unet/lift_joint_flow_matching/latest/best_model.pt",
    )
    DATASET_PATH = os.environ.get(
        "DATASET_PATH",
        "./datasets/lift_annotated_dataset.hdf5",
    )
    RESOLUTION = int(os.environ.get("LANDSCAPE_RESOLUTION", "50"))
    RANGE = float(os.environ.get("LANDSCAPE_RANGE", "0.1"))
    BATCH_SIZE = int(os.environ.get("LANDSCAPE_BATCH_SIZE", "4"))
    # ========================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"地形分辨率: {RESOLUTION}x{RESOLUTION}, 范围: [-{RANGE}, {RANGE}]")
    print(f"加载 Graph-Unet: {CKPT_PATH}")

    if not os.path.isfile(CKPT_PATH):
        base = "./logs/graph_unet"
        found = None
        for root, _, files in os.walk(base):
            if "best_model.pt" in files:
                cand = os.path.join(root, "best_model.pt")
                if found is None or os.path.getmtime(cand) > os.path.getmtime(found):
                    found = cand
        if found:
            CKPT_PATH = found
            print(f"使用找到的 checkpoint: {CKPT_PATH}")
        else:
            print(f"未找到 checkpoint，请设置 CKPT_PATH 或确保 {base} 下有 best_model.pt")
            return

    try:
        checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        cfg = checkpoint["cfg"]
        policy = UnetPolicy(cfg).to(device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    center_weights = copy.deepcopy(policy.state_dict())

    obs_keys = [
        "joint_pos",
        "joint_vel",
        "object_position",
        "object_orientation",
        "ee_position",
        "ee_orientation",
        "actions",
    ]
    if not os.path.isfile(DATASET_PATH):
        print(f"数据集不存在: {DATASET_PATH}，请设置 DATASET_PATH")
        return

    dataset = HDF5DemoDataset(
        DATASET_PATH,
        obs_keys=obs_keys,
        normalize_obs=True,
        normalize_actions=True,
        action_history_length=cfg.action_history_length,
        pred_horizon=cfg.pred_horizon,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=demo_collate_fn,
        shuffle=True,
    )
    fixed_batch = next(iter(dataloader))
    print(f"探针 batch: {len(fixed_batch['lengths'])} demos (显存紧可设 LANDSCAPE_BATCH_SIZE=4)")

    print("生成 Filter Normalized 随机方向...")
    dir_x = get_random_direction(policy)
    dir_y = get_random_direction(policy)

    alphas = np.linspace(-RANGE, RANGE, RESOLUTION)
    betas = np.linspace(-RANGE, RANGE, RESOLUTION)
    losses = np.zeros((RESOLUTION, RESOLUTION))

    print(f"地形扫描 {RESOLUTION}x{RESOLUTION} ...")
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            new_state_dict = {}
            for name, param in center_weights.items():
                if name in dir_x:
                    d = alpha * dir_x[name] + beta * dir_y[name]
                    new_state_dict[name] = param + d
                else:
                    new_state_dict[name] = param
            policy.load_state_dict(new_state_dict, strict=False)
            loss = compute_loss(policy, fixed_batch, device)
            losses[i, j] = loss
            if device == "cuda":
                torch.cuda.empty_cache()
            print(".", end="", flush=True)
        print(f" {i+1}/{RESOLUTION}")

    policy.load_state_dict(center_weights)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("未安装 matplotlib，只保存 losses.npz")
        np.savez("loss_landscape_unet.npz", alphas=alphas, betas=betas, losses=losses)
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(alphas, betas)
    surf = ax.plot_surface(X, Y, losses, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.contour(X, Y, losses, zdir="z", offset=np.min(losses), cmap="viridis", alpha=0.5)
    ax.set_title("Graph-Unet Loss Landscape (Filter Normalized)", fontsize=14)
    ax.set_xlabel("Direction X")
    ax.set_ylabel("Direction Y")
    ax.set_zlabel("Flow Matching Loss")
    ax.set_zlim(0, 0.45)  # 与 DiT 可视化一致，便于对比
    center_loss = losses[RESOLUTION // 2, RESOLUTION // 2]
    ax.scatter(0, 0, center_loss, c="r", s=100, marker="*", label="Best Model")
    ax.legend()
    plt.colorbar(surf, ax=ax, shrink=0.5)
    plt.tight_layout()
    out_path = "loss_landscape_graph_unet.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"已保存: {out_path}")
    np.savez("loss_landscape_unet.npz", alphas=alphas, betas=betas, losses=losses)


if __name__ == "__main__":
    main()

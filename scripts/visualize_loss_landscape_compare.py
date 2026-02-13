# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
"""
Visualize Graph-DiT vs Graph-Unet Loss Landscapes side-by-side (same data, same scale).

Uses one dataset and one probe batch for both models so the comparison is fair.
Run from repo root:  python scripts/visualize_loss_landscape_compare.py

Env (optional):
  CKPT_PATH_DIT   - Graph-DiT best_model.pt (default: auto under ./logs/graph_dit)
  CKPT_PATH_UNET  - Graph-Unet best_model.pt (default: auto under ./logs/graph_unet)
  DATASET_PATH    - HDF5 demo (default: ./datasets/lift_annotated_dataset.hdf5)
  LANDSCAPE_RESOLUTION - grid size (default: 50)
  LANDSCAPE_RANGE - alpha/beta range (default: 0.1)
  LANDSCAPE_BATCH_SIZE - probe batch size (default: 4)
"""

import copy
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
POLICIES_DIR = os.path.join(REPO_ROOT, "source", "SO_101", "SO_101", "policies")
GRAPH_DIT_DIR = os.path.join(SCRIPT_DIR, "graph_dit")
GRAPH_UNET_DIR = os.path.join(SCRIPT_DIR, "graph_unet")
for p in (REPO_ROOT, POLICIES_DIR, GRAPH_DIT_DIR, GRAPH_UNET_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Stub SO_101.policies so checkpoint unpickling does not load Isaac/omni
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
from graph_unet_policy import UnetPolicy
# One dataset for both (same 6-dim, same keys as training)
sys.path.insert(0, GRAPH_DIT_DIR)
from dataset import HDF5DemoDataset, demo_collate_fn


def get_random_direction(model):
    """Filter-normalized random direction: ||d|| = ||w|| per layer."""
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


def compute_loss_dit(policy, batch, device):
    """Flow Matching loss for Graph-DiT (same flatten as train)."""
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


def compute_loss_unet(policy, batch, device):
    """Flow Matching loss for Graph-Unet (same flatten as train)."""
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


def find_best_checkpoint(log_base, name):
    """Find newest best_model.pt under log_base."""
    found = None
    for root, _, files in os.walk(log_base):
        if "best_model.pt" in files:
            cand = os.path.join(root, "best_model.pt")
            if found is None or os.path.getmtime(cand) > os.path.getmtime(found):
                found = cand
    return found


def main():
    # ================= Config (same for both) =================
    CKPT_PATH_DIT = os.environ.get("CKPT_PATH_DIT", None)
    CKPT_PATH_UNET = os.environ.get("CKPT_PATH_UNET", None)
    DATASET_PATH = os.environ.get(
        "DATASET_PATH",
        os.path.join(REPO_ROOT, "datasets", "lift_annotated_dataset.hdf5"),
    )
    RESOLUTION = int(os.environ.get("LANDSCAPE_RESOLUTION", "50"))
    RANGE = float(os.environ.get("LANDSCAPE_RANGE", "0.1"))
    BATCH_SIZE = int(os.environ.get("LANDSCAPE_BATCH_SIZE", "4"))
    # Shared dataset params (match train scripts: 6-dim, same keys)
    ACTION_HISTORY_LENGTH = 10
    PRED_HORIZON = 20
    obs_keys = [
        "joint_pos",
        "joint_vel",
        "object_position",
        "object_orientation",
        "ee_position",
        "ee_orientation",
        "actions",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("========================================")
    print("Loss Landscape Comparison (Graph-DiT vs Graph-Unet)")
    print("Same dataset, same batch, same grid — fair comparison")
    print("========================================")
    print(f"Device: {device}")
    print(f"Grid: {RESOLUTION}x{RESOLUTION}, range [-{RANGE}, {RANGE}]")
    print(f"Dataset: {DATASET_PATH}")

    if not os.path.isfile(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}. Set DATASET_PATH.")
        return

    # Resolve checkpoints
    if not CKPT_PATH_DIT or not os.path.isfile(CKPT_PATH_DIT):
        CKPT_PATH_DIT = find_best_checkpoint(os.path.join(REPO_ROOT, "logs", "graph_dit"), "DiT")
    if not CKPT_PATH_UNET or not os.path.isfile(CKPT_PATH_UNET):
        CKPT_PATH_UNET = find_best_checkpoint(os.path.join(REPO_ROOT, "logs", "graph_unet"), "Unet")
    if not CKPT_PATH_DIT:
        print("No Graph-DiT checkpoint found under ./logs/graph_dit. Set CKPT_PATH_DIT.")
        return
    if not CKPT_PATH_UNET:
        print("No Graph-Unet checkpoint found under ./logs/graph_unet. Set CKPT_PATH_UNET.")
        return
    print(f"Graph-DiT:  {CKPT_PATH_DIT}")
    print(f"Graph-Unet: {CKPT_PATH_UNET}")

    # Load both models
    try:
        checkpoint_dit = torch.load(CKPT_PATH_DIT, map_location=device, weights_only=False)
        cfg_dit = checkpoint_dit["cfg"]
        policy_dit = GraphDiTPolicy(cfg_dit).to(device)
        policy_dit.load_state_dict(checkpoint_dit["policy_state_dict"])
        print("Graph-DiT loaded.")
    except Exception as e:
        print(f"Failed to load Graph-DiT: {e}")
        return

    try:
        checkpoint_unet = torch.load(CKPT_PATH_UNET, map_location=device, weights_only=False)
        cfg_unet = checkpoint_unet["cfg"]
        policy_unet = UnetPolicy(cfg_unet).to(device)
        policy_unet.load_state_dict(checkpoint_unet["policy_state_dict"])
        print("Graph-Unet loaded.")
    except Exception as e:
        print(f"Failed to load Graph-Unet: {e}")
        return

    center_dit = copy.deepcopy(policy_dit.state_dict())
    center_unet = copy.deepcopy(policy_unet.state_dict())

    # One dataset, one fixed batch for both
    dataset = HDF5DemoDataset(
        DATASET_PATH,
        obs_keys=obs_keys,
        normalize_obs=True,
        normalize_actions=True,
        action_history_length=ACTION_HISTORY_LENGTH,
        pred_horizon=PRED_HORIZON,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=demo_collate_fn,
        shuffle=True,
    )
    fixed_batch = next(iter(dataloader))
    print(f"Probe batch: {len(fixed_batch['lengths'])} demos")

    # Directions (per-model, different param names/shapes)
    print("Generating filter-normalized directions...")
    dir_x_dit = get_random_direction(policy_dit)
    dir_y_dit = get_random_direction(policy_dit)
    dir_x_unet = get_random_direction(policy_unet)
    dir_y_unet = get_random_direction(policy_unet)

    alphas = np.linspace(-RANGE, RANGE, RESOLUTION)
    betas = np.linspace(-RANGE, RANGE, RESOLUTION)
    losses_dit = np.zeros((RESOLUTION, RESOLUTION))
    losses_unet = np.zeros((RESOLUTION, RESOLUTION))

    # Grid scan: same (alpha, beta), different models
    print(f"Scanning {RESOLUTION}x{RESOLUTION} for both models...")
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # DiT
            new_dit = {}
            for name, param in center_dit.items():
                if name in dir_x_dit:
                    d = alpha * dir_x_dit[name] + beta * dir_y_dit[name]
                    new_dit[name] = param + d
                else:
                    new_dit[name] = param
            policy_dit.load_state_dict(new_dit, strict=False)
            losses_dit[i, j] = compute_loss_dit(policy_dit, fixed_batch, device)

            # Unet
            new_unet = {}
            for name, param in center_unet.items():
                if name in dir_x_unet:
                    d = alpha * dir_x_unet[name] + beta * dir_y_unet[name]
                    new_unet[name] = param + d
                else:
                    new_unet[name] = param
            policy_unet.load_state_dict(new_unet, strict=False)
            losses_unet[i, j] = compute_loss_unet(policy_unet, fixed_batch, device)

            if device == "cuda":
                torch.cuda.empty_cache()
        print(f"  row {i+1}/{RESOLUTION}")

    policy_dit.load_state_dict(center_dit)
    policy_unet.load_state_dict(center_unet)

    # Same z-axis for direct comparison
    z_max = max(0.45, float(np.nanmax(losses_dit)), float(np.nanmax(losses_unet)))
    z_lim = (0.0, z_max)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d  # noqa: F401 — register 3d projection
    except ImportError:
        print("matplotlib not found; saving npz only.")
        out_npz = os.path.join(REPO_ROOT, "loss_landscape_compare.npz")
        np.savez(out_npz, alphas=alphas, betas=betas, losses_dit=losses_dit, losses_unet=losses_unet)
        print(f"Saved: {out_npz}")
        return

    X, Y = np.meshgrid(alphas, betas)
    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    surf1 = ax1.plot_surface(X, Y, losses_dit, cmap="viridis", edgecolor="none", alpha=0.9)
    ax1.contour(X, Y, losses_dit, zdir="z", offset=np.nanmin(losses_dit), cmap="viridis", alpha=0.5)
    ax1.set_title("Graph-DiT Loss Landscape", fontsize=14)
    ax1.set_xlabel("Direction X")
    ax1.set_ylabel("Direction Y")
    ax1.set_zlabel("Flow Matching Loss")
    ax1.set_zlim(z_lim)
    c1 = losses_dit[RESOLUTION // 2, RESOLUTION // 2]
    ax1.scatter(0, 0, c1, c="r", s=100, marker="*", label="Best")
    ax1.legend()

    surf2 = ax2.plot_surface(X, Y, losses_unet, cmap="viridis", edgecolor="none", alpha=0.9)
    ax2.contour(X, Y, losses_unet, zdir="z", offset=np.nanmin(losses_unet), cmap="viridis", alpha=0.5)
    ax2.set_title("Graph-Unet Loss Landscape", fontsize=14)
    ax2.set_xlabel("Direction X")
    ax2.set_ylabel("Direction Y")
    ax2.set_zlabel("Flow Matching Loss")
    ax2.set_zlim(z_lim)
    c2 = losses_unet[RESOLUTION // 2, RESOLUTION // 2]
    ax2.scatter(0, 0, c2, c="r", s=100, marker="*", label="Best")
    ax2.legend()

    plt.colorbar(surf1, ax=ax1, shrink=0.5)
    plt.colorbar(surf2, ax=ax2, shrink=0.5)
    plt.suptitle("Loss Landscape Comparison (same data, same scale)", fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(REPO_ROOT, "loss_landscape_compare.png")
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Saved: {out_png}")

    out_npz = os.path.join(REPO_ROOT, "loss_landscape_compare.npz")
    np.savez(out_npz, alphas=alphas, betas=betas, losses_dit=losses_dit, losses_unet=losses_unet)
    print(f"Saved: {out_npz}")


if __name__ == "__main__":
    main()

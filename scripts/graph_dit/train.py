# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training script for Graph-DiT Policy.

This script provides a complete training framework for your custom Graph-DiT policy.
Users should replace the placeholder GraphDiTPolicy implementation with their own.

Usage:
    ./isaaclab.sh -p scripts/graph_dit/train.py \
        --task SO-ARM101-Pick-Place-DualArm-IK-Abs-v0 \
        --dataset ./datasets/pick_place.hdf5 \
        --obs_dim 72 \
        --action_dim 8 \
        --epochs 200 \
        --batch_size 256 \
        --lr 1e-4
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import SO_101.tasks  # noqa: F401  # Register environments
import torch
import torch.optim as optim
from SO_101.policies.graph_dit_policy import GraphDiTPolicy, GraphDiTPolicyCfg
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Resolve dataset.py (same directory as this script)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from dataset import HDF5DemoDataset, demo_collate_fn


# HDF5DemoDataset and demo_collate_fn live in dataset.py (imported above).
def train_graph_dit_policy(
    task_name: str,
    dataset_path: str,
    obs_keys: list[str],
    obs_dim: int,
    action_dim: int,
    hidden_dims: list[int] = [256, 256, 128],
    num_layers: int = 6,
    num_heads: int = 8,
    batch_size: int = 256,
    num_epochs: int = 200,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    device: str = "cuda",
    save_dir: str = "./logs/graph_dit",
    log_dir: str = "./logs/graph_dit",
    resume_checkpoint: str | None = None,
    action_history_length: int = 4,
    mode: str = "flow_matching",
    pred_horizon: int = 16,
    exec_horizon: int = 8,
    lr_schedule: str = "constant",
    skip_first_steps: int = 10,
    num_inference_steps: int = 40,
):
    """Train Graph-DiT Policy with Action Chunking.

    Args:
        task_name: Environment task name.
        dataset_path: Path to HDF5 dataset.
        obs_keys: List of observation keys.
        obs_dim: Observation dimension.
        action_dim: Action dimension.
        hidden_dims: Hidden layer dimensions.
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay for optimizer.
        device: Device to train on.
        save_dir: Directory to save checkpoints.
        log_dir: Directory to save logs.
        resume_checkpoint: Path to checkpoint to resume from (optional).
        pred_horizon: Prediction horizon for action chunking (default: 16).
        exec_horizon: Execution horizon for receding horizon control (default: 8).
        lr_schedule: Learning rate schedule: "constant" (stable) or "cosine" (warmup + cosine annealing).
        num_inference_steps: Number of ODE integration steps for flow matching inference (default: 40).
            More steps = smoother predictions but slower. Recommended: 30 for good balance.
    """

    # Create directories with timestamp and mode suffix
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Add mode suffix to save/log directories (e.g., reach_joint_flow_matching)
    base_save_dir = Path(save_dir)
    base_log_dir = Path(log_dir)

    # Append mode suffix to the base directory name
    # e.g., ./logs/graph_dit/reach_joint -> ./logs/graph_dit/reach_joint_flow_matching
    save_dir_with_mode = base_save_dir.parent / f"{base_save_dir.name}_{mode}"
    log_dir_with_mode = base_log_dir.parent / f"{base_log_dir.name}_{mode}"

    # Add timestamp
    save_dir = save_dir_with_mode / timestamp
    log_dir = log_dir_with_mode / timestamp

    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Train] Save dir (with mode and timestamp): {save_dir}")
    print(f"[Train] Log dir (with mode and timestamp): {log_dir}")

    # TensorBoard writer
    tensorboard_dir = log_dir / "tensorboard"
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"[Train] 📊 TensorBoard logs will be saved to: {tensorboard_dir}")

    print(f"[Train] ===== Graph-DiT Policy Training with ACTION CHUNKING =====")
    print(f"[Train] Task: {task_name}")
    print(f"[Train] Dataset: {dataset_path}")
    print(f"[Train] Mode: {mode.upper()} (1-10 steps, fast inference)")
    print(f"[Train] Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"[Train] Action history length: {action_history_length}")
    print(
        f"[Train] ACTION CHUNKING: pred_horizon={pred_horizon}, exec_horizon={exec_horizon}"
    )

    # Load dataset with DEMO-LEVEL loading (complete sequences per demo)
    print(f"\n[Train] Loading dataset with DEMO-LEVEL loading...")
    print(
        f"[Train] CRITICAL: Each sample is a complete demo sequence (not individual timesteps)"
    )
    # ============================================================================
    # 🚑 SINGLE BATCH TEST MODE: Enable for debugging overfitting
    # ============================================================================
    # 🚑 SINGLE BATCH TEST MODE: Enable for debugging overfitting
    # ============================================================================
    # Set to True to test if model can overfit a single demo (classic debugging test)
    # This helps identify if the issue is code logic or data complexity
    single_batch_test = os.getenv("SINGLE_BATCH_TEST", "False").lower() == "true"
    single_batch_size = int(os.getenv("SINGLE_BATCH_SIZE", "16"))

    if single_batch_test:
        print(f"\n🚑 [SINGLE BATCH TEST MODE] Enabled via environment variable!")
        print(
            f"  This will use ONLY the first demo, replicated {single_batch_size} times"
        )
        print(
            f"  Expected outcome: Loss should drop to < 0.001 if code logic is correct"
        )
        print(
            f"  If loss stays > 0.05, there's likely a bug in model architecture or data flow\n"
        )

    dataset = HDF5DemoDataset(
        dataset_path,
        obs_keys,
        normalize_obs=True,
        normalize_actions=True,
        action_history_length=action_history_length,
        pred_horizon=pred_horizon,
        single_batch_test=single_batch_test,
        single_batch_size=single_batch_size,
        skip_first_steps=skip_first_steps,
    )

    # Get normalization stats for saving
    obs_stats = dataset.get_obs_stats()
    action_stats = dataset.get_action_stats()
    node_stats = dataset.get_node_stats()
    joint_stats = dataset.get_joint_stats()

    # Get actual dimensions from first demo
    sample = dataset[0]
    # obs_seq: [T, obs_dim], action_trajectory_seq: [T, pred_horizon, action_dim]
    actual_obs_dim = sample["obs_seq"].shape[-1]
    actual_action_dim = sample["action_seq"].shape[-1]
    actual_pred_horizon = sample["action_trajectory_seq"].shape[1]
    sample_T = sample["length"]

    print(f"[Train] Sample demo length: {sample_T} timesteps")
    print(f"[Train] Obs shape: [T, {actual_obs_dim}]")
    print(
        f"[Train] Action trajectory shape: [T, {actual_pred_horizon}, {actual_action_dim}]"
    )

    # Check if subtask condition is available
    num_subtasks = (
        len(dataset.subtask_order)
        if hasattr(dataset, "subtask_order") and dataset.subtask_order
        else 0
    )
    if num_subtasks > 0:
        print(
            f"[Train] Subtask condition available: {num_subtasks} subtasks ({dataset.subtask_order})"
        )
    else:
        print(f"[Train] No subtask condition found in dataset")

    print(
        f"[Train] Actual obs dim: {actual_obs_dim}, Actual action dim: {actual_action_dim}"
    )

    # Update dimensions if needed
    if obs_dim != actual_obs_dim:
        print(
            f"[Train] Warning: obs_dim mismatch ({obs_dim} vs {actual_obs_dim}), using {actual_obs_dim}"
        )
        obs_dim = actual_obs_dim
    if action_dim != actual_action_dim:
        print(
            f"[Train] Warning: action_dim mismatch ({action_dim} vs {actual_action_dim}), using {actual_action_dim}"
        )
        action_dim = actual_action_dim

    # Create dataloader with demo_collate_fn for variable-length sequence padding
    # IMPORTANT: batch_size now means number of demos per batch, not timesteps!
    # Recommend smaller batch_size (e.g., 4-16) since each demo is ~100 timesteps
    demo_batch_size = min(batch_size, 16)  # Cap at 16 demos per batch for memory
    print(f"[Train] Demo batch size: {demo_batch_size} demos per batch")
    print(
        f"[Train] (Each demo has ~{sample_T} timesteps, so effective batch ≈ {demo_batch_size * sample_T} timesteps)"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=demo_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=demo_collate_fn,  # CRITICAL: Use demo-level collate function
    )

    # Get number of subtasks from dataset
    num_subtasks = (
        len(dataset.subtask_order)
        if hasattr(dataset, "subtask_order") and dataset.subtask_order
        else 0
    )

    # Infer joint_dim from dataset (only joint_pos, joint_vel removed to test if it's noise)
    joint_dim = 0
    if hasattr(dataset, "obs_key_dims"):
        if "joint_pos" in dataset.obs_key_dims:
            joint_dim += dataset.obs_key_dims["joint_pos"]
        # NOTE: joint_vel removed - testing if it's noise
        # if "joint_vel" in dataset.obs_key_dims:
        #     joint_dim += dataset.obs_key_dims["joint_vel"]
    if joint_dim == 0:
        joint_dim = None

    # CRITICAL FIX: Build obs_structure from dataset's obs_key_offsets and obs_key_dims
    # This replaces hardcoded indices with dynamic dictionary-based structure
    obs_structure = None
    if hasattr(dataset, "obs_key_offsets") and hasattr(dataset, "obs_key_dims"):
        obs_structure = {}
        for key in dataset.obs_key_offsets.keys():
            offset = dataset.obs_key_offsets[key]
            dim = dataset.obs_key_dims[key]
            obs_structure[key] = (offset, offset + dim)
        print(f"[Train] ✅ Using dynamic obs_structure from dataset:")
        for key, (start, end) in obs_structure.items():
            print(f"    {key}: [{start}, {end}) (dim={end-start})")
    else:
        print(f"[Train] ⚠️  Dataset doesn't have obs_key_offsets, using default hardcoded indices")

    # Create policy configuration
    # IMPORTANT: num_subtasks must match the actual subtask_condition dimension in data
    # If dataset has subtasks, use that number; otherwise disable subtask conditioning
    cfg = GraphDiTPolicyCfg(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dims[0] if hidden_dims else 256,
        num_layers=num_layers,
        num_heads=num_heads,
        joint_dim=joint_dim,
        num_subtasks=num_subtasks,  # Use actual number from dataset (0 = no subtasks)
        action_history_length=action_history_length,
        pred_horizon=pred_horizon,  # ACTION CHUNKING: predict this many future steps
        exec_horizon=exec_horizon,  # RHC: execute this many steps before re-planning
        device=device,
        obs_structure=obs_structure,  # CRITICAL: Pass dynamic obs_structure instead of hardcoded indices
        num_inference_steps=num_inference_steps,  # Flow matching inference steps (default: 40)
    )

    if num_subtasks > 0:
        print(
            f"[Train] Policy configured with {num_subtasks} subtasks: {dataset.subtask_order}"
        )
        print(f"[Train] Subtask conditioning will be used during training.")
    else:
        print(
            f"[Train] No subtask info found, subtask conditioning disabled (num_subtasks=0)"
        )

    # Create policy network
    print(f"\n[Train] Creating Graph-DiT Policy...")
    policy = GraphDiTPolicy(cfg).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[Train] Total parameters: {total_params:,}")
    print(f"[Train] Trainable parameters: {trainable_params:,}")
    print(f"\n[Train] Policy architecture:")
    print(policy)

    # Create optimizer
    optimizer = optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler
    import math

    if lr_schedule == "cosine":
        # Warmup + Cosine Annealing (original)
        warmup_epochs = max(1, int(num_epochs * 0.1))  # 10% warmup, at least 1 epoch

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup: linearly increase from 0 to learning_rate
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing after warmup
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(
            f"[Train] Using COSINE schedule: {warmup_epochs} warmup epochs ({100*warmup_epochs/num_epochs:.1f}%), then cosine annealing"
        )
    else:
        # Constant learning rate (stable, no scheduling)
        # This is more reliable for debugging and initial training
        scheduler = None
        print(
            f"[Train] Using CONSTANT learning rate: {learning_rate} (no warmup, no decay)"
        )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float("inf")

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n[Train] Resuming from checkpoint: {resume_checkpoint}")
        # weights_only=False is needed for PyTorch 2.6+ to load custom config classes
        checkpoint = torch.load(
            resume_checkpoint, map_location=device, weights_only=False
        )
        policy.load_state_dict(checkpoint["policy_state_dict"])

        # Try to load optimizer and scheduler states (may not exist in best_model.pt)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"[Train] Loaded optimizer state from checkpoint")
        else:
            print(
                f"[Train] Warning: No optimizer state in checkpoint, starting with fresh optimizer"
            )

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"[Train] Loaded scheduler state from checkpoint")

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", checkpoint.get("loss", float("inf")))
        print(f"[Train] Resumed from epoch {start_epoch}, best loss: {best_loss:.6f}")

    # Training loop - DEMO-LEVEL TRAINING
    print(f"\n[Train] Starting DEMO-LEVEL training for {num_epochs} epochs...")
    print(f"[Train] CRITICAL: Each batch contains complete demo sequences")
    print(
        f"[Train] Loss is computed over all timesteps in each demo, with padding masked out"
    )
    policy.train()

    # Track last epoch loss for final model
    last_epoch_loss = None

    for epoch in range(start_epoch, num_epochs):
        epoch_losses = []
        epoch_mse_losses = []
        epoch_total_timesteps = 0  # Track actual timesteps processed

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            # batch is a dictionary from demo_collate_fn
            # All tensors have shape [B, max_T, ...] where max_T is batch-specific
            obs_seq = batch["obs_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, obs_dim]
            action_trajectory_seq = batch["action_trajectory_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, pred_horizon, action_dim]
            action_history_seq = batch["action_history_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, hist_len, action_dim]
            ee_node_history_seq = batch["ee_node_history_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, hist_len, 7]
            object_node_history_seq = batch["object_node_history_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, hist_len, 7]
            joint_states_history_seq = batch["joint_states_history_seq"].to(
                device, non_blocking=True
            )  # [B, max_T, hist_len, joint_dim]
            subtask_condition_seq = batch["subtask_condition_seq"]
            if subtask_condition_seq is not None:
                subtask_condition_seq = subtask_condition_seq.to(
                    device, non_blocking=True
                )  # [B, max_T, num_subtasks]
            lengths = batch["lengths"]  # [B] - original lengths
            mask = batch["mask"].to(
                device, non_blocking=True
            )  # [B, max_T] - True for valid timesteps
            trajectory_mask = batch["trajectory_mask"].to(
                device, non_blocking=True
            )  # [B, max_T, pred_horizon] - horizon-level mask

            B, max_T = obs_seq.shape[:2]
            total_valid_timesteps = mask.sum().item()
            epoch_total_timesteps += total_valid_timesteps

            # Flatten batch for per-timestep processing: [B * max_T, ...]
            # This allows us to process all timesteps in parallel
            obs_flat = obs_seq.reshape(B * max_T, -1)  # [B*max_T, obs_dim]
            actions_flat = action_trajectory_seq.reshape(
                B * max_T, pred_horizon, -1
            )  # [B*max_T, pred_horizon, action_dim]
            action_history_flat = action_history_seq.reshape(
                B * max_T, action_history_length, -1
            )  # [B*max_T, hist_len, action_dim]
            ee_node_history_flat = ee_node_history_seq.reshape(
                B * max_T, action_history_length, -1
            )  # [B*max_T, hist_len, 7]
            object_node_history_flat = object_node_history_seq.reshape(
                B * max_T, action_history_length, -1
            )  # [B*max_T, hist_len, 7]
            joint_states_history_flat = joint_states_history_seq.reshape(
                B * max_T, action_history_length, -1
            )  # [B*max_T, hist_len, joint_dim]
            subtask_condition_flat = None
            if subtask_condition_seq is not None:
                subtask_condition_flat = subtask_condition_seq.reshape(
                    B * max_T, -1
                )  # [B*max_T, num_subtasks]
            trajectory_mask_flat = trajectory_mask.reshape(
                B * max_T, pred_horizon
            )  # [B*max_T, pred_horizon] - horizon-level mask

            # Forward pass - compute loss for ALL timesteps, then mask out padding
            loss_dict = policy.loss(
                obs_flat,
                actions_flat,
                action_history=action_history_flat,
                ee_node_history=ee_node_history_flat,
                object_node_history=object_node_history_flat,
                joint_states_history=joint_states_history_flat,
                subtask_condition=subtask_condition_flat,
                mask=trajectory_mask_flat,  # CRITICAL: Pass horizon-level mask for "half-cut" data handling
            )
            loss = loss_dict["total_loss"]

            # ============================================================================
            # 🕵️ DEBUG: Forensic Analysis - Print diagnostic info
            # ============================================================================
            if batch_idx == 0 and epoch % 100 == 0:  # Print every 100 epochs
                print(f"\n[DEBUG EPOCH {epoch}] ====================")

                # Get debug info if available
                if "debug" in loss_dict:
                    debug = loss_dict["debug"]

                    if "v_pred" in debug and "v_t" in debug:
                        # Flow Matching mode
                        v_pred = debug["v_pred"]
                        v_t = debug["v_t"]
                        actions = debug["actions"]

                        print(f"[Flow Matching Debug]")
                        print(
                            f"  Target (v_t) Mean: {v_t.mean().item():.4f}, Std: {v_t.std().item():.4f}"
                        )
                        print(
                            f"  Target (v_t) Min: {v_t.min().item():.4f}, Max: {v_t.max().item():.4f}"
                        )
                        print(
                            f"  Pred (v_pred) Mean: {v_pred.mean().item():.4f}, Std: {v_pred.std().item():.4f}"
                        )
                        print(
                            f"  Pred (v_pred) Min: {v_pred.min().item():.4f}, Max: {v_pred.max().item():.4f}"
                        )
                        print(
                            f"  Input Action (Data) Min: {actions.min().item():.4f}, Max: {actions.max().item():.4f}"
                        )

                        # Check loss per dimension
                        diff = (v_pred - v_t) ** 2
                        if len(diff.shape) == 3:  # [batch, pred_horizon, action_dim]
                            mse_per_dim = diff.mean(dim=(0, 1)).detach().cpu().numpy()
                            print(f"  Mean MSE per action dim: {mse_per_dim}")

                        # Check time t
                        if "t" in debug:
                            t = debug["t"]
                            print(
                                f"  Time t: Min={t.min().item():.4f}, Max={t.max().item():.4f}, Mean={t.mean().item():.4f}"
                            )

                else:
                    # Fallback: print basic info
                    print(f"  Loss: {loss.item():.6f}")
                    print(f"  Actions shape: {actions_flat.shape}")
                    print(
                        f"  Actions range: [{actions_flat.min().item():.4f}, {actions_flat.max().item():.4f}]"
                    )
                    print(f"  Enable DEBUG_LOSS=true to see detailed diagnostics")

                print("========================================\n")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            # Record losses
            epoch_losses.append(loss.item())
            epoch_mse_losses.append(loss_dict["mse_loss"].item())

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "mse": loss_dict["mse_loss"].item(),
                    "valid_t": total_valid_timesteps,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            # Log to tensorboard
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar(
                "Train/MSE_Loss", loss_dict["mse_loss"].item(), global_step
            )
            writer.add_scalar(
                "Train/LearningRate", optimizer.param_groups[0]["lr"], global_step
            )

        # Update learning rate (if scheduler is enabled)
        if scheduler is not None:
            scheduler.step()

        # Compute epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_mse_loss = np.mean(epoch_mse_losses)

        print(
            f"\n[Epoch {epoch+1}/{num_epochs}] "
            f"Loss: {avg_loss:.6f}, MSE: {avg_mse_loss:.6f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Log to tensorboard
        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Epoch/AverageLoss", avg_loss, epoch)
        writer.add_scalar("Epoch/AverageMSE", avg_mse_loss, epoch)
        writer.add_scalar("Epoch/LearningRate", current_lr, epoch)

        # Track last epoch loss
        last_epoch_loss = avg_loss

        # Save best model only (overwrite if better)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / "best_model.pt"
            # Save with stats for inference
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "cfg": cfg,
                    "obs_stats": obs_stats,
                    "action_stats": action_stats,
                    "node_stats": node_stats,  # CRITICAL: For node feature normalization
                    "joint_stats": joint_stats,  # CRITICAL: For joint state normalization
                    "epoch": epoch,
                    "loss": avg_loss,
                },
                str(best_path),
            )
            print(
                f"[Train] ✅ Saved best model (loss: {best_loss:.6f}, epoch: {epoch+1}) to: {best_path}"
            )

    # Save final model (after all epochs)
    final_loss = last_epoch_loss if last_epoch_loss is not None else best_loss
    final_path = save_dir / "final_model.pt"
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "cfg": cfg,
            "obs_stats": obs_stats,
            "action_stats": action_stats,
            "node_stats": node_stats,  # CRITICAL: For node feature normalization
            "joint_stats": joint_stats,  # CRITICAL: For joint state normalization
            "epoch": num_epochs - 1,
            "loss": final_loss,
        },
        str(final_path),
    )
    print(f"\n[Train] ===== Training Completed! =====")
    print(f"[Train] ✅ Final model saved to: {final_path}")
    print(f"[Train] ✅ Best model saved to: {save_dir / 'best_model.pt'}")
    print(f"[Train] 📁 All models saved in: {save_dir}")
    print(f"[Train] 📊 TensorBoard logs in: {tensorboard_dir}")
    print(f"[Train] 💡 View logs with: tensorboard --logdir {log_dir}")

    writer.close()
    return policy


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Graph-DiT Policy")

    # Dataset arguments
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--dataset", type=str, required=True, help="HDF5 dataset path")

    # Model arguments
    parser.add_argument(
        "--obs_dim",
        type=int,
        default=32,
        help="Observation dimension (reach task: 32 after removing redundant target_object_position)",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=6,
        help="Action dimension (now uses joint_pos[t+5], typically 6 for 6-DoF arm)",
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--action_history_length",
        type=int,
        default=4,
        help="Number of historical actions to use (default: 4)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="flow_matching",
        choices=["flow_matching"],
        help="Training mode: flow_matching (1-10 steps, fast inference)",
    )

    # ACTION CHUNKING arguments (Diffusion Policy's key innovation)
    parser.add_argument(
        "--pred_horizon",
        type=int,
        default=16,
        help="Prediction horizon: number of future action steps to predict (default: 16)",
    )
    parser.add_argument(
        "--exec_horizon",
        type=int,
        default=8,
        help="Execution horizon: number of steps to execute before re-planning (default: 8)",
    )

    # Data preprocessing
    parser.add_argument(
        "--skip_first_steps",
        type=int,
        default=10,
        help="Skip first N steps of each demo (noisy human actions, default: 10)",
    )

    # Learning rate schedule
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine"],
        help="Learning rate schedule: 'constant' (stable) or 'cosine' (warmup + cosine annealing)",
    )

    # Flow matching inference steps
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of ODE integration steps for flow matching inference (default: 30). More steps = smoother but slower.",
    )

    # Paths
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./logs/graph_dit",
        help="Checkpoint save directory",
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs/graph_dit", help="Log directory"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Observation keys (should match your dataset)
    # For reach task, these keys match the observations in reach_env_cfg.py
    # Note: Removed "target_object_position" (7 dims) as it's redundant with object_position + object_orientation
    # In reach task, target is always the current object position, so command is not needed
    obs_keys = [
        "joint_pos",
        "joint_vel",
        "object_position",
        "object_orientation",
        "ee_position",
        "ee_orientation",
        "actions",  # last action for self-attention in Graph DiT
    ]

    # Start training
    train_graph_dit_policy(
        task_name=args.task,
        dataset_path=args.dataset,
        obs_keys=obs_keys,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dims=[args.hidden_dim],
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_checkpoint=args.resume,
        action_history_length=args.action_history_length,
        mode=args.mode,
        pred_horizon=args.pred_horizon,
        exec_horizon=args.exec_horizon,
        lr_schedule=args.lr_schedule,
        skip_first_steps=args.skip_first_steps,
        num_inference_steps=args.num_inference_steps,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
# Copyright (c) 2024-2025, SO-ARM101 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Graph-DiT (Graph Diffusion Transformer) Policy implementation.

This module implements a custom Graph-DiT policy for manipulation tasks.
Architecture:
- Last action self-attention
- EE and Object as node features (position + orientation)
- Edge features: distance + orientation similarity
- Graph attention with edge features
- Cross-attention between action and node features
- Diffusion process for action prediction
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import torch
import torch.nn as nn
import torch.nn.functional as F
from isaaclab.utils import configclass


def _quat_to_axis(quat: torch.Tensor, axis: str = "z") -> torch.Tensor:

    quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    if axis == "z":
        vec_x = 2 * (x * z - w * y)
        vec_y = 2 * (y * z + w * x)
        vec_z = 1 - 2 * (x * x + y * y)
    elif axis == "x":
        vec_x = 1 - 2 * (y * y + z * z)
        vec_y = 2 * (x * y + w * z)
        vec_z = 2 * (x * z - w * y)
    elif axis == "y":
        vec_x = 2 * (x * y - w * z)
        vec_y = 1 - 2 * (x * x + z * z)
        vec_z = 2 * (y * z + w * x)
    elif axis == "-z":
        vec_x = -2 * (x * z - w * y)
        vec_y = -2 * (y * z + w * x)
        vec_z = -(1 - 2 * (x * x + y * y))
    else:
        raise ValueError(f"Unknown axis: {axis}")
    return torch.stack([vec_x, vec_y, vec_z], dim=-1)


class ActionHistoryBuffer:
    """Buffer for maintaining action history across time steps.
    
    This class maintains a fixed-length history of actions for each environment,
    automatically updating when new actions are added and handling environment resets.
    
    Example:
        >>> buffer = ActionHistoryBuffer(history_length=4, action_dim=6, num_envs=128, device="cuda")
        >>> for step in range(num_steps):
        ...     action_history = buffer.get_history()  # [128, 4, 6]
        ...     action = policy.predict(obs, action_history=action_history)
        ...     env.step(action)
        ...     buffer.update(action)  # Update buffer with new action
        ...     if done.any():
        ...         buffer.reset(env_ids=done.nonzero().squeeze())  # Reset done environments
    """
    
    def __init__(
        self,
        history_length: int,
        action_dim: int,
        num_envs: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            history_length: Number of past actions to maintain
            action_dim: Dimension of each action
            num_envs: Number of parallel environments
            device: Device to store buffer on
            dtype: Data type for actions
        """
        self.history_length = history_length
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.dtype = dtype
        
        # PERFORMANCE FIX: Use ring buffer with write index instead of torch.cat
        # Initialize buffer with zeros: [num_envs, history_length, action_dim]
        self.buffer = torch.zeros(
            num_envs, history_length, action_dim, device=self.device, dtype=self.dtype
        )
        # Write index: tracks the next position to write (circular)
        self.write_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Track if buffer has been filled at least once (for proper history ordering)
        self.filled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
    
    def update(self, actions: torch.Tensor):
        """Add new actions to the buffer using ring buffer (no memory allocation).
        
        Args:
            actions: New actions [num_envs, action_dim] or [num_envs, 1, action_dim]
        """
        # Handle both [num_envs, action_dim] and [num_envs, 1, action_dim]
        if len(actions.shape) == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)  # [num_envs, action_dim]
        
        # Ensure correct shape
        assert actions.shape == (
            self.num_envs,
            self.action_dim,
        ), f"Expected actions shape ({self.num_envs}, {self.action_dim}), got {actions.shape}"
        
        # PERFORMANCE FIX: Ring buffer - direct write, no memory allocation
        # Write to current position
        env_indices = torch.arange(self.num_envs, device=self.device)
        self.buffer[env_indices, self.write_idx, :] = actions
        
        # Update write index (circular)
        self.write_idx = (self.write_idx + 1) % self.history_length
        # Mark as filled after first complete cycle
        self.filled = self.filled | (self.write_idx == 0)
    
    def get_history(self) -> torch.Tensor:
        """Get the current action history in chronological order.
        
        Returns:
            Action history [num_envs, history_length, action_dim]
            Ordered from oldest to newest
        """
        # PERFORMANCE FIX: Reorder ring buffer to chronological order
        # If not filled, history starts from index 0; otherwise starts from write_idx
        env_indices = torch.arange(self.num_envs, device=self.device)
        
        if self.history_length == 1:
            return self.buffer
        
        # CRITICAL FIX: Use 0 when not filled yet, otherwise use write_idx
        # This ensures correct chronological order even when buffer is not fully filled
        start = torch.where(
            self.filled, self.write_idx, torch.zeros_like(self.write_idx)
        )  # [num_envs]
        
        # Create index array for reordering
        base = torch.arange(self.history_length, device=self.device).unsqueeze(
            0
        )  # [1, history_length]
        indices = (
            base + start.unsqueeze(1)
        ) % self.history_length  # [num_envs, history_length]
        
        # Reorder buffer using advanced indexing
        reordered = self.buffer[
            env_indices.unsqueeze(1), indices, :
        ]  # [num_envs, history_length, action_dim]
        
        return reordered
    
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset action history for specified environments.
        
        Args:
            env_ids: Environment IDs to reset. If None, resets all environments.
                    Shape: [num_reset_envs] or None
        """
        if env_ids is None:
            # Reset all environments
            self.buffer.zero_()
            self.write_idx.zero_()
            self.filled.zero_()
        else:
            # Reset only specified environments
            if env_ids.dim() == 0:
                env_ids = env_ids.unsqueeze(0)  # Make it 1D
            self.buffer[env_ids].zero_()
            self.write_idx[env_ids] = 0
            self.filled[env_ids] = False
    
    def to(self, device: str | torch.device):
        """Move buffer to a different device.
        
        Args:
            device: Target device
        """
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.buffer = self.buffer.to(self.device)
        self.write_idx = self.write_idx.to(self.device)
        self.filled = self.filled.to(self.device)
        return self
    
    def __repr__(self) -> str:
        return (
            f"ActionHistoryBuffer(history_length={self.history_length}, "
            f"action_dim={self.action_dim}, num_envs={self.num_envs}, "
            f"device={self.device})"
        )


class NodeHistoryBuffer:
    """Buffer for maintaining node features history across time steps.
    
    This class maintains a fixed-length history of EE and Object node features
    for each environment, automatically updating when new observations are added
    and handling environment resets.
    
    Node features are extracted from observations: position(3) + orientation(4) = 7 dims.
    
    Example:
        >>> buffer = NodeHistoryBuffer(history_length=4, node_dim=7, num_envs=128, device="cuda")
        >>> policy = GraphDiTPolicy(cfg)
        >>> for step in range(num_steps):
        ...     obs = env.get_observations()
        ...     ee_node, object_node = policy._extract_node_features(obs)
        ...     buffer.update(ee_node, object_node)
        ...     ee_history, obj_history = buffer.get_history()
        ...     action = policy.predict(obs, ee_node_history=ee_history, object_node_history=obj_history)
        ...     env.step(action)
        ...     if done.any():
        ...         buffer.reset(env_ids=done.nonzero().squeeze())
    """
    
    def __init__(
        self,
        history_length: int,
        node_dim: int = 7,  # position(3) + orientation(4)
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            history_length: Number of past node features to maintain
            node_dim: Dimension of each node feature (default 7: pos 3 + ori 4)
            num_envs: Number of parallel environments
            device: Device to store buffer on
            dtype: Data type for node features
        """
        self.history_length = history_length
        self.node_dim = node_dim
        self.num_envs = num_envs
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.dtype = dtype
        
        # PERFORMANCE FIX: Use ring buffer with write index instead of torch.cat
        # Initialize buffers with zeros: [num_envs, history_length, node_dim]
        self.ee_buffer = torch.zeros(
            num_envs, history_length, node_dim, device=self.device, dtype=self.dtype
        )
        self.object_buffer = torch.zeros(
            num_envs, history_length, node_dim, device=self.device, dtype=self.dtype
        )
        # Write index: tracks the next position to write (circular)
        self.write_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Track if buffer has been filled at least once
        self.filled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
    
    def update(self, ee_node: torch.Tensor, object_node: torch.Tensor):
        """Add new node features to the buffer using ring buffer (no memory allocation).
        
        Args:
            ee_node: EE node features [num_envs, node_dim] or [num_envs, 1, node_dim]
            object_node: Object node features [num_envs, node_dim] or [num_envs, 1, node_dim]
        """
        # Handle both [num_envs, node_dim] and [num_envs, 1, node_dim]
        if len(ee_node.shape) == 3 and ee_node.shape[1] == 1:
            ee_node = ee_node.squeeze(1)  # [num_envs, node_dim]
        if len(object_node.shape) == 3 and object_node.shape[1] == 1:
            object_node = object_node.squeeze(1)  # [num_envs, node_dim]
        
        # Ensure correct shapes
        assert ee_node.shape == (
            self.num_envs,
            self.node_dim,
        ), f"Expected ee_node shape ({self.num_envs}, {self.node_dim}), got {ee_node.shape}"
        assert object_node.shape == (
            self.num_envs,
            self.node_dim,
        ), f"Expected object_node shape ({self.num_envs}, {self.node_dim}), got {object_node.shape}"
        
        # PERFORMANCE FIX: Ring buffer - direct write, no memory allocation
        env_indices = torch.arange(self.num_envs, device=self.device)
        self.ee_buffer[env_indices, self.write_idx, :] = ee_node
        self.object_buffer[env_indices, self.write_idx, :] = object_node
        
        # Update write index (circular)
        self.write_idx = (self.write_idx + 1) % self.history_length
        # Mark as filled after first complete cycle
        self.filled = self.filled | (self.write_idx == 0)
    
    def get_history(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the current node features history in chronological order.
        
        Returns:
            Tuple of (ee_node_history, object_node_history)
            - ee_node_history: [num_envs, history_length, node_dim] (oldest to newest)
            - object_node_history: [num_envs, history_length, node_dim] (oldest to newest)
        """
        # PERFORMANCE FIX: Reorder ring buffer to chronological order
        env_indices = torch.arange(self.num_envs, device=self.device)
        
        if self.history_length == 1:
            return self.ee_buffer, self.object_buffer
        
        # CRITICAL FIX: Use 0 when not filled yet, otherwise use write_idx
        # This ensures correct chronological order even when buffer is not fully filled
        start = torch.where(
            self.filled, self.write_idx, torch.zeros_like(self.write_idx)
        )  # [num_envs]
        
        # Create index array for reordering
        base = torch.arange(self.history_length, device=self.device).unsqueeze(
            0
        )  # [1, history_length]
        indices = (
            base + start.unsqueeze(1)
        ) % self.history_length  # [num_envs, history_length]
        
        # Reorder buffers using advanced indexing
        ee_reordered = self.ee_buffer[
            env_indices.unsqueeze(1), indices, :
        ]  # [num_envs, history_length, node_dim]
        obj_reordered = self.object_buffer[
            env_indices.unsqueeze(1), indices, :
        ]  # [num_envs, history_length, node_dim]
        
        return ee_reordered, obj_reordered
    
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset node history for specified environments.
        
        Args:
            env_ids: Environment IDs to reset. If None, resets all environments.
                    Shape: [num_reset_envs] or None
        """
        if env_ids is None:
            # Reset all environments
            self.ee_buffer.zero_()
            self.object_buffer.zero_()
            self.write_idx.zero_()
            self.filled.zero_()
        else:
            # Reset only specified environments
            if env_ids.dim() == 0:
                env_ids = env_ids.unsqueeze(0)  # Make it 1D
            self.ee_buffer[env_ids].zero_()
            self.object_buffer[env_ids].zero_()
            self.write_idx[env_ids] = 0
            self.filled[env_ids] = False
    
    def to(self, device: str | torch.device):
        """Move buffer to a different device.
        
        Args:
            device: Target device
        """
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.ee_buffer = self.ee_buffer.to(self.device)
        self.object_buffer = self.object_buffer.to(self.device)
        self.write_idx = self.write_idx.to(self.device)
        self.filled = self.filled.to(self.device)
        return self
    
    def __repr__(self) -> str:
        return (
            f"NodeHistoryBuffer(history_length={self.history_length}, "
            f"node_dim={self.node_dim}, num_envs={self.num_envs}, "
            f"device={self.device})"
        )


class JointStateHistoryBuffer:
    """Buffer for maintaining joint states (joint_pos + joint_vel) history across time steps.
    
    This class maintains a fixed-length history of joint states for each environment,
    automatically updating when new observations are added and handling environment resets.
    
    Example:
        >>> buffer = JointStateHistoryBuffer(history_length=4, joint_dim=12, num_envs=128, device="cuda")
        >>> for step in range(num_steps):
        ...     obs = env.get_observations()
        ...     joint_states = extract_joint_states(obs)  # [128, 12]
        ...     buffer.update(joint_states)
        ...     joint_history = buffer.get_history()  # [128, 4, 12]
        ...     action = policy.predict(obs, joint_states_history=joint_history)
        ...     env.step(action)
        ...     if done.any():
        ...         buffer.reset(env_ids=done.nonzero().squeeze())
    """
    
    def __init__(
        self,
        history_length: int,
        joint_dim: int,  # joint_pos_dim + joint_vel_dim (e.g., 6 + 6 = 12)
        num_envs: int = 1,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            history_length: Number of past joint states to maintain
            joint_dim: Dimension of joint states (joint_pos + joint_vel)
            num_envs: Number of parallel environments
            device: Device to store buffer on
            dtype: Data type for joint states
        """
        self.history_length = history_length
        self.joint_dim = joint_dim
        self.num_envs = num_envs
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.dtype = dtype
        
        # PERFORMANCE FIX: Use ring buffer with write index instead of torch.cat
        # Initialize buffer with zeros: [num_envs, history_length, joint_dim]
        self.buffer = torch.zeros(
            num_envs, history_length, joint_dim, device=self.device, dtype=self.dtype
        )
        # Write index: tracks the next position to write (circular)
        self.write_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Track if buffer has been filled at least once
        self.filled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
    
    def update(self, joint_states: torch.Tensor):
        """Add new joint states to the buffer using ring buffer (no memory allocation).
        
        Args:
            joint_states: New joint states [num_envs, joint_dim] or [num_envs, 1, joint_dim]
        """
        # Handle both [num_envs, joint_dim] and [num_envs, 1, joint_dim]
        if len(joint_states.shape) == 3 and joint_states.shape[1] == 1:
            joint_states = joint_states.squeeze(1)  # [num_envs, joint_dim]
        
        # Ensure correct shape
        assert joint_states.shape == (
            self.num_envs,
            self.joint_dim,
        ), f"Expected joint_states shape ({self.num_envs}, {self.joint_dim}), got {joint_states.shape}"
        
        # PERFORMANCE FIX: Ring buffer - direct write, no memory allocation
        env_indices = torch.arange(self.num_envs, device=self.device)
        self.buffer[env_indices, self.write_idx, :] = joint_states
        
        # Update write index (circular)
        self.write_idx = (self.write_idx + 1) % self.history_length
        # Mark as filled after first complete cycle
        self.filled = self.filled | (self.write_idx == 0)
    
    def get_history(self) -> torch.Tensor:
        """Get the current joint states history in chronological order.
        
        Returns:
            Joint states history [num_envs, history_length, joint_dim]
            Ordered from oldest to newest
        """
        # PERFORMANCE FIX: Reorder ring buffer to chronological order
        env_indices = torch.arange(self.num_envs, device=self.device)
        
        if self.history_length == 1:
            return self.buffer
        
        # CRITICAL FIX: Use 0 when not filled yet, otherwise use write_idx
        # This ensures correct chronological order even when buffer is not fully filled
        start = torch.where(
            self.filled, self.write_idx, torch.zeros_like(self.write_idx)
        )  # [num_envs]
        
        # Create index array for reordering
        base = torch.arange(self.history_length, device=self.device).unsqueeze(
            0
        )  # [1, history_length]
        indices = (
            base + start.unsqueeze(1)
        ) % self.history_length  # [num_envs, history_length]
        
        # Reorder buffer using advanced indexing
        reordered = self.buffer[
            env_indices.unsqueeze(1), indices, :
        ]  # [num_envs, history_length, joint_dim]
        
        return reordered
    
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset joint states history for specified environments.
        
        Args:
            env_ids: Environment IDs to reset. If None, resets all environments.
                    Shape: [num_reset_envs] or None
        """
        if env_ids is None:
            # Reset all environments
            self.buffer.zero_()
            self.write_idx.zero_()
            self.filled.zero_()
        else:
            # Reset only specified environments
            if env_ids.dim() == 0:
                env_ids = env_ids.unsqueeze(0)  # Make it 1D
            self.buffer[env_ids].zero_()
            self.write_idx[env_ids] = 0
            self.filled[env_ids] = False
    
    def to(self, device: str | torch.device):
        """Move buffer to a different device.
        
        Args:
            device: Target device
        """
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.buffer = self.buffer.to(self.device)
        self.write_idx = self.write_idx.to(self.device)
        self.filled = self.filled.to(self.device)
        return self
    
    def __repr__(self) -> str:
        return (
            f"JointStateHistoryBuffer(history_length={self.history_length}, "
            f"joint_dim={self.joint_dim}, num_envs={self.num_envs}, "
            f"device={self.device})"
        )


@configclass
class GraphDiTPolicyCfg:
    """Configuration for Graph-DiT Policy."""
    
    obs_dim: int = MISSING
    """Observation dimension (input to policy). Note: obs should be dict or separable."""
    
    action_dim: int = MISSING
    """Action dimension (output from policy). Typically 5 (arm joints only); gripper is handled separately (e.g. gripper_model in play)."""
    
    # Observation structure indices (for backward compatibility with flattened obs)
    # CRITICAL: These should match the actual observation structure
    # NOTE: joint_vel removed - only using joint_pos
    # Default assumes: [joint_pos(6), object_pos(3), object_ori(4),
    #                  ee_pos(3), ee_ori(4), target_object_position(7), actions(6)]
    obs_structure: dict[str, tuple[int, int]] | None = None
    """
    Observation structure mapping for flattened observations.
    If None, uses default indices. If provided, should be a dict like:
    {
        'joint_pos': (0, 6),
        # NOTE: 'joint_vel' removed - no longer used
        'object_position': (12, 15),
        'object_orientation': (15, 19),
        'ee_position': (19, 22),
        'ee_orientation': (22, 26),
        'actions': (26, 32),
    }
    """
    
    # Graph-DiT specific parameters
    hidden_dim: int = 256
    """Hidden dimension for Graph-DiT."""

    z_dim: int = 64
    """Graph latent dimension for RL heads (layer-wise z^k pooled from node features)."""
    
    num_layers: int = 6
    """Number of Graph-DiT layers."""
    
    num_heads: int = 8
    """Number of attention heads."""
    
    graph_edge_dim: int = 128
    """Dimension for graph edge embeddings."""
    
    diffusion_steps: int = 200
    """Number of diffusion steps."""
    
    num_inference_steps: int = 40
    """Number of inference steps for flow matching prediction.
    
    This controls how many ODE integration steps are used during inference.
    More steps = smoother, more accurate predictions but slower.
    Default: 30.
    """
    
    num_subtasks: int = 1
    """Number of subtasks (for conditional generation).
    
    This parameter enables multi-stage task conditioning (e.g., lift_object, place_object).
    During training, the actual number is dynamically read from the dataset.
    If the dataset has no subtasks, this will be set to 0 and subtask conditioning is disabled.
    Default is 2 to support future multi-stage tasks (currently datasets have 1 subtask: 'lift_object').
    """
    
    device: str = "cuda"
    """Device to run on."""
    
    # Node and edge dimensions
    node_dim: int = 7
    """Node feature dimension: position(3) + orientation(4) = 7"""
    
    edge_dim: int = 2
    """Edge feature dimension: distance(1) + orientation_similarity(1) = 2"""

    edge_feature_version: str = "v2"
    """Edge feature version: 'v1' (quat similarity) or 'v2' (best face alignment, robust to object tilt)."""
    
    joint_dim: int | None = None
    """Joint states dimension (input only). Typically 6 (5 arm + 1 gripper). Policy uses this as context; it does not predict gripper. If None, joint states are not used."""

    use_joint_film: bool = False
    """When True, encode joint_states_history and inject into U-Net via FiLM
    (concatenated with graph latent z). Only affects GraphUnetPolicy."""

    # Dynamic graph configuration
    num_nodes: int = 2
    """Number of graph nodes. Default 2 for backward compat (1 EE + 1 Object)."""

    num_node_types: int = 2
    """Number of distinct node types (ee=0, object=1). Drives a learnable type embedding."""

    graph_pool_mode: str = "concat"
    """Node pooling mode for producing graph latent z.
    'concat': [B, N, D] -> reshape [B, N*D] -> MLP (requires fixed N, backward-compat).
    'mean':   [B, N, D] -> mean(dim=1) -> [B, D] -> MLP (works with any N).
    """

    node_configs: list | None = None
    """Describes each node's obs keys for the data pipeline.
    If None, uses the old 2-node (ee + object) code path.
    Example for dual-arm:
    [
        {"name": "left_ee",  "type": 0, "pos_key": "left_ee_position",  "ori_key": "left_ee_orientation"},
        {"name": "right_ee", "type": 0, "pos_key": "right_ee_position", "ori_key": "right_ee_orientation"},
        {"name": "cube_1",   "type": 1, "pos_key": "cube_1_pos",        "ori_key": "cube_1_ori"},
        {"name": "cube_2",   "type": 1, "pos_key": "cube_2_pos",        "ori_key": "cube_2_ori"},
    ]
    """

    # Action history
    action_history_length: int = 10
    """Number of historical actions to use for self-attention."""
    
    # Action Chunking (Receding Horizon Control)
    pred_horizon: int = 20
    """Prediction horizon: number of future action steps to predict at once.
    
    This is the core of Diffusion Policy's "Action Chunking" mechanism.
    The model predicts pred_horizon steps of actions in one forward pass.
    Typical values: 16 for 50Hz control (320ms lookahead).
    """
    
    exec_horizon: int = 10
    """Execution horizon: number of predicted actions to actually execute.
    
    After predicting pred_horizon steps, we execute the first exec_horizon steps,
    then re-predict. This is the "Receding Horizon Control" (RHC) mechanism.
    
    Rule of thumb: exec_horizon = pred_horizon // 2
    This provides temporal consistency while allowing course correction.
    """
    
    # Dual-arm UNet (DualArmUnetPolicy)
    arm_action_dim: int | None = None
    """Per-arm action dimension (e.g. 6 for left/right). Set when action_dim=12 and using DualArmUnetPolicy."""
    cross_arm_heads: int = 4
    """Number of attention heads for CrossArmAttention at bottleneck."""
    use_cross_arm_attn: bool = True
    """Whether to use CrossArmAttention at bottleneck. False: 左右臂独立，仅从 graph EE 推断（stack 任务 pick/stack 顺序执行，无协同）。"""
    use_raw_only: bool = False
    """When True, use DualArmUnetPolicyRawOnly: no graph encoder, only raw node projection into UNet."""
    
    # Edge-Conditioned Modulation (ECC-style)
    use_edge_modulation: bool = True
    """
    Whether to use Edge-Conditioned Modulation (ECC-style).
    
    When True (recommended):
    - Edge features generate gates/scales that directly control Value transformation
    - Edge acts as "controller" rather than just "participant"
    - Stronger inductive bias: Edge directly controls information flow
    
    When False (baseline):
    - Edge features only used as attention bias
    - Network must learn how to use edge information
    """


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization that modulates based on condition.
    
    CRITICAL FIX: Strict shape contract - condition must be [B, C], no silent truncation.
    Uses explicit view_shape for broadcasting instead of implicit unsqueeze loops.
    """
    
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # CRITICAL FIX: Use 2H instead of 6H (only need scale + shift)
        # If you need full DiT-style 6H (gate/scale/shift for MSA/MLP), implement separately
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * hidden_dim, bias=True),  # [B, C] -> [B, 2H]
        )
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim] or [batch, num_nodes, seq_len, hidden_dim] etc.
            condition: Condition tensor [batch, condition_dim] (MUST be 2D, strict contract)
        """
        # CRITICAL FIX: Strict shape validation - no silent truncation
        if condition.dim() != 2:
            raise ValueError(
                f"AdaptiveLayerNorm: condition must be [B, C], got shape {condition.shape}. "
                f"Expected 2D tensor, got {condition.dim()}D."
            )
        if condition.shape[0] != x.shape[0]:
            raise ValueError(
                f"AdaptiveLayerNorm: batch size mismatch - x has {x.shape[0]} samples, "
                f"condition has {condition.shape[0]} samples."
            )
        if condition.shape[1] != self.condition_dim:
            raise ValueError(
                f"AdaptiveLayerNorm: condition feature dim mismatch - expected {self.condition_dim}, "
                f"got {condition.shape[1]}."
            )
        
        # Compute scale and shift from condition
        condition_embed = self.adaLN_modulation(condition)  # [batch, 2*hidden_dim]
        
        # Extract scale and shift
        hidden_dim = x.shape[-1]
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"AdaptiveLayerNorm: x hidden_dim {hidden_dim} != expected {self.hidden_dim}"
            )
        
        # Check condition_embed dimension
        expected_embed_dim = 2 * hidden_dim
        if condition_embed.shape[1] != expected_embed_dim:
            raise ValueError(
                f"AdaptiveLayerNorm: condition_embed shape {condition_embed.shape} != expected [batch, {expected_embed_dim}]. "
                f"condition shape: {condition.shape}, condition_dim: {self.condition_dim}, hidden_dim: {self.hidden_dim}"
            )
        
        scale = condition_embed[:, :hidden_dim]  # [batch, hidden_dim]
        shift = condition_embed[:, hidden_dim : 2 * hidden_dim]  # [batch, hidden_dim]
        
        # CRITICAL FIX: Explicit view_shape for broadcasting (no implicit unsqueeze loops)
        # Construct view_shape: [B, 1, ..., 1, H] to match x's dimensions
        # x shape: [B, dim1, dim2, ..., H] -> view_shape: [B, 1, 1, ..., H]
        view_shape = [x.shape[0]] + [1] * (x.dim() - 2) + [hidden_dim]
        scale = scale.view(*view_shape)  # Explicit reshape for broadcasting
        shift = shift.view(*view_shape)  # Explicit reshape for broadcasting
        
        # Normalize and modulate
        x_norm = self.norm(x)
        x = (1 + scale) * x_norm + shift
        return x


class TemporalAggregator(nn.Module):
    """Learnable temporal aggregation module.
    
    Uses attention mechanism to aggregate temporal features across history dimension,
    replacing hardcoded operations like mean or last-timestep selection.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate temporal features using learnable attention.
        
        Args:
            x: Input tensor [B, N, H, D] where
               B = batch size
               N = number of nodes
               H = history length (temporal dimension)
               D = hidden dimension
        Returns:
            Aggregated tensor [B, N, D] with temporal dimension collapsed
        """
        B, N, H, D = x.shape
        x_flat = x.reshape(B * N, H, D)  # [B*N, H, D] (reshape handles non-contiguous)
        query = self.query.expand(B * N, -1, -1)  # [B*N, 1, D]
        aggregated, _ = self.attn(query, x_flat, x_flat)  # [B*N, 1, D]
        # aggregated is [B*N, 1, D], squeeze(1) -> [B*N, D], then reshape -> [B, N, D]
        aggregated = aggregated.squeeze(1)  # [B*N, D]
        return aggregated.reshape(B, N, D)


class GraphAttentionWithEdgeBias(nn.Module):
    """
    Graph attention layer with edge features as attention bias.
    
    ARCHITECTURE EVOLUTION: This class implements Edge-Conditioned Modulation (ECC-style),
    where Edge features act as "controllers" rather than just "participants".
    
    Key Innovation:
    - Baseline: Edge → Bias (added to attention scores) - Edge is "participant"
    - ECC: Edge → Gate/Scale/Shift (modulates Value) - Edge is "controller"
    
    This provides stronger inductive bias: Edge directly controls information flow,
    rather than requiring the network to learn how to use edge information.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int = 128,
        max_history: int = 20,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_history = max_history
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # ========== NEW: Temporal Components ==========
        # 1. Temporal position encoding
        self.temporal_pos_embedding = nn.Embedding(max_history, hidden_dim)
        
        # 2. Temporal bias
        self.temporal_bias_embedding = nn.Embedding(2 * max_history - 1, num_heads)
        
        # 3. Learnable temporal aggregator
        self.temporal_aggregator = TemporalAggregator(hidden_dim, num_heads)
        # ==============================================
        
        # Edge feature to attention bias (for spatial bias)
        self.edge_to_bias = nn.Linear(edge_dim, num_heads)
        
        # ========== Temporal Edge Modulation (always enabled) ==========
        # Process ALL temporal edge features with GRU to capture dynamics
        # 1. Temporal edge encoder: process edge sequence
        self.edge_temporal_encoder = nn.GRU(
            input_size=edge_dim,
            hidden_size=edge_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        
        # 2. Per-timestep modulation: each timestep gets its own gate/scale/shift
        self.edge_to_gate = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.Sigmoid()
        )
        self.edge_to_scale = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.Tanh()
        )
        self.edge_to_shift = nn.Linear(edge_dim, hidden_dim)
        # 3. Global modulation: aggregated edge info for head scaling
        self.edge_to_head_scale = nn.Sequential(
            nn.Linear(edge_dim, num_heads), nn.Tanh()
        )
        # ========================================================

        self.dropout = nn.Dropout(0.1)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self, 
        node_features: torch.Tensor,   # [B, N, H, D]
        edge_features: torch.Tensor,    # [B, H, edge_dim] (legacy 2-node)
                                        # OR [B, N, N, H, edge_dim] (dynamic N-node)
    ) -> torch.Tensor:
        """
        Graph attention with temporal edge modulation.

        Args:
            node_features: [B, N, H, D]
            edge_features: [B, H, edge_dim] (legacy) or [B, N, N, H, edge_dim] (dynamic)
        Returns:
            [B, N, H, D]
        """
        assert node_features.dim() == 4, f"node_features must be [B,N,H,D], got {node_features.shape}"
        batch_size, num_nodes, history_length, hidden_dim = node_features.shape
        device = node_features.device

        # Detect edge format
        dynamic_edges = edge_features.dim() == 5  # [B, N, N, H, edge_dim]

        if dynamic_edges:
            # Aggregate per-node edge context: for node i, mean of incident edges
            # edge_features: [B, N, N, H, edge_dim]
            # Sum over neighbor dim (j), excluding self (diagonal is 0) -> [B, N, H, edge_dim]
            edge_sum = edge_features.sum(dim=2)  # [B, N, H, edge_dim]
            n_neighbors = (num_nodes - 1) if num_nodes > 1 else 1
            node_edge_ctx = edge_sum / n_neighbors  # [B, N, H, edge_dim]

            # Process each node's edge context through GRU: [B*N, H, edge_dim]
            ctx_flat = node_edge_ctx.reshape(batch_size * num_nodes, history_length, -1)
            ctx_encoded, ctx_hidden = self.edge_temporal_encoder(ctx_flat)
            # ctx_encoded: [B*N, H, edge_dim], ctx_hidden: [1, B*N, edge_dim]

            gate = self.edge_to_gate(ctx_encoded).view(batch_size, num_nodes, history_length, hidden_dim)
            scale = self.edge_to_scale(ctx_encoded).view(batch_size, num_nodes, history_length, hidden_dim)
            shift = self.edge_to_shift(ctx_encoded).view(batch_size, num_nodes, history_length, hidden_dim)

            # Head scale from mean of all node edge summaries
            edge_summary = ctx_hidden.squeeze(0).view(batch_size, num_nodes, -1).mean(dim=1)
            head_scale = self.edge_to_head_scale(edge_summary)  # [B, num_heads]

            # For attention bias, use the raw per-pair edge features
            bias_edge_input = edge_features  # [B, N, N, H, edge_dim]
        else:
            assert edge_features.dim() == 3
            assert edge_features.shape[1] == history_length

            edge_encoded, edge_hidden = self.edge_temporal_encoder(edge_features)

            gate = self.edge_to_gate(edge_encoded).unsqueeze(1)    # [B, 1, H, D]
            scale = self.edge_to_scale(edge_encoded).unsqueeze(1)
            shift = self.edge_to_shift(edge_encoded).unsqueeze(1)

            edge_summary = edge_hidden.squeeze(0)
            head_scale = self.edge_to_head_scale(edge_summary)

            bias_edge_input = edge_features  # [B, H, edge_dim]

        # Temporal position encoding
        time_indices = torch.arange(history_length, device=device)
        temporal_pos = self.temporal_pos_embedding(time_indices)
        node_features = node_features + temporal_pos.view(1, 1, history_length, hidden_dim)

        # Q, K, V
        seq_len = num_nodes * history_length
        node_flat = node_features.view(batch_size, seq_len, hidden_dim)
        Q = self.q_proj(node_flat)
        K = self.k_proj(node_flat)
        V = self.v_proj(node_flat)

        # Edge modulation on V
        V_reshaped = V.view(batch_size, num_nodes, history_length, hidden_dim)
        V_modulated = V_reshaped * (1.0 + scale) + shift
        V_modulated = V_modulated * gate
        V = V_modulated.view(batch_size, seq_len, hidden_dim)

        # Multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if head_scale is not None:
            scores = scores * (1.0 + head_scale.unsqueeze(-1).unsqueeze(-1))

        attention_bias = self._build_attention_bias(
            batch_size, num_nodes, history_length, bias_edge_input, device
        )
        scores = scores + attention_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        out_temporal = out.view(batch_size, num_nodes, history_length, self.hidden_dim)
        out_temporal = self.out_proj(
            out_temporal.view(batch_size * num_nodes * history_length, self.hidden_dim)
        ).view(batch_size, num_nodes, history_length, self.hidden_dim)

        return out_temporal

    def _build_attention_bias(
        self, batch_size, num_nodes, history_length,
        edge_features: torch.Tensor,
        device: torch.device,
    ):
        """Build combined spatial + temporal attention bias.

        Args:
            edge_features: [B, H, edge_dim] (legacy 2-node)
                       OR  [B, N, N, H, edge_dim] (dynamic N-node)
        """
        seq_len = num_nodes * history_length
        attention_bias = torch.zeros(
            batch_size, self.num_heads, seq_len, seq_len, device=device
        )
        t_indices = torch.arange(history_length, device=device)

        if edge_features.dim() == 5:
            # Dynamic: [B, N, N, H, edge_dim]
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    pair_edge = edge_features[:, i, j]  # [B, H, edge_dim]
                    bias_ij = self.edge_to_bias(
                        pair_edge.reshape(-1, pair_edge.shape[-1])
                    ).view(batch_size, history_length, self.num_heads)
                    bias_ij = bias_ij.permute(0, 2, 1)  # [B, heads, H]

                    ni = i * history_length + t_indices
                    nj = j * history_length + t_indices
                    attention_bias[:, :, ni, nj] = bias_ij
                    attention_bias[:, :, nj, ni] = bias_ij
        else:
            # Legacy: [B, H, edge_dim] — single edge for node0↔node1
            edge_bias = self.edge_to_bias(
                edge_features.view(-1, edge_features.shape[-1])
            ).view(batch_size, history_length, self.num_heads)
            edge_bias = edge_bias.permute(0, 2, 1)  # [B, heads, H]

            node0_indices = t_indices
            node1_indices = history_length + t_indices
            attention_bias[:, :, node0_indices, node1_indices] = edge_bias
            attention_bias[:, :, node1_indices, node0_indices] = edge_bias

        # Temporal bias: based on time difference
        time_all = torch.arange(history_length, device=device).repeat(num_nodes)
        time_diff = time_all.unsqueeze(0) - time_all.unsqueeze(1)
        time_diff_shifted = (time_diff + self.max_history - 1).clamp(0, 2 * self.max_history - 2)

        temporal_bias = self.temporal_bias_embedding(time_diff_shifted)
        temporal_bias = temporal_bias.permute(2, 0, 1).unsqueeze(0)

        attention_bias = attention_bias + temporal_bias
        return attention_bias


class GraphDiTUnit(nn.Module):
    """Single Graph DiT unit following the architecture.
    
    Steps:
    1. Last action self-attention → a_new
    2. Node attention with edge features → node_features_new
    3. Cross-attention (a_new × node_features_new) → noise_pred
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_dim: int = 128,
        joint_dim: int | None = None,
        max_history: int = 20,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_history = max_history
        
        # Temporal position encoding for action sequence
        self.action_temporal_pos = nn.Embedding(max_history + 1, hidden_dim)  # +1 for noisy_action
        
        # Step 1: State-action sequence self-attention
        # Now processes (joint_states, action) concatenated sequences
        self.action_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        # AdaLN dynamically adjusts scale and shift based on timestep condition
        self.action_norm1 = AdaptiveLayerNorm(
            hidden_dim, hidden_dim
        )  # condition_dim = hidden_dim
        self.action_norm2 = AdaptiveLayerNorm(hidden_dim, hidden_dim)
        self.action_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1),
        )
        
        # Step 2: Graph attention with edge features (always uses temporal edge modulation)
        self.graph_attention = GraphAttentionWithEdgeBias(
            hidden_dim, num_heads, edge_dim, 
            max_history=max_history,
        )
        # Use AdaLN for node features as well
        self.node_norm1 = AdaptiveLayerNorm(hidden_dim, hidden_dim)
        self.node_norm2 = AdaptiveLayerNorm(hidden_dim, hidden_dim)
        self.node_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1),
        )
        
        # Step 3: Cross-attention (action queries node features + joint velocity memory)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        # Use AdaLN for cross-attention
        self.cross_norm1 = AdaptiveLayerNorm(hidden_dim, hidden_dim)
        self.cross_norm2 = AdaptiveLayerNorm(hidden_dim, hidden_dim)
        self.cross_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1),
        )
    
    def forward(
        self,
        action: torch.Tensor,
        node_features: torch.Tensor,  # [B, 2, H, D] - must be 4D temporal
        edge_features: torch.Tensor,   # [B, H, edge_dim] - must be 3D temporal
        timestep_embed: torch.Tensor | None = None,
        joint_states_history: torch.Tensor | None = None,
    ):
        """
        Simplified forward: only supports temporal mode.
        
        Args:
            action: Action sequence [batch, seq_len, hidden_dim]
            node_features: Node features [batch, 2, history_length, hidden_dim] - must be 4D
            edge_features: Edge features [batch, history_length, edge_dim] - must be 3D temporal
            timestep_embed: Timestep embedding [batch, hidden_dim] (optional)
            joint_states_history: Ignored (kept for API compat). Causal confusion analysis
                showed that injecting joint states causes the network to shortcut via
                auto-regressive echoing, killing gripper learning.
        Returns:
            noise_pred: Predicted noise [batch, seq_len, hidden_dim]
            node_features_new: [batch, 2, history_length, hidden_dim]
        """
        device = action.device
        dtype = action.dtype
        batch_size = action.shape[0]
        condition_emb = timestep_embed if timestep_embed is not None else None
        
        # Ensure action is 3D: [batch, seq_len, hidden_dim]
        if len(action.shape) == 2:
            action = action.unsqueeze(1)  # [batch, hidden_dim] -> [batch, 1, hidden_dim]
        seq_len = action.shape[1]
        
        # Action-only self-attention (no joint states to avoid causal confusion)
        state_action_seq = action  # [batch, seq_len, hidden_dim]
        
        # Add temporal position encoding to action sequence
        actual_seq_len = state_action_seq.shape[1]
        time_indices = torch.arange(actual_seq_len, device=state_action_seq.device)
        # Clamp indices to prevent out-of-bounds access
        max_pos = self.action_temporal_pos.num_embeddings - 1
        time_indices = time_indices.clamp(0, max_pos)
        action_temporal_pos = self.action_temporal_pos(time_indices)  # [seq_len, D]
        state_action_seq = state_action_seq + action_temporal_pos.unsqueeze(0)
        # ============================================================================
        
        # Self-attention on state-action sequence
        state_action_residual = state_action_seq
        cond = self._get_condition(condition_emb, batch_size, device, dtype)
        state_action_seq = self.action_norm1(state_action_seq, cond)
        
        a_new, _ = self.action_self_attn(
            state_action_seq, state_action_seq, state_action_seq
        )  # [batch, seq_len, hidden_dim]
        a_new = a_new + state_action_residual
        
        action_residual = a_new  # [batch, seq_len, hidden_dim]
        a_new = self.action_norm2(a_new, cond)
        a_new = self.action_ff(a_new) + action_residual  # [batch, seq_len, hidden_dim]
        
        # ==================== Step 2: Graph Attention ====================
        # Validate input shapes
        assert node_features.dim() == 4, f"node_features must be [B,N,H,D], got {node_features.shape}"
        assert edge_features.dim() == 3, f"edge_features must be [B,H,edge_dim], got {edge_features.shape}"
        
        B, N, H, D = node_features.shape
        assert edge_features.shape[1] == H, f"Edge history {edge_features.shape[1]} must match node history {H}"
        
        node_residual = node_features  # [B, N, H, D]
        
        # Use AdaLN for node features
        cond = self._get_condition(condition_emb, batch_size, device, dtype)
        node_features_for_attn = self.node_norm1(node_features, cond)

        # Graph attention (always returns [B, N, H, D])
        node_features_new = self.graph_attention(
            node_features_for_attn, edge_features
        )
        
        # Add residual (both are [B, N, H, D])
        node_features_new = node_features_new + node_residual
        
        # FFN: process each timestep independently
        B, N, H, D = node_features_new.shape
        node_features_flat = node_features_new.view(B * N * H, D)
        node_residual_flat = node_features_flat
        
        cond = self._get_condition(condition_emb, batch_size, device, dtype)
        # AdaLN expects 3D input [B, seq_len, D], so reshape temporarily
        node_features_for_norm = node_features_flat.view(B, N * H, D)
        node_features_for_norm = self.node_norm2(node_features_for_norm, cond)
        node_features_flat = node_features_for_norm.view(B * N * H, D)
        node_features_flat = self.node_ff(node_features_flat) + node_residual_flat
        node_features_new = node_features_flat.view(B, N, H, D)
        
        # ==================== Step 3: Cross-Attention ====================
        cross_residual = a_new
        cond = self._get_condition(condition_emb, batch_size, device, dtype)
        a_new_norm = self.cross_norm1(a_new, cond)
        
        # Build KV memory from full temporal node features
        # node_features_new is [B, N, H, D], flatten to [B, N*H, D]
        B, N, H, D = node_features_new.shape
        kv_memory = node_features_new.view(B, N * H, D)
        
        noise_embed, _ = self.cross_attn(a_new_norm, kv_memory, kv_memory)
        noise_embed = noise_embed + cross_residual
        
        cross_residual = noise_embed
        noise_embed = self.cross_norm2(noise_embed, cond)
        noise_embed = self.cross_ff(noise_embed) + cross_residual
        
        return noise_embed, node_features_new
    
    def _get_condition(self, condition_emb, batch_size, device, dtype):
        """Helper to get condition embedding or zeros"""
        if condition_emb is not None:
            return condition_emb
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)


class GraphDiTPolicy(nn.Module):
    """Graph-DiT (Graph Diffusion Transformer) Policy.
    
    Architecture:
    - Extracts EE and Object as node features (position + orientation)
    - Computes edge features (distance + orientation similarity)
    - Uses Graph DiT units with diffusion process
    """
    
    def __init__(self, cfg: GraphDiTPolicyCfg):
        super().__init__()
        self.cfg = cfg
        
        # Node embedding: position(3) + orientation(4) = 7 -> hidden_dim
        self.node_embedding = nn.Sequential(
            nn.Linear(cfg.node_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )

        # Node type embedding: distinguishes EE nodes from Object nodes
        num_node_types = getattr(cfg, "num_node_types", 2)
        self.node_type_embedding = nn.Embedding(num_node_types, cfg.hidden_dim)

        # Dynamic graph params
        num_nodes = getattr(cfg, "num_nodes", 2)
        pool_mode = getattr(cfg, "graph_pool_mode", "concat")

        # Handle backward compatibility: old checkpoints may not have z_dim
        z_dim = getattr(cfg, 'z_dim', 128)  # Default to 128 if not present
        
        # Learnable temporal aggregator for pooling node features
        self.node_temporal_aggregator = TemporalAggregator(cfg.hidden_dim, cfg.num_heads)

        # node_to_z input dim depends on pool mode
        if pool_mode == "mean":
            pool_input_dim = cfg.hidden_dim
        else:  # "concat" (backward-compat default)
            pool_input_dim = cfg.hidden_dim * num_nodes

        self.node_to_z = nn.Sequential(
            nn.Linear(pool_input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, z_dim),
        )
        
        # Edge embedding: [distance, orientation_similarity] -> edge_dim (no delta, avoid extra noise)
        self.edge_embedding = nn.Sequential(
            nn.Linear(cfg.edge_dim, cfg.graph_edge_dim),  # 2 inputs
            nn.LayerNorm(cfg.graph_edge_dim),
            nn.GELU(),
        )
        
        # Action embedding (for single action)
        self.action_embedding = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )
        
        # Action history length and prediction horizon
        self.action_history_length = cfg.action_history_length
        self.pred_horizon = cfg.pred_horizon
        self.exec_horizon = cfg.exec_horizon
        
        # CRITICAL: Prediction Horizon Position Embedding
        # This tells the Transformer which timestep in the future each action token represents
        # Without this, the model cannot distinguish t+1 from t+16
        self.register_buffer(
            "pred_horizon_pos_embed",
            self._get_position_embedding(cfg.pred_horizon, cfg.hidden_dim),
        )
        
        # Note: context_encoder removed - joint states are now handled via joint_states_history
        # which is processed in GraphDiTUnit through state-action sequence self-attention
        
        # Timestep embedding (for diffusion)
        self.timestep_embedding = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim),
        )
        # Sinusoidal position embedding for timesteps
        self.register_buffer(
            "timestep_pos_embed",
            self._get_timestep_embedding(cfg.diffusion_steps, cfg.hidden_dim),
        )
        
        # Subtask condition encoder
        if cfg.num_subtasks > 0:
            self.subtask_encoder = nn.Sequential(
                nn.Linear(cfg.num_subtasks, cfg.hidden_dim // 4),
                nn.LayerNorm(cfg.hidden_dim // 4),
                nn.GELU(),
            )
            # Projection for combined timestep + subtask
            self.condition_proj = nn.Linear(
                cfg.hidden_dim + cfg.hidden_dim // 4, cfg.hidden_dim
            )
            # Separate projection for subtask-only (when no timestep)
            self.subtask_only_proj = nn.Sequential(
                nn.Linear(cfg.hidden_dim // 4, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
            )
        
        # Graph DiT units (stacked layers)
        # CRITICAL: max_history must be large enough to cover action_history_length + pred_horizon
        # The action sequence can be [history] + [pred_horizon] = action_history_length + pred_horizon
        max_history = max(cfg.action_history_length + cfg.pred_horizon, 20)  # Ensure enough capacity
        
        self.graph_dit_units = nn.ModuleList(
            [
            GraphDiTUnit(
                cfg.hidden_dim,
                cfg.num_heads,
                cfg.graph_edge_dim,
                max_history=max_history,
            )
            for _ in range(cfg.num_layers)
            ]
        )
        
        # Final noise prediction head
        # CRITICAL: Now outputs [batch, pred_horizon, action_dim] for trajectory prediction
        self.noise_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _get_timestep_embedding(self, num_steps: int, dim: int):
        """Create sinusoidal timestep embeddings for diffusion timesteps."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = torch.arange(num_steps, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(
            0
        )
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_steps, 1)], dim=1)
        return emb  # [num_steps, hidden_dim]
    
    def _get_position_embedding(self, seq_len: int, dim: int):
        """Create sinusoidal position embeddings for action trajectory positions.
        
        This tells the model which future timestep each action token represents.
        Position 0 = t+1 (next step), Position H-1 = t+H (furthest future).
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        emb = pos * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(seq_len, 1)], dim=1)
        return emb  # [seq_len, hidden_dim]
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
    
    def _extract_node_features(self, obs: torch.Tensor):
        """
        Extract node features from concatenated observations.
        
        CRITICAL FIX: Uses configurable indices instead of hardcoded slicing.
        This prevents silent failures when observation structure changes.
        
        Args:
            obs: [batch, obs_dim]
        Returns:
            ee_node: [batch, 7] - EE position + orientation
            object_node: [batch, 7] - Object position + orientation
        """
        # Use configurable indices if provided, otherwise use defaults
        if self.cfg.obs_structure is not None:
            # Use configured structure
            obs_struct = self.cfg.obs_structure
            object_position = obs[
                :, obs_struct["object_position"][0] : obs_struct["object_position"][1]
            ]
            object_orientation = obs[
                :,
                obs_struct["object_orientation"][0] : obs_struct["object_orientation"][
                    1
                ],
            ]
            ee_position = obs[
                :, obs_struct["ee_position"][0] : obs_struct["ee_position"][1]
            ]
            ee_orientation = obs[
                :, obs_struct["ee_orientation"][0] : obs_struct["ee_orientation"][1]
            ]
        else:
            # Default structure (backward compatibility)
            # WARNING: These hardcoded indices assume a specific observation structure!
            # If your observation structure changes, you MUST update obs_structure in config
            # Default assumes: [joint_pos(6), joint_vel(6), object_pos(3), object_ori(4),
            #                  ee_pos(3), ee_ori(4), target_object_position(7), actions(6)]
            object_position = obs[:, 12:15]  # [batch, 3]
            object_orientation = obs[:, 15:19]  # [batch, 4]
            ee_position = obs[:, 19:22]  # [batch, 3]
            ee_orientation = obs[:, 22:26]  # [batch, 4]
        
        # Construct nodes
        ee_node = torch.cat([ee_position, ee_orientation], dim=-1)  # [batch, 7]
        object_node = torch.cat(
            [object_position, object_orientation], dim=-1
        )  # [batch, 7]
        
        return ee_node, object_node
    
    def _extract_joint_states(self, obs: torch.Tensor):
        """
        Extract joint states (only joint_pos, joint_vel removed to test if it's noise).
        
        Args:
            obs: [batch, obs_dim]
        Returns:
            joint_states: [batch, joint_dim] - only joint_pos (no joint_vel).
                joint_dim is typically 6 (5 arm + 1 gripper); policy output is 5 (arm only).
        """
        # Use configurable indices if provided, otherwise use defaults
        if self.cfg.obs_structure is not None:
            obs_struct = self.cfg.obs_structure
            joint_pos = obs[:, obs_struct["joint_pos"][0] : obs_struct["joint_pos"][1]]
            # NOTE: joint_vel removed - only using joint_pos
        else:
            # Default structure: [joint_pos(6), ...] (joint_vel removed). 6 = 5 arm + 1 gripper.
            joint_pos = obs[:, 0:6]  # [batch, 6]
        
        # Only return joint_pos (no joint_vel). Input 6-dim; policy outputs 5-dim (arm only).
        joint_states = joint_pos  # [batch, joint_dim] where joint_dim = 6
        return joint_states
    
    def _extract_context_features(self, obs: torch.Tensor):
        """
        Extract context features from observations (only joint_pos, joint_vel removed).
        These are used as additional context but not as graph nodes.
        
        Args:
            obs: [batch, obs_dim]
        Returns:
            context_features: [batch, context_dim] - joint_pos + other features (no joint_vel)
        """
        # Use configurable indices if provided, otherwise use defaults
        if self.cfg.obs_structure is not None:
            obs_struct = self.cfg.obs_structure
            # Extract all features except node features (EE, Object) and action
            # NOTE: joint_vel removed - only using joint_pos
            context_parts = []
            if "joint_pos" in obs_struct:
                context_parts.append(
                    obs[:, obs_struct["joint_pos"][0] : obs_struct["joint_pos"][1]]
                )
            # NOTE: joint_vel removed - no longer included
            if "target_object_position" in obs_struct:
                context_parts.append(
                    obs[
                        :,
                        obs_struct["target_object_position"][0] : obs_struct[
                            "target_object_position"
                        ][1],
                    ]
                )
            # Add other context features if needed
            
            if context_parts:
                context_features = torch.cat(context_parts, dim=-1)
            else:
                # Fallback: extract everything except node features and action
                # Default: [joint_pos(6), joint_vel(6), ..., target_object_position(7), ...]
                # Skip: object_pos(3), object_ori(4), ee_pos(3), ee_ori(4), actions(6)
                # This is obs_dim - node_dim*2 - action_dim
                context_dim = (
                    self.cfg.obs_dim - self.cfg.node_dim * 2 - self.cfg.action_dim
                )
                if context_dim > 0:
                    # Extract from beginning (joint_pos, joint_vel) and potentially middle (target_object_position)
                    # This is a simplified extraction - may need adjustment based on actual obs structure
                    context_features = obs[
                        :, :context_dim
                    ]  # Take first context_dim features
                else:
                    # No context features available
                    context_features = torch.zeros(
                        obs.shape[0], 0, device=obs.device, dtype=obs.dtype
                    )
        else:
            # Default structure: [joint_pos(6), object_pos(3), object_ori(4),
            #                     ee_pos(3), ee_ori(4), target_object_position(7), actions(6)]
            # NOTE: joint_vel removed - only using joint_pos
            # Extract: joint_pos(6) + target_object_position(7) = 13 dims (if target_object_position exists)
            # But we skip object_pos, object_ori, ee_pos, ee_ori, actions
            joint_pos = obs[:, 0:6]  # [batch, 6]
            # NOTE: joint_vel removed - no longer extracted
            # target_object_position might be at different position, skip for now
            # Or extract if we know the exact position
            context_features = joint_pos  # [batch, 6] (only joint_pos, no joint_vel)
        
        return context_features
    
    def _pool_node_latent(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Pool node features into a compact graph latent z.

        Args:
            node_features: [B, N, hidden_dim] or [B, N, H, hidden_dim] (full temporal)
        Returns:
            z: [B, z_dim]
        """
        if node_features.dim() == 4:
            # Temporal: [B, N, H, D] -> aggregate to [B, N, D]
            node_features = self.node_temporal_aggregator(node_features)
        elif node_features.dim() == 3:
            pass
        else:
            raise ValueError(
                f"_pool_node_latent expects [B,N,H,D] or [B,N,D], got {node_features.shape}"
            )

        pool_mode = getattr(self.cfg, "graph_pool_mode", "concat")
        B = node_features.shape[0]

        if pool_mode == "mean":
            x = node_features.mean(dim=1)  # [B, D]
        else:
            x = node_features.reshape(B, -1)  # [B, N*D]

        z = self.node_to_z(x)  # [B, z_dim]
        return z
    
    def _compute_pairwise_edge(self, node_a: torch.Tensor, node_b: torch.Tensor) -> torch.Tensor:
        """Compute edge features between two nodes: distance + alignment.

        Args:
            node_a, node_b: [L, 7] (flattened batch*time).
        Returns:
            [L, edge_dim]  (edge_dim=2: distance + alignment)
        """
        version = getattr(self.cfg, "edge_feature_version", "v2")
        pos_a, quat_a = node_a[:, :3], F.normalize(node_a[:, 3:7], p=2, dim=-1, eps=1e-6)
        pos_b, quat_b = node_b[:, :3], F.normalize(node_b[:, 3:7], p=2, dim=-1, eps=1e-6)
        distance = torch.norm(pos_a - pos_b, dim=-1, keepdim=True)

        if version == "v1":
            alignment = torch.sum(quat_a * quat_b, dim=-1, keepdim=True).abs()
        else:
            grasp_dir = _quat_to_axis(quat_a, "-z")
            obj_x = _quat_to_axis(quat_b, "x")
            obj_y = _quat_to_axis(quat_b, "y")
            obj_z = _quat_to_axis(quat_b, "z")
            alignment = torch.stack([
                (grasp_dir * obj_x).sum(-1).abs(),
                (grasp_dir * obj_y).sum(-1).abs(),
                (grasp_dir * obj_z).sum(-1).abs(),
            ], dim=-1).max(dim=-1)[0].unsqueeze(-1)

        return torch.cat([distance, alignment], dim=-1)

    def _compute_edge_features(self, *args, **kwargs):
        """Compute edge features (backward-compat dispatcher).

        Signatures:
            (ee_node, object_node)         -> old 2-node API, returns [B, (H,) edge_dim]
            (node_histories=...,)          -> new N-node API, returns [B, N, N, (H,) edge_dim]
        """
        if "node_histories" in kwargs:
            return self._compute_edge_features_dynamic(kwargs["node_histories"])
        if len(args) == 1 and args[0].dim() >= 3 and args[0].shape[-1] == 7:
            return self._compute_edge_features_dynamic(args[0])
        return self._compute_edge_features_legacy(*args, **kwargs)

    def _compute_edge_features_legacy(self, ee_node: torch.Tensor, object_node: torch.Tensor):
        """Original 2-node edge computation. Returns [B, (H,) edge_dim]."""
        has_time_dim = ee_node.dim() == 3
        if has_time_dim:
            B, T, _ = ee_node.shape
            ee_node = ee_node.reshape(B * T, -1)
            object_node = object_node.reshape(B * T, -1)

        edge_features = self._compute_pairwise_edge(ee_node, object_node)

        if has_time_dim:
            edge_features = edge_features.reshape(B, T, -1)
        return edge_features

    def _compute_edge_features_dynamic(self, node_histories: torch.Tensor):
        """All-pairs edge computation for N nodes.

        Args:
            node_histories: [B, N, H, 7] or [B, N, 7]
        Returns:
            edge_adj: [B, N, N, (H,) edge_dim] — symmetric adjacency of edge features.
                      Diagonal is zero.
        """
        has_time_dim = node_histories.dim() == 4
        if has_time_dim:
            B, N, H, D = node_histories.shape
            flat = node_histories.reshape(B * H, N, D)  # [B*H, N, 7]
        else:
            B, N, D = node_histories.shape
            flat = node_histories  # [B, N, 7]
            H = None

        L = flat.shape[0]  # B*H or B
        edge_dim = getattr(self.cfg, "edge_dim", 2)
        edge_adj = flat.new_zeros(L, N, N, edge_dim)

        for i in range(N):
            for j in range(i + 1, N):
                e = self._compute_pairwise_edge(flat[:, i], flat[:, j])  # [L, edge_dim]
                edge_adj[:, i, j] = e
                edge_adj[:, j, i] = e

        if has_time_dim:
            edge_adj = edge_adj.reshape(B, H, N, N, edge_dim).permute(0, 2, 3, 1, 4)
            # -> [B, N, N, H, edge_dim]
        return edge_adj
    
    def _get_timestep_embed(self, timesteps: torch.Tensor):
        """
        Get timestep embedding.
        
        Args:
            timesteps: [batch] - timestep indices (should be in [0, diffusion_steps-1])
        Returns:
            timestep_embed: [batch, hidden_dim]
        """
        # Clamp timesteps to valid range to prevent index out of bounds
        max_timestep = self.timestep_pos_embed.shape[0] - 1
        timesteps = timesteps.clamp(0, max_timestep)
        
        # Get positional embeddings
        pos_embed = self.timestep_pos_embed[timesteps]  # [batch, hidden_dim]
        # Project through MLP
        timestep_embed = self.timestep_embedding(pos_embed)  # [batch, hidden_dim]
        return timestep_embed
    
    def forward(
        self,
        obs: torch.Tensor,
        noisy_action: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
        joint_states_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict:
        """
        Forward pass of Graph-DiT policy.

        Note: ``node_histories`` / ``node_types`` are accepted for API
        compatibility with UnetPolicy / GraphUnetPolicy but are ignored
        by the base DiT forward (it uses ee_node_history / object_node_history).
        """
        # Extract node features (use history if provided, otherwise extract from current obs)
        if ee_node_history is not None and object_node_history is not None:
            # Use node history: [batch, history_length, 7]
            batch_size, history_length, node_dim = ee_node_history.shape
            
            # Embed node history
            ee_node_history_flat = ee_node_history.view(
                -1, node_dim
            )  # [batch * history_length, 7]
            object_node_history_flat = object_node_history.view(
                -1, node_dim
            )  # [batch * history_length, 7]

            ee_node_embed_flat = self.node_embedding(
                ee_node_history_flat
            )  # [batch * history_length, hidden_dim]
            object_node_embed_flat = self.node_embedding(
                object_node_history_flat
            )  # [batch * history_length, hidden_dim]

            ee_node_embed = ee_node_embed_flat.view(
                batch_size, history_length, self.cfg.hidden_dim
            )  # [batch, history_length, hidden_dim]
            object_node_embed = object_node_embed_flat.view(
                batch_size, history_length, self.cfg.hidden_dim
            )  # [batch, history_length, hidden_dim]
            
            # Stack EE and Object: [batch, history_length, 2, hidden_dim]
            # Then transpose to [batch, 2, history_length, hidden_dim] for Graph Attention
            node_features = torch.stack(
                [ee_node_embed, object_node_embed], dim=2
            )  # [batch, history_length, 2, hidden_dim]
            node_features = node_features.transpose(
                1, 2
            )  # [batch, 2, history_length, hidden_dim]
            
            # Compute edge features for each timestep in history
            # PERFORMANCE FIX: Use vectorized operations instead of Python loop
            # Reshape to [batch * history_length, 7] for batch processing
            ee_history_flat = ee_node_history.view(
                -1, node_dim
            )  # [batch * history_length, 7]
            obj_history_flat = object_node_history.view(
                -1, node_dim
            )  # [batch * history_length, 7]
            
            # Vectorized edge computation for all timesteps at once
            edge_features_raw_all = self._compute_edge_features(
                ee_history_flat, obj_history_flat
            )  # [batch * history_length, 2]
            edge_features_temporal = edge_features_raw_all.view(
                batch_size, history_length, 2
            )  # [batch, history_length, 2]
            # Use raw edge only (distance, orientation_similarity), no delta
            edge_features_temporal_full = edge_features_temporal  # [batch, history_length, 2]
            
        else:
            # Fallback: extract from current obs and convert to H=1 temporal format
            ee_node, object_node = self._extract_node_features(obs)  # [batch, 7] each
            
            # Convert to temporal format with H=1
            ee_node_history = ee_node.unsqueeze(1)  # [batch, 1, 7]
            object_node_history = object_node.unsqueeze(1)  # [batch, 1, 7]
            history_length = 1
            
            # Embed nodes: [batch, 1, 7] → [batch, 1, hidden_dim]
            ee_node_embed = self.node_embedding(ee_node).unsqueeze(1)  # [batch, 1, hidden_dim]
            object_node_embed = self.node_embedding(object_node).unsqueeze(1)  # [batch, 1, hidden_dim]
            
            # Stack: [batch, 2, 1, hidden_dim]
            node_features = torch.stack([ee_node_embed, object_node_embed], dim=1)
            
            # Compute edge (no delta)
            current_edge = self._compute_edge_features(ee_node, object_node)  # [batch, 2]
            edge_features_temporal_full = current_edge.unsqueeze(1)  # [batch, 1, 2]
        
        # Embed temporal edge features: [batch, history_length, 2] → [batch, history_length, edge_dim]
        edge_features_embed = self.edge_embedding(
            edge_features_temporal_full.view(-1, edge_features_temporal_full.shape[-1])
        )  # [batch * history_length, graph_edge_dim]
        edge_features_embed = edge_features_embed.view(
            batch_size, history_length, -1
        )  # [batch, history_length, graph_edge_dim]
        
        # NOTE: joint_states_history is accepted but intentionally NOT passed to
        # GraphDiTUnit. Causal confusion analysis showed joint conditioning creates
        # an auto-regressive shortcut that kills discrete gripper learning.
        
        # ==========================================================================
        # ACTION CHUNKING: Embed noisy_action trajectory + action_history
        # ==========================================================================
        # CRITICAL CHANGE: noisy_action is now [batch, pred_horizon, action_dim] (trajectory)
        # instead of [batch, action_dim] (single step). This is the core of Diffusion Policy's
        # "Action Chunking" mechanism that enables smooth, temporally consistent actions.
        
        batch_size = obs.shape[0]
        
        # Step 1: Embed noisy_action trajectory (the future actions we want to denoise)
        if noisy_action is not None:
            # For diffusion training/inference: noisy_action is [batch, pred_horizon, action_dim]
            if len(noisy_action.shape) == 2:
                # Backward compatibility: single action [batch, action_dim] -> expand to trajectory
                noisy_action = noisy_action.unsqueeze(1).expand(
                    -1, self.pred_horizon, -1
                )  # [batch, pred_horizon, action_dim]
            
            # Embed the action trajectory
            pred_horizon = noisy_action.shape[1]
            noisy_action_flat = noisy_action.reshape(
                -1, self.cfg.action_dim
            )  # [batch * pred_horizon, action_dim]
            noisy_action_embed_flat = self.action_embedding(
                noisy_action_flat
            )  # [batch * pred_horizon, hidden_dim]
            noisy_action_embed = noisy_action_embed_flat.view(
                batch_size, pred_horizon, self.cfg.hidden_dim
            )  # [batch, pred_horizon, hidden_dim]
            
            # CRITICAL: Add position embedding to distinguish future timesteps
            # Position 0 = t+1 (next step), Position H-1 = t+H (furthest future)
            pos_embed = self.pred_horizon_pos_embed[:pred_horizon, :].to(
                noisy_action_embed.device
            )  # [pred_horizon, hidden_dim]
            noisy_action_embed = noisy_action_embed + pos_embed.unsqueeze(
                0
            )  # [batch, pred_horizon, hidden_dim]
        else:
            # Inference mode with no initial noise: start from random noise
            # This shouldn't happen in normal inference, but handle for safety
            pred_horizon = self.pred_horizon
            noisy_action_embed = torch.zeros(
                batch_size,
                pred_horizon,
                self.cfg.hidden_dim,
                device=obs.device,
                dtype=obs.dtype,
            )
            pos_embed = self.pred_horizon_pos_embed[:pred_horizon, :].to(obs.device)
            noisy_action_embed = noisy_action_embed + pos_embed.unsqueeze(0)
        
        # Step 2: Embed action_history as context (CRITICAL: preserve temporal information)
        # History provides context for predicting the future trajectory
        if action_history is not None and action_history.shape[1] > 0:
            # Embed action history [batch, action_history_length, action_dim]
            history_len = action_history.shape[1]
            action_history_flat = action_history.reshape(
                -1, self.cfg.action_dim
            )  # [batch * history_len, action_dim]
            history_embed_flat = self.action_embedding(
                action_history_flat
            )  # [batch * history_len, hidden_dim]
            history_embed = history_embed_flat.view(
                batch_size, history_len, self.cfg.hidden_dim
            )  # [batch, history_len, hidden_dim]
            
            # Concatenate: [history] + [future trajectory to predict]
            # Shape: [batch, history_len + pred_horizon, hidden_dim]
            # Self-attention will see both history and all future tokens
            action_embed = torch.cat([history_embed, noisy_action_embed], dim=1)
        else:
            # No history available, use only future trajectory
            action_embed = noisy_action_embed  # [batch, pred_horizon, hidden_dim]
        
        # Get timestep embedding (if provided)
        timestep_embed = None
        condition_embed = None  # CRITICAL FIX: ensure always defined
        if timesteps is not None:
            timestep_embed = self._get_timestep_embed(timesteps)  # [batch, hidden_dim]
            # Add timestep embedding to action (only injection point for timestep)
            action_embed = action_embed + timestep_embed.unsqueeze(
                1
            )  # [batch, seq_len, hidden_dim] or [batch, 1, hidden_dim]
            # Note: timestep info for nodes will be injected via AdaLN modulation
        
        # Add subtask condition if provided
        if subtask_condition is not None and hasattr(self, "subtask_encoder"):
            subtask_embed = self.subtask_encoder(
                subtask_condition
            )  # [batch, hidden_dim // 4]
            # Combine with timestep embedding or use separately
            if timestep_embed is not None:
                # Both timestep and subtask: concatenate and project
                condition_embed = torch.cat([timestep_embed, subtask_embed], dim=-1)
                condition_embed = self.condition_proj(
                    condition_embed
                )  # [batch, hidden_dim]
            else:
                # Subtask only: use separate projection
                condition_embed = self.subtask_only_proj(
                    subtask_embed
                )  # [batch, hidden_dim]
            
            # Add condition to action (second injection point: subtask condition)
            action_embed = action_embed + condition_embed.unsqueeze(
                1
            )  # [batch, seq_len, hidden_dim] or [batch, 1, hidden_dim]
            # Note: condition info for nodes will be injected via AdaLN modulation
        
        # CRITICAL FIX: Use unified condition_embed for AdaLN (includes both timestep and subtask)
        # This ensures subtask information is properly injected through AdaLN modulation
        condition_for_adaln = (
            condition_embed if condition_embed is not None else timestep_embed
        )
        
        # Process through Graph DiT units
        # CRITICAL FIX: Preserve action sequence across layers for multi-layer temporal modeling
        # Each layer updates the action sequence, but maintains the temporal structure
        # ------------------------------------------------------------
        # NEW: collect layer-wise graph latents z^1...z^K
        # ------------------------------------------------------------
        z_layers: list[torch.Tensor] = []

        for unit in self.graph_dit_units:
            action_embed, node_features_out = unit(
                action_embed, 
                node_features, 
                edge_features_embed, 
                condition_for_adaln,
            )
            # IMPORTANT: propagate updated node features to next layer
            # node_features_out: [B,2,H,D] (full temporal) -> next layer expects [B,2,H,D]
            node_features = node_features_out

            # pool node features into latent z_k: [B,z_dim]
            z_k = self._pool_node_latent(node_features_out)
            z_layers.append(z_k)
        
        # ==========================================================================
        # ACTION CHUNKING OUTPUT: Predict noise/velocity for entire trajectory
        # ==========================================================================
        # Extract the FUTURE tokens (last pred_horizon tokens) for noise prediction
        # The history tokens were context, we only predict the future trajectory
        history_len = (
            self.action_history_length
            if (action_history is not None and action_history.shape[1] > 0)
            else 0
        )
        pred_horizon = self.pred_horizon
        
        if action_embed.shape[1] > pred_horizon:
            # action_embed: [batch, history_len + pred_horizon, hidden_dim]
            # Extract only the future tokens (last pred_horizon)
            future_embed = action_embed[
                :, -pred_horizon:, :
            ]  # [batch, pred_horizon, hidden_dim]
        else:
            # No history was concatenated, all tokens are future
            future_embed = action_embed  # [batch, pred_horizon, hidden_dim]
        
        # Apply noise head to each future timestep
        # Shape: [batch, pred_horizon, hidden_dim] -> [batch, pred_horizon, action_dim]
        noise_pred_flat = self.noise_head(
            future_embed.reshape(-1, self.cfg.hidden_dim)
        )  # [batch * pred_horizon, action_dim]
        noise_pred = noise_pred_flat.view(
            batch_size, pred_horizon, self.cfg.action_dim
        )  # [batch, pred_horizon, action_dim]
        
        if return_dict:
            z_stack = torch.stack(z_layers, dim=1)  # [B, K, z_dim]
            return {
                "noise_pred": noise_pred,  # [batch, pred_horizon, action_dim]
                "node_features": node_features,
                "edge_features": edge_features_embed,  # [batch, history_length, edge_dim]
                "action_embed": action_embed,
                # NEW:
                "z_layers": z_stack,              # [B, K, z_dim]
                "z_final": z_stack[:, -1, :],     # [B, z_dim]
            }
        else:
            return noise_pred  # [batch, pred_horizon, action_dim]
    
    def extract_z(
        self,
        ee_node_history: torch.Tensor,       # [B, H, 7] - must be 3D temporal
        object_node_history: torch.Tensor,   # [B, H, 7] - must be 3D temporal
    ) -> torch.Tensor:
        """
        Extract graph latent z from node histories (for RL high-frequency calls).
        
        Simplified: only supports temporal mode with history.
        This method only runs Graph-Attention layers (no self-attention or cross-attention)
        to extract scene understanding features efficiently.
        
        Args:
            ee_node_history: EE node history [batch, history_length, 7] - must be 3D
            object_node_history: Object node history [batch, history_length, 7] - must be 3D
        
        Returns:
            z_layers: [batch, K, z_dim] - Graph latents from all layers
        """
        # Validate input shapes
        assert ee_node_history.dim() == 3, f"ee_node_history must be [B,H,7], got {ee_node_history.shape}"
        assert object_node_history.dim() == 3, f"object_node_history must be [B,H,7], got {object_node_history.shape}"
        assert ee_node_history.shape[1] >= 1, "History length must be >= 1"
        assert ee_node_history.shape[1] == object_node_history.shape[1], \
            f"History lengths must match: {ee_node_history.shape[1]} vs {object_node_history.shape[1]}"
        
        self.eval()
        
        with torch.no_grad():
            B, H, node_dim = ee_node_history.shape
            
            # 1. Embed nodes: [B, H, 7] → [B, H, hidden_dim]
            ee_flat = ee_node_history.view(B * H, node_dim)
            obj_flat = object_node_history.view(B * H, node_dim)
            
            ee_embed_flat = self.node_embedding(ee_flat)  # [B*H, hidden_dim]
            obj_embed_flat = self.node_embedding(obj_flat)  # [B*H, hidden_dim]
            
            ee_embed = ee_embed_flat.view(B, H, self.cfg.hidden_dim)
            obj_embed = obj_embed_flat.view(B, H, self.cfg.hidden_dim)
            
            # Stack to [B, 2, H, hidden_dim] (Graph-DiT expected format)
            node_features = torch.stack([ee_embed, obj_embed], dim=1)  # [B, 2, H, hidden_dim]
            
            # 2. Compute temporal edge features (no delta)
            edge_list = []
            for t in range(H):
                edge_t = self._compute_edge_features(
                    ee_node_history[:, t, :],
                    object_node_history[:, t, :]
                )  # [B, 2]
                edge_list.append(edge_t)
            edge_temporal = torch.stack(edge_list, dim=1)  # [B, H, 2]
            # Embed temporal edges: [B, H, 2] → [B, H, edge_dim]
            edge_embed_flat = self.edge_embedding(edge_temporal.view(B * H, 2))
            edge_features_embed = edge_embed_flat.view(B, H, self.cfg.graph_edge_dim)
            
            # 3. Run Graph-Attention layers only (no self-attention or cross-attention)
            z_layers: list[torch.Tensor] = []
            zero_condition = torch.zeros(
                B, self.cfg.hidden_dim, 
                device=node_features.device,
                dtype=node_features.dtype
            )
            
            for unit in self.graph_dit_units:
                # Normalize
                node_features_norm = unit.node_norm1(node_features, zero_condition)
                
                # Graph attention (always returns [B, 2, H, D])
                node_features_new = unit.graph_attention(
                    node_features_norm,
                    edge_features_embed
                )
                
                # Residual connection
                node_features = node_features_new + node_features
                
                # FFN (simplified: flatten, apply, reshape)
                B_curr, N, H_curr, D = node_features.shape
                node_flat = node_features.view(B_curr * N * H_curr, D)
                node_for_norm = node_flat.view(B_curr, N * H_curr, D)
                node_features_norm2 = unit.node_norm2(node_for_norm, zero_condition)
                node_flat = node_features_norm2.view(B_curr * N * H_curr, D)
                node_flat = unit.node_ff(node_flat) + node_flat
                node_features = node_flat.view(B_curr, N, H_curr, D)
                
                # Pool to z_k
                z_k = self._pool_node_latent(node_features)  # [B, z_dim]
                z_layers.append(z_k)
            
            return torch.stack(z_layers, dim=1)  # [B, K, z_dim]
    
    def extract_features(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        joint_states_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Extract features from Graph-DiT (for Residual RL).
        
        This method runs the graph attention layers to extract "scene understanding"
        features WITHOUT running the full diffusion process. These features can be
        used by a PPO agent for residual fine-tuning.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            action_history: Action history [batch_size, history_length, action_dim] (optional)
            ee_node_history: EE node history [batch_size, history_length, 7] (optional)
            object_node_history: Object node history [batch_size, history_length, 7] (optional)
            joint_states_history: Joint states history [batch_size, history_length, joint_dim] (optional)
            subtask_condition: Subtask condition (one-hot) [batch_size, num_subtasks] (optional)
            
        Returns:
            dict containing:
                - 'graph_embedding': [batch, hidden_dim] - Aggregated graph features (scene understanding)
                - 'node_features': [batch, 2, hidden_dim] - Node embeddings (EE and Object)
                - 'edge_features': [batch, edge_dim] - Edge embeddings (distance, alignment)
                - 'action_embedding': [batch, hidden_dim] - Action history embedding
        """
        self.eval()
        
        with torch.no_grad():
            batch_size = obs.shape[0]
            device = obs.device
            
            # Extract node features (use history if provided)
            if ee_node_history is not None and object_node_history is not None:
                # Use node history: [batch, history_length, 7]
                history_length = ee_node_history.shape[1]
                node_dim = ee_node_history.shape[2]
                
                # Embed node history
                ee_node_history_flat = ee_node_history.view(-1, node_dim)
                object_node_history_flat = object_node_history.view(-1, node_dim)
                
                ee_node_embed_flat = self.node_embedding(ee_node_history_flat)
                object_node_embed_flat = self.node_embedding(object_node_history_flat)
                
                ee_node_embed = ee_node_embed_flat.view(
                    batch_size, history_length, self.cfg.hidden_dim
                )
                object_node_embed = object_node_embed_flat.view(
                    batch_size, history_length, self.cfg.hidden_dim
                )
                
                # Use most recent embeddings for output
                ee_node_final = ee_node_embed[:, -1, :]  # [batch, hidden_dim]
                object_node_final = object_node_embed[:, -1, :]  # [batch, hidden_dim]
                
                # Stack nodes: [batch, 2, history_length, hidden_dim]
                node_features = torch.stack([ee_node_embed, object_node_embed], dim=2)
                node_features = node_features.transpose(
                    1, 2
                )  # [batch, 2, history_length, hidden_dim]
                
                # Compute edge features
                ee_history_flat = ee_node_history.view(-1, node_dim)
                obj_history_flat = object_node_history.view(-1, node_dim)
                edge_features_raw_all = self._compute_edge_features(
                    ee_history_flat, obj_history_flat
                )
                edge_features_temporal = edge_features_raw_all.view(
                    batch_size, history_length, 2
                )
                
                # Current edge only (no delta)
                edge_features_raw = edge_features_temporal[:, -1, :]  # [batch, 2]
            else:
                # Fallback: extract from current obs
                ee_node, object_node = self._extract_node_features(obs)
                edge_features_raw = self._compute_edge_features(ee_node, object_node)  # [batch, 2]
                
                ee_node_final = self.node_embedding(ee_node)
                object_node_final = self.node_embedding(object_node)
                node_features = torch.stack(
                    [ee_node_final, object_node_final], dim=1
                ).unsqueeze(2)
            
            # Embed edge features
            edge_features_embed = self.edge_embedding(edge_features_raw)
            
            # Embed action history (context for the scene)
            # CRITICAL: Keep full sequence for joint_states_history consistency!
            if action_history is not None and action_history.shape[1] > 0:
                history_len = action_history.shape[1]
                action_history_flat = action_history.reshape(-1, self.cfg.action_dim)
                history_embed_flat = self.action_embedding(action_history_flat)
                history_embed = history_embed_flat.view(
                    batch_size, history_len, self.cfg.hidden_dim
                )
                # Keep full sequence for GraphDiTUnit (matches joint_states_history length)
                action_embed_seq = history_embed  # [batch, history_len, hidden_dim]
                # Also compute mean for output
                action_embedding = history_embed.mean(dim=1)  # [batch, hidden_dim]
            else:
                # Use last action from obs
                action_input = obs[:, -self.cfg.action_dim :]
                action_embedding = self.action_embedding(action_input)
                action_embed_seq = action_embedding.unsqueeze(
                    1
                )  # [batch, 1, hidden_dim]
            
            # Process through Graph DiT units to get "scene understanding"
            # We run the graph attention but without the full diffusion denoising
            # CRITICAL: Must pass joint_states_history for consistent feature extraction!
            # Pass full action history sequence so it matches joint_states_history length
            action_embed = action_embed_seq  # [batch, history_len, hidden_dim]

            # ------------------------------------------------------------
            # NEW: collect layer-wise graph latents z^1...z^K
            # ------------------------------------------------------------
            z_layers: list[torch.Tensor] = []
            
            for unit in self.graph_dit_units:
                action_embed, node_features_out = unit(
                    action_embed, 
                    node_features, 
                    edge_features_embed, 
                    None,
                )
                node_features = node_features_out  # propagate node update

                z_k = self._pool_node_latent(node_features_out)  # [B,z_dim]
                z_layers.append(z_k)

            z_stack = torch.stack(z_layers, dim=1)  # [B,K,z_dim]
            z_final = z_stack[:, -1, :]             # [B,z_dim]
            
            # Stack node embeddings for output
            node_output = torch.stack(
                [ee_node_final, object_node_final], dim=1
            )  # [batch, 2, hidden_dim]
            
            return {
                "graph_embedding": z_final,          # ✅ 直接把 graph_embedding 定义成 z_final（最合理）
                "node_features": node_output,  # [batch, 2, hidden_dim]
                "edge_features": edge_features_embed,  # [batch, edge_dim]
                "action_embedding": action_embedding,  # [batch, hidden_dim]
                # NEW:
                "z_layers": z_stack,                 # [B,K,z_dim]
                "z_final": z_final,                  # [B,z_dim]
            }
    
    def get_base_action(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        joint_states_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
        num_diffusion_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Get base action from DiT (for Residual RL).
        
        This runs the full diffusion process to get the base action,
        which PPO will then add a residual to.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            ... (same as predict)
            
        Returns:
            base_action: [batch_size, action_dim] - First action from predicted trajectory
        """
        # Use predict with defaults
        trajectory = self.predict(
            obs,
            action_history,
            ee_node_history,
            object_node_history,
            joint_states_history,
            subtask_condition,
            num_diffusion_steps,
            deterministic=True,
        )
        
        # Return first action of trajectory [batch, action_dim]
        if len(trajectory.shape) == 3:
            return trajectory[:, 0, :]
        return trajectory

    def predict(
        self, 
        obs: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        joint_states_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
        num_diffusion_steps: int | None = None,
        deterministic: bool = True,
        *,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict action TRAJECTORY from observations (inference mode).
        
        ACTION CHUNKING: Returns pred_horizon steps of actions at once.
        The caller should implement Receding Horizon Control (RHC):
        - Execute first exec_horizon steps
        - Then re-predict
        
        Args:
            obs: Observations [batch_size, obs_dim]
            action_history: Action history [batch_size, history_length, action_dim] (optional)
            ee_node_history: EE node history [batch_size, history_length, 7] (optional)
            object_node_history: Object node history [batch_size, history_length, 7] (optional)
            joint_states_history: Joint states history [batch_size, history_length, joint_dim] (optional)
            subtask_condition: Subtask condition (one-hot) [batch_size, num_subtasks] (optional)
            num_diffusion_steps: Number of steps for inference (default 1-10 for Flow Matching)
            deterministic: If True, use deterministic prediction
            node_histories: [batch_size, N, history_length, 7] (new dynamic graph API)
            node_types: [N] (node type indices)
            
        Returns:
            action_trajectory: Predicted actions [batch_size, pred_horizon, action_dim]
                              First step is t+1, last step is t+pred_horizon
        """
        return self._flow_matching_predict(
            obs,
            action_history,
            ee_node_history,
            object_node_history,
            joint_states_history,
            subtask_condition,
            num_diffusion_steps,
            deterministic,
            node_histories=node_histories,
            node_types=node_types,
        )

    def _flow_matching_predict(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor | None,
        ee_node_history: torch.Tensor | None,
        object_node_history: torch.Tensor | None,
        joint_states_history: torch.Tensor | None,
        subtask_condition: torch.Tensor | None,
        num_diffusion_steps: int | None,
        deterministic: bool,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flow Matching prediction: ODE solving for action trajectory (Euler, 1 NFE per step).

        With Action Chunking, returns [batch, pred_horizon, action_dim].
        """
        self.eval()
        num_steps = (
            num_diffusion_steps
            if num_diffusion_steps is not None
            else self.cfg.num_inference_steps
        )

        with torch.no_grad():
            batch_size = obs.shape[0]
            device = obs.device
            pred_horizon = self.pred_horizon

            action_t = torch.randn(
                batch_size, pred_horizon, self.cfg.action_dim, device=device
            )
            t = torch.zeros(batch_size, device=device)
            dt = 1.0 / num_steps

            for step in range(num_steps):
                timesteps = (
                    (t * (self.cfg.diffusion_steps - 1))
                    .long()
                    .clamp(0, self.cfg.diffusion_steps - 1)
                )
                velocity = self.forward(
                    obs,
                    noisy_action=action_t,
                    action_history=action_history,
                    ee_node_history=ee_node_history,
                    object_node_history=object_node_history,
                    node_histories=node_histories,
                    node_types=node_types,
                    joint_states_history=joint_states_history,
                    subtask_condition=subtask_condition,
                    timesteps=timesteps,
                )
                action_t = action_t + dt * velocity
                t = t + dt

            if not deterministic:
                action_t = action_t + 0.05 * torch.randn_like(action_t)

            return action_t  # [batch, pred_horizon, action_dim]
    
    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        joint_states_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss for training (Flow Matching).

        Accepts either legacy (ee_node_history, object_node_history) or
        new (node_histories, node_types) API for node inputs.
        """
        return self._flow_matching_loss(
            obs,
            actions,
            action_history,
            ee_node_history,
            object_node_history,
            joint_states_history,
            subtask_condition,
            mask,
            node_histories=node_histories,
            node_types=node_types,
        )

    def _flow_matching_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_history: torch.Tensor | None,
        ee_node_history: torch.Tensor | None,
        object_node_history: torch.Tensor | None,
        joint_states_history: torch.Tensor | None,
        subtask_condition: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Flow Matching loss: predict velocity field for action trajectory.
        
        With Action Chunking:
        - actions: [batch, pred_horizon, action_dim] - target trajectory
        - v_pred: [batch, pred_horizon, action_dim] - predicted velocity for trajectory
        - mask: [batch] - True for valid samples, False for padding (used in demo-level training)
        """
        batch_size = obs.shape[0]
        device = obs.device
        
        # Ensure actions is trajectory format [batch, pred_horizon, action_dim]
        if len(actions.shape) == 2:
            # Backward compatibility: single action -> repeat for trajectory
            actions = actions.unsqueeze(1).expand(-1, self.pred_horizon, -1)
        
        pred_horizon = actions.shape[1]
        
        # Sample time t ~ U(0, 1) for flow matching (same t for entire trajectory)
        t = torch.rand(batch_size, device=device)  # [batch_size], each in [0, 1]
        
        # Sample noise (same shape as actions trajectory)
        noise = torch.randn_like(actions)  # [batch, pred_horizon, action_dim]
        
        # CRITICAL FIX: Linear interpolation path: x_t = (1-t) * x_0 + t * x_1
        # x_0 = noise (t=0), x_1 = actions (data, t=1)
        # Reshape t for broadcasting: [batch, 1, 1]
        t_broadcast = t.view(-1, 1, 1)  # [batch, 1, 1]
        x_t = (
            1 - t_broadcast
        ) * noise + t_broadcast * actions  # [batch, pred_horizon, action_dim]
        
        # Ground truth velocity field: v_t = x_1 - x_0 (direction from noise to data)
        v_t = actions - noise  # [batch, pred_horizon, action_dim]
        
        # Convert t to timesteps format for forward (scale to [0, diffusion_steps-1])
        # Clamp to prevent index out of bounds (CUDA error)
        timesteps = (t * (self.cfg.diffusion_steps - 1)).long().clamp(0, self.cfg.diffusion_steps - 1)  # [batch_size]
        
        # Predict velocity field for entire trajectory
        v_pred = self.forward(
            obs,
            noisy_action=x_t,
            action_history=action_history,
            ee_node_history=ee_node_history,
            object_node_history=object_node_history,
            node_histories=node_histories,
            node_types=node_types,
            joint_states_history=joint_states_history,
            subtask_condition=subtask_condition,
            timesteps=timesteps,
        )  # [batch, pred_horizon, action_dim]
        
        # Compute loss (MSE between predicted and ground truth velocity for trajectory)
        # CRITICAL: Apply mask to exclude padding timesteps from loss
        # CRITICAL: Handle "half-cut" data where mask is [batch, pred_horizon] instead of [batch]
        if mask is not None:
            # Handle both mask shapes:
            # - [batch] -> old format (whole sample mask)
            # - [batch, pred_horizon] -> new format (per-horizon-step mask for "half-cut" data)
            if len(mask.shape) == 1:
                # Old format: [batch] -> [batch, 1, 1] for broadcasting
                mask_expanded = mask.float().view(-1, 1, 1)  # [batch, 1, 1]
                # Count valid elements: pred_horizon * action_dim per valid sample
                total_valid_elements = (
                    (mask.float() * pred_horizon * actions.shape[-1]).sum().clamp(min=1)
                )
            else:
                # New format: [batch, pred_horizon] -> [batch, pred_horizon, 1] for broadcasting
                mask_expanded = mask.float().unsqueeze(-1)  # [batch, pred_horizon, 1]
                # Count valid elements: sum over all valid horizon steps, then multiply by action_dim
                total_valid_elements = (mask.float().sum() * actions.shape[-1]).clamp(
                    min=1
                )
            
            # Compute per-element loss
            per_element_loss = F.mse_loss(
                v_pred, v_t, reduction="none"
            )  # [batch, pred_horizon, action_dim]
            # CRITICAL FIX: Zero out padding timesteps BEFORE averaging
            # This prevents padding from contributing to loss at all (handles both demo-level and horizon-level padding)
            masked_loss = (
                per_element_loss * mask_expanded
            )  # [batch, pred_horizon, action_dim]
            # Sum over horizon and action_dim for each sample (padding samples/steps will be 0)
            per_sample_sum = masked_loss.sum(dim=(1, 2))  # [batch]
            # Average over all valid elements (not samples!)
            mse_loss = per_sample_sum.sum() / total_valid_elements
            total_loss = mse_loss
        else:
            # No mask: compute standard MSE loss
            mse_loss = F.mse_loss(v_pred, v_t)  # Average over all dimensions
            total_loss = mse_loss
        
        result = {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
        }
        
        # Add debug info if in debug mode (can be enabled via environment variable)
        import os

        if os.getenv("DEBUG_LOSS", "False").lower() == "true":
            result["debug"] = {
                "v_pred": v_pred.detach(),
                "v_t": v_t.detach(),
                "actions": actions.detach(),
                "noise": noise.detach(),
                "x_t": x_t.detach(),
                "t": t.detach(),
            }
        
        return result
    
    def save(self, path: str):
        """Save policy to file.
        
        Args:
            path: Path to save the model.
        """
        torch.save(
            {
            "policy_state_dict": self.state_dict(),
            "cfg": self.cfg,
            },
            path,
        )
        print(f"[GraphDiTPolicy] Saved model to: {path}")
    
    @classmethod
    def load(cls, path: str, device: str = "cuda"):
        """Load policy from file.
        
        Args:
            path: Path to load the model from.
            device: Device to load the model on.
            
        Returns:
            GraphDiTPolicy: Loaded policy.
        """
        # weights_only=False is needed for PyTorch 2.6+ to load custom config classes
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Load config
        cfg = checkpoint.get("cfg", None)
        if cfg is None:
            raise ValueError(f"No config found in checkpoint: {path}")
        
        # Backward compatibility: add z_dim if missing (for old checkpoints)
        if not hasattr(cfg, 'z_dim'):
            cfg.z_dim = 128
            print(f"[GraphDiTPolicy] Added missing z_dim=128 to config for backward compatibility")
        
        # Create policy
        policy = cls(cfg)
        
        # Load state dict with strict=False to handle missing keys (e.g., node_to_z in old checkpoints)
        state_dict = checkpoint["policy_state_dict"]
        missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
        
        # Handle missing keys (backward compatibility)
        if missing_keys:
            print(f"[GraphDiTPolicy] Warning: Missing keys in checkpoint (will use default initialization):")
            for key in missing_keys:
                print(f"  - {key}")
            # If node_to_z is missing, it will use the default initialization from __init__
            # This is fine for backward compatibility with old checkpoints
        
        if unexpected_keys:
            print(f"[GraphDiTPolicy] Warning: Unexpected keys in checkpoint (ignored):")
            for key in unexpected_keys:
                print(f"  - {key}")
        
        policy.to(device)
        policy.eval()
        
        print(f"[GraphDiTPolicy] Loaded model from: {path}")
        return policy

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
    """Action dimension (output from policy)."""
    
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
    
    diffusion_steps: int = 100
    """Number of diffusion steps."""
    
    num_inference_steps: int = 30
    """Number of inference steps for flow matching prediction.
    
    This controls how many ODE integration steps are used during inference.
    More steps = smoother, more accurate predictions but slower.
    Default: 30 (good balance between quality and speed).
    For faster inference, can use 10-20 steps. For higher quality, use 50-100 steps.
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
    
    joint_dim: int | None = None
    """Joint states dimension (joint_pos + joint_vel). If None, joint states are not used."""
    
    # Action history
    action_history_length: int = 4
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
        use_edge_modulation: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_edge_modulation = use_edge_modulation
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge feature to attention bias (baseline method - kept for backward compatibility)
        self.edge_to_bias = nn.Linear(edge_dim, num_heads)
        
        # CRITICAL INNOVATION: Edge-Conditioned Modulation
        # Edge features generate gates/scales that directly control Value transformation
        if use_edge_modulation:
            # Option 1: Gate mechanism (sigmoid gate controls information flow)
            self.edge_to_gate = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim), nn.Sigmoid()  # Gate in [0, 1]
            )
            
            # Option 2: Scale and Shift (like AdaLN, but conditioned on Edge)
            self.edge_to_scale = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.Tanh(),  # Scale in [-1, 1], can be expanded
            )
            self.edge_to_shift = nn.Linear(edge_dim, hidden_dim)
            
            # Option 3: Per-head modulation (more fine-grained control)
            self.edge_to_head_scale = nn.Sequential(
                nn.Linear(edge_dim, num_heads), nn.Tanh()
            )

        self.dropout = nn.Dropout(0.1)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor):
        """
        Args:
            node_features: [batch, num_nodes, history_length, hidden_dim] or [batch, num_nodes, hidden_dim]
                          (2 nodes: EE, Object, with optional temporal history)
            edge_features: [batch, edge_dim] (distance + orientation_similarity)
        Returns:
            updated_node_features: [batch, num_nodes, hidden_dim] or [batch, num_nodes, history_length, hidden_dim]
        """
        # Handle both temporal and non-temporal cases
        if len(node_features.shape) == 4:
            # Temporal: [batch, num_nodes, history_length, hidden_dim]
            batch_size, num_nodes, history_length, hidden_dim = node_features.shape
            # Reshape to treat (num_nodes, history_length) as sequence
            node_features = node_features.view(
                batch_size, num_nodes * history_length, hidden_dim
            )  # [batch, num_nodes * history_length, hidden_dim]
            temporal_mode = True
        else:
            # Non-temporal: [batch, num_nodes, hidden_dim]
            batch_size, num_nodes, hidden_dim = node_features.shape
            temporal_mode = False
        
        # Project to Q, K, V
        seq_len = node_features.shape[1]  # num_nodes or num_nodes * history_length
        Q = self.q_proj(node_features)  # [batch, seq_len, hidden_dim]
        K = self.k_proj(node_features)  # [batch, seq_len, hidden_dim]
        V = self.v_proj(node_features)  # [batch, seq_len, hidden_dim]
        
        # CRITICAL INNOVATION: Edge-Conditioned Value Modulation
        # Edge features control how Value is transformed BEFORE attention
        if self.use_edge_modulation:
            # Generate modulation signals from edge features
            # Gate: controls information flow [batch, hidden_dim]
            edge_gate = self.edge_to_gate(edge_features)  # [batch, hidden_dim]
            
            # Scale and Shift: fine-grained modulation [batch, hidden_dim]
            edge_scale = self.edge_to_scale(edge_features)  # [batch, hidden_dim]
            edge_shift = self.edge_to_shift(edge_features)  # [batch, hidden_dim]
            
            # Apply modulation to V BEFORE attention
            # Physical meaning:
            # - Gate: "How much information can flow?" (based on distance/orientation)
            # - Scale: "Amplify or dampen specific features" (based on spatial relationship)
            # - Shift: "Bias the feature space" (based on relative pose)
            V_modulated = V * (1.0 + edge_scale.unsqueeze(1)) + edge_shift.unsqueeze(
                1
            )  # [batch, seq_len, hidden_dim]
            V_modulated = V_modulated * edge_gate.unsqueeze(1)  # Apply gate
            
            # Use modulated V for attention
            V = V_modulated
        # If modulation disabled, use original V (backward compatibility)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch, heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = (
            torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        )  # [batch, heads, seq_len, seq_len]
        
        # Add edge bias (baseline method - still useful for attention scores)
        # Edge features -> bias for each head
        edge_bias = self.edge_to_bias(edge_features)  # [batch, num_heads]
        
        # Additional per-head modulation (optional, fine-grained control)
        if self.use_edge_modulation:
            head_scale = self.edge_to_head_scale(edge_features)  # [batch, num_heads]
            # Modulate attention scores per head
            scores = scores * (
                1.0 + head_scale.unsqueeze(-1).unsqueeze(-1)
            )  # [batch, heads, seq_len, seq_len]
        
        if temporal_mode:
            # For temporal mode: create attention bias for spatial connections (between different nodes)
            # at the SAME timestep across all time steps
            # The node order is: [node0(t-3), node0(t-2), ..., node0(t), 
            #                     node1(t-3), node1(t-2), ..., node1(t), ...]
            attention_bias = torch.zeros(
                batch_size,
                self.num_heads,
                seq_len,
                seq_len,
                device=node_features.device,
                dtype=node_features.dtype,
            )
            
            # Set edge bias for spatial connections at the same timestep
            # NOTE: edge_features currently describes the relationship between node0 and node1
            # For num_nodes > 2, this would need to be extended to support multiple edge types
            # Expand edge_bias: [batch, num_heads] -> [batch, num_heads, history_length]
            edge_bias_expanded = edge_bias.unsqueeze(-1).expand(
                -1, -1, history_length
            )  # [batch, num_heads, history_length]
            
            # Connect node0 <-> node1 at the same timestep across all time steps
            # NOTE: Assumes num_nodes=2 (EE and Object), history_length is dynamic
            t_indices = torch.arange(history_length, device=node_features.device)
            node0_indices = t_indices  # [history_length] - node0 at all timesteps
            node1_indices = (
                history_length + t_indices
            )  # [history_length] - node1 at all timesteps
            # Node0 -> Node1 connections (same timestep)
            attention_bias[:, :, node0_indices, node1_indices] = (
                edge_bias_expanded.permute(0, 1, 2)
            )
            # Node1 -> Node0 connections (symmetric, same timestep)
            attention_bias[:, :, node1_indices, node0_indices] = (
                edge_bias_expanded.permute(0, 1, 2)
            )
        else:
            # Non-temporal: simple graph with spatial connections
            attention_bias = torch.zeros(
                batch_size,
                self.num_heads,
                num_nodes,
                num_nodes,
                device=node_features.device,
                dtype=node_features.dtype,
            )
            # Set edge bias for node0 <-> node1 (spatial connections)
            # NOTE: Assumes num_nodes=2 (EE and Object)
            attention_bias[:, :, 0, 1] = edge_bias  # node0 -> node1
            attention_bias[:, :, 1, 0] = edge_bias  # node1 -> node0 (symmetric)
        
        scores = scores + attention_bias
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [batch, heads, num_nodes, head_dim]
        
        # Reshape and project output
        out = out.transpose(
            1, 2
        ).contiguous()  # [batch, num_nodes (* history_length), heads, head_dim]
        
        if temporal_mode:
            # Reshape back to temporal format
            out = out.view(
                batch_size, num_nodes, history_length, self.hidden_dim
            )  # [batch, num_nodes, history_length, hidden_dim]
            # Take the last timestep (most recent) as final node features
            # Each timestep's output already aggregates information from all timesteps via attention,
            # so taking the last one gives us the most up-to-date representation
            out = out[:, :, -1, :]  # [batch, num_nodes, hidden_dim]
        else:
            out = out.view(
                batch_size, num_nodes, self.hidden_dim
            )  # [batch, num_nodes, hidden_dim]
        
        out = self.out_proj(out)  # [batch, num_nodes, hidden_dim]
        
        return out


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
        use_edge_modulation: bool = True,
        joint_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.joint_dim = joint_dim
        self.joint_pos_dim: int | None = None
        self.joint_vel_dim: int | None = None
        
        # Joint states encoder (if joint_dim is provided)
        # NOTE: Only using joint_pos (removed joint_vel to test if it's noise)
        if joint_dim is not None:
            # Now joint_dim only contains joint_pos (no joint_vel)
            # For joint control with 6 DoF: joint_pos_dim = 6, joint_vel_dim = 0
            self.joint_pos_dim = joint_dim  # joint_dim is now only joint_pos
            self.joint_vel_dim = 0  # No joint_vel

            # Encode joint positions for state-action self-attention
            # Only joint_pos (no joint_vel)
            self.joint_pos_encoder = nn.Sequential(
                nn.Linear(self.joint_pos_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.joint_vel_encoder = None  # Disabled - no joint_vel
            # Projection for concatenated (joint_pos_embed, action_embed) sequence
            # 2 * hidden_dim -> hidden_dim (was 3 * hidden_dim when joint_vel was included)
            self.state_action_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.joint_pos_encoder = None
            self.joint_vel_encoder = None
            self.state_action_proj = None
        
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
        
        # Step 2: Graph attention with edge features
        # CRITICAL INNOVATION: Use Edge-Conditioned Modulation (ECC-style)
        # Edge features control Value transformation, not just attention bias
        self.graph_attention = GraphAttentionWithEdgeBias(
            hidden_dim, num_heads, edge_dim, use_edge_modulation=use_edge_modulation
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
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        timestep_embed: torch.Tensor | None = None,
        joint_states_history: torch.Tensor | None = None,
    ):
        """
        Args:
            action: Action sequence [batch, seq_len, hidden_dim] (should be action history)
            node_features: Node features [batch, 2, hidden_dim] (EE, Object)
            edge_features: Edge features [batch, edge_dim]
            timestep_embed: Timestep embedding [batch, hidden_dim] (optional)
            joint_states_history: Joint states history [batch, history_length, joint_dim] (optional)
        Returns:
            noise_pred: Predicted noise [batch, hidden_dim]
        """
        # Prepare condition embedding for AdaLN (timestep embedding)
        condition_emb = timestep_embed if timestep_embed is not None else None
        
        # Ensure action is 3D: [batch, seq_len, hidden_dim]
        if len(action.shape) == 2:
            action = action.unsqueeze(
                1
            )  # [batch, hidden_dim] -> [batch, 1, hidden_dim]
        seq_len = action.shape[1]
        
        # Step 1: State-action sequence self-attention
        # Design (UPDATED):
        #   - Self-attn uses joint positions + joint velocities + action history
        #   - Noisy action token is kept separate (not concatenated with joint states)
        if (
            joint_states_history is not None
            and self.joint_pos_encoder is not None
            and self.joint_pos_dim is not None
        ):
            # Split joint states into position and velocity
            # NOTE: joint_states_history now only contains joint_pos (no joint_vel)
            joint_pos = joint_states_history  # [batch, T, joint_pos_dim] where joint_pos_dim = joint_dim
            # joint_vel removed - no longer extracted
            
            batch_size, history_length, _ = joint_pos.shape
            
            # Encode joint positions: [batch, T, dim] -> [batch, T, hidden_dim]
            joint_pos_embed = self.joint_pos_encoder(
                joint_pos
            )  # [batch, T, hidden_dim]
            # joint_vel_embed removed - no joint_vel
            
            # Determine how many history tokens we have in the action sequence
            # Typical case: seq_len = history_length + 1 (H history + 1 noisy_action)
            if seq_len > history_length:
                # Split action into history and noisy_action
                action_history_embed = action[
                    :, :history_length, :
                ]  # [batch, history_length, hidden_dim]
                noisy_action_embed = action[
                    :, history_length:, :
                ]  # [batch, 1, hidden_dim]
            else:
                # No explicit noisy_action token: treat entire action sequence as history
                action_history_embed = action[:, :history_length, :]
                noisy_action_embed = action[:, 0:0, :]  # empty sequence
            
            # Align joint_pos_embed with action history length (use most recent timesteps if needed)
            if history_length > action_history_embed.shape[1]:
                # Trim joint history to match action history length
                trim_len = action_history_embed.shape[1]
                joint_pos_embed = joint_pos_embed[:, -trim_len:, :]
                history_length = trim_len
            elif history_length < action_history_embed.shape[1]:
                # Trim action history to match joint history length (use most recent)
                trim_len = history_length
                action_history_embed = action_history_embed[:, -trim_len:, :]
                history_length = trim_len
            
            # Concatenate joint pos + action history (NOT with noisy_action token)
            # NOTE: joint_vel removed - only joint_pos + action
            # [batch, history_length, hidden_dim * 2] -> [batch, history_length, hidden_dim]
            state_action_history = torch.cat(
                [joint_pos_embed, action_history_embed], dim=-1
            )
            state_action_history = self.state_action_proj(state_action_history)
            
            # Re-attach noisy_action token (if it exists)
            if noisy_action_embed.shape[1] > 0:
                state_action_seq = torch.cat(
                    [state_action_history, noisy_action_embed], dim=1
                )  # [batch, seq_len, hidden_dim]
            else:
                state_action_seq = (
                    state_action_history  # [batch, history_length, hidden_dim]
                )
        else:
            # Fallback: use only action (no joint states available)
            state_action_seq = action  # [batch, seq_len, hidden_dim]
        
        # Self-attention on state-action sequence
        state_action_residual = state_action_seq
        if condition_emb is not None:
            state_action_seq = self.action_norm1(state_action_seq, condition_emb)
        else:
            zero_condition = torch.zeros(
                state_action_seq.shape[0],
                self.hidden_dim,
                device=state_action_seq.device,
                dtype=state_action_seq.dtype,
            )
            state_action_seq = self.action_norm1(state_action_seq, zero_condition)
        
        a_new, _ = self.action_self_attn(
            state_action_seq, state_action_seq, state_action_seq
        )  # [batch, seq_len, hidden_dim]
        a_new = a_new + state_action_residual
        
        # CRITICAL FIX: Keep full sequence for multi-layer processing
        # Do NOT compress to last token here - preserve sequence structure for downstream layers
        # The sequence information is aggregated through self-attention, but we keep all tokens
        # for cross-attention and subsequent layers
        
        action_residual = a_new  # [batch, seq_len, hidden_dim]
        if condition_emb is not None:
            a_new = self.action_norm2(a_new, condition_emb)
        else:
            zero_condition = torch.zeros(
                a_new.shape[0], self.hidden_dim, device=a_new.device, dtype=a_new.dtype
            )
            a_new = self.action_norm2(a_new, zero_condition)
        a_new = self.action_ff(a_new) + action_residual  # [batch, seq_len, hidden_dim]
        
        # Step 2: Node attention with edge features
        # Handle temporal node features
        if len(node_features.shape) == 4:
            # Temporal: [batch, 2, history_length, hidden_dim]
            # For residual connection: use average of all timesteps as initial state
            # (This is different from attention output, which takes the last timestep)
            node_residual = node_features.mean(dim=2)  # [batch, 2, hidden_dim]
            node_features_for_attn = (
                node_features  # [batch, 2, history_length, hidden_dim]
            )
        else:
            node_residual = node_features  # [batch, 2, hidden_dim]
            # Add temporal dimension for consistency
            node_features_for_attn = node_features.unsqueeze(
                2
            )  # [batch, 2, 1, hidden_dim]
        
        # Use AdaLN for node features
        # CRITICAL FIX: AdaptiveLayerNorm now handles broadcasting internally
        # Always pass condition as [batch, condition_dim], no unsqueeze needed
        if condition_emb is not None:
            node_features_for_attn = self.node_norm1(
                node_features_for_attn, condition_emb
            )
        else:
            zero_condition = torch.zeros(
                node_features_for_attn.shape[0],
                self.hidden_dim,
                device=node_features_for_attn.device,
                dtype=node_features_for_attn.dtype,
            )
            node_features_for_attn = self.node_norm1(
                node_features_for_attn, zero_condition
            )

        node_features_new = self.graph_attention(
            node_features_for_attn, edge_features
        )  # [batch, 2, hidden_dim]
        
        # Add residual
        if len(node_residual.shape) == 3:
            node_features_new = (
                node_features_new + node_residual
            )  # [batch, 2, hidden_dim]
        else:
            # Shouldn't happen, but handle gracefully
            node_features_new = node_features_new
        
        node_residual = node_features_new
        # CRITICAL FIX: AdaptiveLayerNorm now handles broadcasting internally
        if condition_emb is not None:
            node_features_new = self.node_norm2(node_features_new, condition_emb)
        else:
            zero_condition = torch.zeros(
                node_features_new.shape[0],
                self.hidden_dim,
                device=node_features_new.device,
                dtype=node_features_new.dtype,
            )
            node_features_new = self.node_norm2(node_features_new, zero_condition)
        node_features_new = self.node_ff(node_features_new) + node_residual
        
        # Step 3: Cross-attention (a_new as query, node_features_new + joint_vel memory as key/value)
        cross_residual = a_new
        # Use AdaLN for cross-attention
        if condition_emb is not None:
            a_new_norm = self.cross_norm1(a_new, condition_emb)
        else:
            zero_condition = torch.zeros(
                a_new.shape[0], self.hidden_dim, device=a_new.device, dtype=a_new.dtype
            )
            a_new_norm = self.cross_norm1(a_new, zero_condition)
        
        # Prepare key/value memory:
        #   - Graph nodes ONLY (EE / Object):
        #       * temporal mode: [batch, 2, history_length, hidden_dim]
        #       * non-temporal mode: [batch, 2, hidden_dim] (history_length = 1)
        #   - NOTE: joint velocities are NOW used in self-attention with joint positions,
        #     and are no longer concatenated with node features here (previous design proved ineffective).
        if node_features_new.dim() == 4:
            B, num_nodes, history_length, D = node_features_new.shape
            graph_nodes = node_features_new.view(
                B, num_nodes * history_length, D
            )  # [B, 2*H, D]
        elif node_features_new.dim() == 3:
            B, num_nodes, D = node_features_new.shape
            history_length = 1
            graph_nodes = node_features_new.view(
                B, num_nodes * history_length, D
            )  # [B, 2, D]
        else:
            raise ValueError(
                f"Unexpected node_features_new shape {node_features_new.shape}, "
                f"expected [B, 2, H, D] or [B, 2, D]"
            )
        
        # FINAL DESIGN: Only graph nodes as memory for cross-attention
        kv_memory = graph_nodes  # [B, 2*H, D]
        
        noise_embed, _ = self.cross_attn(a_new_norm, kv_memory, kv_memory)
        noise_embed = noise_embed + cross_residual
        
        cross_residual = noise_embed
        if condition_emb is not None:
            noise_embed = self.cross_norm2(noise_embed, condition_emb)
        else:
            zero_condition = torch.zeros(
                noise_embed.shape[0],
                self.hidden_dim,
                device=noise_embed.device,
                dtype=noise_embed.dtype,
            )
            noise_embed = self.cross_norm2(noise_embed, zero_condition)
        noise_embed = self.cross_ff(noise_embed) + cross_residual
        
        # CRITICAL FIX: Return with sequence dimension preserved for multi-layer processing
        # If input action had history, preserve it; otherwise return single token
        # noise_embed: [batch, seq_len, hidden_dim] where seq_len could be 1 or history_length+1
        # For downstream layers, we want to keep the sequence structure
        # NEW: Return both action_out and node_out for graph latent extraction
        return noise_embed, node_features_new  # (action_out, node_out)


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

        # ------------------------------------------------------------
        # NEW: Node latent pooling head (EE/Object nodes -> z in R^z_dim)
        # node_features_k: [B, 2, hidden_dim] -> z_k: [B, z_dim]
        # ------------------------------------------------------------
        # Handle backward compatibility: old checkpoints may not have z_dim
        z_dim = getattr(cfg, 'z_dim', 128)  # Default to 128 if not present
        self.node_to_z = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, z_dim),
        )
        
        # Edge embedding: current(2) + delta(2) = 4 -> edge_dim
        # CRITICAL FIX: Input is now [current_distance, current_alignment, delta_distance, delta_alignment]
        # This includes velocity/trend information for better grasping behavior
        self.edge_embedding = nn.Sequential(
            nn.Linear(
                cfg.edge_dim * 2, cfg.graph_edge_dim
            ),  # 4 inputs (current + delta)
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
        self.graph_dit_units = nn.ModuleList(
            [
                GraphDiTUnit(
                    cfg.hidden_dim,
                    cfg.num_heads,
                    cfg.graph_edge_dim,
                    use_edge_modulation=cfg.use_edge_modulation,
                    joint_dim=cfg.joint_dim,
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
            joint_states: [batch, joint_dim] - only joint_pos (no joint_vel)
        """
        # Use configurable indices if provided, otherwise use defaults
        if self.cfg.obs_structure is not None:
            obs_struct = self.cfg.obs_structure
            joint_pos = obs[:, obs_struct["joint_pos"][0] : obs_struct["joint_pos"][1]]
            # NOTE: joint_vel removed - only using joint_pos
        else:
            # Default structure: [joint_pos(6), ...] (joint_vel removed)
            joint_pos = obs[:, 0:6]  # [batch, 6]
        
        # Only return joint_pos (no joint_vel)
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
            node_features: [B, 2, hidden_dim] or [B, 2, 1, hidden_dim]
        Returns:
            z: [B, z_dim]
        """
        # Handle both [B, 2, hidden_dim] and [B, 2, 1, hidden_dim]
        if node_features.dim() == 4:
            # Temporal: [B, 2, 1, hidden_dim] -> [B, 2, hidden_dim]
            node_features = node_features.squeeze(2)
        elif node_features.dim() == 3:
            # Non-temporal: [B, 2, hidden_dim]
            pass
        else:
            raise ValueError(
                f"_pool_node_latent expects [B,2,H] or [B,2,1,H], got {node_features.shape}"
            )

        if node_features.shape[1] != 2:
            raise ValueError(
                f"_pool_node_latent expects 2 nodes, got {node_features.shape[1]}"
            )

        B = node_features.shape[0]
        x = node_features.reshape(B, -1)  # [B, 2*hidden_dim]
        z = self.node_to_z(x)  # [B, z_dim]
        return z
    
    def _compute_edge_features(self, ee_node: torch.Tensor, object_node: torch.Tensor):
        """
        Compute edge features: distance + orientation similarity.
        
        Args:
            ee_node: [batch, 7] - EE position(3) + orientation(4)
            object_node: [batch, 7] - Object position(3) + orientation(4)
        Returns:
            edge_features: [batch, 2] - [distance, orientation_similarity]
        """
        # Extract positions and orientations
        ee_pos = ee_node[:, :3]  # [batch, 3]
        ee_quat = ee_node[:, 3:7]  # [batch, 4]
        obj_pos = object_node[:, :3]  # [batch, 3]
        obj_quat = object_node[:, 3:7]  # [batch, 4]
        
        # CRITICAL FIX: Normalize quaternions to avoid numerical issues
        # Add eps to prevent NaN when quaternion is near zero vector
        ee_quat = F.normalize(ee_quat, p=2, dim=-1, eps=1e-6)  # [batch, 4]
        obj_quat = F.normalize(obj_quat, p=2, dim=-1, eps=1e-6)  # [batch, 4]
        
        # 1. Distance (L2 norm)
        distance = torch.norm(ee_pos - obj_pos, dim=-1, keepdim=True)  # [batch, 1]
        
        # 2. Orientation similarity (quaternion dot product)
        # Quaternion dot product: q1 · q2 = w1*w2 + x1*x2 + y1*y2 + z1*z2
        quat_dot = torch.sum(ee_quat * obj_quat, dim=-1, keepdim=True)  # [batch, 1]
        # Take absolute value (q and -q represent same rotation)
        orientation_similarity = torch.abs(quat_dot)  # [batch, 1], range [0, 1]
        
        edge_features = torch.cat(
            [distance, orientation_similarity], dim=-1
        )  # [batch, 2]
        return edge_features
    
    def _get_timestep_embed(self, timesteps: torch.Tensor):
        """
        Get timestep embedding.
        
        Args:
            timesteps: [batch] - timestep indices
        Returns:
            timestep_embed: [batch, hidden_dim]
        """
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
        joint_states_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict:
        """
        Forward pass of Graph-DiT policy.
        
        Args:
            obs: Observations [batch_size, obs_dim] - concatenated observations
            noisy_action: Noisy action for diffusion [batch_size, action_dim] (required for diffusion training)
            action_history: Action history [batch_size, history_length, action_dim] (optional)
            ee_node_history: EE node history [batch_size, history_length, 7] (optional)
            object_node_history: Object node history [batch_size, history_length, 7] (optional)
            joint_states_history: Joint states history [batch_size, history_length, joint_dim] (optional)
            subtask_condition: Subtask condition (one-hot) [batch_size, num_subtasks] (optional)
            timesteps: Diffusion timesteps [batch_size] (optional, for training)
            return_dict: If True, return dict with additional info
            
        Returns:
            noise_pred: Predicted noise [batch_size, action_dim]
            or dict with 'noise_pred' and other fields if return_dict=True
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
            
            # CRITICAL FIX: Include both current edge features AND delta (velocity/trend information)
            # "Object is approaching" is more informative than "Object is at 0.5m"
            # edge_features_temporal: [batch, history_length, 2] = (distance, alignment) per timestep
            current_edge = edge_features_temporal[
                :, -1, :
            ]  # [batch, 2] - current (most recent)
            
            if history_length > 1:
                # Compute delta: current - previous (positive = increasing distance, negative = approaching)
                prev_edge = edge_features_temporal[
                    :, -2, :
                ]  # [batch, 2] - previous timestep
                edge_delta = current_edge - prev_edge  # [batch, 2] - velocity/trend
                
                # Concatenate current edge features with delta for richer information
                # [batch, 4] = [current_distance, current_alignment, delta_distance, delta_alignment]
                edge_features_raw = torch.cat(
                    [current_edge, edge_delta], dim=-1
                )  # [batch, 4]
            else:
                # Single timestep: pad with zeros for delta
                edge_delta = torch.zeros_like(current_edge)  # [batch, 2]
                edge_features_raw = torch.cat(
                    [current_edge, edge_delta], dim=-1
                )  # [batch, 4]
            
        else:
            # Fallback: extract from current obs (backward compatibility)
            ee_node, object_node = self._extract_node_features(obs)  # [batch, 7] each
            
            # Compute edge features
            current_edge = self._compute_edge_features(
                ee_node, object_node
            )  # [batch, 2]
            
            # Pad with zeros for delta (no history available)
            edge_delta = torch.zeros_like(current_edge)  # [batch, 2]
            edge_features_raw = torch.cat(
                [current_edge, edge_delta], dim=-1
            )  # [batch, 4]
            
            # Embed nodes (single timestep)
            ee_node_embed = self.node_embedding(ee_node)  # [batch, hidden_dim]
            object_node_embed = self.node_embedding(object_node)  # [batch, hidden_dim]
            # For compatibility: [batch, 2, 1, hidden_dim]
            node_features = torch.stack(
                [ee_node_embed, object_node_embed], dim=1
            ).unsqueeze(
                2
            )  # [batch, 2, 1, hidden_dim]
        
        # Embed edge features (now [batch, 4] = current + delta)
        edge_features_embed = self.edge_embedding(
            edge_features_raw
        )  # [batch, graph_edge_dim]
        
        # Extract joint states history (for state-action sequence self-attention)
        # IMPORTANT: Do NOT fabricate fake histories by repeating the current joint state.
        # If no joint_states_history is provided, we simply skip joint-conditioned self-attention
        # and let the model rely on action history + graph features.
        if joint_states_history is not None:
            # Ensure joint_states_history has correct shape
            if len(joint_states_history.shape) == 2:
                joint_states_history = joint_states_history.unsqueeze(
                    1
                )  # [batch, 1, joint_dim]
            
            # Optionally align length with action_history (only trim if longer; never pad)
            if action_history is not None and action_history.shape[1] > 0:
                target_length = action_history.shape[1]
                current_length = joint_states_history.shape[1]
                if current_length > target_length:
                    # Keep the most recent timesteps to match action history
                    joint_states_history = joint_states_history[:, -target_length:, :]
                elif current_length > target_length:
                    # Truncate to match
                    joint_states_history = joint_states_history[:, :target_length, :]
        
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
            # Add timestep embedding to action
            action_embed = action_embed + timestep_embed.unsqueeze(
                1
            )  # [batch, seq_len, hidden_dim] or [batch, 1, hidden_dim]
            
            # Add to node features (handle both temporal and non-temporal)
            if len(node_features.shape) == 4:
                # Temporal: [batch, 2, history_length, hidden_dim]
                node_features = node_features + timestep_embed.unsqueeze(1).unsqueeze(
                    1
                )  # Broadcast to all timesteps
            else:
                # Non-temporal: [batch, 2, 1, hidden_dim] or [batch, 2, hidden_dim]
                if len(node_features.shape) == 3:
                    node_features = node_features + timestep_embed.unsqueeze(1)
                else:
                    node_features = node_features + timestep_embed.unsqueeze(
                        1
                    ).unsqueeze(1)
        
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
            
            # Add condition to action and nodes
            action_embed = action_embed + condition_embed.unsqueeze(
                1
            )  # [batch, seq_len, hidden_dim] or [batch, 1, hidden_dim]
            
            # Add to node features (handle both temporal and non-temporal)
            if len(node_features.shape) == 4:
                # Temporal: [batch, 2, history_length, hidden_dim]
                node_features = node_features + condition_embed.unsqueeze(1).unsqueeze(
                    1
                )  # Broadcast to all timesteps
            else:
                # Non-temporal
                if len(node_features.shape) == 3:
                    node_features = node_features + condition_embed.unsqueeze(1)
                else:
                    node_features = node_features + condition_embed.unsqueeze(
                        1
                    ).unsqueeze(1)
        
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
            # Each unit processes: action (with joint_states), nodes, edges
            # Pass joint_states_history for state-action sequence self-attention
            # CRITICAL FIX: Pass condition_embed (not just timestep_embed) to AdaLN
            # unit returns (action_out, node_out)
            action_embed, node_features_out = unit(
                action_embed, 
                node_features, 
                edge_features_embed, 
                condition_for_adaln,  # Use unified condition (timestep + subtask)
                joint_states_history=joint_states_history,
            )
            # IMPORTANT: propagate updated node features to next layer
            # node_features_out: [B,2,hidden] -> next layer expects [B,2,hidden] or [B,2,1,hidden]
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
                "edge_features": edge_features_raw,
                "action_embed": action_embed,
                # NEW:
                "z_layers": z_stack,              # [B, K, z_dim]
                "z_final": z_stack[:, -1, :],     # [B, z_dim]
            }
        else:
            return noise_pred  # [batch, pred_horizon, action_dim]
    
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
                
                # Include current edge + delta
                current_edge = edge_features_temporal[:, -1, :]
                if history_length > 1:
                    prev_edge = edge_features_temporal[:, -2, :]
                    edge_delta = current_edge - prev_edge
                    edge_features_raw = torch.cat([current_edge, edge_delta], dim=-1)
                else:
                    edge_delta = torch.zeros_like(current_edge)
                    edge_features_raw = torch.cat([current_edge, edge_delta], dim=-1)
            else:
                # Fallback: extract from current obs
                ee_node, object_node = self._extract_node_features(obs)
                current_edge = self._compute_edge_features(ee_node, object_node)
                edge_delta = torch.zeros_like(current_edge)
                edge_features_raw = torch.cat([current_edge, edge_delta], dim=-1)
                
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
                # Pass joint_states_history so unit can do proper State-Action Self-Attention
                # This ensures extract_features matches the training-time behavior
                # unit returns (action_out, node_out)
                action_embed, node_features_out = unit(
                    action_embed, 
                    node_features, 
                    edge_features_embed, 
                    None,  # timestep_embed (not needed for feature extraction)
                    joint_states_history,  # CRITICAL: Pass joint_states_history!
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

    def extract_z_fast(
        self,
        ee_node: torch.Tensor,      # [B, 7] 当前 EE
        obj_node: torch.Tensor,     # [B, 7] 当前 Object
    ) -> torch.Tensor:
        """
        轻量级 z 提取：只跑 Graph-Attention，不跑 Self-Attn/Cross-Attn
        用于 RL 高频调用
        
        只运行 Graph-Attention 层，输出所有层的 z¹...zᴷ
        
        Args:
            ee_node: [B, 7] - EE position(3) + orientation(4)
            obj_node: [B, 7] - Object position(3) + orientation(4)
        
        Returns:
            z_layers: [B, K, z_dim] - 所有层的 graph latent
        """
        with torch.no_grad():
            # 1. Embed nodes
            ee_embed = self.node_embedding(ee_node)      # [B, hidden]
            obj_embed = self.node_embedding(obj_node)    # [B, hidden]
            node_features = torch.stack([ee_embed, obj_embed], dim=1)  # [B, 2, hidden]
            
            # 2. Compute edge features
            edge_raw = self._compute_edge_features(ee_node, obj_node)  # [B, 2]
            edge_delta = torch.zeros_like(edge_raw)  # 简化，不用 delta
            edge_features_raw = torch.cat([edge_raw, edge_delta], dim=-1)  # [B, 4]
            edge_embed = self.edge_embedding(edge_features_raw)  # [B, graph_edge_dim]
            
            # 3. 遍历所有层，只跑 Graph-Attention，收集 z¹...zᴷ
            z_layers: list[torch.Tensor] = []
            
            for unit in self.graph_dit_units:
                # 准备 node_features 格式: [B, 2, 1, hidden] (temporal format for graph_attention)
                node_features_for_attn = node_features.unsqueeze(2)  # [B, 2, 1, hidden]
                
                # 只运行 Graph-Attention（跳过 Self-Attn 和 Cross-Attn）
                # 使用 zero condition (不需要 timestep/subtask condition)
                zero_condition = torch.zeros(
                    node_features_for_attn.shape[0],
                    self.cfg.hidden_dim,
                    device=node_features_for_attn.device,
                    dtype=node_features_for_attn.dtype,
                )
                
                # Apply normalization (简化版，只做 norm1)
                node_features_for_attn = unit.node_norm1(
                    node_features_for_attn, zero_condition
                )
                
                # Run graph attention
                node_features_new = unit.graph_attention(
                    node_features_for_attn, edge_embed
                )  # [B, 2, 1, hidden] or [B, 2, hidden]
                
                # Handle output shape: graph_attention may return [B, 2, 1, hidden] or [B, 2, hidden]
                if len(node_features_new.shape) == 4:
                    node_features_new = node_features_new.squeeze(2)  # [B, 2, hidden]
                
                # Add residual (简化版，不做 FF)
                node_residual = node_features  # [B, 2, hidden]
                node_features_new = node_features_new + node_residual
                
                # 更新 node_features 用于下一层
                node_features = node_features_new  # [B, 2, hidden]
                
                # Pool to z_k
                z_k = self._pool_node_latent(node_features)  # [B, z_dim]
                z_layers.append(z_k)
            
            # Stack all layers: [B, K, z_dim]
            z_stack = torch.stack(z_layers, dim=1)
            
        return z_stack  # [B, K, z_dim]
    
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
    ) -> torch.Tensor:
        """Flow Matching prediction: ODE solving for action trajectory.
        
        With Action Chunking, returns [batch, pred_horizon, action_dim].
        Flow Matching requires only 2-10 steps for fast inference.
        """
        self.eval()
        # Flow Matching inference steps: use config default (30) if not specified
        num_steps = (
            num_diffusion_steps
            if num_diffusion_steps is not None
            else self.cfg.num_inference_steps
        )
        
        with torch.no_grad():
            batch_size = obs.shape[0]
            device = obs.device
            pred_horizon = self.pred_horizon
            
            # Initialize with random noise TRAJECTORY [batch, pred_horizon, action_dim]
            # Flow Matching: integrate from noise (t=0) to data (t=1)
            action_t = torch.randn(
                batch_size, pred_horizon, self.cfg.action_dim, device=device
            )
            t = torch.zeros(batch_size, device=device)  # Start from t=0 (noise)
            
            # ODE solving: Euler method (forward integration from t=0 to t=1)
            dt = 1.0 / num_steps  # Step size (positive, going forward)
            
            for step in range(num_steps):
                # Convert t (in [0, 1]) to timesteps (in [0, diffusion_steps-1])
                timesteps = (
                    (t * (self.cfg.diffusion_steps - 1))
                    .long()
                    .clamp(0, self.cfg.diffusion_steps - 1)
                )
                
                # Predict velocity field for entire trajectory
                velocity = self.forward(
                    obs,
                    noisy_action=action_t,  # [batch, pred_horizon, action_dim]
                    action_history=action_history,
                    ee_node_history=ee_node_history,
                    object_node_history=object_node_history,
                    joint_states_history=joint_states_history,
                    subtask_condition=subtask_condition,
                    timesteps=timesteps,
                )  # [batch, pred_horizon, action_dim]
                
                # Euler forward step: x_{t+dt} = x_t + dt * v_t
                action_t = action_t + dt * velocity
                t = t + dt  # Move forward in time
            
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
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss for training (Flow Matching).
        
        Args:
            obs: Observations [batch_size, obs_dim]
            actions: Target actions [batch_size, pred_horizon, action_dim]
            action_history: Action history [batch_size, hist_len, action_dim] (optional)
            ee_node_history: EE node history [batch_size, hist_len, 7] (optional)
            object_node_history: Object node history [batch_size, hist_len, 7] (optional)
            joint_states_history: Joint states history [batch_size, hist_len, joint_dim] (optional)
            subtask_condition: Subtask condition (one-hot) [batch_size, num_subtasks] (optional)
            timesteps: Diffusion timesteps [batch_size] (optional, not used in flow matching)
            mask: Boolean mask for valid samples [batch_size] (optional, True = valid, False = padding)
            
        Returns:
            dict: Loss dictionary with 'total_loss' and other losses
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
        timesteps = (t * (self.cfg.diffusion_steps - 1)).long()  # [batch_size]
        
        # Predict velocity field for entire trajectory
        v_pred = self.forward(
            obs,
            noisy_action=x_t,
            action_history=action_history,
            ee_node_history=ee_node_history,
            object_node_history=object_node_history,
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

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


@configclass
class GraphDiTPolicyCfg:
    """Configuration for Graph-DiT Policy."""
    
    obs_dim: int = MISSING
    """Observation dimension (input to policy). Note: obs should be dict or separable."""
    
    action_dim: int = MISSING
    """Action dimension (output from policy)."""
    
    # Observation structure indices (for backward compatibility with flattened obs)
    # CRITICAL: These should match the actual observation structure
    # Default assumes: [joint_pos(6), joint_vel(6), object_pos(3), object_ori(4),
    #                  ee_pos(3), ee_ori(4), target_object_position(7), actions(6)]
    obs_structure: dict[str, tuple[int, int]] | None = None
    """
    Observation structure mapping for flattened observations.
    If None, uses default indices. If provided, should be a dict like:
    {
        'joint_pos': (0, 6),
        'joint_vel': (6, 12),
        'object_position': (12, 15),
        'object_orientation': (15, 19),
        'ee_position': (19, 22),
        'ee_orientation': (22, 26),
        'target_object_position': (26, 33),
        'actions': (33, 39),
    }
    """
    
    # Graph-DiT specific parameters
    hidden_dim: int = 256
    """Hidden dimension for Graph-DiT."""
    
    num_layers: int = 6
    """Number of Graph-DiT layers."""
    
    num_heads: int = 8
    """Number of attention heads."""
    
    graph_edge_dim: int = 128
    """Dimension for graph edge embeddings."""
    
    diffusion_steps: int = 100
    """Number of diffusion steps."""
    
    noise_schedule: str = "cosine"
    """Noise schedule: 'cosine', 'linear'."""
    
    mode: str = "ddpm"
    """Training/inference mode: 'ddpm' (Denoising Diffusion Probabilistic Model) or 'flow_matching'.
    
    - 'ddpm': Standard diffusion process, requires 50-100 steps for inference
    - 'flow_matching': Flow matching (Rectified Flow), requires 1-10 steps for inference (much faster)
    """
    
    num_subtasks: int = 2
    """Number of subtasks (for conditional generation)."""
    
    device: str = "cuda"
    """Device to run on."""
    
    # Node and edge dimensions
    node_dim: int = 7
    """Node feature dimension: position(3) + orientation(4) = 7"""
    
    edge_dim: int = 2
    """Edge feature dimension: distance(1) + orientation_similarity(1) = 2"""
    
    # Action history
    action_history_length: int = 4
    """Number of historical actions to use for self-attention."""
    
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
    """Adaptive Layer Normalization that modulates based on condition."""
    
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 6 * hidden_dim, bias=True)
        )
        self.scale_shift = None  # Will be computed from condition
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            condition: Condition tensor [batch, condition_dim]
        """
        # Compute scale and shift from condition
        condition_embed = self.adaLN_modulation(condition)  # [batch, 6*hidden_dim]
        # Split into 6 parts (could be used for different modulations)
        # For simplicity, we use the first hidden_dim for scale and next for shift
        scale = condition_embed[:, :x.shape[-1]].unsqueeze(1)  # [batch, 1, hidden_dim]
        shift = condition_embed[:, x.shape[-1]:2 * x.shape[-1]].unsqueeze(1)  # [batch, 1, hidden_dim]
        
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
    
    def __init__(self, hidden_dim: int, num_heads: int, edge_dim: int = 128, use_edge_modulation: bool = True):
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
                nn.Linear(edge_dim, hidden_dim),
                nn.Sigmoid()  # Gate in [0, 1]
            )
            
            # Option 2: Scale and Shift (like AdaLN, but conditioned on Edge)
            self.edge_to_scale = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.Tanh()  # Scale in [-1, 1], can be expanded
            )
            self.edge_to_shift = nn.Linear(edge_dim, hidden_dim)
            
            # Option 3: Per-head modulation (more fine-grained control)
            self.edge_to_head_scale = nn.Sequential(
                nn.Linear(edge_dim, num_heads),
                nn.Tanh()
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
            node_features = node_features.view(batch_size, num_nodes * history_length, hidden_dim)  # [batch, num_nodes * history_length, hidden_dim]
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
            V_modulated = V * (1.0 + edge_scale.unsqueeze(1)) + edge_shift.unsqueeze(1)  # [batch, seq_len, hidden_dim]
            V_modulated = V_modulated * edge_gate.unsqueeze(1)  # Apply gate
            
            # Use modulated V for attention
            V = V_modulated
        # If modulation disabled, use original V (backward compatibility)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, heads, seq_len, seq_len]
        
        # Add edge bias (baseline method - still useful for attention scores)
        # Edge features -> bias for each head
        edge_bias = self.edge_to_bias(edge_features)  # [batch, num_heads]
        
        # Additional per-head modulation (optional, fine-grained control)
        if self.use_edge_modulation:
            head_scale = self.edge_to_head_scale(edge_features)  # [batch, num_heads]
            # Modulate attention scores per head
            scores = scores * (1.0 + head_scale.unsqueeze(-1).unsqueeze(-1))  # [batch, heads, seq_len, seq_len]
        
        if temporal_mode:
            # For temporal mode: create attention bias for spatial connections (EE <-> Object) across all time steps
            # PERFORMANCE FIX: Use vectorized operations instead of Python loop
            attention_bias = torch.zeros(
                batch_size, self.num_heads, seq_len, seq_len,
                device=node_features.device, dtype=node_features.dtype
            )
            # EE and Object connect across all time steps
            # PERFORMANCE FIX: Use vectorized indexing instead of Python loop
            t_indices = torch.arange(history_length, device=node_features.device)
            ee_indices = t_indices  # [history_length]
            obj_indices = history_length + t_indices  # [history_length]
            
            # Vectorized assignment: set bias for all timesteps at once
            # Expand edge_bias: [batch, num_heads] -> [batch, num_heads, history_length]
            edge_bias_expanded = edge_bias.unsqueeze(-1).expand(-1, -1, history_length)  # [batch, num_heads, history_length]
            
            # Use advanced indexing to set all connections at once
            # EE -> Object connections
            attention_bias[:, :, ee_indices, obj_indices] = edge_bias_expanded.permute(0, 1, 2)  # [batch, num_heads, history_length]
            # Object -> EE connections (symmetric)
            attention_bias[:, :, obj_indices, ee_indices] = edge_bias_expanded.permute(0, 1, 2)
        else:
            # Non-temporal: simple 2-node graph
            attention_bias = torch.zeros(
                batch_size, self.num_heads, num_nodes, num_nodes,
                device=node_features.device, dtype=node_features.dtype
            )
            # Symmetric edge: both nodes attend to each other with same bias
            attention_bias[:, :, 0, 1] = edge_bias  # EE -> Object
            attention_bias[:, :, 1, 0] = edge_bias  # Object -> EE
        
        scores = scores + attention_bias
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [batch, heads, num_nodes, head_dim]
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous()  # [batch, num_nodes (* history_length), heads, head_dim]
        
        if temporal_mode:
            # Reshape back to temporal format
            out = out.view(batch_size, num_nodes, history_length, self.hidden_dim)  # [batch, num_nodes, history_length, hidden_dim]
            # Average over temporal dimension to get final node features
            out = out.mean(dim=2)  # [batch, num_nodes, hidden_dim]
        else:
            out = out.view(batch_size, num_nodes, self.hidden_dim)  # [batch, num_nodes, hidden_dim]
        
        out = self.out_proj(out)  # [batch, num_nodes, hidden_dim]
        
        return out


class GraphDiTUnit(nn.Module):
    """Single Graph DiT unit following the architecture.
    
    Steps:
    1. Last action self-attention → a_new
    2. Node attention with edge features → node_features_new
    3. Cross-attention (a_new × node_features_new) → noise_pred
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, edge_dim: int = 128, use_edge_modulation: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Step 1: Action self-attention
        self.action_self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        # CRITICAL FIX: Use AdaLN instead of LayerNorm for better diffusion performance
        # AdaLN dynamically adjusts scale and shift based on timestep condition
        self.action_norm1 = AdaptiveLayerNorm(hidden_dim, hidden_dim)  # condition_dim = hidden_dim
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
        
        # Step 3: Cross-attention (action queries node features)
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
    ):
        """
        Args:
            action: Action sequence [batch, seq_len, hidden_dim] (should be action history)
            node_features: Node features [batch, 2, hidden_dim] (EE, Object)
            edge_features: Edge features [batch, edge_dim]
            timestep_embed: Timestep embedding [batch, hidden_dim] (optional)
        Returns:
            noise_pred: Predicted noise [batch, hidden_dim]
        """
        # Ensure action is a sequence (for self-attention to be meaningful)
        if len(action.shape) == 2:
            action = action.unsqueeze(1)  # [batch, hidden_dim] -> [batch, 1, hidden_dim]
        
        # Prepare condition embedding for AdaLN (timestep embedding)
        # CRITICAL FIX: AdaLN requires condition embedding for proper diffusion conditioning
        condition_emb = timestep_embed if timestep_embed is not None else None
        
        # Step 1: Action self-attention (now meaningful with action sequence)
        # action: [batch, seq_len, hidden_dim] where seq_len is action_history_length
        action_residual = action
        # Use AdaLN with condition embedding
        if condition_emb is not None:
            action = self.action_norm1(action, condition_emb)
        else:
            # Fallback: use zero condition if no timestep provided
            zero_condition = torch.zeros(action.shape[0], self.hidden_dim, device=action.device, dtype=action.dtype)
            action = self.action_norm1(action, zero_condition)
        a_new, _ = self.action_self_attn(action, action, action)  # [batch, seq_len, hidden_dim]
        a_new = a_new + action_residual
        
        # Use the last action from sequence for downstream processing
        # This aggregates the sequence information through self-attention, then uses the final action
        if a_new.shape[1] > 1:
            a_new = a_new[:, -1:, :]  # Take last action [batch, 1, hidden_dim]
        
        action_residual = a_new
        if condition_emb is not None:
            a_new = self.action_norm2(a_new, condition_emb)
        else:
            zero_condition = torch.zeros(a_new.shape[0], self.hidden_dim, device=a_new.device, dtype=a_new.dtype)
            a_new = self.action_norm2(a_new, zero_condition)
        a_new = self.action_ff(a_new) + action_residual
        
        # Step 2: Node attention with edge features
        # Handle temporal node features
        if len(node_features.shape) == 4:
            # Temporal: [batch, 2, history_length, hidden_dim]
            # Average over temporal dimension for residual connection
            node_residual = node_features.mean(dim=2)  # [batch, 2, hidden_dim]
            node_features_for_attn = node_features  # [batch, 2, history_length, hidden_dim]
        else:
            node_residual = node_features  # [batch, 2, hidden_dim]
            # Add temporal dimension for consistency
            node_features_for_attn = node_features.unsqueeze(2)  # [batch, 2, 1, hidden_dim]
        
        # Use AdaLN for node features
        if condition_emb is not None:
            # For temporal features, we need to handle the condition properly
            if len(node_features_for_attn.shape) == 4:
                # [batch, 2, history_length, hidden_dim] - expand condition
                condition_expanded = condition_emb.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, hidden_dim]
                node_features_for_attn = self.node_norm1(node_features_for_attn, condition_expanded)
            else:
                condition_expanded = condition_emb.unsqueeze(1)  # [batch, 1, hidden_dim]
                node_features_for_attn = self.node_norm1(node_features_for_attn, condition_expanded)
        else:
            zero_condition = torch.zeros(
                node_features_for_attn.shape[0], self.hidden_dim,
                device=node_features_for_attn.device, dtype=node_features_for_attn.dtype
            )
            if len(node_features_for_attn.shape) == 4:
                zero_condition = zero_condition.unsqueeze(1).unsqueeze(1)
            else:
                zero_condition = zero_condition.unsqueeze(1)
            node_features_for_attn = self.node_norm1(node_features_for_attn, zero_condition)
        
        node_features_new = self.graph_attention(node_features_for_attn, edge_features)  # [batch, 2, hidden_dim]
        
        # Add residual
        if len(node_residual.shape) == 3:
            node_features_new = node_features_new + node_residual  # [batch, 2, hidden_dim]
        else:
            # Shouldn't happen, but handle gracefully
            node_features_new = node_features_new
        
        node_residual = node_features_new
        if condition_emb is not None:
            condition_expanded = condition_emb.unsqueeze(1)  # [batch, 1, hidden_dim]
            node_features_new = self.node_norm2(node_features_new, condition_expanded)
        else:
            zero_condition = torch.zeros(
                node_features_new.shape[0], self.hidden_dim,
                device=node_features_new.device, dtype=node_features_new.dtype
            ).unsqueeze(1)
            node_features_new = self.node_norm2(node_features_new, zero_condition)
        node_features_new = self.node_ff(node_features_new) + node_residual
        
        # Step 3: Cross-attention (a_new as query, node_features_new as key/value)
        cross_residual = a_new
        # Use AdaLN for cross-attention
        if condition_emb is not None:
            a_new_norm = self.cross_norm1(a_new, condition_emb)
        else:
            zero_condition = torch.zeros(a_new.shape[0], self.hidden_dim, device=a_new.device, dtype=a_new.dtype)
            a_new_norm = self.cross_norm1(a_new, zero_condition)
        # Note: We don't modify node_features_new in cross-attention, only use it as key/value
        node_features_new_norm = node_features_new  # Use original node features (or could normalize separately)
        noise_embed, _ = self.cross_attn(a_new_norm, node_features_new_norm, node_features_new_norm)
        noise_embed = noise_embed + cross_residual
        
        cross_residual = noise_embed
        if condition_emb is not None:
            noise_embed = self.cross_norm2(noise_embed, condition_emb)
        else:
            zero_condition = torch.zeros(noise_embed.shape[0], self.hidden_dim, device=noise_embed.device, dtype=noise_embed.dtype)
            noise_embed = self.cross_norm2(noise_embed, zero_condition)
        noise_embed = self.cross_ff(noise_embed) + cross_residual
        
        # Return noise prediction (squeeze sequence dimension)
        noise_pred = noise_embed.squeeze(1)  # [batch, hidden_dim]
        
        return noise_pred


class NoiseScheduler:
    """Noise scheduler for diffusion process."""
    
    def __init__(self, num_steps: int = 100, schedule: str = "cosine"):
        self.num_steps = num_steps
        self.schedule = schedule
        
        if schedule == "cosine":
            # Cosine schedule: alpha_bar(t) = cos^2(π/2 * (t/T))
            self.alphas_cumprod = torch.cos(torch.linspace(0, math.pi / 2, num_steps)) ** 2
        elif schedule == "linear":
            # Linear schedule: alpha_bar(t) = 1 - t/T
            self.alphas_cumprod = torch.linspace(1.0, 0.0, num_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.alphas = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1] / (self.alphas_cumprod[1:] + 1e-8)])
        self.betas = 1.0 - self.alphas
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Precompute for efficiency
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        """Add noise to x at timestep t."""
        # Move scheduler buffers to the same device as x
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(x.device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x.device)
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_x
    
    def get_alpha_t(self, t: torch.Tensor):
        """Get alpha_t for timestep t."""
        return self.alphas.to(t.device)[t]
    
    def get_alpha_bar_t(self, t: torch.Tensor):
        """Get alpha_bar_t (alphas_cumprod) for timestep t."""
        return self.alphas_cumprod.to(t.device)[t]
    
    def get_alpha_bar_t_prev(self, t: torch.Tensor):
        """Get alpha_bar_t_prev for timestep t."""
        return self.alphas_cumprod_prev.to(t.device)[t]


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
        
        # Edge embedding: distance(1) + orientation_similarity(1) = 2 -> edge_dim
        self.edge_embedding = nn.Sequential(
            nn.Linear(cfg.edge_dim, cfg.graph_edge_dim),
            nn.LayerNorm(cfg.graph_edge_dim),
            nn.GELU(),
        )
        
        # Action embedding (for single action)
        self.action_embedding = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )
        
        # Action history length
        self.action_history_length = cfg.action_history_length
        
        # Additional observation encoding (joint_pos, joint_vel, target_object_position)
        # These are used as additional context but not as graph nodes
        self.context_encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim - cfg.node_dim * 2 - cfg.action_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )
        
        # Timestep embedding (for diffusion)
        self.timestep_embedding = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim),
        )
        # Sinusoidal position embedding for timesteps
        self.register_buffer('timestep_pos_embed', self._get_timestep_embedding(cfg.diffusion_steps, cfg.hidden_dim))
        
        # Subtask condition encoder
        if cfg.num_subtasks > 0:
            self.subtask_encoder = nn.Sequential(
                nn.Linear(cfg.num_subtasks, cfg.hidden_dim // 4),
                nn.LayerNorm(cfg.hidden_dim // 4),
                nn.GELU(),
            )
            self.condition_proj = nn.Linear(cfg.hidden_dim + cfg.hidden_dim // 4, cfg.hidden_dim)
        
        # Graph DiT units (stacked layers)
        self.graph_dit_units = nn.ModuleList([
            GraphDiTUnit(
                cfg.hidden_dim, cfg.num_heads, cfg.graph_edge_dim,
                use_edge_modulation=cfg.use_edge_modulation
            )
            for _ in range(cfg.num_layers)
        ])
        
        # Final noise prediction head
        self.noise_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(cfg.diffusion_steps, cfg.noise_schedule)
        
        # Initialize weights
        self._init_weights()
    
    def _get_timestep_embedding(self, num_steps: int, dim: int):
        """Create sinusoidal timestep embeddings."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = torch.arange(num_steps, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_steps, 1)], dim=1)
        return emb  # [num_steps, hidden_dim]
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
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
            object_position = obs[:, obs_struct['object_position'][0]:obs_struct['object_position'][1]]
            object_orientation = obs[:, obs_struct['object_orientation'][0]:obs_struct['object_orientation'][1]]
            ee_position = obs[:, obs_struct['ee_position'][0]:obs_struct['ee_position'][1]]
            ee_orientation = obs[:, obs_struct['ee_orientation'][0]:obs_struct['ee_orientation'][1]]
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
        object_node = torch.cat([object_position, object_orientation], dim=-1)  # [batch, 7]
        
        return ee_node, object_node
    
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
        
        # 1. Distance (L2 norm)
        distance = torch.norm(ee_pos - obj_pos, dim=-1, keepdim=True)  # [batch, 1]
        
        # 2. Orientation similarity (quaternion dot product)
        # Quaternion dot product: q1 · q2 = w1*w2 + x1*x2 + y1*y2 + z1*z2
        quat_dot = torch.sum(ee_quat * obj_quat, dim=-1, keepdim=True)  # [batch, 1]
        # Take absolute value (q and -q represent same rotation)
        orientation_similarity = torch.abs(quat_dot)  # [batch, 1], range [0, 1]
        
        edge_features = torch.cat([distance, orientation_similarity], dim=-1)  # [batch, 2]
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
        subtask_condition: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict:
        """
        Forward pass of Graph-DiT policy.
        
        Args:
            obs: Observations [batch_size, obs_dim] - concatenated observations
            noisy_action: Noisy action for diffusion [batch_size, action_dim] (required for diffusion training)
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
            ee_node_history_flat = ee_node_history.view(-1, node_dim)  # [batch * history_length, 7]
            object_node_history_flat = object_node_history.view(-1, node_dim)  # [batch * history_length, 7]
            
            ee_node_embed_flat = self.node_embedding(ee_node_history_flat)  # [batch * history_length, hidden_dim]
            object_node_embed_flat = self.node_embedding(object_node_history_flat)  # [batch * history_length, hidden_dim]
            
            ee_node_embed = ee_node_embed_flat.view(batch_size, history_length, self.cfg.hidden_dim)  # [batch, history_length, hidden_dim]
            object_node_embed = object_node_embed_flat.view(batch_size, history_length, self.cfg.hidden_dim)  # [batch, history_length, hidden_dim]
            
            # Stack EE and Object: [batch, history_length, 2, hidden_dim]
            # Then transpose to [batch, 2, history_length, hidden_dim] for Graph Attention
            node_features = torch.stack([ee_node_embed, object_node_embed], dim=2)  # [batch, history_length, 2, hidden_dim]
            node_features = node_features.transpose(1, 2)  # [batch, 2, history_length, hidden_dim]
            
            # Compute edge features for each timestep in history
            # PERFORMANCE FIX: Use vectorized operations instead of Python loop
            # Reshape to [batch * history_length, 7] for batch processing
            ee_history_flat = ee_node_history.view(-1, self.node_dim)  # [batch * history_length, 7]
            obj_history_flat = object_node_history.view(-1, self.node_dim)  # [batch * history_length, 7]
            
            # Vectorized edge computation for all timesteps at once
            edge_features_raw_all = self._compute_edge_features(ee_history_flat, obj_history_flat)  # [batch * history_length, 2]
            edge_features_raw = edge_features_raw_all.view(batch_size, history_length, 2)  # [batch, history_length, 2]
            
            # Use average edge features across history (or could use last timestep)
            edge_features_raw = edge_features_raw.mean(dim=1)  # [batch, 2]
            
        else:
            # Fallback: extract from current obs (backward compatibility)
            ee_node, object_node = self._extract_node_features(obs)  # [batch, 7] each
            
            # Compute edge features
            edge_features_raw = self._compute_edge_features(ee_node, object_node)  # [batch, 2]
            
            # Embed nodes (single timestep)
            ee_node_embed = self.node_embedding(ee_node)  # [batch, hidden_dim]
            object_node_embed = self.node_embedding(object_node)  # [batch, hidden_dim]
            # For compatibility: [batch, 2, 1, hidden_dim]
            node_features = torch.stack([ee_node_embed, object_node_embed], dim=1).unsqueeze(2)  # [batch, 2, 1, hidden_dim]
        
        # Embed edge features
        edge_features_embed = self.edge_embedding(edge_features_raw)  # [batch, graph_edge_dim]
        
        # Embed action: noisy_action (for diffusion) + action_history (as context)
        # CRITICAL FIX: Action history should be used as context even when noisy_action is provided
        # This preserves temporal information that is crucial for action prediction
        
        # Step 1: Embed noisy_action (the target we want to denoise)
        if noisy_action is not None:
            # For diffusion training: use noisy_action as the main action to denoise
            action_input = noisy_action  # [batch, action_dim]
            noisy_action_embed = self.action_embedding(action_input)  # [batch, hidden_dim]
            noisy_action_embed = noisy_action_embed.unsqueeze(1)  # [batch, 1, hidden_dim]
        else:
            # Inference mode: extract last action from obs or use action_history
            if action_history is not None and action_history.shape[1] > 0:
                # Use the last action from history
                noisy_action_embed = self.action_embedding(action_history[:, -1, :])  # [batch, hidden_dim]
                noisy_action_embed = noisy_action_embed.unsqueeze(1)  # [batch, 1, hidden_dim]
            else:
                # Fallback: extract last action from obs (backward compatibility)
                last_action = obs[:, -self.cfg.action_dim:]  # [batch, action_dim]
                noisy_action_embed = self.action_embedding(last_action)  # [batch, hidden_dim]
                noisy_action_embed = noisy_action_embed.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Step 2: Embed action_history as context (CRITICAL: preserve temporal information)
        # Even when noisy_action is provided, we should use history as context
        if action_history is not None and action_history.shape[1] > 0:
            # Embed action history [batch, action_history_length, action_dim]
            batch_size, seq_len, action_dim = action_history.shape
            action_history_flat = action_history.view(-1, action_dim)  # [batch * seq_len, action_dim]
            history_embed_flat = self.action_embedding(action_history_flat)  # [batch * seq_len, hidden_dim]
            history_embed = history_embed_flat.view(batch_size, seq_len, self.cfg.hidden_dim)  # [batch, seq_len, hidden_dim]
            
            # Concatenate history with noisy_action: [batch, seq_len+1, hidden_dim]
            # This allows self-attention to see both history and current noisy action
            action_embed = torch.cat([history_embed, noisy_action_embed], dim=1)  # [batch, seq_len+1, hidden_dim]
        else:
            # No history available, use only noisy_action
            action_embed = noisy_action_embed  # [batch, 1, hidden_dim]
        
        # Get timestep embedding (if provided)
        timestep_embed = None
        if timesteps is not None:
            timestep_embed = self._get_timestep_embed(timesteps)  # [batch, hidden_dim]
            # Add timestep embedding to action
            action_embed = action_embed + timestep_embed.unsqueeze(1)  # [batch, seq_len, hidden_dim] or [batch, 1, hidden_dim]
            
            # Add to node features (handle both temporal and non-temporal)
            if len(node_features.shape) == 4:
                # Temporal: [batch, 2, history_length, hidden_dim]
                node_features = node_features + timestep_embed.unsqueeze(1).unsqueeze(1)  # Broadcast to all timesteps
            else:
                # Non-temporal: [batch, 2, 1, hidden_dim] or [batch, 2, hidden_dim]
                if len(node_features.shape) == 3:
                    node_features = node_features + timestep_embed.unsqueeze(1)
                else:
                    node_features = node_features + timestep_embed.unsqueeze(1).unsqueeze(1)
        
        # Add subtask condition if provided
        if subtask_condition is not None and hasattr(self, 'subtask_encoder'):
            subtask_embed = self.subtask_encoder(subtask_condition)  # [batch, hidden_dim // 4]
            # Combine with timestep embedding or use separately
            if timestep_embed is not None:
                condition_embed = torch.cat([timestep_embed, subtask_embed], dim=-1)
                condition_embed = self.condition_proj(condition_embed)  # [batch, hidden_dim]
            else:
                # Expand subtask embed and project
                # Pad with zeros to match expected input size for condition_proj
                batch_size = obs.shape[0]
                expanded_embed = torch.cat([
                    torch.zeros(
                        batch_size, 3 * subtask_embed.shape[-1],
                        device=subtask_embed.device,
                        dtype=subtask_embed.dtype
                    ),
                    subtask_embed
                ], dim=-1)
                condition_embed = self.condition_proj(expanded_embed)  # [batch, hidden_dim]
            
            # Add condition to action and nodes
            action_embed = action_embed + condition_embed.unsqueeze(1)  # [batch, seq_len, hidden_dim] or [batch, 1, hidden_dim]
            
            # Add to node features (handle both temporal and non-temporal)
            if len(node_features.shape) == 4:
                # Temporal: [batch, 2, history_length, hidden_dim]
                node_features = node_features + condition_embed.unsqueeze(1).unsqueeze(1)  # Broadcast to all timesteps
            else:
                # Non-temporal
                if len(node_features.shape) == 3:
                    node_features = node_features + condition_embed.unsqueeze(1)
                else:
                    node_features = node_features + condition_embed.unsqueeze(1).unsqueeze(1)
        
        # Process through Graph DiT units
        for unit in self.graph_dit_units:
            # Each unit processes: action, nodes, edges
            noise_embed = unit(action_embed, node_features, edge_features_embed, timestep_embed)
            # Update action_embed for next layer (residual connection style)
            action_embed = noise_embed.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Final noise prediction
        noise_pred = self.noise_head(noise_embed)  # [batch, action_dim]
        
        if return_dict:
            return {
                "noise_pred": noise_pred,
                "node_features": node_features,
                "edge_features": edge_features_raw,
                "action_embed": action_embed,
            }
        else:
            return noise_pred
    
    def predict(
        self, 
        obs: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
        num_diffusion_steps: int | None = None,
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Predict actions from observations (inference mode).
        
        Args:
            obs: Observations [batch_size, obs_dim]
            action_history: Action history [batch_size, history_length, action_dim] (optional)
            ee_node_history: EE node history [batch_size, history_length, 7] (optional)
            object_node_history: Object node history [batch_size, history_length, 7] (optional)
            subtask_condition: Subtask condition (one-hot) [batch_size, num_subtasks] (optional)
            num_diffusion_steps: Number of steps for inference 
                - DDPM: default 50-100 steps
                - Flow Matching: default 1-10 steps (much faster!)
            deterministic: If True, use deterministic prediction
            
        Returns:
            actions: Predicted actions [batch_size, action_dim]
        """
        if self.cfg.mode == "flow_matching":
            return self._flow_matching_predict(
                obs, action_history, ee_node_history, 
                object_node_history, subtask_condition, 
                num_diffusion_steps, deterministic
            )
        else:  # ddpm
            return self._ddpm_predict(
                obs, action_history, ee_node_history,
                object_node_history, subtask_condition,
                num_diffusion_steps, deterministic
            )
    
    def _ddpm_predict(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor | None,
        ee_node_history: torch.Tensor | None,
        object_node_history: torch.Tensor | None,
        subtask_condition: torch.Tensor | None,
        num_diffusion_steps: int | None,
        deterministic: bool,
    ) -> torch.Tensor:
        """DDPM prediction: iterative denoising."""
        self.eval()
        num_steps = num_diffusion_steps if num_diffusion_steps is not None else self.cfg.diffusion_steps
        
        with torch.no_grad():
            batch_size = obs.shape[0]
            device = obs.device
            
            # Initialize with random noise
            action_t = torch.randn(batch_size, self.cfg.action_dim, device=device)
            
            # Iterative denoising
            for step in range(num_steps):
                t = torch.full((batch_size,), num_steps - step - 1, device=device, dtype=torch.long)
                
                # Predict noise
                noise_pred = self.forward(
                    obs,
                    noisy_action=action_t,
                    action_history=action_history,
                    ee_node_history=ee_node_history,
                    object_node_history=object_node_history,
                    subtask_condition=subtask_condition,
                    timesteps=t
                )
                
                # Denoise step (DDPM)
                if step < num_steps - 1:
                    alpha_bar_t = self.noise_scheduler.get_alpha_bar_t(t)
                    alpha_bar_t_prev = self.noise_scheduler.get_alpha_bar_t_prev(t)
                    
                    # Predict x_0 (clean action)
                    pred_x0 = (action_t - torch.sqrt(1 - alpha_bar_t.unsqueeze(-1)) * noise_pred) / torch.sqrt(alpha_bar_t.unsqueeze(-1))
                    
                    # Compute coefficients
                    pred_dir = torch.sqrt(1 - alpha_bar_t_prev.unsqueeze(-1)) * noise_pred
                    pred_prev = torch.sqrt(alpha_bar_t_prev.unsqueeze(-1)) * pred_x0 + pred_dir
                    
                    action_t = pred_prev
                else:
                    # Last step: directly predict
                    alpha_bar_t = self.noise_scheduler.get_alpha_bar_t(t)
                    action_t = (action_t - torch.sqrt(1 - alpha_bar_t.unsqueeze(-1)) * noise_pred) / torch.sqrt(alpha_bar_t.unsqueeze(-1))
            
            if not deterministic:
                # Add small noise for exploration
                action_t = action_t + 0.05 * torch.randn_like(action_t)
            
            return action_t
    
    def _flow_matching_predict(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor | None,
        ee_node_history: torch.Tensor | None,
        object_node_history: torch.Tensor | None,
        subtask_condition: torch.Tensor | None,
        num_diffusion_steps: int | None,
        deterministic: bool,
    ) -> torch.Tensor:
        """Flow Matching prediction: ODE solving (much faster!)."""
        self.eval()
        # Flow Matching typically needs fewer steps (1-10) vs DDPM (50-100)
        num_steps = num_diffusion_steps if num_diffusion_steps is not None else max(1, self.cfg.diffusion_steps // 10)
        
        with torch.no_grad():
            batch_size = obs.shape[0]
            device = obs.device
            
            # CRITICAL FIX: Flow Matching integration direction
            # In Flow Matching, we integrate from noise (t=0) to data (t=1)
            # Velocity field: v_t = x_1 - x_0 (points from noise to data)
            # ODE: dx/dt = v_t, so x_{t+dt} = x_t + dt * v_t
            
            # Initialize with random noise (t=0, not t=1!)
            action_t = torch.randn(batch_size, self.cfg.action_dim, device=device)
            t = torch.zeros(batch_size, device=device)  # Start from t=0 (noise)
            
            # ODE solving: Euler method (forward integration from t=0 to t=1)
            dt = 1.0 / num_steps  # Step size (positive, going forward)
            
            for step in range(num_steps):
                # Convert t (in [0, 1]) to timesteps (in [0, diffusion_steps-1])
                timesteps = (t * (self.cfg.diffusion_steps - 1)).long().clamp(0, self.cfg.diffusion_steps - 1)
                
                # Predict velocity field (points from noise to data)
                velocity = self.forward(
                    obs,
                    noisy_action=action_t,
                    action_history=action_history,
                    ee_node_history=ee_node_history,
                    object_node_history=object_node_history,
                    subtask_condition=subtask_condition,
                    timesteps=timesteps
                )  # Network outputs velocity field: v_t = data - noise
                
                # Euler forward step: x_{t+dt} = x_t + dt * v_t
                # This moves from noise (t=0) towards data (t=1)
                action_t = action_t + dt * velocity
                t = t + dt  # Move forward in time
            
            if not deterministic:
                # Add small noise for exploration
                action_t = action_t + 0.05 * torch.randn_like(action_t)
            
            return action_t
    
    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss for training (DDPM or Flow Matching).
        
        Args:
            obs: Observations [batch_size, obs_dim]
            actions: Target actions [batch_size, action_dim]
            subtask_condition: Subtask condition (one-hot) [batch_size, num_subtasks] (optional)
            timesteps: Diffusion timesteps [batch_size] (optional, sampled if not provided)
            
        Returns:
            dict: Loss dictionary with 'total_loss' and other losses
        """
        if self.cfg.mode == "flow_matching":
            return self._flow_matching_loss(
                obs, actions, action_history, ee_node_history, 
                object_node_history, subtask_condition
            )
        else:  # ddpm
            return self._ddpm_loss(
                obs, actions, action_history, ee_node_history,
                object_node_history, subtask_condition, timesteps
            )
    
    def _ddpm_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_history: torch.Tensor | None,
        ee_node_history: torch.Tensor | None,
        object_node_history: torch.Tensor | None,
        subtask_condition: torch.Tensor | None,
        timesteps: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """DDPM loss: predict noise."""
        batch_size = obs.shape[0]
        device = obs.device
        
        # Sample timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(0, self.cfg.diffusion_steps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(actions)
        
        # Add noise to actions
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
        
        # Predict noise
        noise_pred = self.forward(
            obs,
            noisy_action=noisy_actions,
            action_history=action_history,
            ee_node_history=ee_node_history,
            object_node_history=object_node_history,
            subtask_condition=subtask_condition,
            timesteps=timesteps
        )
        
        # Compute loss (MSE between predicted and actual noise)
        mse_loss = F.mse_loss(noise_pred, noise)
        
        # Optional: add velocity loss for smoothness
        if batch_size > 1:
            action_diff = actions[1:] - actions[:-1]
            pred_diff = (actions - noise_pred)[1:] - (actions - noise_pred)[:-1]
            smoothness_loss = F.mse_loss(pred_diff, action_diff)
            total_loss = mse_loss + 0.01 * smoothness_loss
        else:
            smoothness_loss = torch.tensor(0.0, device=device)
            total_loss = mse_loss
        
        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "smoothness_loss": smoothness_loss,
        }
    
    def _flow_matching_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_history: torch.Tensor | None,
        ee_node_history: torch.Tensor | None,
        object_node_history: torch.Tensor | None,
        subtask_condition: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Flow Matching loss: predict velocity field."""
        batch_size = obs.shape[0]
        device = obs.device
        
        # Sample time t ~ U(0, 1) for flow matching
        t = torch.rand(batch_size, device=device)  # [batch_size], each in [0, 1]
        
        # Sample noise
        noise = torch.randn_like(actions)  # [batch_size, action_dim]
        
        # Linear interpolation path: x_t = (1-t) * x_1 + t * x_0
        # x_1 = actions (data), x_0 = noise
        x_t = (1 - t.view(-1, 1)) * actions + t.view(-1, 1) * noise  # [batch_size, action_dim]
        
        # Ground truth velocity field: v_t = x_1 - x_0 (direction from noise to data)
        # In Flow Matching: v_t(x_t) = x_1 - x_0 where x_1=data, x_0=noise
        # This is the vector pointing from noise (t=0) to data (t=1)
        v_t = actions - noise  # [batch_size, action_dim]
        
        # Convert t to timesteps format for forward (scale to [0, diffusion_steps-1])
        # For flow matching, t in [0, 1] is mapped to timesteps in [0, diffusion_steps-1]
        timesteps = (t * (self.cfg.diffusion_steps - 1)).long()  # [batch_size]
        
        # Predict velocity field (network outputs velocity, not noise)
        # Reuse the same forward method but interpret output as velocity
        v_pred = self.forward(
            obs,
            noisy_action=x_t,
            action_history=action_history,
            ee_node_history=ee_node_history,
            object_node_history=object_node_history,
            subtask_condition=subtask_condition,
            timesteps=timesteps
        )  # [batch_size, action_dim]
        
        # Compute loss (MSE between predicted and ground truth velocity)
        mse_loss = F.mse_loss(v_pred, v_t)
        
        # Optional: smoothness loss
        if batch_size > 1:
            action_diff = actions[1:] - actions[:-1]
            v_pred_diff = v_pred[1:] - v_pred[:-1]
            smoothness_loss = F.mse_loss(v_pred_diff, action_diff)
            total_loss = mse_loss + 0.01 * smoothness_loss
        else:
            smoothness_loss = torch.tensor(0.0, device=device)
            total_loss = mse_loss
        
        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "smoothness_loss": smoothness_loss,
        }
    
    def save(self, path: str):
        """Save policy to file.
        
        Args:
            path: Path to save the model.
        """
        torch.save({
            "policy_state_dict": self.state_dict(),
            "cfg": self.cfg,
        }, path)
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
        
        # Create policy
        policy = cls(cfg)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.to(device)
        policy.eval()
        
        print(f"[GraphDiTPolicy] Loaded model from: {path}")
        return policy

# Copyright (c) 2024-2025, SO-ARM100 Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Graph-DiT (Graph Diffusion Transformer) Policy implementation.

This module implements a custom Graph-DiT policy for manipulation tasks.
Users can replace this with their own implementation.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch
import torch.nn as nn
from isaaclab.utils import configclass


@configclass
class GraphDiTPolicyCfg:
    """Configuration for Graph-DiT Policy.
    
    This is a placeholder configuration. Users should modify this
    to match their Graph-DiT architecture.
    """
    
    obs_dim: int = MISSING
    """Observation dimension (input to policy)."""
    
    action_dim: int = MISSING
    """Action dimension (output from policy)."""
    
    # Graph-DiT specific parameters (customize based on your architecture)
    hidden_dim: int = 256
    """Hidden dimension for Graph-DiT."""
    
    num_layers: int = 6
    """Number of transformer layers."""
    
    num_heads: int = 8
    """Number of attention heads."""
    
    graph_edge_dim: int = 128
    """Dimension for graph edges."""
    
    diffusion_steps: int = 100
    """Number of diffusion steps."""
    
    noise_schedule: str = "cosine"
    """Noise schedule: 'cosine', 'linear', etc."""
    
    num_subtasks: int = 2
    """Number of subtasks (for conditional generation)."""
    
    device: str = "cuda"
    """Device to run on."""


class GraphDiTPolicy(nn.Module):
    """Graph-DiT (Graph Diffusion Transformer) Policy.
    
    This is a placeholder implementation. Users should replace this
    with their own Graph-DiT architecture.
    
    The policy takes observations as input and outputs actions through
    a diffusion process with graph-based transformer architecture.
    
    Args:
        cfg: Configuration for the Graph-DiT policy.
    """
    
    def __init__(self, cfg: GraphDiTPolicyCfg):
        super().__init__()
        self.cfg = cfg
        
        # ============================================
        # TODO: Replace with your Graph-DiT implementation
        # ============================================
        
        # Placeholder implementation (replace with your architecture)
        # Example: Simple MLP backbone (replace with Graph-DiT)
        self.obs_encoder = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Subtask condition encoder (one-hot encoding of subtask description)
        # Input: one-hot vector [num_subtasks], Output: condition embedding
        self.subtask_encoder = nn.Sequential(
            nn.Linear(cfg.num_subtasks, cfg.hidden_dim // 4),  # Smaller embedding for condition
            nn.LayerNorm(cfg.hidden_dim // 4),
            nn.GELU(),
        )
        
        # Projection layer to combine obs_embed + subtask_embed -> hidden_dim
        self.condition_proj = nn.Linear(cfg.hidden_dim + cfg.hidden_dim // 4, cfg.hidden_dim)
        
        # Placeholder: Graph-DiT layers (replace with your implementation)
        # Example: Simple transformer (replace with Graph-DiT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.graph_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )
        
        # Output projection
        # Note: No Tanh() here - actions need to match dataset ranges
        # Action normalization should be handled in training script or environment
        self.action_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
            # No activation - output raw actions to match dataset ranges
        )
        
        # Diffusion-related components (replace with your implementation)
        self.diffusion_steps = cfg.diffusion_steps
        self.noise_schedule = cfg.noise_schedule
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        obs: torch.Tensor,
        subtask_condition: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict:
        """Forward pass of Graph-DiT policy.
        
        Args:
            obs: Observations [batch_size, obs_dim] or [batch_size, seq_len, obs_dim]
            subtask_condition: Subtask condition (one-hot) [batch_size, num_subtasks] (optional)
            timesteps: Diffusion timesteps [batch_size] (optional, for training)
            return_dict: If True, return dict with additional info
            
        Returns:
            actions: Predicted actions [batch_size, action_dim]
            or dict with 'actions' and other fields if return_dict=True
        """
        # Encode observations
        # obs: [batch_size, obs_dim] -> [batch_size, hidden_dim]
        obs_embed = self.obs_encoder(obs)
        
        # Encode subtask condition if provided
        if subtask_condition is not None:
            # subtask_condition: [batch_size, num_subtasks] -> [batch_size, hidden_dim // 4]
            subtask_embed = self.subtask_encoder(subtask_condition)
            # Concatenate with observation embedding
            # [batch_size, hidden_dim] + [batch_size, hidden_dim // 4] -> [batch_size, hidden_dim + hidden_dim // 4]
            obs_embed = torch.cat([obs_embed, subtask_embed], dim=-1)
            # Project back to hidden_dim
            obs_embed = self.condition_proj(obs_embed)
        
        # Add sequence dimension if needed for transformer
        # [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim]
        if len(obs_embed.shape) == 2:
            obs_embed = obs_embed.unsqueeze(1)
        
        # ============================================
        # TODO: Replace with your Graph-DiT forward pass
        # ============================================
        
        # Placeholder: Graph-DiT processing (replace with your implementation)
        # Example: Transformer encoding
        features = self.graph_transformer(obs_embed)  # [batch_size, seq_len, hidden_dim]
        features = features.squeeze(1)  # [batch_size, hidden_dim]
        
        # ============================================
        # TODO: Add diffusion process here
        # ============================================
        
        # Placeholder: Diffusion process (replace with your implementation)
        # If using diffusion, process features through diffusion steps here
        
        # Predict actions
        actions = self.action_head(features)  # [batch_size, action_dim]
        
        if return_dict:
            return {
                "actions": actions,
                "features": features,
                "obs_embed": obs_embed,
            }
        else:
            return actions
    
    def predict(
        self, 
        obs: torch.Tensor, 
        subtask_condition: torch.Tensor | None = None,
        deterministic: bool = True
    ) -> torch.Tensor:
        """Predict actions from observations (inference mode).
        
        Args:
            obs: Observations [batch_size, obs_dim]
            subtask_condition: Subtask condition (one-hot) [batch_size, num_subtasks] (optional)
            deterministic: If True, use deterministic prediction
            
        Returns:
            actions: Predicted actions [batch_size, action_dim]
        """
        self.eval()
        with torch.no_grad():
            actions = self.forward(obs, subtask_condition=subtask_condition)
            if not deterministic:
                # Add noise for exploration if needed
                actions = actions + 0.1 * torch.randn_like(actions)
                actions = torch.clamp(actions, -1.0, 1.0)
        return actions
    
    def loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        subtask_condition: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss for training.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            actions: Target actions [batch_size, action_dim]
            subtask_condition: Subtask condition (one-hot) [batch_size, num_subtasks] (optional)
            timesteps: Diffusion timesteps [batch_size] (optional)
            
        Returns:
            dict: Loss dictionary with 'total_loss' and other losses
        """
        # Forward pass
        pred_actions = self.forward(obs, subtask_condition=subtask_condition, timesteps=timesteps)
        
        # ============================================
        # TODO: Replace with your Graph-DiT loss function
        # ============================================
        
        # Placeholder: Simple MSE loss (replace with your diffusion loss)
        mse_loss = nn.functional.mse_loss(pred_actions, actions)
        
        # Example: Add diffusion loss if using diffusion
        # diffusion_loss = self._compute_diffusion_loss(...)
        
        total_loss = mse_loss  # + diffusion_loss
        
        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            # "diffusion_loss": diffusion_loss,
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
        checkpoint = torch.load(path, map_location=device)
        
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



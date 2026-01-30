# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
"""
Graph-Unet Policy: Graph encoder (node/edge) + Conditional U-Net 1D backbone.

Replaces the Transformer (DiT) with Conv1D U-Net for Flow Matching.
Conv1D acts as a low-pass filter along the action horizon, reducing jitter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from SO_101.policies.graph_dit_policy import GraphDiTPolicy, GraphDiTPolicyCfg


# ==============================================================================
# 1. Conditional U-Net 1D modules (Diffusion Policy backbone)
# ==============================================================================


def _safe_n_groups(n_groups: int, num_channels: int) -> int:
    """GroupNorm requires num_channels % n_groups == 0 (e.g. action_dim=5 and n_groups=8 fail)."""
    if num_channels <= 0:
        return 1
    for g in (n_groups, 4, 2, 1):
        if g <= num_channels and num_channels % g == 0:
            return g
    return 1


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish"""

    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()
        ng = _safe_n_groups(n_groups, out_channels)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(ng, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """ResNet-style block with FiLM conditioning (scale + shift)."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        # x: [B, in_channels, T], cond: [B, cond_dim]
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)  # [B, 2*C, 1]
        scale, shift = embed.chunk(2, dim=1)
        out = out * (1 + scale) + shift
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """
    Conditional U-Net 1D for action trajectory (Flow Matching).
    Input: [B, action_dim, horizon] (noisy action)
    Output: [B, action_dim, horizon] (predicted velocity)
    Condition: [B, cond_dim] (graph global feature + time embedding)
    """

    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        self.diffusion_step_encoder = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim, start_dim * 4),
            nn.Mish(),
            nn.Linear(start_dim * 4, start_dim),
        )
        cond_dim = global_cond_dim + start_dim

        self.down_modules = nn.ModuleList([])
        for i in range(len(down_dims)):
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(in_dim, out_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(out_dim, out_dim, cond_dim, kernel_size, n_groups),
                nn.Conv1d(out_dim, out_dim, 3, stride=2, padding=1),
            ]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        # Up path: after upsample x has out_dim channels; skip has in_dim channels -> concat = out_dim + in_dim
        # Last up stage must output start_dim (not input_dim) so final_conv can map start_dim -> input_dim
        self.up_modules = nn.ModuleList([])
        for i in reversed(range(len(down_dims))):
            in_dim = all_dims[i + 1]
            out_dim = start_dim if i == 0 else all_dims[i]
            skip_channels = in_dim  # skip from same down level
            self.up_modules.append(nn.ModuleList([
                nn.ConvTranspose1d(in_dim, out_dim, 4, stride=2, padding=1),
                ConditionalResidualBlock1D(out_dim + skip_channels, out_dim, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(out_dim, out_dim, cond_dim, kernel_size, n_groups),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, x, timestep_embed, global_cond):
        # x: [B, C, T], timestep_embed: [B, embed_dim], global_cond: [B, global_cond_dim]
        time_emb = self.diffusion_step_encoder(timestep_embed)
        cond = torch.cat([time_emb, global_cond], dim=-1)

        h = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            h.append(x)
            x = downsample(x)

        for res_block in self.mid_modules:
            x = res_block(x, cond)

        for upsample, res1, res2 in self.up_modules:
            x = upsample(x)
            skip = h.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)

        return self.final_conv(x)


# ==============================================================================
# 2. Graph-Unet Policy (Graph encoder + U-Net backbone)
# ==============================================================================

class GraphUnetPolicy(GraphDiTPolicy):
    """
    Same graph feature extraction as GraphDiT (nodes, edges, pooling to z),
    but replaces the Transformer backbone with Conditional U-Net 1D.
    """

    def __init__(self, cfg: GraphDiTPolicyCfg):
        super().__init__(cfg)

        del self.graph_dit_units
        del self.noise_head
        del self.pred_horizon_pos_embed

        z_dim = getattr(cfg, "z_dim", 64)
        # Smaller U-Net (128,256,512) to fit ~8GB GPU; use [256,512,1024] if you have more VRAM
        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=z_dim,
            diffusion_step_embed_dim=cfg.hidden_dim,
            down_dims=[128, 256, 512],
            kernel_size=5,
            n_groups=8,
        )

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
        if noisy_action is None or timesteps is None:
            raise ValueError("GraphUnetPolicy.forward requires noisy_action and timesteps")

        node_dim = 7

        if ee_node_history is not None and object_node_history is not None:
            B, H, _ = ee_node_history.shape
            ee_flat = ee_node_history.reshape(-1, node_dim)
            obj_flat = object_node_history.reshape(-1, node_dim)
            ee_embed = self.node_embedding(ee_flat).view(B, H, -1)
            obj_embed = self.node_embedding(obj_flat).view(B, H, -1)
            node_features = torch.stack([ee_embed, obj_embed], dim=2)
            node_features = node_features.transpose(1, 2)
        else:
            ee_node, object_node = self._extract_node_features(obs)
            ee_embed = self.node_embedding(ee_node).unsqueeze(1)
            obj_embed = self.node_embedding(object_node).unsqueeze(1)
            node_features = torch.stack([ee_embed, obj_embed], dim=1)

        z = self._pool_node_latent(node_features)

        x = noisy_action.transpose(1, 2)
        ts_embed = self._get_timestep_embed(timesteps)
        noise_pred = self.unet(x, ts_embed, global_cond=z)
        noise_pred = noise_pred.transpose(1, 2)

        if return_dict:
            return {"noise_pred": noise_pred}
        return noise_pred

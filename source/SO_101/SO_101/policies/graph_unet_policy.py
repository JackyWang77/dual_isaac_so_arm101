# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
"""
U-Net / Graph-U-Net Policies for Flow Matching action prediction.

Two variants:
- UnetPolicy: MLP node encoder → pool → z → Conditional U-Net 1D (no real graph attention)
- GraphUnetPolicy: Full graph encoder (GRU edge + Multihead Graph Attention + Edge Modulation)
                   → pool → z → Conditional U-Net 1D

Conv1D U-Net acts as a low-pass filter along the action horizon, reducing jitter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from SO_101.policies.graph_dit_policy import (
    AdaptiveLayerNorm,
    GraphAttentionWithEdgeBias,
    GraphDiTPolicy,
    GraphDiTPolicyCfg,
)


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

class UnetPolicy(GraphDiTPolicy):
    """
    MLP node encoder + Conditional U-Net 1D backbone (no real graph attention).

    Node path: Node History [B,H,7] → MLP_node → TemporalAgg → concat → MLP_z → z
    Action path: noisy_action + z + timestep → U-Net → velocity_pred

    NOTE: This class does NOT use GRU edge encoder, Multihead Graph Attention, or
    Edge-Conditioned Modulation. It simply embeds nodes with an MLP and pools to z.
    For the full graph version, use GraphUnetPolicy instead.
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
            raise ValueError("UnetPolicy.forward requires noisy_action and timesteps")

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

    # ============================================================
    # RL: extract z and features (U-Net has no layer-wise z, single z repeated)
    # ============================================================

    def extract_z(
        self,
        ee_node_history: torch.Tensor,
        object_node_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract graph latent z for RL (high-frequency).

        U-Net has no layer-wise structure; single z is expanded to [B, K, z_dim]
        for compatibility with RL interface.

        Args:
            ee_node_history: [B, H, 7] - EE node history
            object_node_history: [B, H, 7] - Object node history

        Returns:
            z_layers: [B, K, z_dim] - same z repeated K times
        """
        self.eval()
        with torch.no_grad():
            B, H, node_dim = ee_node_history.shape
            ee_flat = ee_node_history.reshape(B * H, node_dim)
            obj_flat = object_node_history.reshape(B * H, node_dim)
            ee_embed = self.node_embedding(ee_flat).view(B, H, -1)
            obj_embed = self.node_embedding(obj_flat).view(B, H, -1)
            node_features = torch.stack([ee_embed, obj_embed], dim=1)
            z = self._pool_node_latent(node_features)
            K = getattr(self.cfg, "num_layers", 6)
            z_layers = z.unsqueeze(1).expand(-1, K, -1).contiguous()
            return z_layers

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
        Extract features for RL (U-Net version).
        Uses graph encoder only; no transformer layers.
        """
        self.eval()
        with torch.no_grad():
            if ee_node_history is not None and object_node_history is not None:
                B, H, node_dim = ee_node_history.shape
                ee_flat = ee_node_history.reshape(B * H, node_dim)
                obj_flat = object_node_history.reshape(B * H, node_dim)
                ee_embed = self.node_embedding(ee_flat).view(B, H, -1)
                obj_embed = self.node_embedding(obj_flat).view(B, H, -1)
                node_features = torch.stack([ee_embed, obj_embed], dim=1)
            else:
                ee_node, object_node = self._extract_node_features(obs)
                ee_embed = self.node_embedding(ee_node)
                obj_embed = self.node_embedding(object_node)
                node_features = torch.stack([ee_embed, obj_embed], dim=1).unsqueeze(2)

            z = self._pool_node_latent(node_features)
            K = getattr(self.cfg, "num_layers", 6)
            z_layers = z.unsqueeze(1).expand(-1, K, -1).contiguous()

            if ee_node_history is not None:
                current_ee = ee_node_history[:, -1, :]
                current_obj = object_node_history[:, -1, :]
            else:
                current_ee, current_obj = self._extract_node_features(obs)
            edge_raw = self._compute_edge_features(current_ee, current_obj)
            edge_embed = self.edge_embedding(edge_raw)

            nf = node_features.mean(dim=2) if node_features.dim() == 4 else node_features
            return {
                "graph_embedding": z,
                "z_layers": z_layers,
                "z_final": z,
                "node_features": nf,
                "edge_features": edge_embed,
            }


# ==============================================================================
# 3. Graph-Unet Policy (Full Graph Attention + U-Net backbone)
# ==============================================================================


class GraphUnetPolicy(GraphDiTPolicy):
    """
    Full Graph encoder + Conditional U-Net 1D backbone for Flow Matching.

    Unlike UnetPolicy (MLP-only node encoder → z → U-Net), this class uses
    the complete graph attention pipeline from DiT to produce a richer z:

        Node History [B,H,7]
            → MLP_node
            → GraphAttention × N (GRU edge encoder + Edge-Conditioned Modulation)
            → TemporalAgg → MLP_z → z
            → U-Net(noisy_action, timestep, z) → velocity_pred

    Key components from DiT that are retained here:
    - GRU temporal edge encoder: captures edge dynamics across history
    - Edge-Conditioned Modulation: gate/scale/shift on Value (ECC-style)
    - Multihead Graph Attention with spatial + temporal bias
    - AdaptiveLayerNorm conditioned on timestep for diffusion awareness

    Key components from DiT that are replaced:
    - Action self-attention → handled by U-Net's ConditionalResidualBlock1D
    - Cross-attention (action × nodes) → replaced by z conditioning in U-Net
    - Noise head (linear) → replaced by U-Net decoder
    """

    def __init__(self, cfg: GraphDiTPolicyCfg):
        super().__init__(cfg)

        # ------------------------------------------------------------------
        # Remove DiT-specific action processing components.
        # Keep: node_embedding, edge_embedding, node_to_z,
        #       node_temporal_aggregator, _compute_edge_features,
        #       _pool_node_latent, timestep_embedding, subtask_encoder, etc.
        # ------------------------------------------------------------------
        del self.graph_dit_units       # replaced by graph_attention_layers below
        del self.noise_head            # replaced by U-Net
        del self.pred_horizon_pos_embed  # U-Net doesn't need action pos embed

        # ==================== Graph-only attention layers ====================
        # Same GRU + Edge-Conditioned Modulation + Multihead Graph Attention
        # as in GraphDiTUnit, but WITHOUT action self-attention and cross-attention.
        # Only processes node features + edge features → enriched node features.
        max_history = max(cfg.action_history_length + cfg.pred_horizon, 20)

        self.graph_attention_layers = nn.ModuleList([
            GraphAttentionWithEdgeBias(
                cfg.hidden_dim, cfg.num_heads, cfg.graph_edge_dim,
                max_history=max_history,
            )
            for _ in range(cfg.num_layers)
        ])
        # Pre-norm (before graph attention) and post-norm (before FFN)
        self.graph_pre_norms = nn.ModuleList([
            AdaptiveLayerNorm(cfg.hidden_dim, cfg.hidden_dim)
            for _ in range(cfg.num_layers)
        ])
        self.graph_post_norms = nn.ModuleList([
            AdaptiveLayerNorm(cfg.hidden_dim, cfg.hidden_dim)
            for _ in range(cfg.num_layers)
        ])
        self.graph_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim),
                nn.Dropout(0.1),
            )
            for _ in range(cfg.num_layers)
        ])

        # ==================== U-Net backbone ====================
        z_dim = getattr(cfg, "z_dim", 64)
        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=z_dim,
            diffusion_step_embed_dim=cfg.hidden_dim,
            down_dims=[128, 256, 512],
            kernel_size=5,
            n_groups=8,
        )

        # Re-initialize weights for newly created modules
        self._init_weights()

    # ------------------------------------------------------------------
    # Core: graph encoder (reused by forward, extract_z, extract_features)
    # ------------------------------------------------------------------

    def _encode_graph(
        self,
        node_features: torch.Tensor,
        edge_embed: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run N layers of graph attention to enrich node features, then pool to z.

        Args:
            node_features: [B, 2, H, D] - embedded node history
            edge_embed:    [B, H, edge_dim] - embedded edge history
            condition:     [B, hidden_dim] or None (timestep + subtask for AdaLN)
        Returns:
            node_features: [B, 2, H, D] - enriched node features
            z:             [B, z_dim]    - graph latent
        """
        B = node_features.shape[0]
        device = node_features.device
        dtype = node_features.dtype

        if condition is None:
            condition = torch.zeros(B, self.cfg.hidden_dim, device=device, dtype=dtype)

        for i in range(len(self.graph_attention_layers)):
            # Pre-norm with AdaLN (timestep-aware normalization)
            residual = node_features
            node_normed = self.graph_pre_norms[i](node_features, condition)

            # Graph attention: GRU edge encoder → Edge-Conditioned Modulation → MHA
            node_features = self.graph_attention_layers[i](node_normed, edge_embed)
            node_features = node_features + residual

            # Post-norm + FFN
            B_n, N, H_n, D = node_features.shape
            residual = node_features
            node_flat = node_features.view(B_n, N * H_n, D)
            node_flat = self.graph_post_norms[i](node_flat, condition)
            node_flat = node_flat.view(B_n * N * H_n, D)
            node_flat = self.graph_ffns[i](node_flat)
            node_features = node_flat.view(B_n, N, H_n, D) + residual

        # Pool: [B, 2, H, D] → TemporalAgg → [B, 2, D] → MLP → [B, z_dim]
        z = self._pool_node_latent(node_features)
        return node_features, z

    # ------------------------------------------------------------------
    # Helper: embed nodes + compute/embed edges
    # ------------------------------------------------------------------

    def _embed_nodes_and_edges(
        self,
        obs: torch.Tensor,
        ee_node_history: torch.Tensor | None,
        object_node_history: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Embed nodes and compute + embed edge features.

        Returns:
            node_features: [B, 2, H, D]
            edge_embed:    [B, H, edge_dim]
        """
        if ee_node_history is not None and object_node_history is not None:
            B, H, node_dim = ee_node_history.shape
            ee_flat = ee_node_history.reshape(-1, node_dim)
            obj_flat = object_node_history.reshape(-1, node_dim)
            ee_embed = self.node_embedding(ee_flat).view(B, H, -1)
            obj_embed = self.node_embedding(obj_flat).view(B, H, -1)
            # [B, H, 2, D] → [B, 2, H, D]
            node_features = torch.stack([ee_embed, obj_embed], dim=2).transpose(1, 2)

            # Temporal edge features: _compute_edge_features handles [B, H, 7]
            edge_raw = self._compute_edge_features(
                ee_node_history, object_node_history
            )  # [B, H, 2]
        else:
            ee_node, object_node = self._extract_node_features(obs)
            ee_embed = self.node_embedding(ee_node).unsqueeze(1)   # [B, 1, D]
            obj_embed = self.node_embedding(object_node).unsqueeze(1)  # [B, 1, D]
            node_features = torch.stack([ee_embed, obj_embed], dim=1)  # [B, 2, 1, D]
            edge_raw = self._compute_edge_features(
                ee_node, object_node
            ).unsqueeze(1)  # [B, 1, 2]

        B = node_features.shape[0]
        H = node_features.shape[2]

        # Embed raw edges [B, H, 2] → [B, H, graph_edge_dim]
        edge_embed = self.edge_embedding(
            edge_raw.reshape(-1, edge_raw.shape[-1])
        ).view(B, H, -1)

        return node_features, edge_embed

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

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
        Forward pass: graph encode → pool to z → U-Net denoise.

        Args:
            obs: [B, obs_dim]
            noisy_action: [B, pred_horizon, action_dim] (required)
            timesteps: [B] diffusion timesteps (required)
            ee_node_history: [B, H, 7] (optional, falls back to obs)
            object_node_history: [B, H, 7] (optional, falls back to obs)
            subtask_condition: [B, num_subtasks] (optional)
            (action_history, joint_states_history are accepted for API compat
             but not used – the U-Net handles action denoising directly)
        Returns:
            velocity_pred: [B, pred_horizon, action_dim]
        """
        if noisy_action is None or timesteps is None:
            raise ValueError("GraphUnetPolicy.forward requires noisy_action and timesteps")

        # 1. Embed nodes + edges
        node_features, edge_embed = self._embed_nodes_and_edges(
            obs, ee_node_history, object_node_history,
        )

        # 2. Build condition for AdaLN (timestep + optional subtask)
        ts_embed = self._get_timestep_embed(timesteps)  # [B, hidden_dim]
        condition = ts_embed

        if subtask_condition is not None and hasattr(self, "subtask_encoder"):
            subtask_embed = self.subtask_encoder(subtask_condition)
            condition = self.condition_proj(
                torch.cat([ts_embed, subtask_embed], dim=-1)
            )

        # 3. Graph attention layers (GRU edge → Edge-Conditioned Modulation → MHA)
        _, z = self._encode_graph(node_features, edge_embed, condition)

        # 4. U-Net: denoise action trajectory
        if noisy_action.dim() == 2:
            noisy_action = noisy_action.unsqueeze(1)  # [B, 1, A] backward compat

        x = noisy_action.transpose(1, 2)          # [B, action_dim, pred_horizon]
        noise_pred = self.unet(x, ts_embed, global_cond=z)
        noise_pred = noise_pred.transpose(1, 2)    # [B, pred_horizon, action_dim]

        if return_dict:
            return {"noise_pred": noise_pred}
        return noise_pred

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def extract_z(
        self,
        ee_node_history: torch.Tensor,
        object_node_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract graph latent z for RL (high-frequency).

        Runs full graph attention pipeline (GRU + Edge Modulation) to produce
        a richer z compared to UnetPolicy's MLP-only pooling.

        Args:
            ee_node_history: [B, H, 7]
            object_node_history: [B, H, 7]
        Returns:
            z_layers: [B, K, z_dim] – same z repeated K times for layer compat
        """
        self.eval()
        with torch.no_grad():
            B, H, node_dim = ee_node_history.shape
            ee_flat = ee_node_history.reshape(-1, node_dim)
            obj_flat = object_node_history.reshape(-1, node_dim)
            ee_embed = self.node_embedding(ee_flat).view(B, H, -1)
            obj_embed = self.node_embedding(obj_flat).view(B, H, -1)
            node_features = torch.stack([ee_embed, obj_embed], dim=2).transpose(1, 2)

            edge_raw = self._compute_edge_features(
                ee_node_history, object_node_history,
            )  # [B, H, 2]
            edge_embed = self.edge_embedding(
                edge_raw.reshape(-1, edge_raw.shape[-1])
            ).view(B, H, -1)

            # Graph encoder (no timestep condition for RL feature extraction)
            _, z = self._encode_graph(node_features, edge_embed, condition=None)

            K = getattr(self.cfg, "num_layers", 6)
            z_layers = z.unsqueeze(1).expand(-1, K, -1).contiguous()
            return z_layers

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
        Extract features for RL (full graph attention version).

        Returns a dict compatible with GraphUnetResidualRLPolicy.
        """
        self.eval()
        with torch.no_grad():
            node_features, edge_embed = self._embed_nodes_and_edges(
                obs, ee_node_history, object_node_history,
            )

            # Graph encoder (no timestep condition for RL)
            node_features_out, z = self._encode_graph(
                node_features, edge_embed, condition=None,
            )

            K = getattr(self.cfg, "num_layers", 6)
            z_layers = z.unsqueeze(1).expand(-1, K, -1).contiguous()

            # Current edge features (for RL residual policy)
            if ee_node_history is not None:
                current_ee = ee_node_history[:, -1, :]
                current_obj = object_node_history[:, -1, :]
            else:
                current_ee, current_obj = self._extract_node_features(obs)
            edge_raw = self._compute_edge_features(current_ee, current_obj)
            edge_feat = self.edge_embedding(edge_raw)

            nf = (
                node_features_out.mean(dim=2)
                if node_features_out.dim() == 4
                else node_features_out
            )
            return {
                "graph_embedding": z,
                "z_layers": z_layers,
                "z_final": z,
                "node_features": nf,
                "edge_features": edge_feat,
            }

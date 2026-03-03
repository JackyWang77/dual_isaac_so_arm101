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
    DisentangledGraphAttention,
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

    Node path: Node History [B,N,H,7] → MLP_node + type_embed → TemporalAgg → pool → MLP_z → z
    Action path: noisy_action + z + timestep → U-Net → velocity_pred

    Supports dynamic N-node graphs via ``node_histories`` kwarg or legacy
    2-node API via ``ee_node_history`` + ``object_node_history``.
    """

    def __init__(self, cfg: GraphDiTPolicyCfg):
        super().__init__(cfg)

        del self.graph_dit_units
        del self.noise_head
        del self.pred_horizon_pos_embed

        self._use_joint_film = getattr(cfg, "use_joint_film", False)
        z_dim = getattr(cfg, "z_dim", 64)
        joint_z_dim = 0
        if self._use_joint_film:
            jd = getattr(cfg, "joint_dim", None) or 6
            hist_len = cfg.action_history_length
            self.joint_encoder = nn.Sequential(
                nn.Linear(jd * hist_len, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, z_dim),
            )
            joint_z_dim = z_dim

        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=z_dim + joint_z_dim,
            diffusion_step_embed_dim=cfg.hidden_dim,
            down_dims=[128, 256, 512],
            kernel_size=5,
            n_groups=8,
        )

        # Option 1: subtask concat into z before U-Net
        num_subtasks = getattr(cfg, "num_subtasks", 0)
        if num_subtasks > 0 and hasattr(self, "subtask_encoder"):
            self.subtask_to_z = nn.Linear(cfg.hidden_dim // 4, z_dim)

    # ------------------------------------------------------------------
    # Shared helper: build node_histories [B, N, H, 7] from any input API
    # ------------------------------------------------------------------

    def _build_node_histories(
        self,
        obs: torch.Tensor,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (node_histories [B, N, H, 7], node_types [N]).

        Accepts either the new unified tensor or the legacy separate tensors.
        Falls back to extracting from obs if nothing is provided.
        """
        num_node_types = getattr(self.cfg, "num_node_types", 2)

        if node_histories is not None:
            # New API: already [B, N, H, 7]
            N = node_histories.shape[1]
            if node_types is None:
                node_types = torch.zeros(N, dtype=torch.long, device=node_histories.device)
            return node_histories, node_types

        if ee_node_history is not None and object_node_history is not None:
            # Legacy 2-node API → stack into [B, N=2, H, 7]
            nh = torch.stack([ee_node_history, object_node_history], dim=1)  # [B, 2, H, 7]
            nt = torch.tensor([0, 1], dtype=torch.long, device=nh.device)
            return nh, nt

        # Fallback: extract from obs (single timestep, H=1)
        ee_node, object_node = self._extract_node_features(obs)
        nh = torch.stack([ee_node.unsqueeze(1), object_node.unsqueeze(1)], dim=1)
        nt = torch.tensor([0, 1], dtype=torch.long, device=nh.device)
        return nh, nt

    def _embed_node_histories(
        self,
        node_histories: torch.Tensor,
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        """Embed raw node histories into feature space with type embeddings.

        Args:
            node_histories: [B, N, H, 7]
            node_types: [N]
        Returns:
            [B, N, H, D]
        """
        B, N, H, node_dim = node_histories.shape
        flat = node_histories.reshape(B * N * H, node_dim)
        embedded = self.node_embedding(flat).view(B, N, H, -1)  # [B, N, H, D]
        type_emb = self.node_type_embedding(node_types)  # [N, D]
        embedded = embedded + type_emb.view(1, N, 1, -1)
        return embedded

    # ------------------------------------------------------------------

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
        if noisy_action is None or timesteps is None:
            raise ValueError("UnetPolicy.forward requires noisy_action and timesteps")

        nh, nt = self._build_node_histories(
            obs, ee_node_history, object_node_history, node_histories, node_types,
        )
        node_features = self._embed_node_histories(nh, nt)  # [B, N, H, D]

        z = self._pool_node_latent(node_features)

        # Option 1: inject subtask into z before U-Net
        if subtask_condition is not None and hasattr(self, "subtask_to_z"):
            subtask_embed = self.subtask_encoder(subtask_condition)
            z = z + self.subtask_to_z(subtask_embed)

        if self._use_joint_film and joint_states_history is not None:
            B_j = joint_states_history.shape[0]
            joint_flat = joint_states_history.reshape(B_j, -1)
            joint_enc = self.joint_encoder(joint_flat)
            z = torch.cat([z, joint_enc], dim=-1)

        x = noisy_action.transpose(1, 2)
        ts_embed = self._get_timestep_embed(timesteps)
        noise_pred = self.unet(x, ts_embed, global_cond=z)
        noise_pred = noise_pred.transpose(1, 2)

        if return_dict:
            return {"noise_pred": noise_pred}
        return noise_pred

    # ============================================================
    # RL: extract z and features
    # ============================================================

    def extract_z(
        self,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        joint_states_history: torch.Tensor | None = None,
        *,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract graph latent z for RL. Accepts old or new node API."""
        self.eval()
        with torch.no_grad():
            dummy_obs = torch.zeros(1, 1, device=(
                ee_node_history.device if ee_node_history is not None
                else node_histories.device
            ))
            nh, nt = self._build_node_histories(
                dummy_obs, ee_node_history, object_node_history, node_histories, node_types,
            )
            node_features = self._embed_node_histories(nh, nt)
            z = self._pool_node_latent(node_features)

            if self._use_joint_film and joint_states_history is not None:
                B = z.shape[0]
                joint_flat = joint_states_history.reshape(B, -1)
                joint_enc = self.joint_encoder(joint_flat)
                z = torch.cat([z, joint_enc], dim=-1)

            K = getattr(self.cfg, "num_layers", 6)
            return z.unsqueeze(1).expand(-1, K, -1).contiguous()

    def extract_features(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        joint_states_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
        *,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract features for RL (U-Net version)."""
        self.eval()
        with torch.no_grad():
            nh, nt = self._build_node_histories(
                obs, ee_node_history, object_node_history, node_histories, node_types,
            )
            node_features = self._embed_node_histories(nh, nt)
            z = self._pool_node_latent(node_features)

            if self._use_joint_film and joint_states_history is not None:
                B_j = joint_states_history.shape[0]
                joint_flat = joint_states_history.reshape(B_j, -1)
                joint_enc = self.joint_encoder(joint_flat)
                z = torch.cat([z, joint_enc], dim=-1)

            K = getattr(self.cfg, "num_layers", 6)
            z_layers = z.unsqueeze(1).expand(-1, K, -1).contiguous()

            # Edge features for RL residual policy (last timestep)
            current_nodes = nh[:, :, -1, :]  # [B, N, 7]
            if current_nodes.shape[1] == 2:
                edge_raw = self._compute_edge_features_legacy(
                    current_nodes[:, 0], current_nodes[:, 1],
                )
            else:
                edge_raw = self._compute_edge_features_dynamic(
                    current_nodes.unsqueeze(2),  # [B, N, 1, 7]
                )
                edge_raw = edge_raw[:, :, :, 0, :]  # [B, N, N, edge_dim]
                # Flatten upper triangle for embedding
                N = edge_raw.shape[1]
                idx = torch.triu_indices(N, N, offset=1)
                edge_raw = edge_raw[:, idx[0], idx[1]]  # [B, num_edges, edge_dim]
                edge_raw = edge_raw.mean(dim=1)  # [B, edge_dim]
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

    Supports dynamic N-node graphs via ``node_histories`` kwarg or legacy
    2-node API via ``ee_node_history`` + ``object_node_history``.

    Pipeline:
        Node History [B,N,H,7] → MLP_node + type_embed
            → GraphAttention × L (GRU edge + Edge-Conditioned Modulation)
            → TemporalAgg → pool → MLP_z → z
            → U-Net(noisy_action, timestep, z) → velocity_pred
    """

    def __init__(self, cfg: GraphDiTPolicyCfg):
        super().__init__(cfg)

        del self.graph_dit_units
        del self.noise_head
        del self.pred_horizon_pos_embed

        max_history = max(cfg.action_history_length + cfg.pred_horizon, 20)

        self.graph_attention_layers = nn.ModuleList([
            GraphAttentionWithEdgeBias(
                cfg.hidden_dim, cfg.num_heads, cfg.graph_edge_dim,
                max_history=max_history,
            )
            for _ in range(cfg.num_layers)
        ])
        self.graph_pre_norms = nn.ModuleList([
            nn.LayerNorm(cfg.hidden_dim)
            for _ in range(cfg.num_layers)
        ])
        self.graph_post_norms = nn.ModuleList([
            nn.LayerNorm(cfg.hidden_dim)
            for _ in range(cfg.num_layers)
        ])
        ffn_expand = 2
        self.graph_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * ffn_expand),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.hidden_dim * ffn_expand, cfg.hidden_dim),
                nn.Dropout(0.1),
            )
            for _ in range(cfg.num_layers)
        ])

        self._use_joint_film = getattr(cfg, "use_joint_film", False)
        z_dim = getattr(cfg, "z_dim", 64)
        joint_z_dim = 0
        if self._use_joint_film:
            jd = getattr(cfg, "joint_dim", None) or 6
            hist_len = cfg.action_history_length
            self.joint_encoder = nn.Sequential(
                nn.Linear(jd * hist_len, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, z_dim),
            )
            joint_z_dim = z_dim

        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=z_dim + joint_z_dim,
            diffusion_step_embed_dim=cfg.hidden_dim,
            down_dims=[128, 256, 512],
            kernel_size=5,
            n_groups=8,
        )

        # Option 1: subtask concat into z before U-Net (subtask_encoder from parent)
        num_subtasks = getattr(cfg, "num_subtasks", 0)
        if num_subtasks > 0 and hasattr(self, "subtask_encoder"):
            self.subtask_to_z = nn.Linear(cfg.hidden_dim // 4, z_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # Shared helpers (inherited from UnetPolicy concept, but re-defined
    # because GraphUnetPolicy inherits from GraphDiTPolicy, not UnetPolicy)
    # ------------------------------------------------------------------

    def _build_node_histories(
        self,
        obs: torch.Tensor,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (node_histories [B, N, H, 7], node_types [N])."""
        if node_histories is not None:
            N = node_histories.shape[1]
            if node_types is None:
                node_types = torch.zeros(N, dtype=torch.long, device=node_histories.device)
            return node_histories, node_types

        if ee_node_history is not None and object_node_history is not None:
            nh = torch.stack([ee_node_history, object_node_history], dim=1)
            nt = torch.tensor([0, 1], dtype=torch.long, device=nh.device)
            return nh, nt

        ee_node, object_node = self._extract_node_features(obs)
        nh = torch.stack([ee_node.unsqueeze(1), object_node.unsqueeze(1)], dim=1)
        nt = torch.tensor([0, 1], dtype=torch.long, device=nh.device)
        return nh, nt

    def _embed_node_histories(
        self,
        node_histories: torch.Tensor,
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        """[B, N, H, 7] → [B, N, H, D] with type embeddings."""
        B, N, H, node_dim = node_histories.shape
        flat = node_histories.reshape(B * N * H, node_dim)
        embedded = self.node_embedding(flat).view(B, N, H, -1)
        type_emb = self.node_type_embedding(node_types)
        embedded = embedded + type_emb.view(1, N, 1, -1)
        return embedded

    def _compute_and_embed_edges(
        self,
        node_histories: torch.Tensor,
    ) -> torch.Tensor:
        """Compute raw edges and embed them.

        Args:
            node_histories: [B, N, H, 7]
        Returns:
            For N==2: [B, H, graph_edge_dim] (legacy format for attention layer)
            For N>2:  [B, N, N, H, graph_edge_dim]
        """
        B, N, H, _ = node_histories.shape

        if N == 2:
            # Legacy path: single edge [B, H, edge_dim]
            edge_raw = self._compute_edge_features_legacy(
                node_histories[:, 0], node_histories[:, 1],
            )  # [B, H, 2]
            edge_embed = self.edge_embedding(
                edge_raw.reshape(-1, edge_raw.shape[-1])
            ).view(B, H, -1)
            return edge_embed
        else:
            # Dynamic: [B, N, N, H, edge_dim]
            edge_raw = self._compute_edge_features_dynamic(node_histories)
            edge_dim = edge_raw.shape[-1]
            edge_embed = self.edge_embedding(
                edge_raw.reshape(-1, edge_dim)
            ).view(B, N, N, H, -1)
            return edge_embed

    # ------------------------------------------------------------------
    # Core graph encoder
    # ------------------------------------------------------------------

    def _encode_graph(
        self,
        node_features: torch.Tensor,
        edge_embed: torch.Tensor,
        condition: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run L layers of graph attention, then pool to z.

        Args:
            node_features: [B, N, H, D]
            edge_embed:    [B, H, edge_dim] (N==2) or [B, N, N, H, edge_dim] (N>2)
            condition:     unused, kept for API compatibility
            return_attention: If True, return attn_weights from last layer
        Returns:
            (enriched node_features, z) or (node_features, z, attn_weights) when return_attention
        """
        attn_weights = None
        for i in range(len(self.graph_attention_layers)):
            residual = node_features
            node_normed = self.graph_pre_norms[i](node_features)
            out = self.graph_attention_layers[i](
                node_normed, edge_embed,
                return_attention=(return_attention and i == len(self.graph_attention_layers) - 1),
            )
            if isinstance(out, tuple):
                node_features, attn_weights = out
            else:
                node_features = out
            node_features = node_features + residual

            B_n, N, H_n, D = node_features.shape
            residual = node_features
            node_flat = node_features.view(B_n * N * H_n, D)
            node_flat = self.graph_post_norms[i](node_flat)
            node_flat = self.graph_ffns[i](node_flat)
            node_features = node_flat.view(B_n, N, H_n, D) + residual

        z = self._pool_node_latent(node_features)
        if return_attention and attn_weights is not None:
            return node_features, z, attn_weights
        return node_features, z

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        return_attention: bool = False,
    ) -> torch.Tensor | dict:
        if noisy_action is None or timesteps is None:
            raise ValueError("GraphUnetPolicy.forward requires noisy_action and timesteps")

        nh, nt = self._build_node_histories(
            obs, ee_node_history, object_node_history, node_histories, node_types,
        )
        node_features = self._embed_node_histories(nh, nt)
        edge_embed = self._compute_and_embed_edges(nh)

        ts_embed = self._get_timestep_embed(timesteps)
        encode_out = self._encode_graph(
            node_features, edge_embed, condition=None,
            return_attention=return_attention,
        )
        if isinstance(encode_out, tuple) and len(encode_out) == 3:
            _, z, attn_weights = encode_out
        else:
            _, z = encode_out
            attn_weights = None

        # Option 1: inject subtask into z before U-Net (subtask 直接影响 action 生成)
        if subtask_condition is not None and hasattr(self, "subtask_to_z"):
            subtask_embed = self.subtask_encoder(subtask_condition)
            z = z + self.subtask_to_z(subtask_embed)

        if self._use_joint_film and joint_states_history is not None:
            B_j = joint_states_history.shape[0]
            joint_flat = joint_states_history.reshape(B_j, -1)
            joint_enc = self.joint_encoder(joint_flat)
            z = torch.cat([z, joint_enc], dim=-1)

        if noisy_action.dim() == 2:
            noisy_action = noisy_action.unsqueeze(1)

        x = noisy_action.transpose(1, 2)
        noise_pred = self.unet(x, ts_embed, global_cond=z)
        noise_pred = noise_pred.transpose(1, 2)

        if return_dict:
            out = {"noise_pred": noise_pred}
            if return_attention and attn_weights is not None:
                out["attention"] = attn_weights
            return out
        return noise_pred

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def extract_z(
        self,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        joint_states_history: torch.Tensor | None = None,
        *,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract graph latent z for RL. Accepts old or new node API."""
        self.eval()
        with torch.no_grad():
            dummy_obs = torch.zeros(1, 1, device=(
                ee_node_history.device if ee_node_history is not None
                else node_histories.device
            ))
            nh, nt = self._build_node_histories(
                dummy_obs, ee_node_history, object_node_history, node_histories, node_types,
            )
            node_features = self._embed_node_histories(nh, nt)
            edge_embed = self._compute_and_embed_edges(nh)
            _, z = self._encode_graph(node_features, edge_embed, condition=None)

            if self._use_joint_film and joint_states_history is not None:
                B = z.shape[0]
                joint_flat = joint_states_history.reshape(B, -1)
                joint_enc = self.joint_encoder(joint_flat)
                z = torch.cat([z, joint_enc], dim=-1)

            K = getattr(self.cfg, "num_layers", 6)
            return z.unsqueeze(1).expand(-1, K, -1).contiguous()

    def extract_features(
        self,
        obs: torch.Tensor,
        action_history: torch.Tensor | None = None,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        joint_states_history: torch.Tensor | None = None,
        subtask_condition: torch.Tensor | None = None,
        *,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract features for RL (full graph attention version)."""
        self.eval()
        with torch.no_grad():
            nh, nt = self._build_node_histories(
                obs, ee_node_history, object_node_history, node_histories, node_types,
            )
            node_features = self._embed_node_histories(nh, nt)
            edge_embed = self._compute_and_embed_edges(nh)
            node_features_out, z = self._encode_graph(
                node_features, edge_embed, condition=None,
            )

            if self._use_joint_film and joint_states_history is not None:
                B_j = joint_states_history.shape[0]
                joint_flat = joint_states_history.reshape(B_j, -1)
                joint_enc = self.joint_encoder(joint_flat)
                z = torch.cat([z, joint_enc], dim=-1)

            K = getattr(self.cfg, "num_layers", 6)
            z_layers = z.unsqueeze(1).expand(-1, K, -1).contiguous()

            # Edge features for RL (summarized)
            current_nodes = nh[:, :, -1, :]  # [B, N, 7]
            if current_nodes.shape[1] == 2:
                edge_raw = self._compute_edge_features_legacy(
                    current_nodes[:, 0], current_nodes[:, 1],
                )
            else:
                edge_raw_full = self._compute_edge_features_dynamic(
                    current_nodes.unsqueeze(2),
                )  # [B, N, N, 1, edge_dim]
                edge_raw_full = edge_raw_full[:, :, :, 0, :]  # [B, N, N, edge_dim]
                N = edge_raw_full.shape[1]
                idx = torch.triu_indices(N, N, offset=1)
                edge_raw = edge_raw_full[:, idx[0], idx[1]].mean(dim=1)
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


# ==============================================================================
# 4. Disentangled Graph-Unet Policy
#    pos/rot 分离 attention + raw residual + v1 edge + no GRU/modulation
# ==============================================================================


class DisentangledGraphUnetPolicy(GraphDiTPolicy):
    """Disentangled Graph encoder + U-Net backbone.

    Key differences from GraphUnetPolicy:
    1. pos(3) and rot(4) use separate attention heads — no cross-type attention
    2. Edge features: distance + v1 quaternion similarity (no v2 axis alignment)
    3. Edge enters as attention bias only — no gate/scale/shift on V
    4. No GRU temporal edge encoding — simple MLP embed
    5. Raw residual: z = graph_z + raw_proj(latest_frame)
    """

    def __init__(self, cfg: GraphDiTPolicyCfg):
        super().__init__(cfg)

        # Remove unused parent modules
        del self.graph_dit_units
        del self.noise_head
        del self.pred_horizon_pos_embed

        max_history = max(cfg.action_history_length + cfg.pred_horizon, 20)
        z_dim = getattr(cfg, "z_dim", 64)

        # Disentangled graph attention (1 layer is usually enough)
        self.disentangled_attn_layers = nn.ModuleList([
            DisentangledGraphAttention(
                cfg.hidden_dim, cfg.num_heads, cfg.graph_edge_dim,
                max_history=max_history,
            )
            for _ in range(cfg.num_layers)
        ])
        self.graph_pre_norms = nn.ModuleList([
            nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.num_layers)
        ])
        self.graph_post_norms = nn.ModuleList([
            nn.LayerNorm(cfg.hidden_dim) for _ in range(cfg.num_layers)
        ])
        self.graph_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
                nn.Dropout(0.1),
            )
            for _ in range(cfg.num_layers)
        ])

        # Separate edge embeddings for distance and similarity
        self.dist_edge_embedding = nn.Sequential(
            nn.Linear(1, cfg.graph_edge_dim),
            nn.LayerNorm(cfg.graph_edge_dim),
            nn.GELU(),
        )
        self.sim_edge_embedding = nn.Sequential(
            nn.Linear(1, cfg.graph_edge_dim),
            nn.LayerNorm(cfg.graph_edge_dim),
            nn.GELU(),
        )

        # Raw projection: latest frame → compact state snapshot
        num_nodes = len(cfg.node_configs) if getattr(cfg, "node_configs", None) else getattr(cfg, "num_nodes", 2)
        raw_node_dim = num_nodes * cfg.node_dim  # N * 7
        raw_proj_dim = z_dim // 2  # half of z_dim for raw, keeps total compact
        self.raw_proj = nn.Linear(raw_node_dim, raw_proj_dim)

        # Joint FiLM (optional)
        self._use_joint_film = getattr(cfg, "use_joint_film", False)
        joint_z_dim = 0
        if self._use_joint_film:
            jd = getattr(cfg, "joint_dim", None) or 6
            hist_len = cfg.action_history_length
            self.joint_encoder = nn.Sequential(
                nn.Linear(jd * hist_len, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, z_dim),
            )
            joint_z_dim = z_dim

        # U-Net backbone: cond = cat([graph_z, raw_proj]) + optional joint
        self.unet = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            global_cond_dim=z_dim + raw_proj_dim + joint_z_dim,
            diffusion_step_embed_dim=cfg.hidden_dim,
            down_dims=[128, 256, 512],
            kernel_size=5,
            n_groups=8,
        )

        # Subtask (optional)
        num_subtasks = getattr(cfg, "num_subtasks", 0)
        if num_subtasks > 0 and hasattr(self, "subtask_encoder"):
            self.subtask_to_z = nn.Linear(cfg.hidden_dim // 4, z_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # Edge computation: v1 only (distance + quaternion similarity)
    # ------------------------------------------------------------------

    def _compute_disentangled_edges(
        self, node_histories: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and embed distance and similarity edges separately.

        Args:
            node_histories: [B, N, H, 7]
        Returns:
            dist_embed: [B, N, N, H, edge_dim] or [B, H, edge_dim]
            sim_embed:  same shape
        """
        B, N, H, _ = node_histories.shape

        if N == 2:
            # Legacy 2-node path
            dist_raw, sim_raw = self._compute_v1_edge_pair(
                node_histories[:, 0], node_histories[:, 1],
            )  # each [B, H, 1]
            dist_embed = self.dist_edge_embedding(
                dist_raw.reshape(-1, 1)
            ).view(B, H, -1)
            sim_embed = self.sim_edge_embedding(
                sim_raw.reshape(-1, 1)
            ).view(B, H, -1)
        else:
            dist_embed, sim_embed = self._compute_v1_edges_dynamic(node_histories)

        return dist_embed, sim_embed

    def _compute_v1_edge_pair(
        self, node_a: torch.Tensor, node_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute v1 edge features for a pair of nodes.

        Args:
            node_a, node_b: [B, H, 7] or [L, 7]
        Returns:
            distance: [B, H, 1] or [L, 1]
            similarity: same shape
        """
        has_time = node_a.dim() == 3
        if has_time:
            B, H, _ = node_a.shape
            node_a = node_a.reshape(B * H, -1)
            node_b = node_b.reshape(B * H, -1)

        pos_a, quat_a = node_a[:, :3], F.normalize(node_a[:, 3:7], p=2, dim=-1, eps=1e-6)
        pos_b, quat_b = node_b[:, :3], F.normalize(node_b[:, 3:7], p=2, dim=-1, eps=1e-6)

        distance = torch.norm(pos_a - pos_b, dim=-1, keepdim=True)
        similarity = torch.sum(quat_a * quat_b, dim=-1, keepdim=True).abs()

        if has_time:
            distance = distance.view(B, H, 1)
            similarity = similarity.view(B, H, 1)
        return distance, similarity

    def _compute_v1_edges_dynamic(
        self, node_histories: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-pairs v1 edge computation → separate embeddings.

        Args:
            node_histories: [B, N, H, 7]
        Returns:
            dist_embed: [B, N, N, H, edge_dim]
            sim_embed:  [B, N, N, H, edge_dim]
        """
        B, N, H, _ = node_histories.shape
        flat = node_histories.reshape(B * H, N, -1)
        L = B * H
        edge_dim = self.dist_edge_embedding[0].out_features

        dist_adj = flat.new_zeros(L, N, N, 1)
        sim_adj = flat.new_zeros(L, N, N, 1)

        for i in range(N):
            for j in range(i + 1, N):
                d, s = self._compute_v1_edge_pair(flat[:, i], flat[:, j])
                dist_adj[:, i, j] = d
                dist_adj[:, j, i] = d
                sim_adj[:, i, j] = s
                sim_adj[:, j, i] = s

        # Embed: [L*N*N, 1] → [L*N*N, edge_dim]
        dist_embed = self.dist_edge_embedding(
            dist_adj.reshape(-1, 1)
        ).view(L, N, N, edge_dim)
        sim_embed = self.sim_edge_embedding(
            sim_adj.reshape(-1, 1)
        ).view(L, N, N, edge_dim)

        # Reshape: [B, H, N, N, edge_dim] → [B, N, N, H, edge_dim]
        dist_embed = dist_embed.view(B, H, N, N, edge_dim).permute(0, 2, 3, 1, 4)
        sim_embed = sim_embed.view(B, H, N, N, edge_dim).permute(0, 2, 3, 1, 4)
        return dist_embed, sim_embed

    # ------------------------------------------------------------------
    # Encode graph
    # ------------------------------------------------------------------

    def _encode_disentangled_graph(
        self,
        node_histories: torch.Tensor,
        dist_embed: torch.Tensor,
        sim_embed: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run disentangled graph attention layers, then pool to z.

        The DisentangledGraphAttention takes raw node_histories directly
        (it has its own pos/rot embedding), so we pass node_histories not
        the embedded features.

        Returns:
            (node_features, z) or (node_features, z, attn_weights)
        """
        B, N, H, _ = node_histories.shape
        # Initial embedding to hidden_dim (needed for residual connections)
        flat = node_histories.reshape(B * N * H, -1)
        node_features = self.node_embedding(flat).view(B, N, H, -1)
        type_emb = self.node_type_embedding(
            torch.zeros(N, dtype=torch.long, device=node_histories.device)
        )
        node_features = node_features + type_emb.view(1, N, 1, -1)

        attn_weights = None
        for i in range(len(self.disentangled_attn_layers)):
            residual = node_features
            # Pre-norm
            B_n, N_n, H_n, D = node_features.shape
            node_normed = self.graph_pre_norms[i](
                node_features.view(B_n * N_n * H_n, D)
            ).view(B_n, N_n, H_n, D)

            # Disentangled attention (uses raw node_histories for pos/rot split,
            # but adds result as residual to the hidden-dim features)
            out = self.disentangled_attn_layers[i](
                node_histories, dist_embed, sim_embed,
                return_attention=(return_attention and i == len(self.disentangled_attn_layers) - 1),
            )
            if isinstance(out, tuple):
                attn_out, attn_weights = out
            else:
                attn_out = out

            node_features = attn_out + residual

            # FFN
            residual = node_features
            B_n, N_n, H_n, D = node_features.shape
            nf_flat = self.graph_post_norms[i](
                node_features.view(B_n * N_n * H_n, D)
            )
            nf_flat = self.graph_ffns[i](nf_flat)
            node_features = nf_flat.view(B_n, N_n, H_n, D) + residual

        z = self._pool_node_latent(node_features)

        if return_attention and attn_weights is not None:
            return node_features, z, attn_weights
        return node_features, z

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        return_attention: bool = False,
    ) -> torch.Tensor | dict:
        if noisy_action is None or timesteps is None:
            raise ValueError("DisentangledGraphUnetPolicy.forward requires noisy_action and timesteps")

        B = obs.shape[0]

        # Build node histories [B, N, H, 7]
        nh, nt = self._build_node_histories(
            obs, ee_node_history, object_node_history, node_histories, node_types,
        )

        # Compute disentangled edges (v1: distance + similarity, separate embeddings)
        dist_embed, sim_embed = self._compute_disentangled_edges(nh)

        # Disentangled graph encode → z
        encode_out = self._encode_disentangled_graph(
            nh, dist_embed, sim_embed,
            return_attention=return_attention,
        )
        if isinstance(encode_out, tuple) and len(encode_out) == 3:
            _, graph_z, attn_weights = encode_out
        else:
            _, graph_z = encode_out
            attn_weights = None

        # Raw projection: latest frame → current state snapshot
        raw_latest = nh[:, :, -1, :].reshape(B, -1)  # [B, N*7]
        raw_feat = self.raw_proj(raw_latest)           # [B, raw_proj_dim]

        # Concat: graph_z (trend) + raw_feat (current state)
        z = torch.cat([graph_z, raw_feat], dim=-1)     # [B, z_dim + raw_proj_dim]

        # Subtask conditioning
        if subtask_condition is not None and hasattr(self, "subtask_to_z"):
            subtask_embed = self.subtask_encoder(subtask_condition)
            z = z + F.pad(self.subtask_to_z(subtask_embed), (0, raw_feat.shape[-1]))

        # Joint FiLM (optional)
        if self._use_joint_film and joint_states_history is not None:
            B_j = joint_states_history.shape[0]
            joint_flat = joint_states_history.reshape(B_j, -1)
            joint_enc = self.joint_encoder(joint_flat)
            z = torch.cat([z, joint_enc], dim=-1)

        # U-Net
        ts_embed = self._get_timestep_embed(timesteps)
        if noisy_action.dim() == 2:
            noisy_action = noisy_action.unsqueeze(1)
        x = noisy_action.transpose(1, 2)
        noise_pred = self.unet(x, ts_embed, global_cond=z)
        noise_pred = noise_pred.transpose(1, 2)

        if return_dict:
            out = {"noise_pred": noise_pred}
            if return_attention and attn_weights is not None:
                out["attention"] = attn_weights
            return out
        return noise_pred

    # ------------------------------------------------------------------
    # _build_node_histories (reuse from parent / UnetPolicy)
    # ------------------------------------------------------------------

    def _build_node_histories(
        self,
        obs: torch.Tensor,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if node_histories is not None:
            N = node_histories.shape[1]
            if node_types is None:
                node_types = torch.zeros(N, dtype=torch.long, device=node_histories.device)
            return node_histories, node_types

        if ee_node_history is not None and object_node_history is not None:
            nh = torch.stack([ee_node_history, object_node_history], dim=1)
            nt = torch.tensor([0, 1], dtype=torch.long, device=nh.device)
            return nh, nt

        ee_node, object_node = self._extract_node_features(obs)
        nh = torch.stack([ee_node.unsqueeze(1), object_node.unsqueeze(1)], dim=1)
        nt = torch.tensor([0, 1], dtype=torch.long, device=nh.device)
        return nh, nt

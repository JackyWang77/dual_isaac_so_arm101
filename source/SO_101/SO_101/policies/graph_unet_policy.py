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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run L layers of graph attention, then pool to z.

        Args:
            node_features: [B, N, H, D]
            edge_embed:    [B, H, edge_dim] (N==2) or [B, N, N, H, edge_dim] (N>2)
            condition:     unused, kept for API compatibility
        Returns:
            (enriched node_features, z)
        """
        for i in range(len(self.graph_attention_layers)):
            residual = node_features
            node_normed = self.graph_pre_norms[i](node_features)
            node_features = self.graph_attention_layers[i](node_normed, edge_embed)
            node_features = node_features + residual

            B_n, N, H_n, D = node_features.shape
            residual = node_features
            node_flat = node_features.view(B_n * N * H_n, D)
            node_flat = self.graph_post_norms[i](node_flat)
            node_flat = self.graph_ffns[i](node_flat)
            node_features = node_flat.view(B_n, N, H_n, D) + residual

        z = self._pool_node_latent(node_features)
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
    ) -> torch.Tensor | dict:
        if noisy_action is None or timesteps is None:
            raise ValueError("GraphUnetPolicy.forward requires noisy_action and timesteps")

        nh, nt = self._build_node_histories(
            obs, ee_node_history, object_node_history, node_histories, node_types,
        )
        node_features = self._embed_node_histories(nh, nt)
        edge_embed = self._compute_and_embed_edges(nh)

        ts_embed = self._get_timestep_embed(timesteps)
        _, z = self._encode_graph(node_features, edge_embed, condition=None)

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
            return {"noise_pred": noise_pred}
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

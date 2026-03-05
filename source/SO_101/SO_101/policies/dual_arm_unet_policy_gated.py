# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
"""
DualArmDisentangledPolicyGated: 门控融合版 Disentangled，不 concat，用可学习 gate 注入 graph。
z_base = raw_feat + gate * graph_z_comp，gate = sigmoid(graph_gate_logit)，初始 -1.7 -> ~0.15 试更快收敛。
loss() 返回 metrics/graph_gate_weight 供 TensorBoard 监控。
"""

import torch

from SO_101.policies.graph_dit_policy import GraphDiTPolicyCfg
from SO_101.policies.dual_arm_unet_policy import DualArmDisentangledPolicy


class DualArmDisentangledPolicyGated(DualArmDisentangledPolicy):
    """Dual-arm disentangled graph + dual U-Net, with learnable gated fusion.

    Replaces raw/graph concat with: z_base = raw_feat + gate * graph_z_comp,
    gate = sigmoid(graph_gate_logit). Initial logit -1.7 (gate ~0.15) for faster convergence.
    Exposes metrics/graph_gate_weight in loss() for TensorBoard.
    """

    def __init__(self, cfg: GraphDiTPolicyCfg):
        super().__init__(cfg)

        z_dim = getattr(cfg, "z_dim", 64)
        raw_node_dim = self.raw_proj.in_features
        graph_z_in = self.graph_z_proj.in_features
        self._z_dim = z_dim
        per_gate = getattr(cfg, "per_gate", False)

        # Full-dim projections (no half concat)
        self.raw_proj = torch.nn.Linear(raw_node_dim, z_dim)
        self.graph_z_proj = torch.nn.Linear(graph_z_in, z_dim)
        # Gate init -1.7 -> sigmoid ~0.15; scalar (1) or per-dim (z_dim) when per_gate=True
        if per_gate:
            self.graph_gate_logit = torch.nn.Parameter(torch.full((z_dim,), -1.7))
        else:
            self.graph_gate_logit = torch.nn.Parameter(torch.tensor([-1.7]))

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
            raise ValueError("DualArmDisentangledPolicyGated.forward requires noisy_action and timesteps")

        B = obs.shape[0]
        device = obs.device

        nh, nt = self._build_node_histories(
            obs, ee_node_history, object_node_history, node_histories, node_types,
        )

        dist_embed, sim_embed = self._compute_disentangled_edges(nh)
        _, graph_z = self._encode_disentangled_graph(nh, dist_embed, sim_embed)

        # Gated fusion: z_base = raw_feat + gate * graph_z_comp (no concat)
        raw_latest = nh[:, :, -1, :].reshape(B, -1)
        raw_feat = self.raw_proj(raw_latest)                    # [B, z_dim]
        graph_z_comp = self.graph_z_proj(graph_z)                # [B, z_dim]
        gate = torch.sigmoid(self.graph_gate_logit).to(device)  # [1] or [z_dim]
        z_base = raw_feat + gate * graph_z_comp                 # [B, z_dim]

        if subtask_condition is not None and hasattr(self, "subtask_to_z"):
            subtask_embed = self.subtask_encoder(subtask_condition)
            z_base = z_base + self.subtask_to_z(subtask_embed)

        z_left = z_right = z_base

        ts_embed = self._get_timestep_embed(timesteps)
        if noisy_action.dim() == 2:
            noisy_action = noisy_action.unsqueeze(1)
        na_left = noisy_action[..., : self.arm_dim].transpose(1, 2)
        na_right = noisy_action[..., self.arm_dim :].transpose(1, 2)

        skips_l, bottleneck_l = self.unet_left.encode(na_left, ts_embed, z_left)
        skips_r, bottleneck_r = self.unet_right.encode(na_right, ts_embed, z_right)

        if self.cross_arm_attn is not None:
            bottleneck_l, bottleneck_r = self.cross_arm_attn(bottleneck_l, bottleneck_r)

        out_l = self.unet_left.decode(bottleneck_l, skips_l, ts_embed, z_left)
        out_r = self.unet_right.decode(bottleneck_r, skips_r, ts_embed, z_right)

        out_l = out_l.transpose(1, 2)
        out_r = out_r.transpose(1, 2)
        noise_pred = torch.cat([out_l, out_r], dim=-1)

        if return_dict:
            return {"noise_pred": noise_pred}
        return noise_pred

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
        result = super().loss(
            obs=obs,
            actions=actions,
            action_history=action_history,
            ee_node_history=ee_node_history,
            object_node_history=object_node_history,
            joint_states_history=joint_states_history,
            subtask_condition=subtask_condition,
            timesteps=timesteps,
            mask=mask,
            node_histories=node_histories,
            node_types=node_types,
        )
        g = torch.sigmoid(self.graph_gate_logit).detach()
        result["metrics/graph_gate_weight"] = g.mean() if g.numel() > 1 else g.squeeze()
        return result

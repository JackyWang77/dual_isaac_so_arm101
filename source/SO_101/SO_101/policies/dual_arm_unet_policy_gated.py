# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
"""
DualArmDisentangledPolicyGated: 门控融合版 Disentangled，不 concat，用可学习 gate 注入 graph。
z_base = raw_feat + gate * graph_z_comp，gate = sigmoid(graph_gate_logit)，初始 -2 -> ~0.12。
loss() 返回 metrics/graph_gate_weight 供 TensorBoard 监控。
"""

import torch

from SO_101.policies.graph_dit_policy import GraphDiTPolicyCfg
from SO_101.policies.dual_arm_unet_policy import DualArmDisentangledPolicy

_NODE_KEY_PAIRS = [
    ("left_ee_position", "left_ee_orientation"),
    ("right_ee_position", "right_ee_orientation"),
    ("cube_1_pos", "cube_1_ori"),
    ("cube_2_pos", "cube_2_ori"),
    ("fork_pos", "fork_ori"),
    ("knife_pos", "knife_ori"),
]


class DualArmDisentangledPolicyGated(DualArmDisentangledPolicy):
    """Dual-arm disentangled graph + dual U-Net, with learnable gated fusion.

    Replaces raw/graph concat with: z_base = raw_feat + gate * graph_z_comp,
    gate = sigmoid(graph_gate_logit). Initial logit -2 (gate ~0.12) for conservative graph injection.
    Exposes metrics/graph_gate_weight in loss() for TensorBoard.
    """

    def __init__(self, cfg: GraphDiTPolicyCfg):
        super().__init__(cfg)

        z_dim = getattr(cfg, "z_dim", 64)
        raw_node_dim = self.raw_proj.in_features
        graph_z_in = self.graph_z_proj.in_features
        self._z_dim = z_dim
        per_gate = getattr(cfg, "per_gate", False)

        self.raw_proj = torch.nn.Linear(raw_node_dim, z_dim)
        self.graph_z_proj = torch.nn.Linear(graph_z_in, z_dim)
        if per_gate:
            self.graph_gate_logit = torch.nn.Parameter(torch.full((z_dim,), -2.0))
        else:
            self.graph_gate_logit = torch.nn.Parameter(torch.tensor([-2.0]))

    def _get_num_nodes(self) -> int:
        """Infer expected number of nodes from raw_proj weight shape."""
        return self.raw_proj.in_features // 7

    def _extract_nodes_from_obs_direct(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract [B, N, 1, 7] from obs using obs_structure, no node_configs needed."""
        obs_struct = getattr(self.cfg, "obs_structure", None)
        if obs_struct is None:
            raise ValueError("obs_structure is required for node extraction")

        nodes = []
        for pos_key, ori_key in _NODE_KEY_PAIRS:
            if pos_key in obs_struct and ori_key in obs_struct:
                pos = obs[:, obs_struct[pos_key][0] : obs_struct[pos_key][1]]
                ori = obs[:, obs_struct[ori_key][0] : obs_struct[ori_key][1]]
                nodes.append(torch.cat([pos, ori], dim=-1))
        num_expected = self._get_num_nodes()
        if len(nodes) < num_expected:
            raise ValueError(
                f"Found {len(nodes)} nodes in obs_structure but raw_proj expects {num_expected}"
            )
        nodes = nodes[:num_expected]
        return torch.stack(nodes, dim=1).unsqueeze(2)  # [B, N, 1, 7]

    def _build_node_histories(
        self,
        obs: torch.Tensor,
        ee_node_history: torch.Tensor | None = None,
        object_node_history: torch.Tensor | None = None,
        node_histories: torch.Tensor | None = None,
        node_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build node histories. Prefers explicit node_histories, else extracts from obs."""
        if node_histories is not None:
            N = node_histories.shape[1]
            if node_types is None:
                node_types = torch.zeros(N, dtype=torch.long, device=node_histories.device)
            return node_histories, node_types

        num_expected = self._get_num_nodes()
        if num_expected > 2:
            nh = self._extract_nodes_from_obs_direct(obs)
            N = nh.shape[1]
            nt = torch.zeros(N, dtype=torch.long, device=nh.device)
            return nh, nt

        if ee_node_history is not None and object_node_history is not None:
            nh = torch.stack([ee_node_history, object_node_history], dim=1)
            nt = torch.tensor([0, 1], dtype=torch.long, device=nh.device)
            return nh, nt

        nh = self._extract_nodes_from_obs_direct(obs)
        N = nh.shape[1]
        nt = torch.zeros(N, dtype=torch.long, device=nh.device)
        return nh, nt

    def _compute_z_base(self, nh: torch.Tensor) -> torch.Tensor:
        """Graph encoder → gated fusion → z_base [B, z_dim].

        Args:
            nh: [B, N, T, 7] — already-normalized node histories.
        """
        B = nh.shape[0]
        dist_embed, sim_embed = self._compute_disentangled_edges(nh)
        _, graph_z = self._encode_disentangled_graph(nh, dist_embed, sim_embed)

        raw_latest = nh[:, :, -1, :].reshape(B, -1)
        raw_feat = self.raw_proj(raw_latest)
        graph_z_comp = self.graph_z_proj(graph_z)
        gate = torch.sigmoid(self.graph_gate_logit).to(nh.device)
        return raw_feat + gate * graph_z_comp

    def extract_z(
        self,
        ee_node_history: torch.Tensor | None = None,
        obj_node_history: torch.Tensor | None = None,
        *,
        obs: torch.Tensor | None = None,
        node_histories: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract gated z. Returns [B, 1, z_dim].

        Prefers pre-normalized node_histories [B, N, T, 7] when available.
        Falls back to extracting from obs (requires obs_structure on cfg).
        """
        if node_histories is not None:
            return self._compute_z_base(node_histories).unsqueeze(1)
        if obs is None:
            raise ValueError("extract_z requires node_histories= or obs=")
        nh = self._extract_nodes_from_obs_direct(obs)
        return self._compute_z_base(nh).unsqueeze(1)

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

        raw_latest = nh[:, :, -1, :].reshape(B, -1)
        raw_feat = self.raw_proj(raw_latest)
        graph_z_comp = self.graph_z_proj(graph_z)
        gate = torch.sigmoid(self.graph_gate_logit).to(device)
        z_base = raw_feat + gate * graph_z_comp

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

# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
"""
DualArmUnetPolicy: 双臂专用策略
- 共享 Graph Encoder → z
- 两个独立的 ConditionalUnet1D（左臂 / 右臂）
- Bottleneck 处加 CrossArmAttention，让两臂互相通信
- 输出: concat [left(6), right(6)] = [B, pred_horizon, 12]

使用方法:
    cfg.use_dual_arm_unet = True
    cfg.arm_action_dim = 6  # 每条臂的 action 维度
    policy = DualArmUnetPolicy(cfg)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from SO_101.policies.graph_dit_policy import GraphDiTPolicyCfg
from SO_101.policies.graph_unet_policy import (
    Conv1dBlock,
    ConditionalResidualBlock1D,
    GraphUnetPolicy,
    UnetPolicy,
    _safe_n_groups,
)


# ==============================================================================
# 1. CrossArmAttention — bottleneck 处的软连接
# ==============================================================================

class CrossArmAttention(nn.Module):
    """
    左右臂 bottleneck feature 互相 cross-attend，实现协调。
    
    输入:  x_left, x_right: [B, mid_dim, T_bottleneck]
    输出:  x_left', x_right': [B, mid_dim, T_bottleneck]
    
    left  queries right → left 知道右臂在做什么
    right queries left  → right 知道左臂在做什么
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.attn_l2r = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.attn_r2l = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.norm_l = nn.LayerNorm(dim)
        self.norm_r = nn.LayerNorm(dim)
        # FFN after cross-attn
        self.ffn_l = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ffn_r = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm_l2 = nn.LayerNorm(dim)
        self.norm_r2 = nn.LayerNorm(dim)

    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x_left, x_right: [B, D, T]
        # 转成 [B, T, D] 给 MultiheadAttention
        xl = x_left.transpose(1, 2)   # [B, T, D]
        xr = x_right.transpose(1, 2)  # [B, T, D]

        # left attends to right
        xl_cross, _ = self.attn_l2r(xl, xr, xr)
        xl = self.norm_l(xl + xl_cross)
        xl = self.norm_l2(xl + self.ffn_l(xl))

        # right attends to left
        xr_cross, _ = self.attn_r2l(xr, xl, xl)
        xr = self.norm_r(xr + xr_cross)
        xr = self.norm_r2(xr + self.ffn_r(xr))

        return xl.transpose(1, 2), xr.transpose(1, 2)


# ==============================================================================
# 2. SplitConditionalUnet1D — 支持 encode / decode 分离的 U-Net
# ==============================================================================

class SplitConditionalUnet1D(nn.Module):
    """
    与 ConditionalUnet1D 相同，但拆分成 encode() + decode()
    以便在 bottleneck 处注入 cross-arm 信息。
    
    encode() → (skip_features, bottleneck)
    decode(bottleneck, skip_features) → output
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
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

        # Down path
        self.down_modules = nn.ModuleList([])
        for i in range(len(down_dims)):
            in_d = all_dims[i]
            out_d = all_dims[i + 1]
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(in_d, out_d, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(out_d, out_d, cond_dim, kernel_size, n_groups),
                nn.Conv1d(out_d, out_d, 3, stride=2, padding=1),
            ]))

        # Mid (bottleneck)
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups),
        ])

        # Up path
        self.up_modules = nn.ModuleList([])
        for i in reversed(range(len(down_dims))):
            in_d = all_dims[i + 1]
            out_d = start_dim if i == 0 else all_dims[i]
            skip_channels = in_d
            self.up_modules.append(nn.ModuleList([
                nn.ConvTranspose1d(in_d, out_d, 4, stride=2, padding=1),
                ConditionalResidualBlock1D(out_d + skip_channels, out_d, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(out_d, out_d, cond_dim, kernel_size, n_groups),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def _build_cond(self, timestep_embed: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        time_emb = self.diffusion_step_encoder(timestep_embed)
        return torch.cat([time_emb, global_cond], dim=-1)

    def encode(
        self,
        x: torch.Tensor,
        timestep_embed: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Down path + bottleneck.
        
        Returns:
            skip_features: list of skip tensors (low→high level order, for decode)
            bottleneck:     [B, mid_dim, T_bottleneck]
        """
        cond = self._build_cond(timestep_embed, global_cond)
        skip_features = []

        for res1, res2, downsample in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skip_features.append(x)
            x = downsample(x)

        for res_block in self.mid_modules:
            x = res_block(x, cond)

        return skip_features, x  # x is bottleneck

    def decode(
        self,
        bottleneck: torch.Tensor,
        skip_features: list[torch.Tensor],
        timestep_embed: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Up path from bottleneck.
        """
        cond = self._build_cond(timestep_embed, global_cond)
        x = bottleneck
        h = list(skip_features)  # copy

        for upsample, res1, res2 in self.up_modules:
            x = upsample(x)
            skip = h.pop()
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)

        return self.final_conv(x)

    def forward(
        self,
        x: torch.Tensor,
        timestep_embed: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Standard forward (no cross-arm, same as ConditionalUnet1D)."""
        skip_features, bottleneck = self.encode(x, timestep_embed, global_cond)
        return self.decode(bottleneck, skip_features, timestep_embed, global_cond)


# ==============================================================================
# 3. DualArmUnetPolicy
# ==============================================================================

class DualArmUnetPolicy(GraphUnetPolicy):
    """
    双臂 Flow Matching 策略。

    继承 GraphUnetPolicy（共享 graph encoder 部分），替换单 U-Net 为双臂结构：

        Graph Encoder (shared)
                ↓ z [B, z_dim]
        ┌────────────────────────┐
        │  UNet_left  (6-dim)    │ ←── z + left_joint_enc
        │  UNet_right (6-dim)    │ ←── z + right_joint_enc
        │       ↕ CrossArmAttn  │  ← bottleneck 处软连接
        └────────────────────────┘
                ↓
        concat [left, right] → [B, pred_horizon, 12]

    Config 新增字段:
        arm_action_dim: int = 6        每臂 action 维度
        cross_arm_heads: int = 4       CrossArmAttention heads
    """

    def __init__(self, cfg: GraphDiTPolicyCfg):
        # 先用父类初始化 graph encoder 部分
        super().__init__(cfg)

        arm_dim = getattr(cfg, "arm_action_dim", cfg.action_dim // 2)
        assert cfg.action_dim == arm_dim * 2, (
            f"DualArmUnetPolicy: action_dim ({cfg.action_dim}) must be 2 × arm_action_dim ({arm_dim})"
        )
        self.arm_dim = arm_dim

        z_dim = getattr(cfg, "z_dim", 64)
        cross_heads = getattr(cfg, "cross_arm_heads", 4)

        # 每臂单独的 joint encoder（只编码自己那侧的 joint history）
        jd_per_arm = arm_dim  # 每臂 joint_dim = arm_action_dim (6: 5 arm + 1 gripper)
        hist_len = cfg.action_history_length
        arm_joint_z_dim = z_dim // 2  # 给每臂留一半 z 空间

        self.left_joint_encoder = nn.Sequential(
            nn.Linear(jd_per_arm * hist_len, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, arm_joint_z_dim),
        )
        self.right_joint_encoder = nn.Sequential(
            nn.Linear(jd_per_arm * hist_len, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, arm_joint_z_dim),
        )

        # 每臂的 UNet（输入 dim = arm_dim，条件 dim = z_dim + arm_joint_z_dim）
        arm_cond_dim = z_dim + arm_joint_z_dim
        down_dims = (128, 256, 512)
        mid_dim = down_dims[-1]

        self.unet_left = SplitConditionalUnet1D(
            input_dim=arm_dim,
            global_cond_dim=arm_cond_dim,
            diffusion_step_embed_dim=cfg.hidden_dim,
            down_dims=down_dims,
            kernel_size=5,
            n_groups=8,
        )
        self.unet_right = SplitConditionalUnet1D(
            input_dim=arm_dim,
            global_cond_dim=arm_cond_dim,
            diffusion_step_embed_dim=cfg.hidden_dim,
            down_dims=down_dims,
            kernel_size=5,
            n_groups=8,
        )

        # Bottleneck cross-arm attention
        self.cross_arm_attn = CrossArmAttention(dim=mid_dim, num_heads=cross_heads)

        # 删掉父类的单 U-Net（不再使用）
        del self.unet

        # 父类如果有 joint_encoder 也删掉（用各臂独立的）
        if hasattr(self, "joint_encoder"):
            del self.joint_encoder

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _split_joint_history(
        self, joint_states_history: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        joint_states_history: [B, H, joint_dim=12]
        split → left [B, H, 6], right [B, H, 6]
        
        Train order: [left_6, right_6]  (同 train.py 里的 is_dual_arm 逻辑)
        """
        if joint_states_history is None:
            return None, None
        jd = joint_states_history.shape[-1]
        half = jd // 2
        return joint_states_history[..., :half], joint_states_history[..., half:]

    def _encode_arm_joint(
        self,
        encoder: nn.Module,
        joint_hist: torch.Tensor | None,
        B: int,
        device: torch.device,
        z_dim_half: int,
    ) -> torch.Tensor:
        if joint_hist is not None:
            flat = joint_hist.reshape(B, -1)
            return encoder(flat)
        return torch.zeros(B, z_dim_half, device=device)

    # ------------------------------------------------------------------
    # forward
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
            raise ValueError("DualArmUnetPolicy.forward requires noisy_action and timesteps")

        B = obs.shape[0]
        device = obs.device

        # 1. 共享 graph encoder → z [B, z_dim] (复用 GraphUnetPolicy 的 _encode_graph)
        nh, nt = self._build_node_histories(
            obs, ee_node_history, object_node_history, node_histories, node_types,
        )
        node_features = self._embed_node_histories(nh, nt)
        edge_embed = self._compute_and_embed_edges(nh)
        ts_embed = self._get_timestep_embed(timesteps)
        condition = ts_embed
        if subtask_condition is not None and hasattr(self, "subtask_encoder"):
            subtask_embed = self.subtask_encoder(subtask_condition)
            condition = self.condition_proj(torch.cat([ts_embed, subtask_embed], dim=-1))
        _, z = self._encode_graph(node_features, edge_embed, condition)

        # 2. 各臂 joint encoding
        left_jh, right_jh = self._split_joint_history(joint_states_history)
        z_dim_half = self.left_joint_encoder[-1].out_features
        left_jenc = self._encode_arm_joint(self.left_joint_encoder, left_jh, B, device, z_dim_half)
        right_jenc = self._encode_arm_joint(self.right_joint_encoder, right_jh, B, device, z_dim_half)

        z_left = torch.cat([z, left_jenc], dim=-1)   # [B, z_dim + z_dim_half]
        z_right = torch.cat([z, right_jenc], dim=-1)  # [B, z_dim + z_dim_half]

        # 3. 拆分 noisy_action
        if noisy_action.dim() == 2:
            noisy_action = noisy_action.unsqueeze(1)  # [B, 1, 12]
        # noisy_action: [B, pred_horizon, 12]
        na_left = noisy_action[..., :self.arm_dim].transpose(1, 2)   # [B, 6, T]
        na_right = noisy_action[..., self.arm_dim:].transpose(1, 2)  # [B, 6, T]

        # 4. Down path (ts_embed from step 1)
        skips_l, bottleneck_l = self.unet_left.encode(na_left, ts_embed, z_left)
        skips_r, bottleneck_r = self.unet_right.encode(na_right, ts_embed, z_right)

        # 5. CrossArmAttention at bottleneck ← 关键
        bottleneck_l, bottleneck_r = self.cross_arm_attn(bottleneck_l, bottleneck_r)

        # 6. Up path
        out_l = self.unet_left.decode(bottleneck_l, skips_l, ts_embed, z_left)   # [B, 6, T]
        out_r = self.unet_right.decode(bottleneck_r, skips_r, ts_embed, z_right) # [B, 6, T]

        # 7. 合并输出
        out_l = out_l.transpose(1, 2)  # [B, T, 6]
        out_r = out_r.transpose(1, 2)  # [B, T, 6]
        noise_pred = torch.cat([out_l, out_r], dim=-1)  # [B, T, 12]

        if return_dict:
            return {"noise_pred": noise_pred}
        return noise_pred


# ==============================================================================
# 4. DualArmUnetPolicyMLP — 双臂 MLP-UNet（无 graph attention）
# ==============================================================================

class DualArmUnetPolicyMLP(UnetPolicy):
    """
    双臂 Flow Matching 策略（MLP 版，无 graph attention）。

    继承 UnetPolicy（共享 MLP node encoder + pool），替换单 U-Net 为双臂结构：

        MLP Node Encoder (shared)
                ↓ z [B, z_dim]
        ┌────────────────────────┐
        │  UNet_left  (6-dim)    │ ←── z + left_joint_enc
        │  UNet_right (6-dim)    │ ←── z + right_joint_enc
        │       ↕ CrossArmAttn  │  ← bottleneck 处软连接
        └────────────────────────┘
                ↓
        concat [left, right] → [B, pred_horizon, 12]

    与 DualArmUnetPolicy 结构相同，区别仅 encoder：MLP pool vs Graph attention。
    """

    def __init__(self, cfg: GraphDiTPolicyCfg):
        super().__init__(cfg)

        arm_dim = getattr(cfg, "arm_action_dim", cfg.action_dim // 2)
        assert cfg.action_dim == arm_dim * 2, (
            f"DualArmUnetPolicyMLP: action_dim ({cfg.action_dim}) must be 2 × arm_action_dim ({arm_dim})"
        )
        self.arm_dim = arm_dim

        z_dim = getattr(cfg, "z_dim", 64)
        cross_heads = getattr(cfg, "cross_arm_heads", 4)

        jd_per_arm = arm_dim
        hist_len = cfg.action_history_length
        arm_joint_z_dim = z_dim // 2

        self.left_joint_encoder = nn.Sequential(
            nn.Linear(jd_per_arm * hist_len, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, arm_joint_z_dim),
        )
        self.right_joint_encoder = nn.Sequential(
            nn.Linear(jd_per_arm * hist_len, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, arm_joint_z_dim),
        )

        arm_cond_dim = z_dim + arm_joint_z_dim
        down_dims = (128, 256, 512)
        mid_dim = down_dims[-1]

        self.unet_left = SplitConditionalUnet1D(
            input_dim=arm_dim,
            global_cond_dim=arm_cond_dim,
            diffusion_step_embed_dim=cfg.hidden_dim,
            down_dims=down_dims,
            kernel_size=5,
            n_groups=8,
        )
        self.unet_right = SplitConditionalUnet1D(
            input_dim=arm_dim,
            global_cond_dim=arm_cond_dim,
            diffusion_step_embed_dim=cfg.hidden_dim,
            down_dims=down_dims,
            kernel_size=5,
            n_groups=8,
        )

        self.cross_arm_attn = CrossArmAttention(dim=mid_dim, num_heads=cross_heads)

        del self.unet
        if hasattr(self, "joint_encoder"):
            del self.joint_encoder

    def _split_joint_history(
        self, joint_states_history: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if joint_states_history is None:
            return None, None
        jd = joint_states_history.shape[-1]
        half = jd // 2
        return joint_states_history[..., :half], joint_states_history[..., half:]

    def _encode_arm_joint(
        self,
        encoder: nn.Module,
        joint_hist: torch.Tensor | None,
        B: int,
        device: torch.device,
        z_dim_half: int,
    ) -> torch.Tensor:
        if joint_hist is not None:
            flat = joint_hist.reshape(B, -1)
            return encoder(flat)
        return torch.zeros(B, z_dim_half, device=device)

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
            raise ValueError("DualArmUnetPolicyMLP.forward requires noisy_action and timesteps")

        B = obs.shape[0]
        device = obs.device

        # 1. MLP encoder → z (复用 UnetPolicy 的 _embed + _pool)
        nh, nt = self._build_node_histories(
            obs, ee_node_history, object_node_history, node_histories, node_types,
        )
        node_features = self._embed_node_histories(nh, nt)
        z = self._pool_node_latent(node_features)  # [B, z_dim]

        # 2. 各臂 joint encoding
        left_jh, right_jh = self._split_joint_history(joint_states_history)
        z_dim_half = self.left_joint_encoder[-1].out_features
        left_jenc = self._encode_arm_joint(self.left_joint_encoder, left_jh, B, device, z_dim_half)
        right_jenc = self._encode_arm_joint(self.right_joint_encoder, right_jh, B, device, z_dim_half)

        z_left = torch.cat([z, left_jenc], dim=-1)
        z_right = torch.cat([z, right_jenc], dim=-1)

        # 3. 拆分 noisy_action
        if noisy_action.dim() == 2:
            noisy_action = noisy_action.unsqueeze(1)
        na_left = noisy_action[..., :self.arm_dim].transpose(1, 2)   # [B, 6, T]
        na_right = noisy_action[..., self.arm_dim:].transpose(1, 2)  # [B, 6, T]

        # 4. timestep embed
        ts_embed = self._get_timestep_embed(timesteps)

        # 5. Down path
        skips_l, bottleneck_l = self.unet_left.encode(na_left, ts_embed, z_left)
        skips_r, bottleneck_r = self.unet_right.encode(na_right, ts_embed, z_right)

        # 6. CrossArmAttention at bottleneck
        bottleneck_l, bottleneck_r = self.cross_arm_attn(bottleneck_l, bottleneck_r)

        # 7. Up path
        out_l = self.unet_left.decode(bottleneck_l, skips_l, ts_embed, z_left)   # [B, 6, T]
        out_r = self.unet_right.decode(bottleneck_r, skips_r, ts_embed, z_right) # [B, 6, T]

        # 8. 合并输出
        out_l = out_l.transpose(1, 2)  # [B, T, 6]
        out_r = out_r.transpose(1, 2)  # [B, T, 6]
        noise_pred = torch.cat([out_l, out_r], dim=-1)  # [B, T, 12]

        if return_dict:
            return {"noise_pred": noise_pred}
        return noise_pred

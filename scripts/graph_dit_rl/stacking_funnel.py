"""Shared stacking funnel mask for train_graph_rl (rollout + expert) and play_graph_rl (alignment gating).

Must stay in sync: same geometry as GraphDiTRLTrainer._stacking_funnel_mask_from_cubes.
"""
import torch

FUNNEL_HELD_Z_MIN_M = 0.015  # held cube lifted above table
FUNNEL_XY_ALIGN_M = 0.003  # 3 mm — inner "aligned"
FUNNEL_XY_MAX_M = 0.02  # 20 mm — outer xy bound for approach band
FUNNEL_Z_SEP_MIN_M = 0.005  # min vertical separation (stack phase branch)


def stacking_funnel_mask_from_cubes(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """Where RL residual α·δ matches training: approach band OR aligned+z branch."""
    c1_z, c2_z = c1[:, 2], c2[:, 2]
    target_xy = torch.where((c1_z < c2_z).unsqueeze(-1), c1[:, :2], c2[:, :2])
    held_z = torch.max(c1_z, c2_z)
    held_xy = torch.where((c1_z < c2_z).unsqueeze(-1), c2[:, :2], c1[:, :2])
    xy_error_norm = (held_xy - target_xy).norm(dim=-1)
    z_sep = torch.abs(c1_z - c2_z)
    hz = held_z > FUNNEL_HELD_Z_MIN_M
    approach_xy = (xy_error_norm > FUNNEL_XY_ALIGN_M) & (xy_error_norm < FUNNEL_XY_MAX_M)
    aligned_xy = xy_error_norm <= FUNNEL_XY_ALIGN_M
    near_target = hz & approach_xy
    aligned_near_stack = hz & aligned_xy & (z_sep > FUNNEL_Z_SEP_MIN_M)
    return near_target | aligned_near_stack

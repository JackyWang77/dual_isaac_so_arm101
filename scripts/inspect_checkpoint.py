#!/usr/bin/env python3
"""Inspect a .pt checkpoint: top-level keys, policy_state_dict keys, cfg, obs_key_offsets/dims.
Usage: python scripts/inspect_checkpoint.py [path_to_best_model.pt]
"""
import sys
import torch

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        path = "logs/graph_unet_full/stack_joint_t1_gripper_flow_matching/high_quality_data/best_model.pt"
        print(f"Using default: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print("=== Top-level keys ===")
    print(list(ckpt.keys()))
    print()
    state = ckpt.get("policy_state_dict", {})
    keys = list(state.keys())
    print("=== policy_state_dict keys (all) ===")
    for k in sorted(keys):
        t = state[k]
        sh = t.shape if hasattr(t, "shape") else type(t).__name__
        print(f"  {k}: {sh}")
    print()
    print("=== Policy type (what play.py will choose) ===")
    cfg = ckpt.get("cfg", None)
    print("  cfg present:", cfg is not None)
    print("  cfg.arm_action_dim:", getattr(cfg, "arm_action_dim", None))
    print("  cfg.use_raw_only:", getattr(cfg, "use_raw_only", None))
    print("  'graph_gate_logit' in state:", any("graph_gate_logit" in k for k in keys))
    print("  'graph_attention_layers' in state:", any("graph_attention_layers" in k for k in keys))
    print("  'disentangled_attn_layers' in state:", any("disentangled_attn_layers" in k for k in keys))
    print()
    print("=== Obs layout (for play alignment) ===")
    print("  obs_key_offsets:", ckpt.get("obs_key_offsets"))
    print("  obs_key_dims:", ckpt.get("obs_key_dims"))
    if cfg is not None and hasattr(cfg, "obs_key_offsets") and cfg.obs_key_offsets:
        print("  cfg.obs_key_offsets:", cfg.obs_key_offsets)
    print()
    print("=== Expected obs dim (sum of obs_key_dims) ===")
    dims = ckpt.get("obs_key_dims")
    if dims:
        total = sum(dims.values()) if isinstance(dims, dict) else sum(dims)
        print("  total:", total)
    else:
        print("  (not in checkpoint)")

if __name__ == "__main__":
    main()

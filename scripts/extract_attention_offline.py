#!/usr/bin/env python3
"""Offline: load obs records, run through graph encoder, extract attention.

Supports both GraphUnetPolicy and DualArmUnetPolicy (auto-detected from checkpoint).

Usage:
    python scripts/extract_attention_offline.py \
        --obs_records attention_records.json \
        --checkpoint ./logs/graph_unet_full/.../best_model.pt \
        --output attention_with_attn.json \
        --success_only
"""
import argparse
import json

import numpy as np
import torch

from SO_101.policies.graph_unet_policy import GraphUnetPolicy
from SO_101.policies.graph_dit_policy import GraphDiTPolicyCfg
from SO_101.policies.dual_arm_unet_policy import DualArmUnetPolicy, DualArmUnetPolicyMLP


def _resolve_policy_class(checkpoint: dict):
    """Pick the right PolicyClass from checkpoint metadata (mirrors play.py logic)."""
    cfg = checkpoint.get("cfg", None)
    state_keys = checkpoint.get("policy_state_dict", {}).keys()

    if cfg is not None and getattr(cfg, "arm_action_dim", None) is not None:
        if getattr(cfg, "use_raw_only", False):
            raise ValueError(
                "Checkpoint is DualArmUnetPolicyRawOnly (no graph attention). "
                "Cannot extract attention from this model."
            )
        if getattr(cfg, "use_graph_encoder", False):
            return DualArmUnetPolicy
        has_graph = any("graph_attention_layers" in k for k in state_keys)
        if has_graph:
            return DualArmUnetPolicy
        return DualArmUnetPolicyMLP
    has_graph = any("graph_attention_layers" in k for k in state_keys)
    if has_graph:
        return GraphUnetPolicy
    raise ValueError("Cannot determine policy class from checkpoint")


def aggregate_attn(attn_weights, num_nodes, history_length):
    """[B, heads, seq, seq] -> [B, N, N] node-to-node (mean over heads & time)."""
    B = attn_weights.shape[0]
    out = torch.zeros(B, num_nodes, num_nodes, device=attn_weights.device)
    for i in range(num_nodes):
        for j in range(num_nodes):
            qi = slice(i * history_length, (i + 1) * history_length)
            kj = slice(j * history_length, (j + 1) * history_length)
            out[:, i, j] = attn_weights[:, :, qi, kj].mean(dim=(1, 2, 3))
    return out.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs_records", required=True, help="JSON from play.py --record_attention")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint with graph attention layers")
    parser.add_argument("--output", default=None, help="Output JSON with attention added")
    parser.add_argument("--success_only", action="store_true", help="Only process successful episodes")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    with open(args.obs_records) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records")

    if args.success_only:
        records = [r for r in records if r.get("success", False)]
        print(f"Filtered to {len(records)} successful records")

    if not records:
        print("No records to process")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint.get("cfg", None)

    PolicyClass = _resolve_policy_class(checkpoint)
    print(f"Detected policy: {PolicyClass.__name__}")

    policy = PolicyClass.load(args.checkpoint, device=device)
    policy.eval()
    print(f"Loaded {PolicyClass.__name__} from {args.checkpoint}")

    node_stats = checkpoint.get("node_stats", None)
    ee_mean = ee_std = obj_mean = obj_std = None
    if node_stats:
        for key in ("ee_mean", "ee_std", "object_mean", "object_std"):
            val = node_stats.get(key)
            if val is None:
                continue
            t = torch.from_numpy(val).squeeze().to(device) if isinstance(val, np.ndarray) else val.squeeze().to(device)
            if key == "ee_mean":
                ee_mean = t
            elif key == "ee_std":
                ee_std = t
            elif key == "object_mean":
                obj_mean = t
            elif key == "object_std":
                obj_std = t

    node_configs = getattr(cfg, "node_configs", None) if cfg else None
    if node_configs:
        node_types = torch.tensor([nc.get("type", 0) for nc in node_configs], dtype=torch.long, device=device)
        num_nodes = len(node_configs)
        node_names = [nc["name"] for nc in node_configs]
    else:
        node_types = torch.tensor([0, 1], dtype=torch.long, device=device)
        num_nodes = 2
        node_names = ["ee", "object"]

    n_extracted = 0
    for idx, rec in enumerate(records):
        obs = rec.get("obs", {})
        node_feats_raw = obs.get("node_features")
        if node_feats_raw is None:
            if idx == 0:
                print("Warning: no node_features in records, skipping attention extraction")
            continue

        nh = torch.tensor(node_feats_raw, dtype=torch.float32, device=device).unsqueeze(0)
        H = nh.shape[2]

        if ee_mean is not None:
            for n_idx in range(nh.shape[1]):
                ntype = node_types[n_idx].item()
                if ntype == 0 and ee_std is not None:
                    nh[:, n_idx] = (nh[:, n_idx] - ee_mean) / ee_std
                elif ntype == 1 and obj_mean is not None and obj_std is not None:
                    nh[:, n_idx] = (nh[:, n_idx] - obj_mean) / obj_std

        with torch.no_grad():
            node_embedded = policy._embed_node_histories(nh, node_types)
            edge_embed = policy._compute_and_embed_edges(nh)
            result = policy._encode_graph(node_embedded, edge_embed, return_attention=True)
            if isinstance(result, tuple) and len(result) == 3:
                _, _, attn_weights = result
                attn_nxn = aggregate_attn(attn_weights, num_nodes, H)
                rec["attention"] = attn_nxn[0].tolist()
                rec["node_names"] = node_names
                n_extracted += 1

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(records)}")

    print(f"Extracted attention for {n_extracted}/{len(records)} records")

    out_path = args.output or args.obs_records.replace(".json", "_with_attn.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

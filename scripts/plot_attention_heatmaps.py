#!/usr/bin/env python3
"""Plot position-only attention heatmaps (Turbo cmap, waffle grid, serif, Gen. Prob. caption)."""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Top-tier paper: Turbo (full spectrum, perceptually uniform)
try:
    CMAP = plt.get_cmap("turbo")
except Exception:
    CMAP = plt.get_cmap("RdYlBu_r")
MASK_COLOR = "#f0f0f0"

# Short axis labels: left_ee->left, right_ee->right, cube_1->cube1, cube_2->cube2 (no underscores)
def _tick_labels(node_names):
    m = {"left_ee": "left", "right_ee": "right", "cube_1": "cube1", "cube_2": "cube2"}
    return [m.get(n, n.replace("_", "")) for n in node_names]

# Expected node order for stack task (must match train NODE_CONFIGS → checkpoint → extract)
EXPECTED_NODE_NAMES_4 = ["left_ee", "right_ee", "cube_1", "cube_2"]

PHASE_LABELS = [
    "1_left_pick_cube1",
    "2_left_release",
    "3_right_pick_cube2",
    "4_stack_done",
]


def main():
    parser = argparse.ArgumentParser(description="Plot position attention heatmaps (diagonal gray, norm from off-diagonal).")
    parser.add_argument("--input", default="attention_output/obs_with_attn_record.json", help="JSON with attention_pos per record")
    parser.add_argument("--output_dir", default="attention_output/heatmaps", help="Directory to save figures")
    parser.add_argument("--format", default="pdf", choices=["png", "pdf"], help="Output format")
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Global vmin/vmax from all records (off-diagonal) so every PDF uses the same color scale
    mats_display = []
    for rec in records:
        mat = np.array(rec.get("attention_pos"), dtype=float)
        if mat.size == 0:
            mats_display.append(None)
            continue
        md = mat.copy()
        n = mat.shape[0]
        md[np.diag_indices(n)] = np.nan
        mats_display.append(md)
    valid = [m for m in mats_display if m is not None]
    if not valid:
        print("No attention_pos data")
        return
    vmin = float(np.nanmin([np.nanmin(m) for m in valid]))
    vmax = float(np.nanmax([np.nanmax(m) for m in valid]))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    plt.rcParams["font.family"] = "serif"
    cmap = CMAP.copy()
    cmap.set_bad(MASK_COLOR)

    for i, rec in enumerate(records):
        mat_display = mats_display[i] if i < len(mats_display) else None
        if mat_display is None:
            continue
        node_names = rec.get("node_names", ["left_ee", "right_ee", "cube_1", "cube_2"])
        if len(node_names) == 4 and node_names != EXPECTED_NODE_NAMES_4:
            print(f"  WARNING: node_names {node_names} != expected {EXPECTED_NODE_NAMES_4}; axes may be mislabeled.")
        n = mat_display.shape[0]
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(mat_display, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
        labels = _tick_labels(node_names)
        ax.set_xticks(range(len(node_names)))
        ax.set_yticks(range(len(node_names)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=20)
        ax.set_yticklabels(labels, fontsize=20)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both", length=0)
        for x in range(1, n):
            ax.axvline(x - 0.5, color="white", linewidth=0.5)
            ax.axhline(x - 0.5, color="white", linewidth=0.5)
        ax.axvline(n - 0.5, color="white", linewidth=0.5)
        ax.axhline(n - 0.5, color="white", linewidth=0.5)
        cbar = plt.colorbar(im, ax=ax, label="")
        cbar.ax.tick_params(labelsize=16)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f"phase{i+1}_pos.{args.format}")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    print(f"Done. {len(records)} figures in {args.output_dir}/")


if __name__ == "__main__":
    main()

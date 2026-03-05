#!/usr/bin/env python3
"""Animate position attention heatmaps (Turbo cmap, waffle grid, progress bar, phase label)."""
import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

# Top-tier paper style: Turbo (full spectrum, perceptually better than Jet)
try:
    CMAP = plt.get_cmap("turbo")
except Exception:
    CMAP = plt.get_cmap("RdYlBu_r")
MASK_COLOR = "#f0f0f0"  # light gray for diagonal, focus on cross-node interaction

# Short axis labels: left_ee->left, right_ee->right, cube_1->cube1, cube_2->cube2 (no underscores)
def _tick_labels(node_names):
    m = {"left_ee": "left", "right_ee": "right", "cube_1": "cube1", "cube_2": "cube2"}
    return [m.get(n, n.replace("_", "")) for n in node_names]


def main():
    parser = argparse.ArgumentParser(description="Animate attention heatmaps over records (GIF).")
    parser.add_argument("--input", default="attention_output/obs_with_attn_record.json", help="JSON with attention_pos per record")
    parser.add_argument("--output", default="attention_output/attention_animation.gif", help="Output GIF path")
    parser.add_argument("--fps", type=int, default=5, help="FPS (default 5 = record every exec_horizon=10 steps, 10*0.02s=0.2s per frame)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for GIF frames")
    parser.add_argument("--figsize", type=float, nargs=2, default=[7, 6], metavar=("W", "H"), help="Figure size (default 7 6)")
    parser.add_argument("--indices_json", default=None, help="epN_indices.json for phase labels on progress bar")
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)

    if not records:
        print("No records")
        return

    # Build list of display matrices (diagonal = nan for gray), global vmin/vmax from off-diagonal
    mats_display = []
    for rec in records:
        mat = np.array(rec.get("attention_pos"), dtype=float)
        if mat.size == 0:
            continue
        md = mat.copy()
        n = mat.shape[0]
        md[np.diag_indices(n)] = np.nan
        mats_display.append(md)

    if not mats_display:
        print("No attention_pos data")
        return

    vmin = float(np.nanmin([np.nanmin(m) for m in mats_display]))
    vmax = float(np.nanmax([np.nanmax(m) for m in mats_display]))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    node_names = records[0].get("node_names", ["left_ee", "right_ee", "cube_1", "cube_2"])
    n_nodes = len(node_names)
    cmap = CMAP.copy()
    cmap.set_bad(MASK_COLOR)

    # Phase labels from indices JSON (step -> human-readable phase)
    phase_by_step = {}
    if getattr(args, "indices_json", None) and os.path.isfile(args.indices_json):
        with open(args.indices_json) as f:
            idx_data = json.load(f)
        phases = idx_data.get("phases", {})
        for k, v in phases.items():
            if isinstance(v, dict) and "step" in v:
                phase_by_step[v["step"]] = k.replace("_", " ").title()
    def get_phase_label(step):
        if not phase_by_step:
            return f"Step {step}"
        # nearest phase by step
        best = min(phase_by_step.keys(), key=lambda s: abs(s - step))
        return phase_by_step.get(best, f"Step {step}")

    plt.rcParams["font.family"] = "serif"
    w, h = args.figsize
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(mats_display[0], cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    labels = _tick_labels(node_names)
    ax.set_xticks(range(n_nodes))
    ax.set_yticks(range(n_nodes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=20)
    ax.set_yticklabels(labels, fontsize=20)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", length=0)
    # Waffle: white grid between cells
    for x in range(1, n_nodes):
        ax.axvline(x - 0.5, color="white", linewidth=0.5)
        ax.axhline(x - 0.5, color="white", linewidth=0.5)
    ax.axvline(n_nodes - 0.5, color="white", linewidth=0.5)
    ax.axhline(n_nodes - 0.5, color="white", linewidth=0.5)
    title = ax.set_title("", fontsize=18)
    cbar = plt.colorbar(im, ax=ax, label="")
    cbar.ax.tick_params(labelsize=16)
    # Progress bar (top): full width strip, fill grows with frame
    ax_prog = fig.add_axes([0.12, 0.92, 0.76, 0.03])
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.axis("off")
    prog_bg = Rectangle((0, 0), 1, 1, facecolor="#e0e0e0", edgecolor="none")
    ax_prog.add_patch(prog_bg)
    prog_fill = Rectangle((0, 0), 0, 1, facecolor="#2196F3", edgecolor="none")
    ax_prog.add_patch(prog_fill)
    phase_text = ax_prog.text(0.5, 0.5, "", ha="center", va="center", fontsize=12, fontfamily="serif")
    plt.tight_layout(rect=[0, 0.04, 1, 0.88])

    def update(frame):
        im.set_array(mats_display[frame])
        step = records[frame].get("step", frame)
        title.set_text(f"Frame {frame + 1}/{len(mats_display)}  step={step}")
        prog_fill.set_width((frame + 1) / len(mats_display))
        phase_text.set_text(get_phase_label(step))
        return im, title

    anim = FuncAnimation(fig, update, frames=len(mats_display), blit=False)
    writer = PillowWriter(fps=args.fps)
    anim.save(args.output, writer=writer, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {args.output} ({len(mats_display)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()

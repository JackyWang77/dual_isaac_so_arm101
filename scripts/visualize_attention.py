#!/usr/bin/env python3
"""Visualize graph attention across 4 task phases.

Reads classified JSON (from classify_attention_by_distance.py) and generates:
  - Per-episode figure: 4 heatmaps (one per phase) + 4 directed graphs
  - Averaged figure: mean attention across all episodes

Usage:
    python scripts/visualize_attention.py classified.json --output ./figs
    python scripts/visualize_attention.py classified.json --episode 0
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

PHASE_ORDER = ["left_pick", "release", "right_pick", "right_stack"]
PHASE_LABELS = {
    "left_pick": "① Left Pick",
    "release": "② Release",
    "right_pick": "③ Right Pick",
    "right_stack": "④ Right Stack",
}
PHASE_COLORS = {
    "left_pick": "#4CAF50",
    "release": "#FF9800",
    "right_pick": "#2196F3",
    "right_stack": "#E91E63",
}


def plot_heatmap(attn: np.ndarray, node_names: list[str], title: str, ax, phase_color: str):
    """Plot NxN attention as annotated heatmap."""
    n = len(node_names)
    im = ax.imshow(attn, cmap="YlOrRd", vmin=0, vmax=max(attn.max(), 0.01), aspect="equal")
    for i in range(n):
        for j in range(n):
            val = attn[i, j]
            color = "white" if val > attn.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11, fontweight="bold", color=color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [_short(nm) for nm in node_names]
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("Key (attended to)", fontsize=9)
    ax.set_ylabel("Query (attending from)", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", color=phase_color, pad=8)
    return im


def plot_graph(attn: np.ndarray, node_names: list[str], title: str, ax, phase_color: str):
    """Plot NxN attention as directed graph with edge thickness."""
    n = len(node_names)
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    pos = {i: (np.cos(a) * 0.8, np.sin(a) * 0.8) for i, a in enumerate(angles)}

    node_colors = ["#81C784", "#64B5F6", "#FFB74D", "#F48FB1"]
    max_w = max(attn.max(), 1e-6)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = attn[i, j] / max_w
            if w < 0.05:
                continue
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            dist = np.hypot(dx, dy)
            shrink = 0.18
            sx = pos[i][0] + dx * shrink / dist
            sy = pos[i][1] + dy * shrink / dist
            ex = pos[j][0] - dx * shrink / dist
            ey = pos[j][1] - dy * shrink / dist
            ax.annotate(
                "",
                xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.3,head_length=0.15",
                    color=phase_color,
                    lw=1 + w * 4,
                    alpha=0.3 + w * 0.6,
                    connectionstyle="arc3,rad=0.1",
                ),
            )

    for i, (x, y) in pos.items():
        c = node_colors[i % len(node_colors)]
        ax.scatter(x, y, s=700, c=c, edgecolors="black", linewidths=1.5, zorder=5)
        ax.text(x, y, _short(node_names[i]), ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=phase_color, pad=6)


def _short(name: str) -> str:
    """Shorten node names for display."""
    mapping = {"left_ee": "L_EE", "right_ee": "R_EE", "cube_1": "C1", "cube_2": "C2"}
    return mapping.get(name, name)


def plot_episode(phases_data: dict, node_names: list[str], ep_label: str, save_path: str):
    """Plot one episode: top row heatmaps, bottom row graphs."""
    present = [p for p in PHASE_ORDER if p in phases_data]
    n = len(present)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 9))
    if n == 1:
        axes = axes.reshape(2, 1)

    for k, phase in enumerate(present):
        attn = np.array(phases_data[phase])
        color = PHASE_COLORS.get(phase, "black")
        label = PHASE_LABELS.get(phase, phase)
        plot_heatmap(attn, node_names, label, axes[0, k], color)
        plot_graph(attn, node_names, label, axes[1, k], color)

    fig.suptitle(f"Graph Attention — {ep_label}", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_average(all_phases: dict, node_names: list[str], save_path: str):
    """Plot average attention across all episodes."""
    present = [p for p in PHASE_ORDER if p in all_phases and len(all_phases[p]) > 0]
    if not present:
        return

    fig, axes = plt.subplots(2, len(present), figsize=(4.5 * len(present), 9))
    if len(present) == 1:
        axes = axes.reshape(2, 1)

    for k, phase in enumerate(present):
        attn = np.mean(all_phases[phase], axis=0)
        color = PHASE_COLORS.get(phase, "black")
        n_ep = len(all_phases[phase])
        label = f"{PHASE_LABELS.get(phase, phase)} (n={n_ep})"
        plot_heatmap(attn, node_names, label, axes[0, k], color)
        plot_graph(attn, node_names, label, axes[1, k], color)

    fig.suptitle("Graph Attention — Average Across Episodes", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Classified JSON (from classify_attention_by_distance.py)")
    parser.add_argument("--output", type=str, default="./figs", help="Output directory")
    parser.add_argument("--episode", type=int, default=None, help="Plot only this episode index (0-based)")
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)
    if not isinstance(records, list):
        records = [records]

    print(f"Loaded {len(records)} classified episodes")
    out_dir = args.output

    # Collect for averaging
    all_phases = {p: [] for p in PHASE_ORDER}

    indices = [args.episode] if args.episode is not None else range(len(records))
    for idx in indices:
        if idx >= len(records):
            continue
        rec = records[idx]
        phases_data = rec.get("phases", {})
        node_names = rec.get("node_names", ["left_ee", "right_ee", "cube_1", "cube_2"])
        ep_id = rec.get("episode_id", idx)
        env_id = rec.get("env_id", 0)
        ep_label = f"Episode {ep_id} (env {env_id})"

        plot_episode(phases_data, node_names, ep_label, os.path.join(out_dir, f"attention_ep{idx}.png"))

        for phase in PHASE_ORDER:
            if phase in phases_data:
                all_phases[phase].append(np.array(phases_data[phase]))

    if len(indices) > 1 and any(len(v) > 0 for v in all_phases.values()):
        node_names = records[0].get("node_names", ["left_ee", "right_ee", "cube_1", "cube_2"])
        plot_average(all_phases, node_names, os.path.join(out_dir, "attention_average.png"))

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()

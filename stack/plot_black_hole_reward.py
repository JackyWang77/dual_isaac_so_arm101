#!/usr/bin/env python3
"""
Visualize new black_hole reward: inverse-distance potential Φ = 1/(d_3d + ε).

Matches mdp.black_hole_attraction + CubeStackRLRewardsCfg (reward = Φ' - Φ per step).

Two figures: 3D surface + top view. Color uses LogNorm — Φ diverges at d→0 ("更野").
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# === Match CubeStackRLRewardsCfg.black_hole + black_hole_attraction ===
TARGET_Z_OFFSET = 0.012
EPS = 0.0005
REWARD_WEIGHT = 20.0  # env scales ΔΦ by this; static plot shows Φ (see note)

XY_RANGE = 0.01  # ±1 cm slice: cube1 at target height (z_err=0) → d_3d = sqrt(dx^2+dy^2)
N_GRID = 220


def potential_inverse_r(dx: np.ndarray, dy: np.ndarray, z_err: float = 0.0) -> np.ndarray:
    """Φ = 1/(d_3d + ε), d_3d = ||(dx,dy,z_err)||."""
    d3 = np.sqrt(dx**2 + dy**2 + z_err**2)
    return 1.0 / (d3 + EPS)


def _save(fig: plt.Figure, base: str, out_dir: str) -> None:
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{base}.{ext}")
        fig.savefig(
            path,
            format=ext,
            dpi=220 if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.22,
        )
        print(f"Saved: {path}")


def main():
    xs = np.linspace(-XY_RANGE, XY_RANGE, N_GRID)
    ys = np.linspace(-XY_RANGE, XY_RANGE, N_GRID)
    X, Y = np.meshgrid(xs, ys)

    Phi = potential_inverse_r(X, Y, z_err=0.0)
    phi_min = float(np.nanmin(Phi))
    phi_max = float(np.nanmax(Phi))

    # LogNorm: 避免除零附近一条线霸占整张色条
    norm = LogNorm(vmin=max(phi_min, 1.0 / (XY_RANGE * np.sqrt(2) + EPS) * 0.98), vmax=phi_max)

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------ 3D (Z = Φ, 物理尺度 — 中心很尖)
    fig3d = plt.figure(figsize=(7.2, 5.0))
    ax3d = fig3d.add_subplot(111, projection="3d")
    rs = max(2, N_GRID // 60)
    surf = ax3d.plot_surface(
        X,
        Y,
        Phi,
        cmap=cm.coolwarm,
        linewidth=0.06,
        edgecolor="0.35",
        antialiased=True,
        alpha=0.9,
        rstride=rs,
        cstride=rs,
        shade=True,
        norm=norm,
    )
    ax3d.set_xlabel(r"$\Delta x$ (m)", labelpad=6)
    ax3d.set_ylabel(r"$\Delta y$ (m)", labelpad=6)
    ax3d.set_zlabel(r"$\Phi = 1/(d+\varepsilon)$", labelpad=10)
    ax3d.set_title(
        rf"Inverse-distance potential ($z_{{\rm err}}=0$, $\varepsilon={EPS}$)",
        fontsize=10,
        pad=10,
    )
    ax3d.view_init(elev=14, azim=-52)
    ax3d.set_box_aspect((1, 1, 0.55))
    try:
        ax3d.set_proj_type("persp")
    except Exception:
        pass
    ax3d.dist = 11.5
    ax3d.xaxis.set_major_locator(MaxNLocator(5))
    ax3d.yaxis.set_major_locator(MaxNLocator(5))
    ax3d.zaxis.set_major_locator(MaxNLocator(5))

    fig3d.subplots_adjust(left=0.02, right=0.74, top=0.90, bottom=0.08)
    cax3d = fig3d.add_axes([0.80, 0.10, 0.022, 0.78])
    cb3d = fig3d.colorbar(surf, cax=cax3d)
    cb3d.set_label(r"$\Phi$ (log scale)")

    _save(fig3d, "black_hole_reward_3d", out_dir)
    plt.close(fig3d)

    # ------------------------------------------------------------------ Top
    fig2d = plt.figure(figsize=(5.8, 5.0))
    ax2d = fig2d.add_subplot(111)
    lev = np.geomspace(max(phi_min, 1e-6), phi_max, 40)
    cf = ax2d.contourf(X, Y, Phi, levels=lev, cmap=cm.coolwarm, norm=norm)
    ax2d.contour(X, Y, Phi, levels=np.geomspace(lev[2], lev[-2], 12), colors="k", linewidths=0.2, alpha=0.35)
    ax2d.set_xlabel(r"$\Delta x$ (m)")
    ax2d.set_ylabel(r"$\Delta y$ (m)")
    ax2d.set_title(rf"Top: $\Phi$ ($\pm${XY_RANGE * 100:.1f} cm, $z_{{\rm err}}=0$)", fontsize=10, pad=10)
    ax2d.set_aspect("equal")
    ax2d.xaxis.set_major_locator(MaxNLocator(5))
    ax2d.yaxis.set_major_locator(MaxNLocator(5))
    ax2d.tick_params(axis="both", which="major", labelsize=9)

    divider2d = make_axes_locatable(ax2d)
    cax2d = divider2d.append_axes("right", size="4.5%", pad=0.12)
    cb2d = fig2d.colorbar(cf, cax=cax2d)
    cb2d.set_label(r"$\Phi = 1/(d+\varepsilon)$")

    plt.tight_layout()
    _save(fig2d, "black_hole_reward_top", out_dir)
    plt.close(fig2d)

    print(
        f"params: target_z_offset={TARGET_Z_OFFSET}, eps={EPS}, reward_weight={REWARD_WEIGHT} (on ΔΦ)"
    )
    print(f"Φ at d→0: ~{1.0/EPS:.1f} (=1/ε); Φ at xy={XY_RANGE}m (z_err=0): {1.0/(XY_RANGE+EPS):.2f}")
    print("Training reward per step: r = w · (Φ(s') - Φ(s))")


if __name__ == "__main__":
    main()

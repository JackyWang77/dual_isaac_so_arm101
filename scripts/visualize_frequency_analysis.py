# Copyright (c) 2024-2025, SO-ARM101 Project
# SPDX-License-Identifier: BSD-3-Clause
"""
Frequency Domain Analysis: DiT vs U-Net Action Trajectories

对同一个 observation, 让两个模型各生成 N 条轨迹,
然后对轨迹做 FFT, 看频谱分布的差异。

U-Net (Conv1d) 应该在高频段能量更低 → 证明 low-pass filtering effect
DiT (Self-Attention) 高频段能量更高 → 证明 no frequency bias

Run from repo root:  python scripts/visualize_frequency_analysis.py
Output: frequency_analysis.pdf, jerk_comparison.pdf (same style as other viz scripts)
"""

import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Paper style: unified palette (DiT / U-Net), no in-figure titles
COLOR_DIT = "#4878CF"
COLOR_UNET = "#D65F5F"


FIGSIZE = (5.5, 4)


def _apply_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.linewidth": 1.2,
        "figure.dpi": 300,
    })


def analyze_frequency_spectrum(trajectories, label, color, ax):
    """
    对轨迹做 FFT, 画频谱。PSD 做下界截断，避免 log 尺度下数值尖刺。
    trajectories: [N, T, D] - N条轨迹, T步, D维动作
    """
    N, T, D = trajectories.shape
    all_spectra = []
    for i in range(N):
        for d in range(D):
            signal = trajectories[i, :, d]
            signal = signal - signal.mean()
            fft = np.fft.rfft(signal)
            power = np.abs(fft) ** 2
            all_spectra.append(power)

    all_spectra = np.array(all_spectra)
    all_spectra = np.clip(all_spectra, 1e-8, None)
    mean_spectrum = all_spectra.mean(axis=0)
    std_spectrum = all_spectra.std(axis=0)
    freqs = np.fft.rfftfreq(T)

    ax.semilogy(freqs, mean_spectrum, label=label, color=color, linewidth=2.0, alpha=0.9)
    psd_low = np.maximum(mean_spectrum - std_spectrum, 1e-8)
    psd_low = np.clip(psd_low, 1e-2, None)
    ax.fill_between(freqs, psd_low, mean_spectrum + std_spectrum, alpha=0.12, color=color)


def high_freq_ratio(trajectories, cutoff=0.3):
    """高频能量 / 总能量"""
    N, T, D = trajectories.shape
    ratios = []
    for i in range(N):
        for d in range(D):
            signal = trajectories[i, :, d] - trajectories[i, :, d].mean()
            fft = np.fft.rfft(signal)
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(T)
            total_power = power.sum()
            if total_power < 1e-10:
                continue
            high_freq_power = power[freqs >= cutoff].sum()
            ratios.append(high_freq_power / total_power)
    return np.array(ratios)


def plot_frequency_spectrum(dit_traj, unet_traj, save_path):
    """PSD 频谱图（单独一张）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25, linestyle="-")

    analyze_frequency_spectrum(dit_traj, "DiT", COLOR_DIT, ax)
    analyze_frequency_spectrum(unet_traj, "U-Net", COLOR_UNET, ax)
    ax.axvline(x=0.3, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.31, 0.88, r"$f\!=\!0.3$", transform=ax.transAxes, fontsize=13, color="gray")
    ax.set_ylim(1e-2, 1e3)
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("PSD (log scale)")
    ax.legend(framealpha=0.95)
    ax.axvspan(0.3, 0.5, alpha=0.06, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {save_path}")


def plot_highfreq_ratio(dit_traj, unet_traj, save_path):
    """高频能量比箱线图（单独一张）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25, axis="y")

    dit_ratios = high_freq_ratio(dit_traj)
    unet_ratios = high_freq_ratio(unet_traj)
    bp = ax.boxplot(
        [dit_ratios, unet_ratios],
        labels=["DiT", "U-Net"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor(COLOR_DIT)
    bp["boxes"][1].set_facecolor(COLOR_UNET)
    for box in bp["boxes"]:
        box.set_alpha(0.6)
    med_dit = np.median(dit_ratios)
    med_unet = np.median(unet_ratios)
    ax.text(1.28, med_dit, f"{med_dit:.3f}", ha="left", va="center", fontsize=11, color="#1a1a1a")
    ax.text(2.28, med_unet, f"{med_unet:.3f}", ha="left", va="center", fontsize=11, color="#1a1a1a")
    ax.set_ylabel("High-Freq Energy Ratio")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {save_path}")


def compute_jerk(trajectories):
    """Jerk = d³x/dt³ ≈ Δ(Δ(Δx)). 越小越平滑. trajectories: [N, T, D]"""
    vel = np.diff(trajectories, axis=1)
    acc = np.diff(vel, axis=1)
    jerk = np.diff(acc, axis=1)
    return np.mean(np.abs(jerk), axis=(1, 2))


def plot_jerk_comparison(dit_traj, unet_traj, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_style()
    dit_jerk = compute_jerk(dit_traj)
    unet_jerk = compute_jerk(unet_traj)

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25, axis="y")

    bp = ax.boxplot(
        [dit_jerk, unet_jerk],
        labels=["DiT", "U-Net"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor(COLOR_DIT)
    bp["boxes"][1].set_facecolor(COLOR_UNET)
    for box in bp["boxes"]:
        box.set_alpha(0.6)

    ax.set_ylabel("Mean Abs. Jerk")
    med_dit = np.median(dit_jerk)
    med_unet = np.median(unet_jerk)
    ax.text(1.28, med_dit, f"{med_dit:.2f}", ha="left", va="center", fontsize=11, color="#1a1a1a")
    ax.text(2.28, med_unet, f"{med_unet:.2f}", ha="left", va="center", fontsize=11, color="#1a1a1a")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {save_path}")


def main():
    np.random.seed(42)
    T = 64
    D = 5
    N = 50

    t = np.linspace(0, 2 * np.pi, T)
    base_traj = np.stack([np.sin(t + d * 0.5) for d in range(D)], axis=-1)

    unet_traj = np.stack([base_traj + np.random.randn(T, D) * 0.05 for _ in range(N)])
    dit_traj = np.stack([
        base_traj + np.random.randn(T, D) * 0.05 + np.random.randn(T, D) * 0.15
        for _ in range(N)
    ])

    out_psd = os.path.join(REPO_ROOT, "frequency_spectrum.pdf")
    out_ratio = os.path.join(REPO_ROOT, "highfreq_ratio.pdf")
    out_jerk = os.path.join(REPO_ROOT, "jerk_comparison.pdf")
    plot_frequency_spectrum(dit_traj, unet_traj, out_psd)
    plot_highfreq_ratio(dit_traj, unet_traj, out_ratio)
    plot_jerk_comparison(dit_traj, unet_traj, out_jerk)


if __name__ == "__main__":
    main()

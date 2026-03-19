#!/usr/bin/env python3
"""
Plot dual_arm_rl training curves from TensorBoard events.
Usage: python scripts/plot_dual_arm_rl.py [log_dir] [--events 1|2] [--no-show]
  --events 2   use 2nd events file (default when 2+ files exist)
  --no-show    save only, don't display

Requires: pip install tensorboard  (or tensorflow, or tbparse)
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

# Try multiple backends for reading tfevents
def _get_events_files(log_dir):
    """List events files, sorted by size (largest first). 1=largest, 2=2nd largest."""
    files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    return sorted(files, key=lambda f: os.path.getsize(f), reverse=True)

def _get_loader():
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        def load(log_dir, tag, max_steps, events_file=None):
            path = events_file if events_file else log_dir
            ea = EventAccumulator(path)
            ea.Reload()
            if tag not in ea.scalars.Keys():
                return np.array([]), np.array([])
            events = ea.scalars.Items(tag)
            steps = np.array([e.step for e in events])
            values = np.array([e.value for e in events])
            mask = steps <= max_steps
            return steps[mask], values[mask]
        return load
    except ImportError:
        pass
    try:
        from tensorflow.python.summary.summary_iterator import summary_iterator
        def load(log_dir, tag, max_steps, events_file=None):
            steps, values = [], []
            files = [events_file] if events_file else sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
            for f in files:
                if not os.path.isfile(f):
                    continue
                for e in summary_iterator(f):
                    if not e.summary.value:
                        continue
                    for v in e.summary.value:
                        if v.tag == tag:
                            steps.append(e.step)
                            values.append(v.simple_value)
                            break
            steps, values = np.array(steps), np.array(values)
            if len(steps) == 0:
                return np.array([]), np.array([])
            order = np.argsort(steps)
            steps, values = steps[order], values[order]
            mask = steps <= max_steps
            return steps[mask], values[mask]
        return load
    except ImportError:
        pass
    try:
        from tbparse import SummaryReader
        def load(log_dir, tag, max_steps, events_file=None):
            path = events_file if events_file else log_dir
            reader = SummaryReader(path)
            df = reader.scalars
            if df is None or len(df) == 0:
                return np.array([]), np.array([])
            sub = df[df["tag"] == tag]
            if len(sub) == 0:
                return np.array([]), np.array([])
            steps = sub["step"].values
            values = sub["value"].values
            mask = steps <= max_steps
            return steps[mask], values[mask]
        return load
    except ImportError:
        pass
    raise ImportError("Need tensorboard, tensorflow, or tbparse: pip install tensorboard")

# Config
MAX_STEPS = 5_000_000  # 5M
SMOOTH = 0.85  # EMA: y_smooth = SMOOTH * y_prev + (1-SMOOTH) * y_current
TAGS = [
    "main/explained_variance",
    "main/loss_critic_bar",
    "main/ep_reward_mean",
    "main/success_rate",
]
TAG_LABELS = {
    "main/explained_variance": "Explained Variance",
    "main/loss_critic_bar": "Critic Loss",
    "main/ep_reward_mean": "Reward Mean",
    "main/success_rate": "Success Rate",
}


def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average. alpha=0.8 -> 80% from prev, 20% from current."""
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out




def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    default_dir = os.path.join(
        os.path.dirname(__file__), "..",
        "logs/dual_arm_rl/SO-ARM101-Dual-Cube-Stack-RL-v0/env512_epochs3_beta2.0_cEnt0.01_cDelta10.0_adaptEntTrue"
    )
    log_dir = args[0] if args else default_dir
    log_dir = os.path.abspath(log_dir)
    if not os.path.isdir(log_dir):
        print(f"Error: {log_dir} not found")
        sys.exit(1)

    # --events 2: use 2nd events file (1=largest, 2=2nd). Default 1.
    events_files = _get_events_files(log_dir)
    events_index = 1
    if "--events" in sys.argv:
        idx = sys.argv.index("--events")
        if idx + 1 < len(sys.argv) and sys.argv[idx + 1].isdigit():
            events_index = int(sys.argv[idx + 1])
    events_file = events_files[events_index - 1] if events_index <= len(events_files) else None
    if events_file and len(events_files) > 1:
        print(f"Using events file [{events_index}/{len(events_files)}]: {os.path.basename(events_file)}")
    print(f"Loading from: {log_dir}")
    print(f"Max steps: {MAX_STEPS/1e6:.1f}M, Smooth: {SMOOTH}")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 14
    load_scalars = _get_loader()
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    out_paths = []

    for tag, label in zip(TAGS, subplot_labels):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        steps, values = load_scalars(log_dir, tag, MAX_STEPS, events_file=events_file)
        if len(steps) == 0:
            ax.text(0.5, 0.5, f"No data: {tag}", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(TAG_LABELS.get(tag, tag))
        else:
            values_smooth = ema_smooth(values, SMOOTH)
            v_min, v_max = float(values_smooth.min()), float(values_smooth.max())
            if v_max <= v_min:
                v_max = v_min + 0.1
            ax.set_ylim(bottom=v_min, top=v_max * 1.05)
            x = steps / 1e6
            ax.fill_between(x, v_min, values, alpha=0.06, color="gray")  # very faint shadow
            ax.plot(x, values_smooth, color="#5b84c1", linewidth=1.5)
            ax.set_ylabel(TAG_LABELS.get(tag, tag))
            ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="left")
        ax.set_xlabel("Steps (M)")
        ax.grid(True, alpha=0.3, linestyle="-")
        plt.tight_layout()
        base = tag.replace("main/", "").replace("/", "_").replace("_", "-")
        out_path = os.path.join(log_dir, f"plot_{base}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        out_paths.append(out_path)
        plt.close(fig)

    for p in out_paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()

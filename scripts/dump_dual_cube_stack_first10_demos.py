#!/usr/bin/env python3
"""Dump first 10 complete demos from dual cube stack annotated dataset.
Every input is labeled including subsignals (pick_cube, stack_cube).

Usage:
    python scripts/dump_dual_cube_stack_first10_demos.py
    python scripts/dump_dual_cube_stack_first10_demos.py -o datasets/dual_cube_stack_first10_demos_detail.txt
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

OBS_KEYS = [
    "left_joint_pos", "left_joint_vel", "right_joint_pos", "right_joint_vel",
    "left_ee_position", "left_ee_orientation", "right_ee_position", "right_ee_orientation",
    "cube_1_pos", "cube_1_ori", "cube_2_pos", "cube_2_ori", "last_action_all",
]


def main():
    parser = argparse.ArgumentParser(description="Dump first 10 demos with full detail")
    default_path = Path(__file__).resolve().parent.parent / "datasets" / "dual_cube_stack_annotated_dataset.hdf5"
    parser.add_argument("hdf5_path", nargs="?", default=str(default_path))
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument("-n", "--num_demos", type=int, default=10)
    args = parser.parse_args()

    hdf5_path = args.hdf5_path
    if not Path(hdf5_path).exists():
        print(f"Error: File not found: {hdf5_path}")
        return 1

    out_path = args.output or str(Path(hdf5_path).parent / "dual_cube_stack_first10_demos_detail.txt")
    lines = []

    def w(s=""):
        lines.append(s)

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]) if "_" in x else 0,
        )[: args.num_demos]

        w("=" * 100)
        w(f"Dual Cube Stack - First {args.num_demos} Demos - Full Input Detail (every field labeled)")
        w("=" * 100)
        w(f"Dataset: {hdf5_path}")
        w(f"Demos: {demo_keys}")
        w()

        for demo_key in demo_keys:
            demo = f[f"data/{demo_key}"]
            obs_container = demo.get("obs", demo.get("observations", None))
            if not obs_container:
                w(f"[{demo_key}] No obs container")
                continue

            actions = np.array(demo["actions"])
            T = actions.shape[0]

            # Handle 2x obs length (annotated duplication)
            def get_obs_arr(key):
                if key not in obs_container:
                    return None
                arr = np.array(obs_container[key])
                if arr.shape[0] == 2 * T:
                    return arr[::2]
                return arr[:T]

            # Subtask signals from datagen_info
            subtask_order = []
            pick_cube_signal = None
            stack_cube_signal = None
            if "datagen_info" in obs_container:
                dgi = obs_container["datagen_info"]
                if "subtask_term_signals" in dgi:
                    sig_grp = dgi["subtask_term_signals"]
                    subtask_order = list(sig_grp.keys())
                    if "pick_cube" in sig_grp:
                        pick_cube_signal = np.array(sig_grp["pick_cube"])
                        if pick_cube_signal.shape[0] == 2 * T:
                            pick_cube_signal = pick_cube_signal[::2]
                        pick_cube_signal = pick_cube_signal[:T]
                    if "stack_cube" in sig_grp:
                        stack_cube_signal = np.array(sig_grp["stack_cube"])
                        if stack_cube_signal.shape[0] == 2 * T:
                            stack_cube_signal = stack_cube_signal[::2]
                        stack_cube_signal = stack_cube_signal[:T]

            w("")
            w("#" * 100)
            w(f"# {demo_key}  |  T={T} steps  |  subtask_order={subtask_order}")
            w("#" * 100)

            for step in range(T):
                w("")
                w("-" * 80)
                w(f"  [{demo_key}] Step {step} / {T}")
                w("-" * 80)

                # Observations
                w("  [OBSERVATIONS]")
                for key in OBS_KEYS:
                    arr = get_obs_arr(key)
                    if arr is not None:
                        val = arr[step]
                        if val.size <= 12:
                            w(f"    {key}: {val.tolist()}")
                        else:
                            w(f"    {key}: shape={val.shape} {val.flatten()[:16].tolist()}...")

                # Subtask signals (raw from datagen_info)
                w("  [SUBSIGNALS - raw from datagen_info/subtask_term_signals]")
                if pick_cube_signal is not None:
                    pv = pick_cube_signal[step] if pick_cube_signal.ndim > 0 else pick_cube_signal
                    w(f"    pick_cube (done=1): {float(pv):.4f}")
                if stack_cube_signal is not None:
                    sv = stack_cube_signal[step] if stack_cube_signal.ndim > 0 else stack_cube_signal
                    w(f"    stack_cube (done=1): {float(sv):.4f}")

                # Derived subtask_condition (one-hot: active_idx = first False)
                if subtask_order:
                    active_idx = len(subtask_order) - 1
                    for idx, name in enumerate(subtask_order):
                        sig = None
                        if name == "pick_cube" and pick_cube_signal is not None:
                            sig = pick_cube_signal[step]
                        elif name == "stack_cube" and stack_cube_signal is not None:
                            sig = stack_cube_signal[step]
                        if sig is not None and float(sig) < 0.5:
                            active_idx = idx
                            break
                    cond = np.zeros(len(subtask_order), dtype=np.float32)
                    cond[active_idx] = 1.0
                    w("  [SUBTASK_CONDITION - one-hot, active=first not-done]")
                    w(f"    active_idx={active_idx} ({subtask_order[active_idx]})")
                    w(f"    one_hot: {cond.tolist()}")

                # Action
                w("  [ACTION]")
                act = actions[step]
                w(f"    action[0:6] (right arm): {act[:6].tolist()}")
                w(f"    action[6:12] (left arm): {act[6:12].tolist()}")

            w("")

        w("=" * 100)
        w("End of dump")
        w("=" * 100)

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Wrote {len(lines)} lines to {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())

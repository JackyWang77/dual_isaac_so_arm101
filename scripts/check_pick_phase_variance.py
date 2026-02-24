#!/usr/bin/env python3
"""Check right-arm action variance during pick phase (data quality verification).

Usage:
    python scripts/check_pick_phase_variance.py
    python scripts/check_pick_phase_variance.py datasets/dual_cube_stack_annotated_dataset.hdf5
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Check pick-phase action variance")
    default_path = Path(__file__).resolve().parent.parent / "datasets" / "dual_cube_stack_annotated_dataset.hdf5"
    parser.add_argument("hdf5_path", nargs="?", default=str(default_path), help="Path to HDF5 dataset")
    parser.add_argument("--demo", type=int, default=None, help="Demo index (default: random)")
    parser.add_argument("--all", action="store_true", help="Aggregate over all demos")
    args = parser.parse_args()

    if not Path(args.hdf5_path).exists():
        print(f"Error: File not found: {args.hdf5_path}")
        sys.exit(1)

    with h5py.File(args.hdf5_path, "r") as f:
        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
        if not demo_keys:
            print("No demos found")
            sys.exit(1)

        if args.all:
            demo_indices = list(range(len(demo_keys)))
        else:
            demo_idx = args.demo if args.demo is not None else np.random.randint(0, len(demo_keys))
            demo_indices = [demo_idx]

        all_right_pick = []
        all_left_pick = []

        for demo_idx in demo_indices:
            demo_key = demo_keys[demo_idx]
            demo = f[f"data/{demo_key}"]
            actions = np.array(demo["actions"])
            T = actions.shape[0]

            obs_container = demo.get("obs", demo.get("observations", None))
            if not obs_container or "datagen_info" not in obs_container:
                continue
            dgi = obs_container["datagen_info"]
            if "subtask_term_signals" not in dgi:
                continue

            pick_done = np.array(dgi["subtask_term_signals"]["pick_cube"])
            if pick_done.shape[0] == 2 * T:
                pick_done = pick_done[::2]
            pick_done = pick_done[:T].astype(bool)

            pick_phase = actions[~pick_done]
            if len(pick_phase) > 0:
                all_right_pick.append(pick_phase[:, 0:6])
                all_left_pick.append(pick_phase[:, 6:12])

        if not all_right_pick:
            print("No pick-phase data found")
            sys.exit(1)

        right_arm = np.vstack(all_right_pick)
        left_arm = np.vstack(all_left_pick)

        n_pick = len(right_arm)
        n_demos = len(all_right_pick)
        print(f"Dataset: {args.hdf5_path}")
        print(f"Demos: {n_demos}, Total pick-phase steps: {n_pick}")
        print()
        print("Right arm (indices 0:6) variance during pick phase:")
        print(f"  std:  {right_arm.std(axis=0)}")
        print(f"  mean: {right_arm.mean(axis=0)}")
        print()
        print("Left arm (indices 6:12) variance during pick phase:")
        print(f"  std:  {left_arm.std(axis=0)}")
        print(f"  mean: {left_arm.mean(axis=0)}")
        print()
        print("Gripper (index 5 per arm) during pick:")
        print(f"  Right gripper: mean={right_arm[:, 5].mean():.4f}, std={right_arm[:, 5].std():.4f}")
        print(f"  Left gripper:  mean={left_arm[:, 5].mean():.4f}, std={left_arm[:, 5].std():.4f}")
        print()
        print("(Low variance = consistent behavior during pick, model should learn)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze cube position distribution in cube stack HDF5 dataset.

- pick + stack: 都用每个 demo 最后 N 步 (已叠放=目标位置).
- pick target_xy = 叠放位置中心, target_z = 底座 z (桌面).

Usage:
    python scripts/analyze_cube_stack_subtask_params.py datasets/dual_cube_stack_annotated_dataset.hdf5
    python scripts/analyze_cube_stack_subtask_params.py datasets/dual_cube_stack_annotated_dataset.hdf5 --output cube_stack_subtask_params.py
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def analyze(hdf5_path: str, obs_keys: list | None = None, output_path: str | None = None):
    """Analyze cube positions when pick_cube and stack_cube are True."""
    if obs_keys is None:
        obs_keys = [
            "left_joint_pos", "left_joint_vel", "right_joint_pos", "right_joint_vel",
            "left_ee_position", "left_ee_orientation", "right_ee_position", "right_ee_orientation",
            "cube_1_pos", "cube_1_ori", "cube_2_pos", "cube_2_ori", "last_action_all",
        ]

    demo_keys = []
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            print("No 'data' group found")
            return
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]) if "_" in x else 0,
        )

    # pick + stack: 都用每个 demo 最后几步 (已叠放, 那里就是目标位置)
    # pick target = 叠放位置的 xy + 底座 cube 的 z (桌面高度, 证明被放下)
    # stack = 高度差 + xy 对齐
    LAST_N = 5  # 每个 demo 最后 N 步
    pick_xy_list = []
    pick_z_list = []  # 底座 z (min of c1_z, c2_z)
    stack_height_diffs = []
    stack_xy_dists = []

    with h5py.File(hdf5_path, "r") as f:
        for demo_key in demo_keys:
            demo = f[f"data/{demo_key}"]
            obs_container = demo.get("obs", demo.get("observations", None))
            if obs_container is None:
                continue

            if "cube_1_pos" not in obs_container or "cube_2_pos" not in obs_container:
                continue

            c1_pos = np.array(obs_container["cube_1_pos"])
            c2_pos = np.array(obs_container["cube_2_pos"])
            actions = demo["actions"]
            T = actions.shape[0]
            if c1_pos.shape[0] == 2 * T:
                c1_pos = c1_pos[::2]
                c2_pos = c2_pos[::2]
            if c1_pos.shape[0] != T:
                c1_pos = c1_pos[:T]
                c2_pos = c2_pos[:T]

            # 最后 N 步: 已叠放, 那里就是目标位置
            for i in range(max(0, T - LAST_N), T):
                c1 = np.pad(c1_pos[i].flatten(), (0, 3))[:3]
                c2 = np.pad(c2_pos[i].flatten(), (0, 3))[:3]
                # pick target: 叠放位置的 xy (两 cube 中心), 底座 z
                xy_center = (c1[:2] + c2[:2]) / 2
                base_z = min(c1[2], c2[2])
                pick_xy_list.append(xy_center)
                pick_z_list.append(base_z)
                # stack
                stack_height_diffs.append(np.abs(c1[2] - c2[2]))
                stack_xy_dists.append(np.linalg.norm(c1[:2] - c2[:2]))

    # Compute statistics
    print("=" * 70)
    print("Cube Stack Subtask Parameter Analysis")
    print("=" * 70)
    print(f"Dataset: {hdf5_path}")
    print(f"Demos: {len(demo_keys)}")
    print()

    suggested = {}

    # Pick: 最后几步的叠放位置 = 目标位置 (xy + 底座 z)
    if pick_xy_list and pick_z_list:
        pick_xy = np.array(pick_xy_list)
        pick_z = np.array(pick_z_list)
        target_xy = (float(np.median(pick_xy[:, 0])), float(np.median(pick_xy[:, 1])))
        target_z = float(np.median(pick_z))  # 底座 z = 桌面高度
        target_eps_xy = float(np.percentile(np.linalg.norm(pick_xy - target_xy, axis=1), 95))
        target_eps_z = float(np.percentile(np.abs(pick_z - target_z), 95))

        suggested["pick_cube"] = {
            "target_xy": tuple(round(x, 4) for x in target_xy),
            "target_z": round(target_z, 4),
            "target_eps_xy": round(max(target_eps_xy, 0.02), 4),
            "target_eps_z": round(max(target_eps_z, 0.002), 4),
        }

        print("每个 demo 最后 N 步 (已叠放=目标位置) → pick target:")
        print(f"  样本数: {len(pick_xy_list)} (最后{LAST_N}步 × {len(demo_keys)} demos)")
        print(f"  XY: median={target_xy}, eps_xy(95%)= {target_eps_xy:.4f}")
        print(f"  Z:  median={target_z:.4f}, eps_z(95%)= {target_eps_z:.4f}")
        print(f"  建议: target_xy={suggested['pick_cube']['target_xy']}, target_z={suggested['pick_cube']['target_z']}")
        print(f"        target_eps_xy={suggested['pick_cube']['target_eps_xy']}, target_eps_z={suggested['pick_cube']['target_eps_z']}")
    else:
        print("pick_cube: 无数据")
        suggested["pick_cube"] = None

    print()

    # Stack: 同样用最后几步 (已叠放)
    if stack_height_diffs and stack_xy_dists:
        expected_height = float(np.median(stack_height_diffs))
        eps_z = float(np.percentile(np.abs(np.array(stack_height_diffs) - expected_height), 95))
        eps_xy = float(np.percentile(stack_xy_dists, 95))

        suggested["stack_cube"] = {
            "expected_height": round(expected_height, 4),
            "eps_z": round(max(eps_z, 0.001), 4),
            "eps_xy": round(max(eps_xy, 0.005), 4),
        }

        print("每个 demo 最后 N 步 cube 分布 (已叠放):")
        print(f"  样本数: {len(stack_height_diffs)}")
        print(f"  高度差 |c1_z-c2_z|: median={expected_height:.4f}, eps_z(95%)= {eps_z:.4f}")
        print(f"  XY 距离 ||c1_xy-c2_xy||: median={np.median(stack_xy_dists):.4f}, eps_xy(95%)= {eps_xy:.4f}")
        print(f"  建议: expected_height={suggested['stack_cube']['expected_height']}, eps_z={suggested['stack_cube']['eps_z']}, eps_xy={suggested['stack_cube']['eps_xy']}")
    else:
        print("stack_cube: 无数据")
        suggested["stack_cube"] = None

    print()
    print("=" * 70)

    if output_path:
        with open(output_path, "w") as out:
            out.write("# Auto-generated from analyze_cube_stack_subtask_params.py\n")
            out.write("# Paste into CubeStackSubtaskCfg in cube_stack_env_cfg.py\n\n")
            if suggested["pick_cube"]:
                p = suggested["pick_cube"]
                out.write(f'# pick_cube (either_cube_placed_at_target)\n')
                out.write(f'"target_xy": {p["target_xy"]},\n')
                out.write(f'"target_z": {p["target_z"]},\n')
                out.write(f'"target_eps_xy": {p["target_eps_xy"]},\n')
                out.write(f'"target_eps_z": {p["target_eps_z"]},\n')
            if suggested["stack_cube"]:
                s = suggested["stack_cube"]
                out.write(f'\n# stack_cube (two_cubes_stacked_aligned)\n')
                out.write(f'"expected_height": {s["expected_height"]},\n')
                out.write(f'"eps_z": {s["eps_z"]},\n')
                out.write(f'"eps_xy": {s["eps_xy"]},\n')
        print(f"Wrote suggested params to {output_path}")

    return suggested


def main():
    parser = argparse.ArgumentParser(description="Analyze cube stack subtask params from HDF5 data")
    parser.add_argument("hdf5_path", type=str, help="Path to cube stack annotated HDF5")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file for suggested params")
    args = parser.parse_args()

    if not Path(args.hdf5_path).exists():
        print(f"Error: File not found: {args.hdf5_path}")
        sys.exit(1)

    analyze(args.hdf5_path, output_path=args.output)


if __name__ == "__main__":
    main()

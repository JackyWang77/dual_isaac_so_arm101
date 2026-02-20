#!/usr/bin/env python3
"""Merge multiple Isaac Lab HDF5 dataset files into one.

Uses only h5py - no Isaac Lab dependency. Preserves full structure.

Usage:
    python scripts/merge_hdf5_datasets.py \
        datasets/dual_cube_stack_joint_states_mimic_dataset_39.hdf5 \
        datasets/dual_cube_stack_joint_states_mimic_dataset_59.hdf5 \
        datasets/dual_cube_stack_joint_states_mimic_dataset_100.hdf5 \
        -o datasets/dual_cube_stack_joint_states_mimic_dataset.hdf5
"""

import argparse
import os
import sys

import h5py


def copy_group(src_group, dst_group):
    """Recursively copy an HDF5 group (including datasets) to another group."""
    for key in src_group.keys():
        src_item = src_group[key]
        if isinstance(src_item, h5py.Group):
            new_group = dst_group.create_group(key)
            copy_group(src_item, new_group)
        else:
            data = src_item[()]
            dst_group.create_dataset(key, data=data, compression="gzip")


def merge_hdf5_datasets(input_files: list[str], output_file: str):
    """Merge multiple HDF5 dataset files into one.

    Args:
        input_files: List of input HDF5 file paths.
        output_file: Output merged HDF5 file path.
    """
    if not input_files:
        print("Error: No input files specified.")
        sys.exit(1)

    for f in input_files:
        if not os.path.exists(f):
            print(f"Error: Input file not found: {f}")
            sys.exit(1)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    total_episodes = 0
    total_steps = 0

    with h5py.File(output_file, "w") as out_f:
        data_group = out_f.create_group("data")
        data_group.attrs["total"] = 0
        env_args = None

        for inp in input_files:
            with h5py.File(inp, "r") as in_f:
                if "data" not in in_f:
                    print(f"Warning: No 'data' group in {inp}, skipping.")
                    continue

                in_data = in_f["data"]
                if env_args is None and "env_args" in in_data.attrs:
                    env_args = in_data.attrs["env_args"]
                    data_group.attrs["env_args"] = env_args

                demo_keys = sorted(
                    [k for k in in_data.keys() if k.startswith("demo_")],
                    key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0,
                )

                for demo_key in demo_keys:
                    new_key = f"demo_{total_episodes}"
                    src_demo = in_data[demo_key]
                    dst_demo = data_group.create_group(new_key)

                    if "num_samples" in src_demo.attrs:
                        dst_demo.attrs["num_samples"] = src_demo.attrs["num_samples"]
                        total_steps += src_demo.attrs["num_samples"]
                    for attr in ("seed", "success"):
                        if attr in src_demo.attrs:
                            dst_demo.attrs[attr] = src_demo.attrs[attr]

                    copy_group(src_demo, dst_demo)
                    total_episodes += 1

        data_group.attrs["total"] = total_steps

    print(f"Merged {total_episodes} episodes ({total_steps} steps) from {len(input_files)} files -> {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Isaac Lab HDF5 dataset files into one."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input HDF5 dataset files to merge.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output merged HDF5 file path.",
    )
    args = parser.parse_args()

    merge_hdf5_datasets(args.input_files, args.output)


if __name__ == "__main__":
    main()

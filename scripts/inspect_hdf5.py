#!/usr/bin/env python3
"""Simple script to inspect HDF5 dataset structure.

Usage:
    python scripts/inspect_hdf5.py <hdf5_file_path>
    python scripts/inspect_hdf5.py datasets/generated_dataset_ik_abs.hdf5
"""

import sys

import h5py
import numpy as np


def print_structure(name, obj, indent=0, max_depth=5):
    """Recursively print HDF5 file structure."""
    if indent > max_depth:
        return

    prefix = "  " * indent

    if isinstance(obj, h5py.Group):
        print(f"{prefix}📁 {name}/")
        for key in sorted(obj.keys()):
            print_structure(key, obj[key], indent + 1, max_depth)
    elif isinstance(obj, h5py.Dataset):
        shape = obj.shape
        dtype = obj.dtype
        size_mb = obj.nbytes / (1024 * 1024)

        # Check if it looks like image data
        is_image = False
        if len(shape) >= 3:
            # Could be (H, W, C) or (N, H, W, C) or (N, C, H, W)
            if any(dim > 100 for dim in shape[-3:]):
                is_image = True

        icon = "🖼️ " if is_image else "📄"
        print(
            f"{prefix}{icon} {name}: shape={shape}, dtype={dtype}, size={size_mb:.2f}MB"
        )

        # Print sample data for small arrays
        if len(shape) <= 2 and shape[0] <= 5:
            sample = obj[:]
            print(f"{prefix}   Sample: {sample}")


def inspect_hdf5(file_path):
    """Inspect HDF5 file structure."""
    print("=" * 70)
    print(f"Inspecting: {file_path}")
    print("=" * 70)

    try:
        with h5py.File(file_path, "r") as f:
            print_structure("root", f)

            # Summary statistics
            print("\n" + "=" * 70)
            print("Summary:")
            print("=" * 70)

            # Count demos
            if "data" in f:
                demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
                print(f"Number of demonstrations: {len(demo_keys)}")

                if demo_keys:
                    # Get first demo for analysis
                    first_demo = f[f"data/{demo_keys[0]}"]

                    # Check observations
                    if "obs" in first_demo:
                        obs_keys = list(first_demo["obs"].keys())
                        print(f"Observation keys: {obs_keys}")

                    # Check actions
                    if "actions" in first_demo:
                        actions = first_demo["actions"]
                        print(f"Actions shape: {actions.shape}")
                        print(f"Actions dtype: {actions.dtype}")
                        if len(actions) > 0:
                            print(
                                f"Action range: [{actions[:].min():.4f}, {actions[:].max():.4f}]"
                            )

                    # Check for images
                    print("\nChecking for image data...")
                    has_images = False

                    def check_for_images(name, obj):
                        nonlocal has_images
                        if isinstance(obj, h5py.Dataset):
                            if len(obj.shape) >= 3 and any(
                                dim > 100 for dim in obj.shape[-3:]
                            ):
                                has_images = True
                                print(
                                    f"  Found potential image: {name}, shape={obj.shape}"
                                )

                    def walk_group(name, obj):
                        if isinstance(obj, h5py.Group):
                            for key in obj.keys():
                                walk_group(f"{name}/{key}", obj[key])
                        else:
                            check_for_images(name, obj)

                    walk_group("data", f["data"])

                    if not has_images:
                        print("  ❌ No image data found in this dataset")
                        print("  ✅ Dataset contains only state/observation data")

    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_hdf5.py <hdf5_file_path>")
        print("\nExample:")
        print("  python scripts/inspect_hdf5.py datasets/generated_dataset_ik_abs.hdf5")
        sys.exit(1)

    inspect_hdf5(sys.argv[1])

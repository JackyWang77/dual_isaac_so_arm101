#!/usr/bin/env python3
"""Interactive HDF5 Data Browser

A friendly tool to browse and explore HDF5 dataset files.

Usage:
    python hdf5_reader.py <hdf5_file_path>
    python hdf5_reader.py datasets/dataset.hdf5
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def format_size(size_bytes):
    """Format bytes to human readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def print_structure(name, obj, indent=0, max_depth=5, show_samples=True):
    """Recursively print HDF5 file structure."""
    if indent > max_depth:
        return

    prefix = "  " * indent

    if isinstance(obj, h5py.Group):
        print(f"{prefix}📁 {name}/")
        for key in sorted(obj.keys()):
            print_structure(key, obj[key], indent + 1, max_depth, show_samples)
    elif isinstance(obj, h5py.Dataset):
        shape = obj.shape
        dtype = obj.dtype
        size_str = format_size(obj.nbytes)

        # Check if it looks like image data
        is_image = False
        if len(shape) >= 3:
            if any(dim > 100 for dim in shape[-3:]):
                is_image = True

        icon = "🖼️ " if is_image else "📄"
        print(f"{prefix}{icon} {name}: shape={shape}, dtype={dtype}, size={size_str}")

        # Print statistics for numeric data
        if np.issubdtype(dtype, np.number):
            try:
                data = obj[:]
                if data.size > 0:
                    print(
                        f"{prefix}   └─ min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}, std={data.std():.4f}"
                    )
                    if show_samples and data.size <= 100:
                        print(f"{prefix}   └─ sample: {data.flatten()[:10]}")
            except Exception:
                pass


def show_demo_steps(f, demo_key, demo_idx, max_steps=None):
    """Show observation and action for first N steps of a demo.

    Args:
        f: HDF5 file handle
        demo_key: Key of the demo (e.g., 'demo_0')
        demo_idx: Index of the demo (0-based)
        max_steps: Maximum number of steps to show. If None, show all steps.
    """
    demo = f[f"data/{demo_key}"]

    print("\n" + "=" * 80)
    print(f"📊 Demo {demo_idx + 1}: {demo_key}")
    print("=" * 80)

    # Get observations
    obs_container = demo.get("obs", demo.get("observations", None))
    if not obs_container:
        print("⚠️  No observations found")
        return

    # Get actions
    if "actions" not in demo:
        print("⚠️  No actions found")
        return

    actions = demo["actions"]
    if max_steps is None:
        num_steps = actions.shape[0]
        print(f"\nShowing all {num_steps} steps:\n")
    else:
        num_steps = min(max_steps, actions.shape[0])
        print(f"\nShowing first {num_steps} steps:\n")

    # Get all observation keys (exclude 'actions' from obs since we show it separately)
    obs_keys = sorted([k for k in obs_container.keys() if k != "actions"])

    for step in range(num_steps):
        print(f"\nStep {step}:")
        print("  Observation:")
        for obs_key in obs_keys:
            obs_data = obs_container[obs_key]
            if isinstance(obs_data, h5py.Dataset):
                if len(obs_data.shape) == 2 and step < obs_data.shape[0]:
                    # 2D observation (steps, features)
                    obs_values = obs_data[step]
                    # Format array nicely
                    if len(obs_values) <= 10:
                        print(f"    {obs_key}: {obs_values}")
                    else:
                        print(
                            f"    {obs_key}: {obs_values[:10]}... (shape: {obs_values.shape})"
                        )
                elif len(obs_data.shape) == 1 and step < obs_data.shape[0]:
                    # 1D observation
                    print(f"    {obs_key}: {obs_data[step]}")
                elif len(obs_data.shape) > 2:
                    # Higher dimensional, just show shape
                    print(f"    {obs_key}: shape={obs_data.shape}")

        # Show action
        if step < actions.shape[0]:
            action_values = actions[step]
            print(f"  Action: {action_values}")


def browse_demo(f, demo_key, demo_idx, total_demos):
    """Browse a specific demonstration."""
    demo = f[f"data/{demo_key}"]

    print("\n" + "=" * 80)
    print(f"📊 Demo {demo_idx + 1}/{total_demos}: {demo_key}")
    print("=" * 80)

    # Get demo metadata
    if "meta" in demo:
        print("\n📋 Metadata:")
        for key, value in demo["meta"].attrs.items():
            print(f"   {key}: {value}")

    # Check observations
    obs_container = demo.get("obs", demo.get("observations", None))
    if obs_container:
        print(f"\n👁️  Observations ({len(obs_container.keys())} keys):")
        for obs_key in sorted(obs_container.keys()):
            obs_data = obs_container[obs_key]
            if isinstance(obs_data, h5py.Dataset):
                shape = obs_data.shape
                dtype = obs_data.dtype
                size_str = format_size(obs_data.nbytes)
                print(f"   • {obs_key}: shape={shape}, dtype={dtype}, size={size_str}")

                # Show statistics for numeric data
                if np.issubdtype(dtype, np.number) and obs_data.size > 0:
                    try:
                        data = obs_data[:]
                        if data.size < 10000:  # Only for small arrays
                            print(
                                f"     └─ min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}"
                            )
                            if len(shape) == 1 and shape[0] <= 20:
                                print(f"     └─ values: {data}")
                            elif len(shape) == 2 and shape[0] <= 5:
                                print(
                                    f"     └─ first few rows:\n{data[:min(3, shape[0])]}"
                                )
                    except Exception:
                        pass

    # Check actions
    if "actions" in demo:
        actions = demo["actions"]
        print("\n🎮 Actions:")
        print(f"   shape: {actions.shape}")
        print(f"   dtype: {actions.dtype}")
        if actions.size > 0:
            try:
                action_data = actions[:]
                print(f"   range: [{action_data.min():.4f}, {action_data.max():.4f}]")
                print(
                    f"   mean: {action_data.mean():.4f}, std: {action_data.std():.4f}"
                )

                # Show first few actions
                if len(actions.shape) == 2:
                    print("\n   First 5 actions:")
                    for i in range(min(5, actions.shape[0])):
                        print(f"     step {i}: {action_data[i]}")
            except Exception:
                pass

    # Check rewards
    if "rewards" in demo:
        rewards = demo["rewards"]
        print("\n⭐ Rewards:")
        print(f"   shape: {rewards.shape}")
        if rewards.size > 0:
            try:
                reward_data = rewards[:]
                print(f"   range: [{reward_data.min():.4f}, {reward_data.max():.4f}]")
                print(
                    f"   mean: {reward_data.mean():.4f}, sum: {reward_data.sum():.4f}"
                )
            except Exception:
                pass

    # Check for images
    print("\n🖼️  Image Data:")
    has_images = False

    def check_images(name, obj):
        nonlocal has_images
        if isinstance(obj, h5py.Dataset):
            if len(obj.shape) >= 3 and any(dim > 100 for dim in obj.shape[-3:]):
                has_images = True
                print(f"   ✓ Found: {name}, shape={obj.shape}")

    def walk_for_images(name, obj):
        if isinstance(obj, h5py.Group):
            for key in obj.keys():
                walk_for_images(f"{name}/{key}", obj[key])
        else:
            check_images(name, obj)

    walk_for_images("", demo)
    if not has_images:
        print("   ✗ No image data found")

    # Episode length
    if "actions" in demo:
        print(f"\n📏 Episode Length: {demo['actions'].shape[0]} steps")


def main():
    """Main function to browse HDF5 file."""
    if len(sys.argv) < 2:
        print(
            "Usage: python hdf5_reader.py <hdf5_file_path> [--interactive] [--demo N]"
        )
        print("\nExample:")
        print(
            "  python hdf5_reader.py datasets/dataset.hdf5  # Print first 5 demos, 10 steps each"
        )
        print(
            "  python hdf5_reader.py datasets/dataset.hdf5 --interactive  # Interactive mode"
        )
        print(
            "  python hdf5_reader.py datasets/dataset.hdf5 --demo 1  # Show complete demo 1"
        )
        sys.exit(1)

    file_path = sys.argv[1]
    # Default to printing all demos in detail, use --interactive for interactive mode
    print_all = "--interactive" not in sys.argv

    # Check if user wants to show a specific demo
    demo_num = None
    if "--demo" in sys.argv:
        try:
            demo_idx = sys.argv.index("--demo")
            if demo_idx + 1 < len(sys.argv):
                demo_num = int(sys.argv[demo_idx + 1])
        except (ValueError, IndexError):
            print("❌ Error: --demo requires a demo number (e.g., --demo 1)")
            sys.exit(1)

    if not Path(file_path).exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)

    print("=" * 80)
    print("🔍 HDF5 Data Browser")
    print("=" * 80)
    print(f"📁 File: {file_path}")
    print(f"📦 Size: {format_size(Path(file_path).stat().st_size)}")
    print("=" * 80)

    try:
        with h5py.File(file_path, "r") as f:
            # Print file structure
            print("\n📂 File Structure:")
            print("-" * 80)
            print_structure("root", f, max_depth=3, show_samples=False)

            # Summary statistics
            print("\n" + "=" * 80)
            print("📊 Summary Statistics")
            print("=" * 80)

            # Count demos
            if "data" in f:
                # Sort demo keys numerically (demo_0, demo_1, ..., demo_10, ...)
                demo_keys = sorted(
                    [k for k in f["data"].keys() if k.startswith("demo_")],
                    key=lambda x: int(x.split("_")[1]) if "_" in x else 0,
                )
                print(f"\n📈 Total Demonstrations: {len(demo_keys)}")

                if demo_keys:
                    # Analyze all demos
                    total_steps = 0
                    demo_lengths = []

                    for demo_key in demo_keys:
                        demo = f[f"data/{demo_key}"]
                        if "actions" in demo:
                            length = demo["actions"].shape[0]
                            demo_lengths.append(length)
                            total_steps += length

                    if demo_lengths:
                        print(f"📏 Total Steps: {total_steps}")
                        print(
                            f"📏 Average Demo Length: {np.mean(demo_lengths):.1f} steps"
                        )
                        print(f"📏 Min Length: {min(demo_lengths)} steps")
                        print(f"📏 Max Length: {max(demo_lengths)} steps")

                    # Get observation keys from first demo
                    first_demo = f[f"data/{demo_keys[0]}"]
                    obs_container = first_demo.get(
                        "obs", first_demo.get("observations", None)
                    )
                    if obs_container:
                        obs_keys = sorted(list(obs_container.keys()))
                        print(f"\n👁️  Observation Keys ({len(obs_keys)}):")
                        for key in obs_keys:
                            obs_data = obs_container[key]
                            if isinstance(obs_data, h5py.Dataset):
                                print(f"   • {key}: {obs_data.shape}, {obs_data.dtype}")

                    # Check actions
                    if "actions" in first_demo:
                        actions = first_demo["actions"]
                        print("\n🎮 Actions:")
                        print(f"   shape: {actions.shape}")
                        print(f"   dtype: {actions.dtype}")
                        if actions.size > 0:
                            action_data = actions[:]
                            print(
                                f"   dimension: {actions.shape[1] if len(actions.shape) > 1 else 1}"
                            )

                    # Print all demos in detail if --all flag is set
                    if print_all:
                        if demo_num is not None:
                            # Show specific demo completely
                            if 1 <= demo_num <= len(demo_keys):
                                demo_idx = demo_num - 1
                                demo_key = demo_keys[demo_idx]
                                print("\n" + "=" * 80)
                                print(f"📋 Complete Demo {demo_num}: {demo_key}")
                                print("=" * 80)
                                show_demo_steps(f, demo_key, demo_idx, max_steps=None)
                            else:
                                print(
                                    f"❌ Error: Demo {demo_num} not found. Available demos: 1-{len(demo_keys)}"
                                )
                        else:
                            # Show first 5 demos, each with first 10 steps
                            print("\n" + "=" * 80)
                            print("📋 First 5 Demonstrations - First 10 Steps Each")
                            print("=" * 80)
                            for i, demo_key in enumerate(demo_keys[:5]):
                                show_demo_steps(f, demo_key, i, max_steps=10)
                    else:
                        # Interactive browsing
                        print("\n" + "=" * 80)
                        print("🔍 Interactive Browsing")
                        print("=" * 80)
                        print("\nOptions:")
                        print("  1. Browse all demos (summary)")
                        print("  2. Browse specific demo (detailed)")
                        print("  3. Exit")
                        print(
                            "\n💡 Tip: Use --all flag to print all demos in detail without interaction"
                        )

                        while True:
                            try:
                                choice = input("\nEnter choice (1-3): ").strip()

                                if choice == "1":
                                    print("\n" + "=" * 80)
                                    print("📋 All Demonstrations Summary")
                                    print("=" * 80)
                                    for i, demo_key in enumerate(demo_keys):
                                        demo = f[f"data/{demo_key}"]
                                        length = (
                                            demo["actions"].shape[0]
                                            if "actions" in demo
                                            else 0
                                        )
                                        print(f"\n  Demo {i+1}: {demo_key}")
                                        print(f"    Length: {length} steps")
                                        if "meta" in demo:
                                            meta_items = list(
                                                demo["meta"].attrs.items()
                                            )
                                            if meta_items:
                                                print(
                                                    f"    Metadata: {dict(meta_items)}"
                                                )

                                elif choice == "2":
                                    print(f"\nAvailable demos (1-{len(demo_keys)}):")
                                    demo_num = input(
                                        f"Enter demo number (1-{len(demo_keys)}): "
                                    ).strip()
                                    try:
                                        idx = int(demo_num) - 1
                                        if 0 <= idx < len(demo_keys):
                                            browse_demo(
                                                f, demo_keys[idx], idx, len(demo_keys)
                                            )
                                        else:
                                            print("❌ Invalid demo number")
                                    except ValueError:
                                        print("❌ Please enter a valid number")

                                elif choice == "3":
                                    print("\n👋 Goodbye!")
                                    break

                                else:
                                    print("❌ Invalid choice. Please enter 1, 2, or 3.")

                            except KeyboardInterrupt:
                                print("\n\n👋 Goodbye!")
                                break
                            except EOFError:
                                print("\n\n👋 Goodbye!")
                                break
                else:
                    print("\n⚠️  No demonstrations found in 'data' group")
            else:
                print("\n⚠️  No 'data' group found in file")
                print("\nFull structure:")
                print_structure("root", f, max_depth=10)

    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

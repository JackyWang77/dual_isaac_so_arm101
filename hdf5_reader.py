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


def print_structure(name, obj, output_file, indent=0, max_depth=5, show_samples=True):
    """Recursively print HDF5 file structure."""
    if indent > max_depth:
        return

    prefix = "  " * indent

    if isinstance(obj, h5py.Group):
        output_file.write(f"{prefix}📁 {name}/\n")
        for key in sorted(obj.keys()):
            print_structure(key, obj[key], output_file, indent + 1, max_depth, show_samples)
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
        output_file.write(f"{prefix}{icon} {name}: shape={shape}, dtype={dtype}, size={size_str}\n")

        # Print statistics for numeric data
        if np.issubdtype(dtype, np.number):
            try:
                data = obj[:]
                if data.size > 0:
                    output_file.write(
                        f"{prefix}   └─ min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}, std={data.std():.4f}\n"
                    )
                    if show_samples and data.size <= 100:
                        output_file.write(f"{prefix}   └─ sample: {data.flatten()[:10]}\n")
            except Exception:
                pass


def show_demo_steps(f, demo_key, demo_idx, output_file, max_steps=None):
    """Show observation and action for first N steps of a demo.

    Args:
        f: HDF5 file handle
        demo_key: Key of the demo (e.g., 'demo_0')
        demo_idx: Index of the demo (0-based)
        output_file: File handle to write output to
        max_steps: Maximum number of steps to show. If None, show all steps.
    """
    demo = f[f"data/{demo_key}"]

    output_file.write("\n" + "=" * 80 + "\n")
    output_file.write(f"📊 Demo {demo_idx + 1}: {demo_key}\n")
    output_file.write("=" * 80 + "\n")

    # Get observations
    obs_container = demo.get("obs", demo.get("observations", None))
    if not obs_container:
        output_file.write("⚠️  No observations found\n")
        return

    # Get actions
    if "actions" not in demo:
        output_file.write("⚠️  No actions found\n")
        return

    actions = demo["actions"]
    if max_steps is None:
        num_steps = actions.shape[0]
        output_file.write(f"\nShowing all {num_steps} steps:\n\n")
    else:
        num_steps = min(max_steps, actions.shape[0])
        output_file.write(f"\nShowing first {num_steps} steps:\n\n")

    # Get all observation keys (exclude 'actions' from obs since we show it separately)
    obs_keys = sorted([k for k in obs_container.keys() if k != "actions"])

    for step in range(num_steps):
        output_file.write(f"\nStep {step}:\n")
        output_file.write("  Observation:\n")
        for obs_key in obs_keys:
            obs_data = obs_container[obs_key]
            if isinstance(obs_data, h5py.Dataset):
                if len(obs_data.shape) == 2 and step < obs_data.shape[0]:
                    # 2D observation (steps, features)
                    obs_values = obs_data[step]
                    # Format array nicely
                    if len(obs_values) <= 10:
                        output_file.write(f"    {obs_key}: {obs_values}\n")
                    else:
                        output_file.write(
                            f"    {obs_key}: {obs_values[:10]}... (shape: {obs_values.shape})\n"
                        )
                elif len(obs_data.shape) == 1 and step < obs_data.shape[0]:
                    # 1D observation
                    output_file.write(f"    {obs_key}: {obs_data[step]}\n")
                elif len(obs_data.shape) > 2:
                    # Higher dimensional, just show shape
                    output_file.write(f"    {obs_key}: shape={obs_data.shape}\n")

        # Show action
        if step < actions.shape[0]:
            action_values = actions[step]
            output_file.write(f"  Action: {action_values}\n")


def browse_demo(f, demo_key, demo_idx, total_demos, output_file):
    """Browse a specific demonstration."""
    demo = f[f"data/{demo_key}"]

    output_file.write("\n" + "=" * 80 + "\n")
    output_file.write(f"📊 Demo {demo_idx + 1}/{total_demos}: {demo_key}\n")
    output_file.write("=" * 80 + "\n")

    # Get demo metadata
    if "meta" in demo:
        output_file.write("\n📋 Metadata:\n")
        for key, value in demo["meta"].attrs.items():
            output_file.write(f"   {key}: {value}\n")

    # Check observations
    obs_container = demo.get("obs", demo.get("observations", None))
    if obs_container:
        output_file.write(f"\n👁️  Observations ({len(obs_container.keys())} keys):\n")
        for obs_key in sorted(obs_container.keys()):
            obs_data = obs_container[obs_key]
            if isinstance(obs_data, h5py.Dataset):
                shape = obs_data.shape
                dtype = obs_data.dtype
                size_str = format_size(obs_data.nbytes)
                output_file.write(f"   • {obs_key}: shape={shape}, dtype={dtype}, size={size_str}\n")

                # Show statistics for numeric data
                if np.issubdtype(dtype, np.number) and obs_data.size > 0:
                    try:
                        data = obs_data[:]
                        if data.size < 10000:  # Only for small arrays
                            output_file.write(
                                f"     └─ min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}\n"
                            )
                            if len(shape) == 1 and shape[0] <= 20:
                                output_file.write(f"     └─ values: {data}\n")
                            elif len(shape) == 2 and shape[0] <= 5:
                                output_file.write(
                                    f"     └─ first few rows:\n{data[:min(3, shape[0])]}\n"
                                )
                    except Exception:
                        pass

    # Check actions
    if "actions" in demo:
        actions = demo["actions"]
        output_file.write("\n🎮 Actions:\n")
        output_file.write(f"   shape: {actions.shape}\n")
        output_file.write(f"   dtype: {actions.dtype}\n")
        if actions.size > 0:
            try:
                action_data = actions[:]
                output_file.write(f"   range: [{action_data.min():.4f}, {action_data.max():.4f}]\n")
                output_file.write(
                    f"   mean: {action_data.mean():.4f}, std: {action_data.std():.4f}\n"
                )

                # Show first few actions
                if len(actions.shape) == 2:
                    output_file.write("\n   First 5 actions:\n")
                    for i in range(min(5, actions.shape[0])):
                        output_file.write(f"     step {i}: {action_data[i]}\n")
            except Exception:
                pass

    # Check rewards
    if "rewards" in demo:
        rewards = demo["rewards"]
        output_file.write("\n⭐ Rewards:\n")
        output_file.write(f"   shape: {rewards.shape}\n")
        if rewards.size > 0:
            try:
                reward_data = rewards[:]
                output_file.write(f"   range: [{reward_data.min():.4f}, {reward_data.max():.4f}]\n")
                output_file.write(
                    f"   mean: {reward_data.mean():.4f}, sum: {reward_data.sum():.4f}\n"
                )
            except Exception:
                pass

    # Check for images
    output_file.write("\n🖼️  Image Data:\n")
    has_images = False

    def check_images(name, obj):
        nonlocal has_images
        if isinstance(obj, h5py.Dataset):
            if len(obj.shape) >= 3 and any(dim > 100 for dim in obj.shape[-3:]):
                has_images = True
                output_file.write(f"   ✓ Found: {name}, shape={obj.shape}\n")

    def walk_for_images(name, obj):
        if isinstance(obj, h5py.Group):
            for key in obj.keys():
                walk_for_images(f"{name}/{key}", obj[key])
        else:
            check_images(name, obj)

    walk_for_images("", demo)
    if not has_images:
        output_file.write("   ✗ No image data found\n")

    # Episode length
    if "actions" in demo:
        output_file.write(f"\n📏 Episode Length: {demo['actions'].shape[0]} steps\n")


def main():
    """Main function to browse HDF5 file."""
    if len(sys.argv) < 2:
        print(
            "Usage: python hdf5_reader.py <hdf5_file_path> [--interactive] [--demo N] [--output OUTPUT_FILE]"
        )
        print("\nExample:")
        print(
            "  python hdf5_reader.py datasets/dataset.hdf5  # Save to dataset_info.txt"
        )
        print(
            "  python hdf5_reader.py datasets/dataset.hdf5 --interactive  # Interactive mode (prints to console)"
        )
        print(
            "  python hdf5_reader.py datasets/dataset.hdf5 --demo 1  # Show complete demo 1"
        )
        print(
            "  python hdf5_reader.py datasets/dataset.hdf5 --output custom_output.txt  # Custom output file"
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

    # Check for custom output file
    output_file_path = None
    if "--output" in sys.argv:
        try:
            output_idx = sys.argv.index("--output")
            if output_idx + 1 < len(sys.argv):
                output_file_path = sys.argv[output_idx + 1]
        except (ValueError, IndexError):
            print("❌ Error: --output requires a file path (e.g., --output output.txt)")
            sys.exit(1)

    if not Path(file_path).exists():
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)

    # Generate output file path if not specified
    if output_file_path is None and not print_all:
        # Interactive mode: print to console
        output_file = None
    else:
        # Non-interactive mode: save to file
        if output_file_path is None:
            # Auto-generate output filename based on input file
            input_path = Path(file_path)
            output_file_path = input_path.parent / f"{input_path.stem}_info.txt"
        
        output_file = open(output_file_path, "w", encoding="utf-8")
        print(f"📝 Writing output to: {output_file_path}")
    
    # Helper class for console output in interactive mode
    class ConsoleOutput:
        def write(self, text):
            print(text, end="")

    try:
        if output_file:
            output_file.write("=" * 80 + "\n")
            output_file.write("🔍 HDF5 Data Browser\n")
            output_file.write("=" * 80 + "\n")
            output_file.write(f"📁 File: {file_path}\n")
            output_file.write(f"📦 Size: {format_size(Path(file_path).stat().st_size)}\n")
            output_file.write("=" * 80 + "\n")
        else:
            # Interactive mode: print to console
    print("=" * 80)
    print("🔍 HDF5 Data Browser")
    print("=" * 80)
    print(f"📁 File: {file_path}")
    print(f"📦 Size: {format_size(Path(file_path).stat().st_size)}")
    print("=" * 80)
        with h5py.File(file_path, "r") as f:
            # Print file structure
            if output_file:
                output_file.write("\n📂 File Structure:\n")
                output_file.write("-" * 80 + "\n")
                print_structure("root", f, output_file, max_depth=3, show_samples=False)
            else:
            print("\n📂 File Structure:")
            print("-" * 80)
                # For interactive mode, we still need to print to console
                # Create a wrapper that prints to console
                class ConsoleOutput:
                    def write(self, text):
                        print(text, end="")
                console_output = ConsoleOutput()
                print_structure("root", f, console_output, max_depth=3, show_samples=False)

            # Summary statistics
            if output_file:
                output_file.write("\n" + "=" * 80 + "\n")
                output_file.write("📊 Summary Statistics\n")
                output_file.write("=" * 80 + "\n")
            else:
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
                if output_file:
                    output_file.write(f"\n📈 Total Demonstrations: {len(demo_keys)}\n")
                else:
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
                        if output_file:
                            output_file.write(f"📏 Total Steps: {total_steps}\n")
                            output_file.write(
                                f"📏 Average Demo Length: {np.mean(demo_lengths):.1f} steps\n"
                            )
                            output_file.write(f"📏 Min Length: {min(demo_lengths)} steps\n")
                            output_file.write(f"📏 Max Length: {max(demo_lengths)} steps\n")
                        else:
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
                        if output_file:
                            output_file.write(f"\n👁️  Observation Keys ({len(obs_keys)}):\n")
                            for key in obs_keys:
                                obs_data = obs_container[key]
                                if isinstance(obs_data, h5py.Dataset):
                                    output_file.write(f"   • {key}: {obs_data.shape}, {obs_data.dtype}\n")
                        else:
                        print(f"\n👁️  Observation Keys ({len(obs_keys)}):")
                        for key in obs_keys:
                            obs_data = obs_container[key]
                            if isinstance(obs_data, h5py.Dataset):
                                print(f"   • {key}: {obs_data.shape}, {obs_data.dtype}")

                    # Check actions
                    if "actions" in first_demo:
                        actions = first_demo["actions"]
                        if output_file:
                            output_file.write("\n🎮 Actions:\n")
                            output_file.write(f"   shape: {actions.shape}\n")
                            output_file.write(f"   dtype: {actions.dtype}\n")
                            if actions.size > 0:
                                action_data = actions[:]
                                output_file.write(
                                    f"   dimension: {actions.shape[1] if len(actions.shape) > 1 else 1}\n"
                                )
                        else:
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
                                if output_file:
                                    output_file.write("\n" + "=" * 80 + "\n")
                                    output_file.write(f"📋 Complete Demo {demo_num}: {demo_key}\n")
                                    output_file.write("=" * 80 + "\n")
                                    show_demo_steps(f, demo_key, demo_idx, output_file, max_steps=None)
                                else:
                                print("\n" + "=" * 80)
                                print(f"📋 Complete Demo {demo_num}: {demo_key}")
                                print("=" * 80)
                                    show_demo_steps(f, demo_key, demo_idx, ConsoleOutput(), max_steps=None)
                            else:
                                error_msg = f"❌ Error: Demo {demo_num} not found. Available demos: 1-{len(demo_keys)}"
                                if output_file:
                                    output_file.write(error_msg + "\n")
                                else:
                                    print(error_msg)
                        else:
                            # Show first 5 demos, each with first 10 steps
                            if output_file:
                                output_file.write("\n" + "=" * 80 + "\n")
                                output_file.write("📋 First 5 Demonstrations - First 10 Steps Each\n")
                                output_file.write("=" * 80 + "\n")
                                for i, demo_key in enumerate(demo_keys[:5]):
                                    show_demo_steps(f, demo_key, i, output_file, max_steps=10)
                            else:
                            print("\n" + "=" * 80)
                            print("📋 First 5 Demonstrations - First 10 Steps Each")
                            print("=" * 80)
                            for i, demo_key in enumerate(demo_keys[:5]):
                                    show_demo_steps(f, demo_key, i, ConsoleOutput(), max_steps=10)
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
                                            # Interactive mode: print to console
                                            class ConsoleOutput:
                                                def write(self, text):
                                                    print(text, end="")
                                            browse_demo(
                                                f, demo_keys[idx], idx, len(demo_keys), ConsoleOutput()
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
                    msg = "\n⚠️  No demonstrations found in 'data' group"
                    if output_file:
                        output_file.write(msg + "\n")
                    else:
                        print(msg)
            else:
                msg = "\n⚠️  No 'data' group found in file"
                if output_file:
                    output_file.write(msg + "\n")
                    output_file.write("\nFull structure:\n")
                    print_structure("root", f, output_file, max_depth=10)
                else:
                    print(msg)
                print("\nFull structure:")
                    class ConsoleOutput:
                        def write(self, text):
                            print(text, end="")
                    print_structure("root", f, ConsoleOutput(), max_depth=10)

    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if output_file:
            output_file.close()
            print(f"✅ Output saved to: {output_file_path}")


if __name__ == "__main__":
    main()

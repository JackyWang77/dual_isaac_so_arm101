# SO-ARM101 Dual Arm Manipulation with Graph DiT Policy

A robotic manipulation framework built on **Isaac Lab** that combines **Graph Neural Networks**, **Diffusion Transformers**, and **Reinforcement Learning** to train SO-ARM101 dual-arm robots for complex manipulation tasks.

## Overview

This project implements a **Graph Diffusion Transformer (Graph DiT)** policy for learning manipulation skills from demonstrations, with optional RL fine-tuning. The framework supports multiple manipulation tasks including lifting, pick-and-place, cube stacking, and table setting — all running in NVIDIA Isaac Lab simulation with massively parallel environments.

## Architecture

### Graph DiT Policy

The core policy models manipulation as a **graph-based structure**, where spatial relationships between the end-effector and objects are explicitly represented.

```
Observations (32D)
       │
       ▼
┌──────────────────────────────┐
│  Graph Construction          │
│  ├─ EE Node (7D: pos+quat)  │
│  ├─ Object Node (7D)        │
│  └─ Edge (2D: dist+orient)  │
└──────────┬───────────────────┘
           │
           ▼  × N layers
┌──────────────────────────────┐
│  Graph DiT Unit              │
│  1. Action Self-Attention    │  ← temporal action history
│  2. Graph Attention + Edge   │  ← spatial EE-object reasoning
│  3. Cross-Attention          │  ← fuse action & scene info
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Flow Matching Diffusion     │  ← 1-10 steps (real-time)
│  or DDPM (50-100 steps)      │
└──────────┬───────────────────┘
           │
           ▼
      Actions (6D)
```

**Key design choices:**
- **Graph nodes**: EE and object, each with 7D features (position + quaternion)
- **Graph edges**: Euclidean distance + orientation similarity (2D)
- **Temporal history**: Configurable window (default 10 steps) for capturing motion patterns
- **Flow Matching**: Recommended for real-time inference (1-10 diffusion steps, 10x faster than DDPM)

### Alternative Policies

| Policy | Description |
|--------|-------------|
| `graph_dit_policy.py` | Graph DiT with flow matching / DDPM (primary) |
| `graph_unet_policy.py` | Graph encoder + U-Net 1D backbone |
| `graph_unet_residual_rl_policy.py` | Graph U-Net with residual RL fine-tuning |
| `dual_arm_unet_policy.py` | Separate left/right arm networks |

## Tasks

Built-in Isaac Lab environments for SO-ARM101:

| Task | Description | Variants |
|------|-------------|----------|
| **Reach** | End-effector reaching to target poses | IK absolute, joint position |
| **Lift** | Single-arm object lifting | Joint position, IK |
| **Pick & Place** | Object pick-and-place manipulation | 12 variants (IK/joint, single/dual-arm, mimic) |
| **Cube Stack** | Dual-arm cube stacking | IK relative, joint states, mimic |
| **Table Setting** | Dual-arm table arrangement | Joint states, mimic |

All tasks support **mimic** (learning from demonstrations) and **RL** training modes.

## Training Pipeline

### Stage 1: Supervised Learning from Demonstrations

Train the Graph DiT backbone on collected demonstration data (HDF5 format):

```bash
bash train_graph_dit.sh flow_matching
```

**Configuration:**
- Dataset: HDF5 with observation-action trajectories
- Prediction horizon: 20 steps, execution horizon: 10 steps
- Action history: 10 steps
- Architecture: hidden_dim=64, num_layers=2, num_heads=4
- Training: 500 epochs, batch_size=32, lr=3e-4

### Stage 2: RL Fine-tuning (Optional)

Fine-tune with PPO using a **head-only** strategy — the Graph DiT/U-Net backbone is frozen, and only a small RL head is trained:

```bash
bash train_residual_rl.sh <pretrained_checkpoint> [num_envs] [max_iterations]
```

```
Frozen Backbone (78M params)     Trainable RL Head (~400K params, <1%)
┌─────────────────────────┐     ┌──────────────────────────────┐
│ Graph DiT / U-Net       │────▶│ Residual Actor (MLP)         │
│ (no gradients)          │     │ Z Adapter                    │
└─────────────────────────┘     │ Value Critic (MLP)           │
                                └──────────────────────────────┘
```

**Features:**
- PPO via RSL-RL framework
- 512-4096 parallel environments
- Expert Jacobian intervention (DAgger-style)
- Adaptive regularization

### Evaluation

```bash
bash play_graph_dit_rl.sh <rl_checkpoint> [pretrained_checkpoint] [task]
```

Auto-detects the latest Graph DiT checkpoint if not specified.

## Project Structure

```
├── source/SO_101/SO_101/
│   ├── policies/           # Graph DiT, U-Net, dual-arm policies
│   ├── tasks/              # Reach, Lift, Pick-Place, Cube Stack, Table Setting
│   ├── robots/             # SO-ARM101 robot configurations
│   └── devices/            # Device drivers
├── scripts/
│   ├── graph_dit/          # Supervised training scripts
│   ├── graph_dit_rl/       # RL fine-tuning scripts
│   └── *.py                # Analysis, visualization, dataset tools
├── datasets/               # HDF5 demonstration datasets
├── docs/                   # Technical documentation
├── train_graph_dit.sh      # Stage 1: supervised learning
├── train_residual_rl.sh    # Stage 2: RL fine-tuning
├── play_graph_dit_rl.sh    # Evaluation / playback
└── test_*.sh               # Task-specific test scripts
```

## Quick Start

### Prerequisites

- **NVIDIA Isaac Lab** (Isaac Sim)
- **Python 3.10+**
- **PyTorch** with CUDA
- **Git LFS** (for large dataset files)

### Installation

```bash
git clone https://github.com/JackyWang77/dual_isaac_so_arm101.git
cd dual_isaac_so_arm101
pip install -e source/SO_101
```

### Train from Demonstrations

```bash
# Prepare demonstration dataset in HDF5 format
# Train Graph DiT with Flow Matching
bash train_graph_dit.sh flow_matching
```

### Fine-tune with RL

```bash
bash train_residual_rl.sh ./logs/graph_dit/lift_joint/best_model.pt 512 500
```

### Run Trained Policy

```bash
bash play_graph_dit_rl.sh ./logs/gr_dit/.../policy_iter_300.pt
```

### Record Demonstrations

```bash
python scripts/record_demos.py --task SO-ARM101-Lift-Cube-v0 --teleop_device keyboard
```

## Analysis & Visualization

The `scripts/` directory includes tools for:

- **Attention visualization**: `visualize_attention.py`, `plot_attention_heatmaps.py`, `plot_attention_animation.py`
- **Frequency analysis**: `visualize_frequency_analysis.py`
- **Input sensitivity**: `visualize_input_sensitivity.py`
- **Loss landscape**: `visualize_loss_landscape_compare.py`
- **Dataset inspection**: `inspect_hdf5.py`, `inspect_checkpoint.py`
- **RL training curves**: `plot_dual_arm_rl.py`

## Documentation

- [`docs/disentangled_graph_flow.md`](docs/disentangled_graph_flow.md) — Disentangled graph attention technical flow
- [`docs/train_play_input_alignment.md`](docs/train_play_input_alignment.md) — Input alignment strategies
- [`docs/play_reset_stuck_analysis.md`](docs/play_reset_stuck_analysis.md) — Troubleshooting stuck states during playback

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Requires Git LFS and pre-commit hooks.

## License

BSD-3-Clause

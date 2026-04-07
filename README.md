# Graph-Based Diffusion Policy with Residual Reinforcement Learning for Dual-Arm Manipulation

This repository implements the **Graph-Conditioned Dual-Frequency Framework** for dual-arm robotic manipulation, built on NVIDIA Isaac Lab. The framework explicitly decouples **low-frequency trajectory planning** (behavior cloning with a Graph U-Net) from **high-frequency reactive correction** (residual RL with Counterfactual AWR), resolving the inherent temporal mismatch between long-horizon reasoning and closed-loop execution.

> MSc Thesis — Eindhoven University of Technology, Department of Electrical Engineering
> In collaboration with Unseq B.V.

## Method Overview

```
Stage 1: Behavior Cloning                    Stage 2: Residual RL Fine-tuning

  Teleoperation Data                           512 Parallel Environments (Isaac Lab)
        │                                              │
        ▼                                              ▼
┌─────────────────────────┐              ┌──────────────────────────────────┐
│  Spatio-Temporal Graph  │              │  Frozen Graph U-Net Backbone     │
│  Encoder                │              │  → base trajectory (a_base)      │
│  ├─ EE nodes (7D)       │              └──────────────┬───────────────────┘
│  ├─ Object nodes (7D)   │                             │
│  └─ Edges (dist+orient) │                             ▼
└──────────┬──────────────┘              ┌──────────────────────────────────┐
           │                             │  Residual RL Agent               │
           ▼                             │  ├─ Z-Adapter (near-identity)    │
┌─────────────────────────┐              │  ├─ Counterfactual AWR Actor     │
│  1D Convolutional U-Net │              │  ├─ Expectile Critic             │
│  (intrinsic low-pass    │              │  └─ Expert DAgger (training only)│
│   filter)               │              └──────────────┬───────────────────┘
└──────────┬──────────────┘                             │
           │                                            ▼
           ▼                                   a = a_base + ω * ω_t
   Flow Matching (K=15)                        (bounded high-freq correction)
   → base trajectory
```

### Key Contributions

1. **Spatio-temporal graph encoder** with edge-conditioned modulation — preserves 3D geometric topology that flat-vector representations destroy. Learnable graph gate converges to 0.4-0.7, confirming complementary geometric reasoning.

2. **Frequency-domain analysis** of diffusion architectures — demonstrates mathematically and empirically that 1D convolutional U-Nets act as intrinsic low-pass filters, generating smoother, hardware-safe trajectories compared to Diffusion Transformers (DiT).

3. **Residual RL with Counterfactual AWR** — resolves credit assignment flaws in standard value baselines. Near-identity Z-Adapter and hard L2 projection (ω_max = 0.25) ensure the residual operates strictly as a local corrector without destroying the pre-trained kinematic prior.

4. **Automated expert intervention (DAgger)** — Jacobian-based geometric expert provides zero-cost corrective anchors in simulation, bypassing the scalability bottleneck of human teleoperation. Expert ratio decays from 1.0 to 0.0 during training.

## Architecture Details

### Graph Encoder

- **Nodes**: End-effector and object entities, each with 7D pose (position + quaternion)
- **Edges**: Translational proximity (L2 distance) + rotational alignment (quaternion inner product)
- **Edge-conditioned modulation**: FiLM-style modulation of Value vectors before graph attention
- **Temporal aggregation**: GRU + learnable query token over history window

### Diffusion Backbone (Graph U-Net)

- 1D convolutional U-Net with FiLM conditioning from graph latent z
- Flow Matching with K=15 inference steps
- Receding Horizon Control: execute T_p/2 steps, then re-plan
- Acts as natural low-pass filter — smoother trajectories than DiT (verified via PSD analysis)

### Residual RL Agent

- **Z-Adapter**: Near-identity transform (frozen z → RL-friendly z), prevents catastrophic forgetting
- **Counterfactual AWR**: Advantage computed against Q(s, a_base) instead of V(s), isolating residual contribution
- **Hard clamp**: ω_max = 0.25 for arm joints, ω = 0.0 for gripper (separate binary head)
- **Expectile critic** (τ = 0.7) with log-compressed advantages
- **Adaptive temperature**: Maintains target ESR of 0.4 to prevent mode collapse

### Expert Intervention (Training Only)

- **Stacking funnel**: Jacobian-based IK expert activates within spatial proximity (xy_error 0.5-10mm)
- **Two-phase strategy**: Align XY at hover height (25mm), then descend to stack height (19mm)
- **Gripper expert**: Triggers release when stacking geometry is satisfied (Δz ∈ 16-22mm, Δp_xy < 3mm)
- **DAgger decay**: Expert ratio decays from 1.0 → 0.0 with ρ = 0.95

## Tasks

Evaluated in Isaac Lab with 512 parallel environments on two SO-ARM101 robots (5-DOF arm + 1-DOF gripper each, 12-DOF composite action space):

| Task | Arms | Demos | Description |
|------|------|-------|-------------|
| **Single-Arm Pick** | 1 | — | Backbone architecture benchmark (U-Net vs DiT) |
| **Table Pick & Place** | 2 | 177 | Dual-arm fork/knife manipulation |
| **Dual Stack** | 2 | 300 | Contact-rich cube stacking with millimeter precision |

### Results (1000 eval episodes, 5 seeds)

| Method | Dual Pick | Dual Stack | Novel Spawns (Pick) | Novel Spawns (Stack) |
|--------|-----------|------------|---------------------|----------------------|
| MLP-UNet | 92.6% | 44.2% | 27.2% | 5.8% |
| Graph-UNet | **96.7%** | **58.0%** | **35.7%** | **8.5%** |
| + Residual RL + Expert | — | **62.3%** | — | — |

## Training Pipeline

### Stage 1: Behavior Cloning

Train the Graph U-Net backbone on teleoperation demonstrations (HDF5):

```bash
bash train_graph_dit.sh flow_matching
```

Key hyperparameters (see Appendix Table IV in thesis):
- Hidden dim: 32, GAT layers: 1, heads: 4
- U-Net channels: (256, 512, 1024), kernel size: 5
- Batch size: 16 (demo-level), epochs: 1000, lr: 3e-4

### Stage 2: Residual RL Fine-tuning

Fine-tune with Counterfactual AWR in 512 parallel environments:

```bash
bash train_residual_rl.sh <pretrained_checkpoint> [num_envs] [max_iterations]
```

Key hyperparameters (see Appendix Table V in thesis):
- Actor/Critic hidden: (256, 256), Z-Adapter: (256,)
- 512 envs, 405 steps/env, 3 PPO epochs/iteration
- Critic warmup: 5 iterations, max iterations: 25
- Expert decay: 1.0 → 0.0 with ρ = 0.95

### Evaluation

```bash
bash play_graph_dit_rl.sh <rl_checkpoint> [pretrained_checkpoint] [task]
```

## Project Structure

```
├── source/SO_101/SO_101/
│   ├── policies/           # Graph U-Net, Graph DiT, residual RL policies
│   ├── tasks/              # Reach, Lift, Pick-Place, Cube Stack, Table Setting
│   ├── robots/             # SO-ARM101 configurations (dual-arm)
│   └── devices/            # Device drivers
├── scripts/
│   ├── graph_dit/          # Stage 1: supervised training
│   ├── graph_dit_rl/       # Stage 2: residual RL fine-tuning
│   └── *.py                # Analysis & visualization tools
├── datasets/               # HDF5 teleoperation demonstrations
├── docs/                   # Technical documentation
├── train_graph_dit.sh      # Stage 1 entry point
├── train_residual_rl.sh    # Stage 2 entry point
└── play_graph_dit_rl.sh    # Evaluation / playback
```

## Analysis & Visualization

```bash
# Frequency spectrum & high-freq energy ratio
python scripts/visualize_frequency_analysis.py

# Graph attention heatmaps across task phases
python scripts/plot_attention_heatmaps.py

# Loss landscape comparison (DiT vs U-Net)
python scripts/visualize_loss_landscape_compare.py

# Input sensitivity under sensor noise
python scripts/visualize_input_sensitivity.py

# RL training curves (reward, success rate, critic loss)
python scripts/plot_dual_arm_rl.py
```

## Quick Start

### Prerequisites

- NVIDIA Isaac Lab (Isaac Sim)
- Python 3.10+, PyTorch with CUDA
- Git LFS (for dataset files)

### Installation

```bash
git clone https://github.com/JackyWang77/dual_isaac_so_arm101.git
cd dual_isaac_so_arm101
pip install -e source/SO_101
```

### Full Pipeline

```bash
# 1. Record demonstrations via teleoperation
python scripts/record_demos.py --task SO-ARM101-Lift-Cube-v0 --teleop_device keyboard

# 2. Train Graph U-Net backbone (behavior cloning)
bash train_graph_dit.sh flow_matching

# 3. Fine-tune with residual RL
bash train_residual_rl.sh ./logs/graph_unet/lift_joint/best_model.pt 512 25

# 4. Evaluate
bash play_graph_dit_rl.sh ./logs/graph_unet_rl/.../policy_iter_25.pt
```

## Citation

```
@mastersthesis{wang2026graph,
  title={Graph-Based Diffusion Policy with Residual Reinforcement Learning for Dual-Arm Manipulation},
  author={Wang, Jiaqi},
  school={Eindhoven University of Technology},
  year={2026}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Requires Git LFS and pre-commit hooks.

## License

BSD-3-Clause

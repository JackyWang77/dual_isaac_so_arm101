# Isaac Lab – SO‑ARM100 / SO‑ARM101 Project

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0/5.0.0-green.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.2.0-green.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![DOI](https://zenodo.org/badge/968772915.svg)](https://zenodo.org/badge/latestdoi/968772915)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![ci-test](https://img.shields.io/github/actions/workflow/status/MuammerBay/isaac_so_arm101/ci-test.yml?branch=main)](https://github.com/MuammerBay/isaac_so_arm101/actions/workflows/ci-test.yml)

This repository implements tasks for the SO‑ARM100 and SO‑ARM101 robots using Isaac Lab. It serves as the foundation for several tutorials in the LycheeAI Hub series [Project: SO‑ARM101 × Isaac Sim × Isaac Lab](https://lycheeai-hub.com/project-so-arm101-x-isaac-sim-x-isaac-lab-tutorial-series).

### 📰 News featuring this repository:

- **10 June 2025:** 🎥 LycheeAI Channel Premiere: SO-ARM101 tutorial series announcement! [🔗 Watch on YouTube](https://www.youtube.com/watch?v=2uH7Zn4SAVI)
- **23 April 2025:** 🤖 NVIDIA Omniverse Livestream: Training a Robot from Scratch in Simulation (URDF → OpenUSD). [🔗 Watch on YouTube](https://www.youtube.com/watch?v=_HMk7I-vSBQ)
- **19 April 2025:** 🎥 LycheeAI Tutorial: How to Create External Projects in Isaac Lab. [🔗 Watch on YouTube](https://www.youtube.com/watch?v=i51krqsk8ps)

### 🤖 Don’t have a SO-ARM? → Buy one with a Discount: `LYCHEEAI5` !

There are official vendors who sell all the required parts and already assembled kits such as WowRobo who sponsor this project. Use code `LYCHEEAI5` at checkout for a 5% discount on the [SO-ARM101](https://shop.wowrobo.com/?sca_ref=8879221). Now also for [LeKiwi](https://shop.wowrobo.com/products/lekiwi-mobile-robot).


### 🎬 Watch the Lift Task in action

![rl-video-step-0](https://github.com/user-attachments/assets/890e3a9d-5cbd-46a5-9317-37d0f2511684)

## 🛠️ Installation

1. Install Isaac Lab by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) (using conda).
2. Clone this repository **outside** the `IsaacLab` directory.
3. Install the package:

   ```bash
   python -m pip install -e source/SO_100
   ```

## 🚀 Quickstart

To list all available environments:

```bash
python scripts/list_envs.py
```

## 🐞 Debugging Tasks

Two scripts can help verify your setup:

**Zero Agent**

Sends zero commands to all robots, confirming that the environment loads correctly:

```bash
python scripts/zero_agent.py --task SO-ARM100-Reach-Play-v0
```

**Random Agent**

Sends random commands to all robots, confirming proper actuation:

```bash
python scripts/random_agent.py --task SO-ARM100-Reach-Play-v0
```

## 🧠 Graph DiT (Graph Diffusion Transformer) Policy

This repository implements a **Graph Diffusion Transformer (Graph DiT)** policy for manipulation tasks. The Graph DiT architecture combines graph neural networks, transformers, and diffusion models for learning robust manipulation policies from demonstrations and fine-tuning with reinforcement learning.

### Architecture Overview

The Graph DiT policy models the manipulation task as a graph-based structure, where spatial relationships between the end-effector and objects are explicitly represented as graph edges.

#### Graph Structure

- **Nodes**: Two nodes representing key entities in manipulation:
  - **End-Effector (EE) Node**: 
    - Features: Position (3D) + Orientation (4D quaternion) = 7D
    - Represents the robot's end-effector state in the scene
  - **Object Node**: 
    - Features: Position (3D) + Orientation (4D quaternion) = 7D
    - Represents the target object state in the scene

- **Edges**: Represent spatial relationships between nodes:
  - **Distance**: Euclidean distance between EE and object (1D)
  - **Orientation Similarity**: Cosine similarity of quaternion orientations (1D)
  - Total edge features: 2D
  - Edges enable the model to reason about spatial relationships

- **Temporal History**: 
  - Both nodes and actions maintain a history window (configurable, default: 4 steps)
  - Allows the model to capture temporal dependencies and motion patterns

#### Graph DiT Unit

Each Graph DiT layer (`GraphDiTUnit`) consists of three sequential components:

1. **Action Self-Attention**: 
   - Processes historical action sequences using multi-head self-attention
   - Captures temporal dependencies and patterns in action sequences
   - Input: `[batch, action_history_length, hidden_dim]`
   - Output: Refined action features `a_new`

2. **Graph Attention with Edge Features**:
   - Performs attention between EE and Object nodes using graph attention mechanism
   - Uses edge features (distance + orientation similarity) as attention bias
   - Enables the model to dynamically weight node interactions based on spatial relationships
   - Input: Node features `[batch, 2, hidden_dim]` + Edge features `[batch, edge_dim]`
   - Output: Updated node features `node_features_new`

3. **Cross-Attention**:
   - Queries updated node features using refined action features
   - Generates noise predictions for the diffusion process
   - Integrates action and scene information for denoising
   - Input: Action features (queries) × Node features (keys/values)
   - Output: Predicted noise for action denoising

This three-step process is repeated across multiple Graph DiT layers to enable complex reasoning about manipulation tasks.

#### Diffusion Process

The policy uses a diffusion-based generative model for action prediction, similar to Diffusion Policy:

- **DDPM Mode** (Denoising Diffusion Probabilistic Model):
  - Standard diffusion process with noise scheduling
  - Requires 50-100 diffusion steps for inference
  - More accurate but slower
  - Suitable for offline planning or high-precision tasks

- **Flow Matching Mode** (Rectified Flow):
  - Faster inference with straight-line paths in latent space
  - Requires only 1-10 steps for inference (10x faster than DDPM)
  - Maintains competitive performance
  - Recommended for real-time applications and interactive control

Both modes support conditional generation based on subtask conditions for multi-task learning.

### Training Pipeline

The Graph DiT policy supports a two-stage training approach:

1. **Supervised Learning from Demonstrations**:
   - Train on collected demonstration data (HDF5 format)
   - Learn to generate actions that match expert demonstrations
   - Uses action prediction loss (MSE for Flow Matching, standard diffusion loss for DDPM)
   - Script: `train_graph_dit.sh`
   - Output: Checkpoint with trained Graph DiT backbone

2. **RL Fine-tuning** (Optional):
   - Fine-tune the policy using reinforcement learning (RSL-RL framework with PPO)
   - Freezes the Graph DiT backbone (feature extractor)
   - Trains only a small RL head on top (action mean/std prediction)
   - Trains a separate value network for advantage estimation
   - Improves policy performance through environment interaction
   - Script: `train_graph_dit_rl_rsl.sh`
   - Supports efficient multi-environment parallel training (512-4096 envs)

### RL Fine-tuning: Head-Only Training

RL fine-tuning is an optional second stage that improves the policy through reinforcement learning. This approach is particularly effective for adapting pre-trained models to specific tasks or improving performance beyond what demonstrations can provide.

#### Why RL Fine-tuning?

- **Compensate for Demonstration Limitations**: Demonstrations may not cover all scenarios or optimal behaviors
- **Optimize for Task-Specific Rewards**: Learn behaviors that maximize task-specific rewards rather than just imitating
- **Adapt to Environment Dynamics**: Fine-tune to account for simulator-to-simulator or sim-to-real gaps
- **Efficient Learning**: Leverages pre-trained representations for faster convergence

#### Architecture: Head-Only Fine-tuning

The RL fine-tuning approach uses a **head-only** strategy, where the Graph DiT backbone is frozen and only a small output layer is trained:

```
┌─────────────────────────────────────────┐
│   Frozen Graph DiT Backbone             │
│   (78M parameters, pre-trained)         │
│                                         │
│   - Node embeddings                     │
│   - Graph attention layers              │
│   - Diffusion process                   │
│   - All weights frozen (no gradients)   │
└───────────────┬─────────────────────────┘
                │
                │ Extract features
                ▼
┌─────────────────────────────────────────┐
│   Trainable RL Head                     │
│   (~400K parameters, <1% of total)      │
│                                         │
│   ┌───────────────────────────────┐    │
│   │  RL Action Head (MLP)         │    │
│   │  Graph DiT features →         │    │
│   │  Action mean + std            │    │
│   └───────────────────────────────┘    │
│                                         │
│   ┌───────────────────────────────┐    │
│   │  Value Network (MLP)          │    │
│   │  Observations → Value est.    │    │
│   └───────────────────────────────┘    │
└─────────────────────────────────────────┘
```

#### Key Components

1. **Frozen Graph DiT Backbone**:
   - Pre-trained on demonstration data
   - Serves as a feature extractor
   - Processes observations to extract rich spatial-temporal features
   - Parameters: ~78M (frozen, no gradients)

2. **Trainable RL Action Head**:
   - Small MLP (e.g., [128, 64] hidden dims)
   - Maps Graph DiT features → action distribution
   - Outputs: action mean and log_std for PPO
   - Parameters: ~200K (trainable)

3. **Independent Value Network**:
   - Separate MLP for value estimation
   - Takes full observations (39D) as input
   - Outputs: value estimates for advantage calculation
   - Parameters: ~200K (trainable)

#### Integration with RSL-RL

The RL fine-tuning is fully integrated with the RSL-RL framework:

- **Standard RSL-RL Interface**: Uses `ActorCritic` interface compatible with RSL-RL's `OnPolicyRunner`
- **PPO Algorithm**: Standard Proximal Policy Optimization (PPO) from RSL-RL
- **Multi-Environment Training**: Supports parallel training with 512-4096 environments
- **Automatic Configuration**: Uses Hydra configuration system for easy customization

#### Training Process

1. **Load Pre-trained Backbone**:
   ```python
   # Graph DiT backbone weights loaded from checkpoint
   # All parameters set to requires_grad=False
   graph_dit_backbone.eval()  # Set to eval mode
   ```

2. **Initialize RL Head**:
   ```python
   # Random initialization for RL head
   # Value network also randomly initialized
   ```

3. **PPO Training Loop**:
   - **Rollout Phase**: Collect trajectories using current policy
   - **Update Phase**: 
     - Compute advantages using value network
     - Update action head using PPO loss
     - Update value network using value loss
     - Graph DiT backbone remains frozen (no gradients)

4. **Observation Handling**:
   - Graph DiT receives 32D observations (target_object_position removed)
   - Value network receives full 39D observations
   - Automatic dimension handling ensures compatibility

#### Advantages

- **Parameter Efficiency**: Only ~0.5% of parameters are trainable
- **Fast Training**: Smaller trainable network converges faster
- **Stable Learning**: Pre-trained features provide good initialization
- **Memory Efficient**: Frozen backbone doesn't store gradients
- **Scalable**: Supports large-scale parallel training (4096+ envs)

#### Configuration

RL fine-tuning can be configured via the `ReachCubeGraphDiTRLRunnerCfg`:

- **PPO Hyperparameters**:
  - Learning rate: `3.0e-4` (default)
  - Clip parameter: `0.2`
  - Entropy coefficient: `0.01`
  - Value loss coefficient: `1.0`

- **RL Head Architecture**:
  - Hidden dims: `[128, 64]` (default)
  - Activation: `elu`
  - Initial noise std: `0.5`

- **Value Network**:
  - Hidden dims: `[256, 128, 64]` (default)
  - Takes full observation space (39D)

#### Training Tips

- **Start with More Environments**: Use 512-1024 envs for faster data collection
- **Monitor Reward Curves**: Use TensorBoard to track training progress
- **Adjust Learning Rate**: If training is unstable, reduce learning rate (e.g., `1.0e-4`)
- **Training Iterations**: Usually 200-500 iterations are sufficient (may vary by task)
- **Checkpoint Frequency**: Save checkpoints regularly to avoid losing progress

### Key Features

- **Graph-based Representation**: Explicitly models spatial relationships between robot and objects
- **Temporal Modeling**: Captures action history and temporal node state evolution
- **Fast Inference**: Flow Matching mode enables real-time control (1-10 steps)
- **RL-Compatible**: Integrates seamlessly with RSL-RL for efficient multi-env training
- **Head-only Fine-tuning**: Freezes pre-trained backbone for efficient RL adaptation

### Usage

**Train Graph DiT from demonstrations:**
```bash
bash train_graph_dit.sh flow_matching  # or 'ddpm'
```

**Fine-tune with RL:**
```bash
bash train_graph_dit_rl_rsl.sh <pretrained_checkpoint> [num_envs] [iterations]
# Example: bash train_graph_dit_rl_rsl.sh ./logs/graph_dit/.../best_model.pt 512 200
```

**Play trained policy:**
```bash
bash play_graph_dit_rl.sh [checkpoint_path]
# Auto-detects latest checkpoint if path not provided
```

## 🏋️‍♂️ Training and Playback

You can train a policy for SO‑ARM100 / SO‑ARM101 tasks (for example, the **Reach** task, which is a basic RL-based IK) with the `rsl_rl` and/or `skrl` library:

```bash
python scripts/rsl_rl/train.py --task SO-ARM100-Reach-v0 --headless
```

After training, validate the learned policy:

```bash
python scripts/rsl_rl/play.py --task SO-ARM100-Reach-Play-v0
```

This ensures that your policy performs as expected in Isaac Lab before attempting real‑world transfer.

## 🧩 Sim2Sim Transfer

_Work in progress._

## 🛠️ Sim2Real Transfer

_Work in progress._

## 🤝 Contributing

We welcome contributions of all kinds!
Please read our [Contributing Guide](CONTRIBUTING.md) to learn how to set up your environment, follow our coding style, and submit pull requests.

## 📄 License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This project builds upon the excellent work of several open-source projects and communities:

- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** - The foundational robotics simulation framework that powers this project
- **[NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)** - The underlying physics simulation platform
- **[RSL-RL](https://github.com/leggedrobotics/rsl_rl)** - Reinforcement learning library used for training policies
- **[SKRL](https://github.com/Toni-SM/skrl)** - Alternative RL library integration
- **[SO-ARM100/SO-ARM101 Robot](https://github.com/TheRobotStudio/SO-ARM100)** - The hardware platform that inspired this simulation environment

Special thanks to:

- The Isaac Lab development team at NVIDIA for providing the simulation framework
- Hugging Face and The Robot Studio for the SO‑ARM robot series
- The LycheeAI Hub community for tutorials and support

## 📚 Citation

If you use this work, please cite it as:

```bibtex
@software{Louis_Isaac_Lab_2025,
   author = {Louis, Le Lay and Muammer, Bay},
   doi = {https://doi.org/10.5281/zenodo.16794229},
   license = {BSD-3-Clause},
   month = apr,
   title = {Isaac Lab – SO‑ARM100 / SO‑ARM101 Project},
   url = {https://github.com/MuammerBay/isaac_so_arm101},
   version = {1.1.0},
   year = {2025}
}

#!/bin/bash
# ============================================================
# Train GR-DiT: Graph Residual Diffusion Transformer
# ============================================================
#
# This script trains a GR-DiT policy using our method:
#   - GraphDiT backbone (frozen): base action + layer-wise graph latents z¹...zᴷ
#   - GateNet (trainable): softmax aggregation → z_bar
#   - Z Adapter (trainable): frozen z → RL-friendly z
#   - Residual Actor (trainable): delta_a ~ π(δa | obs, a_base, z_bar)
#   - Deep Value Critic (trainable): V^k(z^k) for deep supervision
#
# Key innovation:
#   - High-frequency z: extract_z_fast() every step (real-time scene understanding)
#   - Low-frequency base action: DiT trajectory every exec_horizon steps
#   - Advantage-weighted regression (NOT PPO)
#
# Usage:
#   ./train_gr_dit.sh <pretrained_dit_checkpoint> [options]
#
# ============================================================

set -e

# ============================================================
# Default Arguments
# ============================================================
PRETRAINED_CHECKPOINT="${1:-}"
NUM_ENVS="${2:-64}"
MAX_ITERATIONS="${3:-200}"
STEPS_PER_ENV="${4:-130}"  # Increased from 24 to 130 to allow episodes to complete
MINI_BATCH_SIZE="${5:-64}"
NUM_EPOCHS="${6:-5}"
SEED="${7:-42}"
HEADLESS="${8:-false}"  # Default: false (enable visualization)

# Task (can be overridden)
TASK="${TASK:-SO-ARM101-Lift-Cube-v0}"

# ============================================================
# Validate Arguments
# ============================================================
if [ -z "$PRETRAINED_CHECKPOINT" ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                    GR-DiT Training Script                      ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "❌ Error: Missing required argument: pretrained_dit_checkpoint"
    echo ""
    echo "Usage:"
    echo "  $0 <pretrained_dit_checkpoint> [num_envs] [max_iter] [steps_per_env] [mini_batch] [epochs] [seed] [headless]"
    echo ""
    echo "Arguments:"
    echo "  pretrained_dit_checkpoint  Path to pre-trained GraphDiT checkpoint (required)"
    echo "  num_envs                   Number of parallel environments (default: 64)"
    echo "  max_iterations             Maximum training iterations (default: 500)"
    echo "  steps_per_env              Rollout steps per env per iteration (default: 24)"
    echo "  mini_batch_size            Mini-batch size for updates (default: 64)"
    echo "  num_epochs                 Epochs per iteration (default: 5)"
    echo "  seed                       Random seed (default: 42)"
    echo "  headless                   Enable headless mode: true/false (default: false, enables visualization)"
    echo ""
    echo "Environment Variables:"
    echo "  TASK                       Task name (default: SO-ARM101-Lift-Cube-v0)"
    echo ""
    echo "Examples:"
    echo "  # Basic usage"
    echo "  $0 ./logs/graph_dit/best_model.pt"
    echo ""
    echo "  # Custom settings"
    echo "  $0 ./logs/graph_dit/best_model.pt 128 1000 32 128 8 123"
    echo ""
    echo "  # Different task"
    echo "  TASK=SO-ARM101-Pick-Place-v0 $0 ./logs/graph_dit/best_model.pt"
    echo ""
    exit 1
fi

if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $PRETRAINED_CHECKPOINT"
    exit 1
fi

# ============================================================
# Check Environment
# ============================================================
echo ""
echo "Checking environment..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found!"
    exit 1
fi

# Check PyTorch
if ! python -c "import torch" 2>/dev/null; then
    echo "❌ Error: PyTorch not found!"
    echo "Please activate your conda environment:"
    echo "  conda activate env_isaaclab"
    exit 1
fi

# Check CUDA
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')")
    echo "✅ CUDA available: $GPU_NAME ($GPU_MEM)"
else
    echo "⚠️  Warning: CUDA not available, training will be slow!"
fi

# Set headless flag
if [ "$HEADLESS" = "true" ] || [ "$HEADLESS" = "1" ]; then
    HEADLESS_FLAG="--headless"
else
    HEADLESS_FLAG=""
fi

# ============================================================
# Print Configuration
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              GR-DiT: Graph Residual DiT Training               ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║ Architecture:                                                  ║"
echo "║   GraphDiT (frozen) → base action + z_layers [z¹...zᴷ]        ║"
echo "║   Z Adapter (trainable) → RL-friendly z                        ║"
echo "║   GateNet (trainable) → softmax aggregation → z_bar            ║"
echo "║   Residual Actor (trainable) → δa ~ π(δa | obs, a_base, z_bar) ║"
echo "║   Deep Value Critic (trainable) → V^k(z^k) supervision         ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║ Key Innovation:                                                ║"
echo "║   • High-freq z: extract_z_fast() every step                   ║"
echo "║   • Low-freq base: DiT trajectory every exec_horizon steps     ║"
echo "║   • Loss: Advantage-weighted regression (NOT PPO)              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Task:                 $TASK"
echo "  Pretrained DiT:       $PRETRAINED_CHECKPOINT"
echo "  Num envs:             $NUM_ENVS"
echo "  Max iterations:       $MAX_ITERATIONS"
echo "  Steps per env:        $STEPS_PER_ENV"
echo "  Mini-batch size:      $MINI_BATCH_SIZE"
echo "  Epochs per iter:      $NUM_EPOCHS"
echo "  Seed:                 $SEED"
echo "  Headless:             $HEADLESS"
echo ""
echo "Computed:"
echo "  Batch size:           $((NUM_ENVS * STEPS_PER_ENV))"
echo "  Updates per iter:     $((NUM_ENVS * STEPS_PER_ENV / MINI_BATCH_SIZE * NUM_EPOCHS))"
echo ""

# ============================================================
# Confirm
# ============================================================
read -p "Start training? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Aborted."
    exit 0
fi

# ============================================================
# Run Training
# ============================================================
echo ""
echo "Starting training..."
echo "========================================"

# Use isaaclab.sh to properly initialize Isaac Sim
python scripts/graph_dit_rl/train_graph_dit_rl.py \
    --task "$TASK" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --steps_per_env "$STEPS_PER_ENV" \
    --mini_batch_size "$MINI_BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --seed "$SEED" \
    --log_dir "./logs/gr_dit" \
    --save_interval 50 \
    $HEADLESS_FLAG

echo ""
echo "========================================"
echo "✅ Training completed!"
echo ""
echo "Logs saved to: ./logs/gr_dit/$TASK/"
echo "========================================"
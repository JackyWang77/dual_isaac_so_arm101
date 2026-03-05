#!/bin/bash
# Attention pipeline: Step 1 = record obs (play with record_attention).
# Step 2–4 (extract attn, classify, visualize) run later after choosing episode(s).
#
# Model/params aligned with play_disentangled_graph_gated. Collects 10 episodes by default.
#
# Usage:
#   ./stack/run_attention_pipeline.sh [checkpoint_path] [num_episodes] [output_dir]
#   ./stack/run_attention_pipeline.sh
#   ./stack/run_attention_pipeline.sh ./logs/graph_unet_full/.../gate_graph/best_model.pt 20
set -e
cd "$(dirname "$0")/.."

ATTN_CHECKPOINT="./logs/graph_unet_full/stack_joint_t1_gripper_flow_matching/gate_graph/best_model.pt"
CHECKPOINT="${1:-$ATTN_CHECKPOINT}"
NUM_EPISODES="${2:-10}"
OUT_DIR="${3:-./attention_output}"

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Usage: $0 [checkpoint_path] [num_episodes=10] [output_dir=./attention_output]"
    echo "  Checkpoint default: $ATTN_CHECKPOINT"
    exit 1
fi

# Same as play_disentangled_graph_gated.sh
NUM_ENVS="${NUM_ENVS:-1}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-10}"
EXEC_HORIZON="${EXEC_HORIZON:-10}"
EMA="${EMA:-0.8}"

mkdir -p "$OUT_DIR"

OBS_JSON="$OUT_DIR/obs_records.json"

echo "========================================================"
echo "Attention Pipeline — Step 1: Record observations"
echo "  Model:       disentangled_graph_unet_gated (same as play_disentangled_graph_gated)"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Episodes:    $NUM_EPISODES (envs=$NUM_ENVS)"
echo "  Output:      $OBS_JSON"
echo "========================================================"

# ── Step 1: Record observations ──────────────────────────────
echo ""
echo "═══ Step 1: Recording observations ═══"
python scripts/graph_unet/play.py \
    --task SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-Play-v0 \
    --checkpoint "$CHECKPOINT" \
    --policy_type disentangled_graph_unet_gated \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --episode_length_s "$EPISODE_LENGTH_S" \
    --num_diffusion_steps 15 \
    --exec_horizon "$EXEC_HORIZON" \
    --ema "$EMA" \
    --record_attention \
    --attention_output "$OBS_JSON"

if [ ! -f "$OBS_JSON" ]; then
    echo "ERROR: Observation recording failed (no output file)"
    exit 1
fi

N_TOTAL=$(python3 -c "import json; print(len(json.load(open('$OBS_JSON'))))")
N_SUCCESS=$(python3 -c "import json; print(sum(1 for r in json.load(open('$OBS_JSON')) if r.get('success')))")
echo "Recorded: $N_TOTAL total records, $N_SUCCESS from successful episodes"
echo ""
echo "========================================================"
echo "Step 1 done. Raw obs: $OBS_JSON"
echo "Step 2–4 (extract attn, classify, visualize) to be run after choosing episode(s)."
echo "========================================================"

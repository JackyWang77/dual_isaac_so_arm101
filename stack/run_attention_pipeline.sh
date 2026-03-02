#!/bin/bash
# End-to-end attention analysis pipeline:
#   1. Play (record obs) → 2. Extract attention → 3. Classify phases → 4. Visualize
#
# Usage:
#   ./stack/run_attention_pipeline.sh <checkpoint_path> [num_episodes] [output_dir]
#   ./stack/run_attention_pipeline.sh ./logs/graph_unet_full/.../best_model.pt
#   ./stack/run_attention_pipeline.sh ./logs/graph_unet_full/.../best_model.pt 20
#   NUM_ENVS=2 ./stack/run_attention_pipeline.sh ./logs/.../best_model.pt 20 ./attn_results
set -e
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-}"
NUM_EPISODES="${2:-10}"
OUT_DIR="${3:-./attention_output}"

if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -t ./logs/graph_unet_full/stack_joint*/*/best_model.pt 2>/dev/null | head -1)
fi
if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "Usage: $0 <checkpoint_path> [num_episodes=10] [output_dir=./attention_output]"
    exit 1
fi

NUM_ENVS="${NUM_ENVS:-1}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-10}"
EXEC_HORIZON="${EXEC_HORIZON:-10}"
EMA="${EMA:-0.9}"

mkdir -p "$OUT_DIR"

OBS_JSON="$OUT_DIR/obs_records.json"
ATTN_JSON="$OUT_DIR/obs_with_attn.json"
CLASS_JSON="$OUT_DIR/classified.json"
FIG_DIR="$OUT_DIR/figs"

echo "========================================================"
echo "Attention Analysis Pipeline"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Episodes:    $NUM_EPISODES (envs=$NUM_ENVS)"
echo "  Output dir:  $OUT_DIR"
echo "========================================================"

# ── Step 1: Record observations ──────────────────────────────
echo ""
echo "═══ Step 1/4: Recording observations ═══"
python scripts/graph_unet/play.py \
    --task SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-Play-v0 \
    --checkpoint "$CHECKPOINT" \
    --policy_type graph_unet \
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

if [ "$N_SUCCESS" -eq 0 ]; then
    echo "WARNING: No successful episodes. Attention extraction will be empty."
    echo "Try increasing num_episodes or check model quality."
    exit 1
fi

# ── Step 2: Extract attention (offline, through graph encoder) ──
echo ""
echo "═══ Step 2/4: Extracting attention weights ═══"
python scripts/extract_attention_offline.py \
    --obs_records "$OBS_JSON" \
    --checkpoint "$CHECKPOINT" \
    --output "$ATTN_JSON" \
    --success_only

# ── Step 3: Classify phases by distance ──────────────────────
echo ""
echo "═══ Step 3/4: Classifying phases ═══"
python scripts/classify_attention_by_distance.py "$ATTN_JSON" -o "$CLASS_JSON"

# ── Step 4: Visualize ────────────────────────────────────────
echo ""
echo "═══ Step 4/4: Generating figures ═══"
python scripts/visualize_attention.py "$CLASS_JSON" --output "$FIG_DIR"

echo ""
echo "========================================================"
echo "Pipeline complete!"
echo "  Raw obs:      $OBS_JSON"
echo "  With attn:    $ATTN_JSON"
echo "  Classified:   $CLASS_JSON"
echo "  Figures:      $FIG_DIR/"
echo "========================================================"

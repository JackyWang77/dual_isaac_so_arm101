#!/bin/bash
# Choose episode -> output separate step images (PDF per frame). Reads obs_records.json.
# Usage: ./stack/attention_episode_to_frames.sh [episode_id=0] [output_dir]
set -e
cd "$(dirname "$0")/.."

OBS_JSON="attention_output/obs_records.json"
CHECKPOINT="./logs/graph_unet_full/stack_joint_t1_gripper_flow_matching/gate_graph/best_model.pt"
EPISODE_ID="${1:-0}"
OUT_DIR="${2:-attention_output/heatmaps_ep${EPISODE_ID}}"

if [ ! -f "$OBS_JSON" ]; then
    echo "Missing $OBS_JSON — run ./stack/run_attention_pipeline.sh first"
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "=== Episode $EPISODE_ID -> per-step PDFs ==="
python scripts/select_episode0_phases_by_distance.py \
    --obs_records "$OBS_JSON" \
    --episode_id "$EPISODE_ID" \
    --output "attention_output/ep${EPISODE_ID}_indices.json"

IDX_JSON="attention_output/ep${EPISODE_ID}_indices.json"
ALL_INDICES=$(python3 -c "import json; d=json.load(open('$IDX_JSON')); print(' '.join(map(str, d['all_global_indices'])))")
python scripts/extract_attention_offline.py \
    --obs_records "$OBS_JSON" \
    --checkpoint "$CHECKPOINT" \
    --output "attention_output/obs_with_attn_ep${EPISODE_ID}.json" \
    --record_index $ALL_INDICES

python scripts/plot_attention_heatmaps.py \
    --input "attention_output/obs_with_attn_ep${EPISODE_ID}.json" \
    --output_dir "$OUT_DIR" \
    --format pdf

echo "Done: $OUT_DIR/phase1_pos.pdf, phase2_pos.pdf, ..."

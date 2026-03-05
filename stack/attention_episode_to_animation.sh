#!/bin/bash
# Choose episode -> output attention animation (GIF). Reads obs_records.json.
# Usage: ./stack/attention_episode_to_animation.sh [episode_id=0] [output.gif]
set -e
cd "$(dirname "$0")/.."

OBS_JSON="attention_output/obs_records.json"
CHECKPOINT="./logs/graph_unet_full/stack_joint_t1_gripper_flow_matching/gate_graph/best_model.pt"
EPISODE_ID="${1:-0}"
OUT_GIF="${2:-attention_output/attention_ep${EPISODE_ID}_animation.gif}"

if [ ! -f "$OBS_JSON" ]; then
    echo "Missing $OBS_JSON — run ./stack/run_attention_pipeline.sh first"
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "=== Episode $EPISODE_ID -> attention animation ==="
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

# record at every replan = exec_horizon=10 steps; step_dt=0.02s -> 5 fps for real-time sync
# --indices_json enables phase label on progress bar (e.g. Left Pick Cube1, Stacking)
python scripts/plot_attention_animation.py \
    --input "attention_output/obs_with_attn_ep${EPISODE_ID}.json" \
    --output "$OUT_GIF" \
    --fps 5 \
    --indices_json "attention_output/ep${EPISODE_ID}_indices.json"

echo "Done: $OUT_GIF"

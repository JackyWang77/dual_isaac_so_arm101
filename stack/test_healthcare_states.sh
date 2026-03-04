#!/bin/bash
# Record demonstrations for healthcare table setting task (elderly care use case).
# Dual-arm: left hand places fork, right hand places knife onto tray beside plate.
#
# Usage:
#   bash stack/test_healthcare_states.sh                          # default: 200 demos
#   bash stack/test_healthcare_states.sh 50                       # 50 demos
#   bash stack/test_healthcare_states.sh 100 spacemouse           # 100 demos with spacemouse
#
cd "$(dirname "$0")/.."

NUM_DEMOS="${1:-200}"
TELEOP_DEVICE="${2:-joint_states}"
DATASET_FILE="${3:-./datasets/table_setting_dataset.hdf5}"

TASK="SO-ARM101-Dual-Table-Setting-Joint-States-Mimic-v0"

echo "========================================"
echo "Healthcare Table Setting - Data Collection"
echo "========================================"
echo "Task:          $TASK"
echo "Demos:         $NUM_DEMOS"
echo "Teleop Device: $TELEOP_DEVICE"
echo "Dataset:       $DATASET_FILE"
echo "========================================"
echo ""
echo "Scene layout:"
echo "  - Tray + Plate: center (static)"
echo "  - Fork: spawns LEFT  -> left arm places LEFT of plate"
echo "  - Knife: spawns RIGHT -> right arm places RIGHT of plate"
echo "========================================"

python scripts/record_demos.py \
    --task "$TASK" \
    --teleop_device "$TELEOP_DEVICE" \
    --dataset_file "$DATASET_FILE" \
    --num_demos "$NUM_DEMOS" \
    --num_success_steps 10

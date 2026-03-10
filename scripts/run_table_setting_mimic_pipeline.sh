#!/bin/bash
# IsaacLab Mimic Pipeline for Dual-Arm Table Setting Task
# Merge datasets 1-10 -> annotate with subtask signals (place_fork, place_knife)
#
# IMPORTANT: Use the SAME task as record_demos so the same success termination
# (table_setting_env_cfg.TerminationsCfg.success = both_placed_stable, 10 steps) is used.
# Recording: python scripts/record_demos.py --task SO-ARM101-Dual-Table-Setting-Joint-States-Mimic-v0 ...
# This script passes that same task to annotate_demos.

TASK_ANNOTATE="SO-ARM101-Dual-Table-Setting-Joint-States-Mimic-v0"

INPUT_FILE="./datasets/table_setting_dataset.hdf5"
ANNOTATED_FILE="./datasets/table_setting_annotated_dataset.hdf5"
DEVICE="cpu"

# Component datasets to merge (1-10). If your files are named
# table_setting_dataset1.hdf5 (no underscore), use dataset1, dataset2, ... dataset10.
DATASETS=(
    "./datasets/table_setting_dataset_1.hdf5"
    "./datasets/table_setting_dataset_2.hdf5"
    "./datasets/table_setting_dataset_3.hdf5"
    "./datasets/table_setting_dataset_4.hdf5"
    "./datasets/table_setting_dataset_5.hdf5"
    "./datasets/table_setting_dataset_6.hdf5"
    "./datasets/table_setting_dataset_7.hdf5"
    "./datasets/table_setting_dataset_8.hdf5"
    "./datasets/table_setting_dataset_9.hdf5"
    "./datasets/table_setting_dataset_10.hdf5"
)

echo "========================================"
echo "IsaacLab Mimic Pipeline - Dual Table Setting"
echo "========================================"
echo "Annotation Task: $TASK_ANNOTATE"
echo "Input: $INPUT_FILE (merge of datasets 1-10)"
echo "Output: $ANNOTATED_FILE"
echo "========================================"

echo ""
echo "Merging datasets 1-10..."
echo "========================================"
python scripts/merge_hdf5_datasets.py \
    "${DATASETS[@]}" \
    -o "$INPUT_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Merge failed!"
    exit 1
fi
echo ""

echo ""
echo "Annotating demos with subtask signals (place_fork, place_knife)..."
echo "========================================"
python scripts/isaaclab_mimic/annotate_demos.py \
    --device "$DEVICE" \
    --task "$TASK_ANNOTATE" \
    --auto \
    --headless \
    --input_file "$INPUT_FILE" \
    --output_file "$ANNOTATED_FILE"
    

if [ $? -ne 0 ]; then
    echo "Error: Annotation failed!"
    exit 1
fi

echo ""
echo "Annotation complete!"
echo "========================================"
echo "Output: $ANNOTATED_FILE"

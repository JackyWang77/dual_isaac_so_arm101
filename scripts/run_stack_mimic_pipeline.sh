#!/bin/bash
# IsaacLab Mimic Pipeline Script for Dual-Arm Cube Stack Task
# Merge datasets (39+59+100) -> annotate with subtask signals (pick_cube, stack_cube)
# (No generation step - collect more data directly instead)

TASK_ANNOTATE="SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-v0"

INPUT_FILE="./datasets/dual_cube_stack_joint_states_mimic_dataset.hdf5"
ANNOTATED_FILE="./datasets/dual_cube_stack_annotated_dataset.hdf5"
DEVICE="cpu"

# Component datasets to merge (if merged file doesn't exist)
DATASET_39="./datasets/dual_cube_stack_joint_states_mimic_dataset_39.hdf5"
DATASET_59="./datasets/dual_cube_stack_joint_states_mimic_dataset_59.hdf5"
DATASET_100="./datasets/dual_cube_stack_joint_states_mimic_dataset_100.hdf5"

echo "========================================"
echo "IsaacLab Mimic Pipeline - Dual Cube Stack"
echo "========================================"
echo "Annotation Task: $TASK_ANNOTATE"
echo "Input: $INPUT_FILE"
echo "Output: $ANNOTATED_FILE"
echo "========================================"

# Merge component datasets if merged file doesn't exist
if [ ! -f "$INPUT_FILE" ] && [ -f "$DATASET_39" ] && [ -f "$DATASET_59" ] && [ -f "$DATASET_100" ]; then
    echo ""
    echo "Merging datasets (39+59+100)..."
    echo "========================================"
    python scripts/merge_hdf5_datasets.py \
        "$DATASET_39" "$DATASET_59" "$DATASET_100" \
        -o "$INPUT_FILE"
    if [ $? -ne 0 ]; then
        echo "Error: Merge failed!"
        exit 1
    fi
    echo ""
fi

echo ""
echo "Annotating demos with subtask signals..."
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

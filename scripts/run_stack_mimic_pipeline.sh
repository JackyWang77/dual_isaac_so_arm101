#!/bin/bash
# IsaacLab Mimic Pipeline Script for Dual-Arm Cube Stack Task
# Merge datasets 1-14 -> annotate with subtask signals (pick_cube, stack_cube)

TASK_ANNOTATE="SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-v0"

INPUT_FILE="./datasets/dual_cube_stack_joint_states_mimic_dataset.hdf5"
ANNOTATED_FILE="./datasets/dual_cube_stack_annotated_dataset.hdf5"
DEVICE="cpu"

# Component datasets to merge (1-14)
DATASETS=(
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_1.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_2.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_3.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_4.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_5.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_6.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_7.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_8.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_9.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_10.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_11.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_12.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_13.hdf5"
    "./datasets/dual_cube_stack_joint_states_mimic_dataset_14.hdf5"
)

echo "========================================"
echo "IsaacLab Mimic Pipeline - Dual Cube Stack"
echo "========================================"
echo "Annotation Task: $TASK_ANNOTATE"
echo "Input: $INPUT_FILE (merge of datasets 1-14)"
echo "Output: $ANNOTATED_FILE"
echo "========================================"

echo ""
echo "Merging datasets 1-14..."
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

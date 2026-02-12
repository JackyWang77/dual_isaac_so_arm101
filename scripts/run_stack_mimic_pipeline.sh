#!/bin/bash
# IsaacLab Mimic Pipeline Script for Dual-Arm Cube Stack Task
# Annotate demos with subtask signals (pick_cube, stack_cube)
# (No generation step - collect more data directly instead)

TASK_ANNOTATE="SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-v0"

INPUT_FILE="./datasets/dual_cube_stack_joint_states_mimic_dataset.hdf5"
ANNOTATED_FILE="./datasets/dual_cube_stack_annotated_dataset.hdf5"
DEVICE="cpu"

echo "========================================"
echo "IsaacLab Mimic Pipeline - Dual Cube Stack"
echo "========================================"
echo "Annotation Task: $TASK_ANNOTATE"
echo "Input: $INPUT_FILE"
echo "Output: $ANNOTATED_FILE"
echo "========================================"

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

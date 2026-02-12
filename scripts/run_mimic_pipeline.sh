#!/bin/bash
# IsaacLab Mimic Pipeline Script
# 1. Annotate demos with subtask signals (using Joint States env to replay recorded joint actions)
# 2. Generate augmented dataset (using IK Abs env to generate EEF pose actions)

# Annotation uses Joint States env (to replay recorded joint state actions)
TASK_ANNOTATE="SO-ARM100-Pick-Place-Joint-States-Mimic-v0"

# Generation uses IK Abs env (to output EEF pose actions for IK Abs policy training)
TASK_GENERATE="SO-ARM100-Pick-Place-IK-Abs-Mimic-v0"

INPUT_FILE="./datasets/dataset.hdf5"
ANNOTATED_FILE="./datasets/annotated_dataset.hdf5"
OUTPUT_FILE="./datasets/generated_dataset_ik_abs.hdf5"
DEVICE="cpu"
NUM_ENVS=10
NUM_TRIALS=3

echo "========================================"
echo "IsaacLab Mimic Pipeline"
echo "========================================"
echo "Annotation Task: $TASK_ANNOTATE"
echo "Generation Task: $TASK_GENERATE"
echo "Input: $INPUT_FILE"
echo "Annotated: $ANNOTATED_FILE"
echo "Output: $OUTPUT_FILE"
echo "========================================"

# Step 1: Annotate demos (using Joint States env)
echo ""
echo "[Step 1/2] Annotating demos with Joint States env..."
echo "========================================"
python scripts/isaaclab_mimic/annotate_demos.py \
    --device "$DEVICE" \
    --task "$TASK_ANNOTATE" \
    --auto \
    --headless \
    --enable_cameras \
    --input_file "$INPUT_FILE" \
    --output_file "$ANNOTATED_FILE"

echo ""
echo "[Step 1/2] Annotation complete!"
echo "========================================"

# Step 2: Generate dataset (using IK Abs env)
echo ""
echo "[Step 2/2] Generating IK Abs dataset..."
echo "========================================"
# python scripts/isaaclab_mimic/generate_dataset.py \
#     --device "$DEVICE" \
#     --num_envs "$NUM_ENVS" \
#     --generation_num_trials "$NUM_TRIALS" \
#     --task "$TASK_GENERATE" \
#     --headless \
#     --enable_cameras \
#     --input_file "$ANNOTATED_FILE" \
#     --output_file "$OUTPUT_FILE"

echo ""
echo "[Step 2/2] Generation complete!"
echo "========================================"
echo ""
echo "Pipeline finished successfully!"
echo "Output file: $OUTPUT_FILE"
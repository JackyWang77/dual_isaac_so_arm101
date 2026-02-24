#!/bin/bash
# Generate augmented dataset from dual_cube_stack_annotated_dataset.hdf5
# Uses Joint-States-Mimic env to replay/interpolate demos with different cube poses

TASK_GENERATE="SO-ARM101-Dual-Cube-Stack-Joint-States-Mimic-v0"

INPUT_FILE="./datasets/dual_cube_stack_annotated_dataset.hdf5"
OUTPUT_FILE="./datasets/dual_cube_stack_generated_dataset.hdf5"

DEVICE="cpu"
NUM_ENVS=100
NUM_TRIALS=3

echo "========================================"
echo "Generate Stack Dataset Pipeline"
echo "========================================"
echo "Task: $TASK_GENERATE"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Num envs: $NUM_ENVS"
echo "Num trials: $NUM_TRIALS"
echo "========================================"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo "Run run_stack_mimic_pipeline.sh first to create annotated dataset."
    exit 1
fi

echo ""
echo "Generating dataset..."
echo "========================================"
python scripts/isaaclab_mimic/generate_dataset.py \
    --device "$DEVICE" \
    --num_envs "$NUM_ENVS" \
    --generation_num_trials "$NUM_TRIALS" \
    --task "$TASK_GENERATE" \
    --headless \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Generation failed!"
    exit 1
fi

echo ""
echo "Generation complete!"
echo "========================================"
echo "Output: $OUTPUT_FILE"

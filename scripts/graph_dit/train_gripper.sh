#!/bin/bash
# Training script for gripper prediction model

# Default values
DATASET="./datasets/lift_annotated_dataset.hdf5"
EPOCHS=200
BATCH_SIZE=512
LR=1e-3
DEVICE="cuda"
SAVE_PATH="./logs/gripper_model/gripper_model.pt"
HIDDEN_DIMS="128 64 64"
DROPOUT=0.1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --save-path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --hidden-dims)
            HIDDEN_DIMS="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create save directory
mkdir -p "$(dirname "$SAVE_PATH")"

# Run training
python scripts/graph_dit/train_gripper.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device "$DEVICE" \
    --save-path "$SAVE_PATH" \
    --hidden-dims $HIDDEN_DIMS \
    --dropout "$DROPOUT"

echo "Training completed! Model saved to: $SAVE_PATH"

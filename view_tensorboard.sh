#!/bin/bash
# View TensorBoard for Graph DiT training logs
#
# Usage:
#   bash view_tensorboard.sh                    # View all logs
#   bash view_tensorboard.sh reach              # View reach task logs
#   bash view_tensorboard.sh lift               # View lift task logs
#   bash view_tensorboard.sh flow_matching      # View all Flow Matching logs
#   bash view_tensorboard.sh <path>             # View specific directory

TASK="${1:-}"

# Handle arguments
if [ -z "$TASK" ]; then
    # No arguments: view all logs
    LOG_DIR="./logs/graph_dit"
elif [ "$TASK" = "flow_matching" ]; then
    # Mode specified: view all logs
    LOG_DIR="./logs/graph_dit"
elif [ "$TASK" = "reach" ] || [ "$TASK" = "lift" ]; then
    # Task specified: view flow_matching logs for that task
    LOG_DIR="./logs/graph_dit/${TASK}_joint_flow_matching"
else
    # Treat as directory path
    LOG_DIR="$TASK"
fi

echo "========================================"
echo "Starting TensorBoard"
echo "========================================"
echo "Log directory: $LOG_DIR"
echo ""
echo "TensorBoard will open at: http://localhost:6006"
echo "Press Ctrl+C to stop"
echo ""

# Check if log directory exists
if [ ! -d "$LOG_DIR" ] && [ ! -d "$(dirname "$LOG_DIR")" ]; then
    echo "❌ Error: Log directory not found: $LOG_DIR"
    echo ""
    echo "Usage:"
    echo "  $0                                    # View all logs"
    echo "  $0 reach                              # View reach task logs"
    echo "  $0 lift                               # View lift task logs"
    echo "  $0 <log_directory>                    # View specific directory"
    echo ""
    echo "Examples:"
    echo "  $0                                    # View all logs"
    echo "  $0 reach                              # View all reach logs"
    echo "  $0 lift                               # View lift Flow Matching logs"
    echo "  $0 ./logs/graph_dit/lift_joint_flow_matching/2025-01-10_15-30-45  # View specific run"
    exit 1
fi

# Start TensorBoard
tensorboard --logdir "$LOG_DIR" --port 6006

#!/bin/bash
# View TensorBoard for Graph DiT training logs
#
# Usage:
#   bash view_tensorboard.sh                    # View all logs
#   bash view_tensorboard.sh reach [mode]       # View reach task logs (mode: ddpm/flow_matching)
#   bash view_tensorboard.sh lift [mode]        # View lift task logs (mode: ddpm/flow_matching)
#   bash view_tensorboard.sh ddpm               # View all DDPM logs
#   bash view_tensorboard.sh flow_matching      # View all Flow Matching logs
#   bash view_tensorboard.sh <path>             # View specific directory

TASK="${1:-}"
MODE="${2:-}"

# Handle mode-only arguments (backward compatibility)
if [ -z "$TASK" ]; then
    # No arguments: view all logs
    LOG_DIR="./logs/graph_dit"
elif [ "$TASK" = "ddpm" ] || [ "$TASK" = "flow_matching" ]; then
    # Mode specified directly (backward compatibility): view all logs (TensorBoard will show all)
    LOG_DIR="./logs/graph_dit"
elif [ "$TASK" = "reach" ] || [ "$TASK" = "lift" ]; then
    # Task specified
    if [ -z "$MODE" ]; then
        # No mode: view all logs (TensorBoard will show all subdirs)
        LOG_DIR="./logs/graph_dit"
    elif [ "$MODE" = "ddpm" ] || [ "$MODE" = "flow_matching" ]; then
        # Task + mode: view specific
        LOG_DIR="./logs/graph_dit/${TASK}_joint_${MODE}"
    else
        echo "❌ Error: Invalid mode '$MODE'. Use 'ddpm' or 'flow_matching'"
        exit 1
    fi
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
    echo "  $0 reach [ddpm|flow_matching]        # View reach task logs"
    echo "  $0 lift [ddpm|flow_matching]         # View lift task logs"
    echo "  $0 ddpm                               # View all DDPM logs (backward compat)"
    echo "  $0 flow_matching                      # View all Flow Matching logs (backward compat)"
    echo "  $0 <log_directory>                    # View specific directory"
    echo ""
    echo "Examples:"
    echo "  $0                                    # View all logs"
    echo "  $0 reach                              # View all reach logs"
    echo "  $0 lift flow_matching                 # View lift Flow Matching logs"
    echo "  $0 ./logs/graph_dit/lift_joint_ddpm/2025-01-10_15-30-45  # View specific run"
    exit 1
fi

# Start TensorBoard
tensorboard --logdir "$LOG_DIR" --port 6006

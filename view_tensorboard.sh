#!/bin/bash
# View TensorBoard for Graph DiT training logs

MODE="${1:-}"

# Auto-detect if mode is specified
if [ -z "$MODE" ]; then
    # Default: view all logs (both modes)
    LOG_DIR="./logs/graph_dit"
elif [ "$MODE" = "ddpm" ] || [ "$MODE" = "flow_matching" ]; then
    # View specific mode
    LOG_DIR="./logs/graph_dit/reach_joint_${MODE}"
else
    # Treat as directory path
    LOG_DIR="$MODE"
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
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ Error: Log directory not found: $LOG_DIR"
    echo ""
    echo "Usage:"
    echo "  $0                          # View all logs (both modes)"
    echo "  $0 ddpm                     # View DDPM logs only"
    echo "  $0 flow_matching            # View Flow Matching logs only"
    echo "  $0 <log_directory>          # View specific directory"
    echo ""
    echo "Examples:"
    echo "  $0                                    # View all logs"
    echo "  $0 ddpm                                # View DDPM logs"
    echo "  $0 ./logs/graph_dit/reach_joint_ddpm/2025-01-10_15-30-45  # View specific run"
    exit 1
fi

# Start TensorBoard
tensorboard --logdir "$LOG_DIR" --port 6006

#!/bin/bash
# 直接运行 RL 训练，使用最新的模型

# 找到最新的 Graph-DiT checkpoint
LATEST_DIT=$(find ./logs/graph_dit -name "best_model.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')

if [ -z "$LATEST_DIT" ]; then
    echo "❌ Error: No Graph-DiT checkpoint found in ./logs/graph_dit/"
    exit 1
fi

echo "📦 Using Graph-DiT checkpoint: $LATEST_DIT"

# 尝试找到最新的 gripper model（可选）
LATEST_GRIPPER=$(find ./logs/gripper -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')

if [ -n "$LATEST_GRIPPER" ]; then
    echo "📦 Using Gripper model: $LATEST_GRIPPER"
    GRIPPER_ARG="GRIPPER_MODEL=$LATEST_GRIPPER"
else
    echo "⚠️  No gripper model found, will use Graph-DiT for gripper"
    GRIPPER_ARG=""
fi

# 运行训练
echo ""
echo "🚀 Starting RL training..."
echo ""

$GRIPPER_ARG ./train_residual_rl.sh "$LATEST_DIT" 64 500 130 64 5 42 false


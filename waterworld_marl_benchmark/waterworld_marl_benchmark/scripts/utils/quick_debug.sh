#!/bin/bash

# 快速调试脚本
# 用法: ./quick_debug.sh

echo "🐛 快速调试模式"
echo "================="

# Stage 1.1 调试（只训练PPO，1000步）
python scripts/training/train_stage1_1.py \
    --mode debug \
    --algos PPO \
    --timesteps 1000

echo "✅ 调试完成"
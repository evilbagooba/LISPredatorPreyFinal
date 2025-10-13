#!/bin/bash

# 完整训练流程
# 用法: ./run_full_pipeline.sh [mode]
# mode: debug / dryrun / prod (默认: prod)

MODE=${1:-prod}

echo "🚀 开始完整训练流程"
echo "===================="
echo "运行模式: $MODE"
echo ""

# Stage 1.1: Prey预热
echo "📍 Stage 1.1: Prey预热训练"
python scripts/training/train_stage1_1.py --mode $MODE
if [ $? -ne 0 ]; then
    echo "❌ Stage 1.1 失败"
    exit 1
fi

echo ""
echo "✅ Stage 1.1 完成"
echo ""

# Stage 1.2: Predator引导
echo "📍 Stage 1.2: Predator引导训练"
python scripts/training/train_stage1_2.py --mode $MODE
if [ $? -ne 0 ]; then
    echo "❌ Stage 1.2 失败"
    exit 1
fi

echo ""
echo "✅ Stage 1.2 完成"
echo ""

# Stage 1.3: 共进化
echo "📍 Stage 1.3: 共进化训练"
python scripts/training/train_stage1_3.py --mode $MODE
if [ $? -ne 0 ]; then
    echo "❌ Stage 1.3 失败"
    exit 1
fi

echo ""
echo "✅ Stage 1.3 完成"
echo ""

echo "🎉 完整训练流程完成！"
echo "===================="
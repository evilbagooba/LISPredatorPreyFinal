#!/bin/bash
# ============================================================================
# 一键运行完整评估流程
# ============================================================================

set -e  # 遇到错误立即退出

# 默认参数
MODE="test"
N_EPISODES=20
ALGOS="PPO A2C SAC TD3 RANDOM"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --n-episodes)
            N_EPISODES="$2"
            shift 2
            ;;
        --algos)
            ALGOS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================================================="
echo "🚀 运行完整评估流程"
echo "=============================================================================="
echo "模式:        $MODE"
echo "Episodes:    $N_EPISODES"
echo "算法:        $ALGOS"
echo "=============================================================================="
echo ""

# Step 1: 交叉评估
echo "📊 Step 1: 执行交叉评估..."
echo "------------------------------------------------------------------------------"

python scripts/evaluation/cross_eval.py \
    --mode $MODE \
    --n-episodes $N_EPISODES \
    --algos $ALGOS

if [ $? -ne 0 ]; then
    echo "❌ 交叉评估失败"
    exit 1
fi

echo ""
echo "✅ 交叉评估完成"
echo ""

# Step 2: 查找最新的评估结果目录
echo "📂 Step 2: 查找评估结果..."
echo "------------------------------------------------------------------------------"

EVAL_DIR="${MODE}_outputs/evaluation_results"
LATEST_RESULT=$(ls -td $EVAL_DIR/cross_eval_* 2>/dev/null | head -1)

if [ -z "$LATEST_RESULT" ]; then
    echo "❌ 未找到评估结果目录"
    exit 1
fi

echo "找到最新结果: $LATEST_RESULT"
echo ""

# Step 3: 生成可视化
echo "🎨 Step 3: 生成可视化..."
echo "------------------------------------------------------------------------------"

python scripts/analysis/plot_results.py \
    --input "$LATEST_RESULT"

if [ $? -ne 0 ]; then
    echo "❌ 可视化生成失败"
    exit 1
fi

echo ""
echo "✅ 可视化生成完成"
echo ""

# Step 4: 显示结果摘要
echo "=============================================================================="
echo "📋 评估结果摘要"
echo "=============================================================================="

if [ -f "$LATEST_RESULT/results_summary.json" ]; then
    echo ""
    echo "Catch Rate Matrix (部分):"
    echo "------------------------------------------------------------------------------"
    head -10 "$LATEST_RESULT/catch_rate_matrix.csv" | column -t -s,
    echo ""
fi

if [ -f "$LATEST_RESULT/adaptability_scores.json" ]; then
    echo ""
    echo "Algorithm Adaptability Ranking:"
    echo "------------------------------------------------------------------------------"
    python -c "
import json
with open('$LATEST_RESULT/adaptability_scores.json') as f:
    data = json.load(f)
    ranking = data.get('ranking', [])
    print(f\"{'Rank':<6} {'Algorithm':<10} {'Adaptability':<15} {'Performance Drop':<18}\")
    print('-' * 50)
    for r in ranking:
        print(f\"{r['rank']:<6} {r['algorithm']:<10} {r['adaptability_score']:<15.3f} {r['performance_drop']:<18.3f}\")
" 2>/dev/null || echo "  (无法解析排名数据)"
    echo ""
fi

echo "=============================================================================="
echo "✅ 完整评估流程已完成！"
echo "=============================================================================="
echo ""
echo "📁 结果目录: $LATEST_RESULT"
echo ""
echo "生成的文件："
echo "  📊 数据文件："
echo "     - raw_results.pkl              (完整评估数据)"
echo "     - results_summary.json         (主要指标摘要)"
echo "     - adaptability_scores.json     (自适应性得分)"
echo "     - catch_rate_matrix.csv        (性能矩阵)"
echo ""
echo "  📈 可视化图表："
echo "     - heatmap_catch_rate.png       (性能热力图)"
echo "     - generalization_curves.png    (泛化曲线)"
echo "     - ranking_comparison.png       (算法排名对比)"
echo "     - performance_distribution.png (性能分布)"
echo ""
echo "查看图表："
echo "  open $LATEST_RESULT/*.png         # macOS"
echo "  xdg-open $LATEST_RESULT/*.png     # Linux"
echo ""
echo "=============================================================================="
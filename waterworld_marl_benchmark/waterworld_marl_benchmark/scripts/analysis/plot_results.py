"""
可视化分析结果
生成热力图、泛化曲线、雷达图等
"""

import sys
import argparse
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import config_loader


def parse_args():
    parser = argparse.ArgumentParser(description='可视化评估结果')
    parser.add_argument('--input', type=str, required=True,
                        help='交叉评估结果目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录（默认为input目录）')
    return parser.parse_args()


def load_results(results_dir: Path):
    """加载评估结果"""
    # 加载原始结果
    raw_path = results_dir / "raw_results.pkl"
    if raw_path.exists():
        with open(raw_path, 'rb') as f:
            results_matrix = pickle.load(f)
    else:
        print(f"❌ 未找到原始结果: {raw_path}")
        return None
    
    # 加载自适应性得分
    adapt_path = results_dir / "adaptability_scores.json"
    if adapt_path.exists():
        with open(adapt_path, 'r') as f:
            adaptability_data = json.load(f)
    else:
        adaptability_data = None
    
    return results_matrix, adaptability_data


def plot_heatmap(results_matrix, output_path, metric='catch_rate'):
    """
    绘制热力图
    
    Args:
        results_matrix: 结果矩阵
        output_path: 输出路径
        metric: 要显示的指标
    """
    # 提取算法列表
    algos = list(results_matrix.keys())
    
    # 构建矩阵
    matrix = np.zeros((len(algos), len(algos)))
    
    for i, pred_algo in enumerate(algos):
        for j, prey_algo in enumerate(algos):
            metrics = results_matrix[pred_algo].get(prey_algo)
            if metrics:
                matrix[i, j] = metrics.get(metric, 0)
            else:
                matrix[i, j] = np.nan
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    
    # 创建mask来标记对角线
    mask = np.zeros_like(matrix, dtype=bool)
    
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        xticklabels=algos,
        yticklabels=algos,
        vmin=0, vmax=1,
        cbar_kws={'label': 'Predator Catch Rate'},
        linewidths=0.5,
        linecolor='gray',
        mask=mask
    )
    
    # 标记对角线（In-Distribution）
    for i in range(len(algos)):
        plt.gca().add_patch(plt.Rectangle((i, i), 1, 1,
                            fill=False, edgecolor='blue', lw=3))
    
    plt.xlabel('Prey Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Predator Algorithm', fontsize=12, fontweight='bold')
    plt.title(f'Cross-Algorithm Performance Matrix\n'
              f'Metric: {metric.replace("_", " ").title()}\n'
              f'Blue boxes = In-Distribution (trained together)',
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 热力图: {output_path}")


def plot_generalization_curves(results_matrix, adaptability_data, output_path):
    """绘制泛化曲线"""
    
    # 加载策略距离配置
    eval_config = config_loader.load_yaml('evaluation/cross_eval.yaml')
    policy_distances = eval_config['analysis']['policy_distance']['distances']
    
    plt.figure(figsize=(10, 6))
    
    algos = [a for a in results_matrix.keys() if a != 'RANDOM']
    
    for pred_algo in algos:
        # 收集(距离, 性能)点对
        points = []
        
        for prey_algo in algos:
            if prey_algo in policy_distances.get(pred_algo, {}):
                distance = policy_distances[pred_algo][prey_algo]
                
                metrics = results_matrix[pred_algo].get(prey_algo)
                if metrics:
                    performance = metrics['catch_rate']
                    points.append((distance, performance))
        
        # 按距离排序
        points.sort(key=lambda x: x[0])
        
        if points:
            distances = [p[0] for p in points]
            performances = [p[1] for p in points]
            
            plt.plot(distances, performances, marker='o', linewidth=2,
                    label=f'{pred_algo}', markersize=8)
    
    plt.xlabel('Policy Distance from Training Distribution', fontsize=12)
    plt.ylabel('Catch Rate (Performance)', fontsize=12)
    plt.title('Generalization Curves: Performance vs Opponent Distance',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5,
                label='Balance Line (50%)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 泛化曲线: {output_path}")


def plot_ranking_bars(adaptability_data, output_path):
    """绘制排名柱状图"""
    
    if not adaptability_data:
        print("  ⚠️  跳过排名图: 无自适应性数据")
        return
    
    ranking = adaptability_data['ranking']
    
    algos = [r['algorithm'] for r in ranking]
    adapt_scores = [r['adaptability_score'] for r in ranking]
    in_dist = [r['in_dist_performance'] for r in ranking]
    ood_avg = [r['ood_avg_performance'] for r in ranking]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 自适应性得分
    ax1 = axes[0]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(algos)))
    bars1 = ax1.barh(algos, adapt_scores, color=colors)
    ax1.set_xlabel('Adaptability Score', fontsize=12)
    ax1.set_title('Algorithm Adaptability Ranking', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1.1)
    
    # 添加数值标签
    for bar, score in zip(bars1, adapt_scores):
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10)
    
    # 子图2: In-Dist vs OOD 性能
    ax2 = axes[1]
    x = np.arange(len(algos))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, in_dist, width, label='In-Distribution',
                    color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, ood_avg, width, label='Out-of-Distribution',
                    color='coral', alpha=0.8)
    
    ax2.set_ylabel('Catch Rate', fontsize=12)
    ax2.set_title('In-Distribution vs OOD Performance', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algos)
    ax2.legend()
    ax2.set_ylim(0, 0.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 排名图: {output_path}")


def plot_performance_distribution(results_matrix, output_path):
    """绘制性能分布箱线图"""
    
    algos = [a for a in results_matrix.keys() if a != 'RANDOM']
    
    # 收集每个算法的OOD性能
    ood_performances = {algo: [] for algo in algos}
    
    for pred_algo in algos:
        for prey_algo in algos:
            if prey_algo != pred_algo and prey_algo != 'RANDOM':
                metrics = results_matrix[pred_algo].get(prey_algo)
                if metrics:
                    ood_performances[pred_algo].append(metrics['catch_rate'])
    
    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    
    data = [ood_performances[algo] for algo in algos]
    positions = range(1, len(algos) + 1)
    
    bp = plt.boxplot(data, positions=positions, labels=algos,
                     patch_artist=True, widths=0.6)
    
    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(algos)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel('Catch Rate (OOD Performance)', fontsize=12)
    plt.xlabel('Predator Algorithm', fontsize=12)
    plt.title('Out-of-Distribution Performance Distribution\n'
              '(Performance against unseen opponents)',
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5,
                label='Balance Line')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 性能分布图: {output_path}")


def main():
    args = parse_args()
    
    results_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("📊 生成可视化结果")
    print("="*70)
    print(f"输入目录: {results_dir}")
    print(f"输出目录: {output_dir}")
    print("="*70 + "\n")
    
    # 加载结果
    print("📂 加载数据...")
    results_data = load_results(results_dir)
    
    if results_data is None:
        print("❌ 加载失败")
        return
    
    results_matrix, adaptability_data = results_data
    print("  ✓ 数据加载完成\n")
    
    # 生成各种图表
    print("🎨 生成图表...")
    
    # 1. 热力图
    plot_heatmap(
        results_matrix,
        output_dir / "heatmap_catch_rate.png",
        metric='catch_rate'
    )
    
    # 2. 泛化曲线
    plot_generalization_curves(
        results_matrix,
        adaptability_data,
        output_dir / "generalization_curves.png"
    )
    
    # 3. 排名柱状图
    if adaptability_data:
        plot_ranking_bars(
            adaptability_data,
            output_dir / "ranking_comparison.png"
        )
    
    # 4. 性能分布箱线图
    plot_performance_distribution(
        results_matrix,
        output_dir / "performance_distribution.png"
    )
    
    print("\n" + "="*70)
    print("✅ 可视化完成！")
    print("="*70)
    print(f"\n📁 图表保存在: {output_dir}")
    print("\n生成的图表:")
    print("  1. heatmap_catch_rate.png         - 性能热力图")
    print("  2. generalization_curves.png      - 泛化曲线")
    print("  3. ranking_comparison.png         - 算法排名对比")
    print("  4. performance_distribution.png   - 性能分布")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
"""
交叉评估主脚本
执行 5×5 算法对战矩阵评估
"""

import sys
import argparse
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from eval_single_matchup import evaluate_single_matchup
from metrics_calculator import MetricsCalculator
from src.utils.config_loader import config_loader


def parse_args():
    parser = argparse.ArgumentParser(description='交叉评估')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'dryrun', 'prod'],
                        help='评估模式')
    parser.add_argument('--n-episodes', type=int, default=None,
                        help='每个组合的评估episode数')
    parser.add_argument('--algos', type=str, nargs='+',
                        default=['PPO', 'A2C', 'SAC', 'TD3', 'RANDOM'],
                        help='要评估的算法列表')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    return parser.parse_args()


def find_model_path(base_dir: Path, algo: str, side: str) -> Path:
    """
    查找模型文件
    
    优先级：
    1. 最新的模型（按时间戳）
    2. 固定池中的模型
    """
    # 搜索路径
    search_patterns = [
        base_dir / f"stage1.*_{side}_*" / f"{algo}_{side}_*.zip",
        base_dir / f"*{side}*" / f"{algo}_{side}_*.zip",
        base_dir / f"**/{algo}_{side}_*.zip",
    ]
    
    found_models = []
    for pattern in search_patterns:
        found_models.extend(base_dir.glob(str(pattern.relative_to(base_dir))))
    
    if not found_models:
        return None
    
    # 返回最新的
    found_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return found_models[0]


def main():
    args = parse_args()
    
    # 加载配置
    eval_config = config_loader.load_yaml('evaluation/cross_eval.yaml')
    mode_config = eval_config['evaluation']['model_pools'][args.mode]
    
    # 参数
    algos = args.algos
    n_episodes = args.n_episodes or eval_config['evaluation']['n_episodes']
    output_dir = Path(args.output_dir) if args.output_dir else \
                 Path(f"{args.mode}_outputs/evaluation_results")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"cross_eval_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("🧪 交叉评估 (Cross-Algorithm Evaluation)")
    print("="*70)
    print(f"模式:        {args.mode}")
    print(f"算法:        {', '.join(algos)}")
    print(f"Episodes:    {n_episodes}")
    print(f"输出目录:    {run_dir}")
    print("="*70 + "\n")
    
    # 查找所有模型
    print("📂 查找模型文件...")
    
    saved_models_base = Path(mode_config['saved_models_base'])
    
    predator_models = {}
    prey_models = {}
    
    for algo in algos:
        if algo == 'RANDOM':
            predator_models[algo] = None
            prey_models[algo] = None
            print(f"  ✓ {algo}: 使用随机策略")
            continue
        
        # 查找 predator
        pred_path = find_model_path(saved_models_base, algo, 'predator')
        if pred_path:
            predator_models[algo] = pred_path
            print(f"  ✓ {algo}_predator: {pred_path.name}")
        else:
            print(f"  ✗ {algo}_predator: 未找到")
            predator_models[algo] = None
        
        # 查找 prey
        prey_path = find_model_path(saved_models_base, algo, 'prey')
        if prey_path:
            prey_models[algo] = prey_path
            print(f"  ✓ {algo}_prey: {prey_path.name}")
        else:
            print(f"  ✗ {algo}_prey: 未找到")
            prey_models[algo] = None
    
    # 执行交叉评估
    print(f"\n{'='*70}")
    print(f"🎯 开始交叉评估 ({len(algos)}×{len(algos)} = {len(algos)**2} 组合)")
    print(f"{'='*70}\n")
    
    results_matrix = {}
    total_matchups = len(algos) * len(algos)
    current_matchup = 0
    
    for pred_algo in algos:
        results_matrix[pred_algo] = {}
        
        for prey_algo in algos:
            current_matchup += 1
            print(f"\n[{current_matchup}/{total_matchups}] "
                  f"评估: {pred_algo}_pred vs {prey_algo}_prey")
            print("-" * 70)
            
            # 获取模型路径
            pred_path = predator_models.get(pred_algo)
            prey_path = prey_models.get(prey_algo)
            
            # 检查模型是否存在
            if pred_path is None and pred_algo != 'RANDOM':
                print(f"  ⚠️  跳过: {pred_algo}_predator 模型未找到")
                results_matrix[pred_algo][prey_algo] = None
                continue
            
            if prey_path is None and prey_algo != 'RANDOM':
                print(f"  ⚠️  跳过: {prey_algo}_prey 模型未找到")
                results_matrix[pred_algo][prey_algo] = None
                continue
            
            # 执行评估
            try:
                metrics = evaluate_single_matchup(
                    predator_model_path=pred_path,
                    prey_model_path=prey_path,
                    predator_algo=pred_algo,
                    prey_algo=prey_algo,
                    env_config_name='waterworld_fast',
                    n_episodes=n_episodes,
                    deterministic=True,
                    verbose=1
                )
                
                results_matrix[pred_algo][prey_algo] = metrics
                
            except Exception as e:
                print(f"  ❌ 评估失败: {e}")
                import traceback
                traceback.print_exc()
                results_matrix[pred_algo][prey_algo] = None
    
    # 保存原始结果
    print(f"\n{'='*70}")
    print("💾 保存结果...")
    print(f"{'='*70}")
    
    # 保存为pickle（完整数据）
    raw_results_path = run_dir / "raw_results.pkl"
    with open(raw_results_path, 'wb') as f:
        pickle.dump(results_matrix, f)
    print(f"  ✓ 原始结果: {raw_results_path}")
    
    # 保存为JSON（主要指标）
    json_results = {}
    for pred_algo, prey_dict in results_matrix.items():
        json_results[pred_algo] = {}
        for prey_algo, metrics in prey_dict.items():
            if metrics is not None:
                # 只保存主要指标
                json_results[pred_algo][prey_algo] = {
                    'catch_rate': metrics['catch_rate'],
                    'survival_rate': metrics['survival_rate'],
                    'pred_avg_reward': metrics['pred_avg_reward'],
                    'prey_avg_reward': metrics['prey_avg_reward'],
                    'balance_score': metrics['balance_score'],
                    'is_ood': metrics['is_ood']
                }
            else:
                json_results[pred_algo][prey_algo] = None
    
    json_path = run_dir / "results_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON摘要: {json_path}")
    
    # 计算自适应性得分
    print(f"\n{'='*70}")
    print("📊 计算自适应性得分...")
    print(f"{'='*70}\n")
    
    # 过滤掉None结果
    valid_results = {}
    for pred_algo, prey_dict in results_matrix.items():
        if pred_algo == 'RANDOM':
            continue
        valid_results[pred_algo] = {}
        for prey_algo, metrics in prey_dict.items():
            if metrics is not None:
                valid_results[pred_algo][prey_algo] = metrics
    
    if valid_results:
        adaptability_scores = MetricsCalculator.compute_adaptability_scores(valid_results)
        ranking = MetricsCalculator.compute_ranking(adaptability_scores)
        
        # 打印排名
        print(f"{'='*80}")
        print("Algorithm Adaptability Ranking (Predator)")
        print(f"{'='*80}")
        print(f"{'Rank':<6} {'Algorithm':<10} {'In-Dist':<10} {'OOD Avg':<10} "
              f"{'Adapt':<8} {'Drop':<8} {'Std':<8}")
        print("-"*80)
        
        for scores in ranking:
            print(f"{scores['rank']:<6} {scores['algorithm']:<10} "
                  f"{scores['in_dist_performance']:<10.3f} "
                  f"{scores['ood_avg_performance']:<10.3f} "
                  f"{scores['adaptability_score']:<8.3f} "
                  f"{scores['performance_drop']:<8.3f} "
                  f"{scores['ood_std']:<8.3f}")
        
        print(f"{'='*80}\n")
        
        # 保存自适应性得分
        adapt_path = run_dir / "adaptability_scores.json"
        with open(adapt_path, 'w', encoding='utf-8') as f:
            json.dump({
                'scores': adaptability_scores,
                'ranking': ranking
            }, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 自适应性得分: {adapt_path}")
    
    # 生成性能矩阵（CSV格式）
    print(f"\n{'='*70}")
    print("📋 生成性能矩阵...")
    print(f"{'='*70}\n")
    
    # Catch Rate矩阵
    catch_rate_matrix = []
    header = ['Predator\\Prey'] + algos
    catch_rate_matrix.append(header)
    
    for pred_algo in algos:
        row = [pred_algo]
        for prey_algo in algos:
            metrics = results_matrix[pred_algo].get(prey_algo)
            if metrics:
                row.append(f"{metrics['catch_rate']:.3f}")
            else:
                row.append("N/A")
        catch_rate_matrix.append(row)
    
    # 保存为CSV
    csv_path = run_dir / "catch_rate_matrix.csv"
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(catch_rate_matrix)
    print(f"  ✓ Catch Rate矩阵: {csv_path}")
    
    # 打印矩阵预览
    print("\n  Catch Rate Matrix Preview:")
    print("  " + "-" * 60)
    for row in catch_rate_matrix[:6]:  # 只显示前6行
        print("  " + " | ".join(f"{cell:>12}" for cell in row))
    print("  " + "-" * 60)
    
    # 保存配置快照
    config_snapshot = {
        'mode': args.mode,
        'algorithms': algos,
        'n_episodes': n_episodes,
        'timestamp': timestamp,
        'model_paths': {
            'predators': {k: str(v) if v else None for k, v in predator_models.items()},
            'preys': {k: str(v) if v else None for k, v in prey_models.items()}
        }
    }
    
    config_path = run_dir / "eval_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 配置快照: {config_path}")
    
    # 完成
    print(f"\n{'='*70}")
    print("✅ 交叉评估完成！")
    print(f"{'='*70}")
    print(f"\n📁 结果保存在: {run_dir}")
    print("\n下一步：")
    print(f"  1. 查看结果: cat {json_path}")
    print(f"  2. 生成可视化: python scripts/analysis/plot_results.py --input {run_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
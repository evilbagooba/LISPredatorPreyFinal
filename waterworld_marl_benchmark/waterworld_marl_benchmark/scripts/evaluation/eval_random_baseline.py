"""
Random Baseline 评估脚本
在训练前先评估随机策略的性能，建立基线
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.trainer import MultiAgentTrainer
from src.utils.config_loader import get_training_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Random Baseline 评估')
    
    parser.add_argument('--stage', type=str, required=True,
                        choices=['1.1', '1.2'],
                        help='训练阶段（1.1=Prey预热, 1.2=Predator引导）')
    
    parser.add_argument('--n-episodes', type=int, default=20,
                        help='评估episode数')
    
    parser.add_argument('--env-config', type=str, default='waterworld_standard',
                        help='环境配置')
    
    parser.add_argument('--device', type=str, default='cpu',
                        help='计算设备')
    
    parser.add_argument('--output-dir', type=str, default='outputs/baselines',
                        help='结果保存目录')
    
    return parser.parse_args()


def evaluate_random_baseline(stage: str, args, stage_config: dict):
    """
    评估 Random Baseline
    
    Args:
        stage: 训练阶段
        args: 命令行参数
        stage_config: 阶段配置
    """
    
    print(f"\n{'='*70}")
    print(f"🎲 Random Baseline 评估 - Stage {stage}")
    print(f"{'='*70}\n")
    
    # 根据阶段确定训练方和对手
    if stage == '1.1':
        train_side = 'prey'
        opponent_config = {
            'type': 'algorithm',
            'side': 'predator',
            'algorithm': 'RANDOM',
            'freeze': True
        }
        scenario = "Random Prey vs Random Predator"
    
    elif stage == '1.2':
        train_side = 'predator'
        prey_pool = 'outputs/fixed_pools/prey_pool_v1'
        
        # 检查 prey_pool 是否存在
        if not Path(prey_pool).exists():
            print(f"❌ Prey池不存在: {prey_pool}")
            print(f"   请先运行 Stage 1.1 训练")
            sys.exit(1)
        
        opponent_config = {
            'type': 'mixed_pool',
            'side': 'prey',
            'pool_path': prey_pool,
            'mix_strategy': {
                'fixed_ratio': 0.7,
                'sampling': 'uniform'
            },
            'freeze': True
        }
        scenario = "Random Predator vs prey_pool_v1"
    
    else:
        raise ValueError(f"未知的阶段: {stage}")
    
    print(f"场景: {scenario}")
    print(f"评估Episodes: {args.n_episodes}\n")
    
    # 创建训练器（使用 RANDOM 算法）
    trainer = MultiAgentTrainer(
        train_side=train_side,
        train_algo='RANDOM',
        opponent_config=opponent_config,
        experiment_name=f"random_baseline_stage{stage}",
        stage_name=f"baseline_stage{stage}",
        generation=0,
        version='baseline',
        run_mode='dryrun',  # 使用 dryrun 模式
        total_timesteps=0,  # 不训练，只评估
        device=args.device
    )
    
    # 设置环境
    trainer.setup()
    
    print(f"开始评估...")
    print(f"{'-'*70}\n")
    
    # 评估
    eval_results = trainer.evaluate(n_episodes=args.n_episodes)
    
    # 清理
    trainer.cleanup()
    
    return eval_results, scenario


def save_baseline_results(stage: str, scenario: str, results: dict, output_dir: Path):
    """保存基线结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"random_baseline_stage{stage}_{timestamp}.json"
    filepath = output_dir / filename
    
    # 构建完整记录
    record = {
        'timestamp': datetime.now().isoformat(),
        'stage': stage,
        'scenario': scenario,
        'baseline_type': 'RANDOM',
        'results': results
    }
    
    # 保存
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 基线结果已保存: {filepath}")
    
    return filepath


def print_baseline_summary(stage: str, scenario: str, results: dict):
    """打印基线摘要"""
    print(f"\n{'='*70}")
    print(f"📊 Random Baseline 结果 - Stage {stage}")
    print(f"{'='*70}")
    print(f"\n场景: {scenario}")
    print(f"\n性能指标:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  最大奖励: {results['max_reward']:.2f}")
    print(f"  最小奖励: {results['min_reward']:.2f}")
    print(f"  平均Episode长度: {results['mean_length']:.0f} ± {results['std_length']:.0f}")
    print(f"  评估Episodes: {results['n_episodes']}")
    print(f"\n{'='*70}")
    print(f"\n💡 提示:")
    print(f"   - 训练后的算法应该显著超过这个基线")
    print(f"   - 如果训练后性能 < 基线，说明训练可能有问题")
    print(f"   - 建议提升幅度: > 20% 为良好，> 50% 为优秀")
    print(f"{'='*70}\n")


def main():
    """主函数"""
    args = parse_args()
    
    # 加载阶段配置
    if args.stage == '1.1':
        stage_config = get_training_config('stage1_1_prey_warmup')
    elif args.stage == '1.2':
        stage_config = get_training_config('stage1_2_pred_guided')
    else:
        raise ValueError(f"未知的阶段: {args.stage}")
    
    # 评估
    eval_results, scenario = evaluate_random_baseline(args.stage, args, stage_config)
    
    # 保存结果
    save_baseline_results(args.stage, scenario, eval_results, args.output_dir)
    
    # 打印摘要
    print_baseline_summary(args.stage, scenario, eval_results)


if __name__ == "__main__":
    main()
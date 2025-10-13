"""
Stage 1.1: Prey预热训练
训练所有算法的Prey对抗RANDOM Predator
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.trainer import MultiAgentTrainer
from src.utils.config_loader import get_training_config, get_env_config  # ← 加上 get_env_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Stage 1.1: Prey预热训练')
    
    parser.add_argument('--mode', type=str, default='prod',
                        choices=['debug', 'dryrun', 'test', 'prod'],
                        help='运行模式')
    
    parser.add_argument('--algos', type=str, nargs='+',
                        default=['PPO', 'A2C', 'SAC', 'TD3'],
                        help='要训练的算法列表')
    
    parser.add_argument('--env-config', type=str, default='waterworld_standard',
                        help='环境配置')
    
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备')
    
    parser.add_argument('--timesteps', type=int, default=None,
                        help='总训练步数（覆盖配置）')
    
    return parser.parse_args()


def train_one_prey_algo(algo: str, args, stage_config: dict):
    """训练单个Prey算法"""
    
    print(f"\n{'='*70}")
    print(f"{'记录' if algo == 'RANDOM' else '训练'} {algo}_prey vs RANDOM_predator")
    print(f"{'='*70}\n")
    
    # 构建对手配置
    opponent_config = {
        'type': 'algorithm',
        'side': 'predator',
        'algorithm': 'RANDOM',
        'freeze': True
    }
    
    # 确定训练步数
    if algo == 'RANDOM':
        # RANDOM 只需要运行足够的 episodes 来记录基线
        # 例如运行 50k 步（大约 50 个 episodes）
        timesteps = 5000
    else:
        # 正常算法的训练步数
        timesteps = args.timesteps
    
    # 创建训练器
    # ✅ 根据模式选择环境配置
    if args.mode == 'test':
        env_config_name = 'waterworld_fast'
        print(f"🏃 测试模式，使用快速环境: max_cycles=500")
    else:
        env_config_name = 'waterworld_standard'
    
    # 创建训练器
    trainer = MultiAgentTrainer(
        train_side='prey',
        train_algo=algo,
        opponent_config=opponent_config,
        experiment_name=f"{algo}_prey_warmup",
        stage_name=stage_config['stage']['name'],
        generation=stage_config['stage']['generation'],
        version='v1',
        run_mode=args.mode,
        env_config=get_env_config(env_config_name),  # ✅ 使用动态环境
        total_timesteps=timesteps,
        device=args.device
    )
    
    # 获取冻结条件
    freeze_config = stage_config.get('freeze_on_success', {})
    freeze_criteria = freeze_config.get('criteria', {})
    
    # 运行训练
    # RANDOM 不需要保存到池
    save_to_pool = freeze_config.get('enabled', False) if algo != 'RANDOM' else False
    
    eval_results = trainer.run(
        save_to_pool=save_to_pool,
        pool_name=freeze_config.get('save_to_pool'),
        check_freeze=freeze_config.get('enabled', False) if algo != 'RANDOM' else False,
        freeze_criteria=freeze_criteria
    )
    
    return eval_results

def main():
    """主函数"""
    args = parse_args()
    
    # 加载Stage 1.1配置
    stage_config = get_training_config('stage1_1_prey_warmup')
    
    # 获取算法列表
    algos_to_train = args.algos or stage_config.get('algorithms_to_train', ['PPO', 'A2C', 'SAC', 'TD3'])
    
    print(f"\n{'='*70}")
    print(f"🎯 Stage 1.1: Prey预热训练")
    print(f"{'='*70}")
    print(f"运行模式: {args.mode}")
    print(f"训练算法: {', '.join(algos_to_train)}")
    print(f"{'='*70}\n")
    
    # ========== 新增：先运行 RANDOM 作为基线 ==========
    print(f"\n{'='*70}")
    print(f"步骤 0: 记录 RANDOM Baseline")
    print(f"{'='*70}\n")
    
    # 将 RANDOM 添加到训练列表的最前面
    all_algos = ['RANDOM'] + algos_to_train
    
    # ========== 训练所有算法（包括 RANDOM）==========
    results = {}
    for algo in all_algos:
        try:
            eval_results = train_one_prey_algo(algo, args, stage_config)
            results[algo] = eval_results
        except KeyboardInterrupt:
            print(f"\n⚠️  {algo} 训练被中断")
            break
        except Exception as e:
            print(f"\n❌ {algo} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印汇总（突出显示与 RANDOM 的对比）
    print(f"\n{'='*70}")
    print(f"✅ Stage 1.1 完成")
    print(f"{'='*70}")
    
    # 显示 RANDOM 基线
    if 'RANDOM' in results and results['RANDOM']:
        baseline_reward = results['RANDOM'].get('mean_reward', 0)
        print(f"\n📊 Random Baseline: {baseline_reward:.2f}")
        
        print(f"\n训练结果（相对于 Random）:")
        for algo in algos_to_train:  # 只显示训练的算法
            if algo in results and results[algo]:
                reward = results[algo].get('mean_reward', 0)
                improvement = ((reward - baseline_reward) / abs(baseline_reward) * 100) if baseline_reward != 0 else 0
                status = "✅" if improvement > 0 else "❌"
                print(f"  {status} {algo}: {reward:.2f} ({improvement:+.1f}% vs Random)")
    else:
        print(f"\n训练结果:")
        for algo, result in results.items():
            if result:
                print(f"  {algo}: 平均奖励 = {result.get('mean_reward', 'N/A'):.2f}")
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
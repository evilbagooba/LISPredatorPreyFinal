"""
Stage 1.3: 共进化训练
Predator和Prey交替训练，共同进化
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
    parser = argparse.ArgumentParser(description='Stage 1.3: 共进化训练')
    
    parser.add_argument('--mode', type=str, default='prod',
                        choices=['debug', 'dryrun','test',  'prod'],
                        help='运行模式')
    
    parser.add_argument('--max-generations', type=int, default=20,
                        help='最大代数')
    
    parser.add_argument('--start-generation', type=int, default=2,
                        help='起始代数')
    
    parser.add_argument('--algos', type=str, nargs='+',
                        default=['PPO', 'A2C', 'SAC', 'TD3'],
                        help='要训练的算法列表')
    
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备')
    
    parser.add_argument('--timesteps-per-gen', type=int, default=None,
                        help='每代的训练步数')
    
    return parser.parse_args()


def train_one_generation(
    generation: int,
    train_side: str,
    opponent_pool_path: str,
    algos: list,
    args,
    stage_config: dict
):
    """训练一代"""
    
    print(f"\n{'='*70}")
    print(f"🔄 Generation {generation}: 训练 {train_side.upper()}")
    print(f"{'='*70}\n")
    
    results = {}
    
    for algo in algos:
        print(f"\n{'-'*70}")
        print(f"训练 {algo}_{train_side} (Gen {generation})")
        print(f"{'-'*70}\n")
        
        # 构建对手配置
        opponent_side = 'prey' if train_side == 'predator' else 'predator'
        opponent_config = {
            'type': 'mixed_pool',
            'side': opponent_side,
            'pool_path': opponent_pool_path,
            'mix_strategy': {
                'fixed_ratio': 0.7,
                'sampling': 'uniform'
            },
            'freeze': True
        }
        
        # ✅ 根据模式选择环境配置
        if args.mode == 'test':
            env_config_name = 'waterworld_fast'
            print(f"🏃 测试模式，使用快速环境: max_cycles=500")
        else:
            env_config_name = 'waterworld_standard'
        
        # 创建训练器
        trainer = MultiAgentTrainer(
            train_side=train_side,
            train_algo=algo,
            opponent_config=opponent_config,
            experiment_name=f"{algo}_{train_side}_coevo",
            stage_name=f"{stage_config['stage']['name']}/Gen_{generation}",
            generation=generation,
            version=f"v{generation}",
            run_mode=args.mode,
            env_config=get_env_config(env_config_name),  # ✅ 使用动态环境
            total_timesteps=args.timesteps_per_gen,
            device=args.device
        )
        
        # 获取冻结条件
        freeze_config = stage_config.get('freeze_on_success', {})
        freeze_criteria = freeze_config.get('criteria', {}).get(train_side, {})
        
        # 运行训练
        try:
            eval_results = trainer.run(
                save_to_pool=freeze_config.get('enabled', False),
                pool_name=f"{train_side}_pool_v{generation}",
                check_freeze=freeze_config.get('enabled', False),
                freeze_criteria=freeze_criteria
            )
            results[algo] = eval_results
        
        except Exception as e:
            print(f"\n❌ {algo} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            results[algo] = None
    
    return results


def check_convergence(generation_results: list, threshold: float = 0.03) -> bool:
    """
    检查是否收敛
    
    Args:
        generation_results: 最近几代的结果列表
        threshold: 性能变化阈值
    
    Returns:
        是否收敛
    """
    if len(generation_results) < 5:
        return False
    
    # 取最近5代的平均奖励
    recent_rewards = []
    for gen_result in generation_results[-5:]:
        rewards = [r.get('mean_reward', 0) for r in gen_result.values() if r]
        if rewards:
            recent_rewards.append(sum(rewards) / len(rewards))
    
    if len(recent_rewards) < 5:
        return False
    
    # 计算变化率
    mean_reward = sum(recent_rewards) / len(recent_rewards)
    max_deviation = max(abs(r - mean_reward) for r in recent_rewards)
    
    change_rate = max_deviation / (abs(mean_reward) + 1e-6)
    
    print(f"\n📊 收敛检查:")
    print(f"  最近5代平均奖励: {recent_rewards}")
    print(f"  均值: {mean_reward:.2f}")
    print(f"  最大偏差: {max_deviation:.2f}")
    print(f"  变化率: {change_rate:.2%}")
    print(f"  阈值: {threshold:.2%}")
    
    return change_rate < threshold


def main():
    """主函数"""
    args = parse_args()
    # ✅ 根据运行模式选择配置文件
    if args.mode == 'test':
        config_name = 'stage1_3_coevolution_test'  # 使用测试配置
        print("🧪 TEST模式：使用测试配置（1代，2算法）")
    else:
        config_name = 'stage1_3_coevolution'        # 使用正式配置
    # 检查初始池是否存在
    prey_pool_path = Path("outputs/fixed_pools/prey_pool_v1")
    pred_pool_path = Path("outputs/fixed_pools/predator_pool_v1")
    
    if not prey_pool_path.exists() or not pred_pool_path.exists():
        print("❌ 初始对手池不存在")
        print(f"  Prey池: {prey_pool_path} - {'✓' if prey_pool_path.exists() else '✗'}")
        print(f"  Predator池: {pred_pool_path} - {'✓' if pred_pool_path.exists() else '✗'}")
        print("\n请先运行 Stage 1.1 和 Stage 1.2")
        sys.exit(1)
    
    # 加载Stage 1.3配置
    stage_config = get_training_config(config_name)
    
    # 获取配置
    coevo_config = stage_config.get('coevolution', {})
    max_generations = args.max_generations or coevo_config.get('max_generations', 20)
    start_generation = args.start_generation or coevo_config.get('start_generation', 2)
    
    algos_to_train = args.algos or stage_config.get('algorithms_to_train', ['PPO', 'A2C', 'SAC', 'TD3'])
    
    print(f"\n{'='*70}")
    print(f"🎯 Stage 1.3: 共进化训练")
    print(f"{'='*70}")
    print(f"运行模式: {args.mode}")
    print(f"训练算法: {', '.join(algos_to_train)}")
    print(f"代数范围: {start_generation} - {max_generations}")
    print(f"{'='*70}\n")
    
    # 记录所有代的结果
    all_results = []
    
    # 共进化循环
    for generation in range(start_generation, max_generations + 1):
        
        # 奇偶代交替训练
        if generation % 2 == 0:
            # 偶数代：训练Predator
            train_side = 'predator'
            opponent_pool = str(prey_pool_path)
        else:
            # 奇数代：训练Prey
            train_side = 'prey'
            opponent_pool = str(pred_pool_path)
        
        # 训练当前代
        try:
            gen_results = train_one_generation(
                generation=generation,
                train_side=train_side,
                opponent_pool_path=opponent_pool,
                algos=algos_to_train,
                args=args,
                stage_config=stage_config
            )
            
            all_results.append(gen_results)
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Generation {generation} 被中断")
            break
        
        except Exception as e:
            print(f"\n❌ Generation {generation} 失败: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # 检查收敛
        convergence_config = coevo_config.get('convergence', {})
        if convergence_config.get('enabled', True):
            if check_convergence(
                all_results,
                threshold=convergence_config.get('performance_change_threshold', 0.03)
            ):
                print(f"\n✅ 在 Generation {generation} 达到收敛")
                break
    
    # 打印最终汇总
    print(f"\n{'='*70}")
    print(f"✅ Stage 1.3 完成")
    print(f"{'='*70}")
    print(f"总共训练了 {len(all_results)} 代")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
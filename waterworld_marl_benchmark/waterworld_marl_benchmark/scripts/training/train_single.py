"""
单次训练脚本（通用）
可用于训练任意配置的单个模型
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.trainer import MultiAgentTrainer
from src.utils.config_loader import get_env_config, get_training_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='单次训练脚本')
    
    # 核心参数
    parser.add_argument('--train-side', type=str, required=True,
                        choices=['predator', 'prey'],
                        help='训练方')
    parser.add_argument('--train-algo', type=str, required=True,
                        choices=['PPO', 'A2C', 'SAC', 'TD3', 'RANDOM'],
                        help='训练算法')
    parser.add_argument('--opponent-type', type=str, required=True,
                        choices=['algorithm', 'fixed_model', 'mixed_pool'],
                        help='对手类型')
    
    # 对手配置
    parser.add_argument('--opponent-algo', type=str, default='RANDOM',
                        help='对手算法（当opponent-type=algorithm时使用）')
    parser.add_argument('--opponent-model', type=str, default=None,
                        help='对手模型路径（当opponent-type=fixed_model时使用）')
    parser.add_argument('--opponent-pool', type=str, default=None,
                        help='对手池路径（当opponent-type=mixed_pool时使用）')
    parser.add_argument('--fixed-ratio', type=float, default=0.7,
                        help='固定对手占比（当opponent-type=mixed_pool时使用）')
    
    # 实验元数据
    parser.add_argument('--experiment-name', type=str, required=True,
                        help='实验名称')
    parser.add_argument('--stage-name', type=str, required=True,
                        help='训练阶段名称')
    parser.add_argument('--version', type=str, default='v1',
                        help='版本号')
    parser.add_argument('--generation', type=int, default=0,
                        help='代数')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='prod',
                        choices=['debug', 'dryrun','test',  'prod'],
                        help='运行模式')
    
    # 环境配置
    parser.add_argument('--env-config', type=str, default='waterworld_standard',
                        help='环境配置名称')
    
    # 训练配置（覆盖默认值）
    parser.add_argument('--timesteps', type=int, default=None,
                        help='总训练步数')
    parser.add_argument('--n-envs', type=int, default=None,
                        help='并行环境数')
    parser.add_argument('--eval-freq', type=int, default=None,
                        help='评估频率')
    parser.add_argument('--checkpoint-freq', type=int, default=None,
                        help='检查点频率')
    
    # 其他
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备（auto/cuda/cpu）')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--notes', type=str, default='',
                        help='实验备注')
    
    # 保存选项
    parser.add_argument('--save-to-pool', action='store_true',
                        help='是否保存到固定池')
    parser.add_argument('--pool-name', type=str, default=None,
                        help='固定池名称')
    parser.add_argument('--check-freeze', action='store_true',
                        help='是否检查冻结条件')
    
    return parser.parse_args()


def build_opponent_config(args):
    """根据参数构建对手配置"""
    
    # 对手角色（与训练方相反）
    opponent_side = 'prey' if args.train_side == 'predator' else 'predator'
    
    if args.opponent_type == 'algorithm':
        return {
            'type': 'algorithm',
            'side': opponent_side,
            'algorithm': args.opponent_algo,
            'freeze': True
        }
    
    elif args.opponent_type == 'fixed_model':
        if not args.opponent_model:
            raise ValueError("使用fixed_model时必须指定--opponent-model")
        
        return {
            'type': 'fixed_model',
            'side': opponent_side,
            'path': args.opponent_model,
            'freeze': True
        }
    
    elif args.opponent_type == 'mixed_pool':
        if not args.opponent_pool:
            raise ValueError("使用mixed_pool时必须指定--opponent-pool")
        
        return {
            'type': 'mixed_pool',
            'side': opponent_side,
            'pool_path': args.opponent_pool,
            'mix_strategy': {
                'fixed_ratio': args.fixed_ratio,
                'sampling': 'uniform'
            },
            'freeze': True
        }
    
    else:
        raise ValueError(f"未知的对手类型: {args.opponent_type}")


def main():
    """主函数"""
    args = parse_args()
    
    # 构建对手配置
    opponent_config = build_opponent_config(args)
    
    # 加载环境配置
    env_config = get_env_config(args.env_config)
    
    # 创建训练器
    trainer = MultiAgentTrainer(
        train_side=args.train_side,
        train_algo=args.train_algo,
        opponent_config=opponent_config,
        experiment_name=args.experiment_name,
        stage_name=args.stage_name,
        generation=args.generation,
        version=args.version,
        run_mode=args.mode,
        env_config=env_config,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device,
        seed=args.seed,
        notes=args.notes
    )
    
    # 运行训练
    freeze_criteria = None
    if args.check_freeze:
        from src.utils.config_loader import config_loader
        run_modes_config = config_loader.load_yaml("run_modes.yaml")
        freeze_criteria = run_modes_config['freeze_criteria'][args.train_side]
    
    trainer.run(
        save_to_pool=args.save_to_pool,
        pool_name=args.pool_name,
        check_freeze=args.check_freeze,
        freeze_criteria=freeze_criteria
    )
    
    print("\n✅ 训练完成！")


if __name__ == "__main__":
    main()
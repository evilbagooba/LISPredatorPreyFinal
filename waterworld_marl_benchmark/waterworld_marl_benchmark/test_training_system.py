"""
训练系统综合测试
验证所有组件是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有模块导入"""
    print("\n" + "="*70)
    print("测试模块导入")
    print("="*70)
    
    try:
        # 核心模块
        from src.core import (
            WaterworldEnvManager,
            OpponentPool,
            MultiAgentTrainer,
            AgentManager
        )
        print("  ✓ 核心模块")
        
        # 算法模块
        from src.algorithms import (
            create_algorithm,
            PPOWrapper,
            A2CWrapper,
            SACWrapper,
            TD3Wrapper,
            RandomPolicy
        )
        print("  ✓ 算法模块")
        
        # 回调模块
        from src.callbacks import (
            MultiAgentTensorBoardCallback,
            CheckpointCallback,
            EvalCallback,
            FreezeCallback,
            ProgressBarCallback
        )
        print("  ✓ 回调模块")
        
        # 工具模块
        from src.utils.config_loader import get_mode_config, get_env_config, get_algo_config
        from src.utils.path_manager import PathManager
        from src.utils.naming import FileNaming
        from src.utils.logger import create_logger
        from src.utils.config_validator import validator
        print("  ✓ 工具模块")
        
        print("\n✅ 所有模块导入成功！\n")
        return True
    
    except Exception as e:
        print(f"\n❌ 模块导入失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_config_system():
    """测试配置系统"""
    print("="*70)
    print("测试配置系统")
    print("="*70)
    
    try:
        from src.utils.config_loader import (
            get_mode_config,
            get_env_config,
            get_algo_config,
            get_training_config
        )
        
        # 测试运行模式配置
        debug_config = get_mode_config("debug")
        assert 'total_timesteps' in debug_config
        print("  ✓ 运行模式配置")
        
        # 测试环境配置
        env_config = get_env_config("waterworld_standard")
        assert 'environment' in env_config
        print("  ✓ 环境配置")
        
        # 测试算法配置
        ppo_config = get_algo_config("PPO")
        assert 'hyperparameters' in ppo_config
        print("  ✓ 算法配置")
        
        # 测试训练配置
        stage_config = get_training_config("stage1_1_prey_warmup")
        assert 'stage' in stage_config
        print("  ✓ 训练阶段配置")
        
        print("\n✅ 配置系统测试通过！\n")
        return True
    
    except Exception as e:
        print(f"\n❌ 配置系统测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_environment_creation():
    """测试环境创建"""
    print("="*70)
    print("测试环境创建")
    print("="*70)
    
    try:
        from src.core import WaterworldEnvManager
        from src.utils.config_loader import get_env_config
        
        # 创建环境管理器
        env_config = get_env_config("waterworld_fast")  # 使用快速环境
        env_manager = WaterworldEnvManager(env_config)
        print("  ✓ 环境管理器创建")
        
        # 创建环境
        env = env_manager.create_env()
        print("  ✓ 环境创建")
        
        # 测试重置
        obs, infos = env.reset()
        print(f"  ✓ 环境重置 (agents: {len(env.agents)})")
        
        # 测试单步
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        print("  ✓ 环境单步")
        
        # 清理
        env_manager.close()
        print("  ✓ 环境清理")
        
        print("\n✅ 环境创建测试通过！\n")
        return True
    
    except Exception as e:
        print(f"\n❌ 环境创建测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_algorithm_creation():
    """测试算法创建"""
    print("="*70)
    print("测试算法创建")
    print("="*70)
    
    try:
        from src.algorithms import create_algorithm
        from src.core import WaterworldEnvManager
        from src.utils.config_loader import get_env_config, get_algo_config
        import gymnasium as gym
        
        # 创建环境获取空间信息
        env_config = get_env_config("waterworld_fast")
        env_manager = WaterworldEnvManager(env_config)
        env_manager.create_env()
        
        obs_space = env_manager.get_observation_space("predator")
        action_space = env_manager.get_action_space("predator")
        
        # 测试每个算法
        for algo_name in ['PPO', 'A2C', 'SAC', 'TD3', 'RANDOM']:
            algo_config = get_algo_config(algo_name)
            algorithm = create_algorithm(
                algo_name=algo_name,
                observation_space=obs_space,
                action_space=action_space,
                config=algo_config,
                device='cpu'
            )
            print(f"  ✓ {algo_name} 算法创建")
        
        env_manager.close()
        
        print("\n✅ 算法创建测试通过！\n")
        return True
    
    except Exception as e:
        print(f"\n❌ 算法创建测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_initialization():
    """测试训练器初始化"""
    print("="*70)
    print("测试训练器初始化")
    print("="*70)
    
    try:
        from src.core import MultiAgentTrainer
        
        # 创建最小配置的训练器
        trainer = MultiAgentTrainer(
            train_side='prey',
            train_algo='PPO',
            opponent_config={
                'type': 'algorithm',
                'side': 'predator',
                'algorithm': 'RANDOM',
                'freeze': True
            },
            experiment_name='test_experiment',
            stage_name='test_stage',
            generation=0,
            version='v1',
            run_mode='debug',
            total_timesteps=100  # 极少步数
        )
        print("  ✓ 训练器初始化")
        
        # 测试setup（不实际运行训练）
        trainer.setup()
        print("  ✓ 训练器设置")
        
        # 清理
        trainer.cleanup()
        print("  ✓ 训练器清理")
        
        print("\n✅ 训练器初始化测试通过！\n")
        return True
    
    except Exception as e:
        print(f"\n❌ 训练器初始化测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training():
    """测试完整的迷你训练流程"""
    print("="*70)
    print("测试迷你训练流程")
    print("="*70)
    print("这将运行一个超短的训练（200步）来验证完整流程")
    print("-"*70)
    
    try:
        from src.core import MultiAgentTrainer
        
        # 创建训练器
        trainer = MultiAgentTrainer(
            train_side='prey',
            train_algo='PPO',
            opponent_config={
                'type': 'algorithm',
                'side': 'predator',
                'algorithm': 'RANDOM',
                'freeze': True
            },
            experiment_name='mini_test',
            stage_name='test_mini_training',
            generation=0,
            version='v1',
            run_mode='debug',
            total_timesteps=200,  # 只训练200步
            n_envs=1,
            eval_freq=-1,  # 禁用评估
            checkpoint_freq=-1  # 禁用检查点
        )
        
        print("\n开始迷你训练...")
        
        # 设置
        trainer.setup()
        
        # 训练
        trainer.train()
        
        # 评估
        eval_results = trainer.evaluate(n_episodes=2)
        
        # 保存（但不加入池）
        trainer.save_model(save_to_pool=False)
        
        # 保存摘要
        trainer.save_training_summary()
        
        # 清理
        trainer.cleanup()
        
        print("\n✅ 迷你训练流程测试通过！")
        mean_reward = eval_results.get('mean_reward', None)
        if mean_reward is not None:
            print(f"   平均奖励: {mean_reward:.2f}")
        else:
            print(f"   平均奖励: N/A")
        print()
        return True
    
    except Exception as e:
        print(f"\n❌ 迷你训练流程测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🧪 训练系统综合测试")
    print("="*70 + "\n")
    
    tests = [
        ("模块导入", test_imports),
        ("配置系统", test_config_system),
        ("环境创建", test_environment_creation),
        ("算法创建", test_algorithm_creation),
        ("训练器初始化", test_trainer_initialization),
        ("迷你训练流程", test_mini_training),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}测试崩溃: {e}")
            results[test_name] = False
    
    # 打印汇总
    print("="*70)
    print("测试结果汇总")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20s}: {status}")
    
    print("="*70)
    
    # 统计
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"\n总计: {total} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    
    if failed == 0:
        print("\n🎉 所有测试通过！系统已准备就绪！")
        return 0
    else:
        print(f"\n⚠️  {failed} 个测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
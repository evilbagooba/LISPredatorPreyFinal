"""
测试配置系统是否正常工作
"""

from src.utils.config_loader import (
    get_mode_config,
    get_env_config,
    get_algo_config,
    get_training_config
)
from src.utils.path_manager import PathManager
from src.utils.naming import FileNaming
from src.utils.banner import print_mode_banner

def test_config_loading():
    """测试配置加载"""
    print("="*70)
    print("测试配置加载")
    print("="*70)
    
    # 测试运行模式配置
    print("\n1. 加载debug模式配置...")
    debug_config = get_mode_config("debug")
    print(f"   ✓ Debug训练步数: {debug_config['total_timesteps']}")
    
    # 测试环境配置
    print("\n2. 加载环境配置...")
    env_config = get_env_config("waterworld_standard")
    print(f"   ✓ Predator数量: {env_config['environment']['n_predators']}")
    
    # 测试算法配置
    print("\n3. 加载算法配置...")
    ppo_config = get_algo_config("PPO")
    print(f"   ✓ PPO学习率: {ppo_config['hyperparameters']['learning_rate']}")
    
    # 测试训练配置
    print("\n4. 加载训练配置...")
    stage_config = get_training_config("stage1_1_prey_warmup")
    print(f"   ✓ Stage名称: {stage_config['stage']['name']}")
    
    print("\n✅ 配置加载测试通过！\n")


def test_path_management():
    """测试路径管理"""
    print("="*70)
    print("测试路径管理")
    print("="*70)
    
    for mode in ["debug", "dryrun", "prod"]:
        print(f"\n{mode.upper()} 模式:")
        pm = PathManager(mode, "test_experiment")
        print(f"  模型目录: {pm.get_model_dir()}")
        print(f"  日志目录: {pm.get_tensorboard_dir()}")
    
    print("\n✅ 路径管理测试通过！\n")


def test_naming():
    """测试文件命名"""
    print("="*70)
    print("测试文件命名")
    print("="*70)
    
    naming = FileNaming()
    
    # 测试模型文件名生成
    filename = naming.generate_model_filename(
        train_algo="PPO",
        train_side="prey",
        version="v1",
        opponent_info="RANDOM_pred",
        run_mode="debug"
    )
    print(f"\nDebug模式文件名: {filename}")
    
    filename = naming.generate_model_filename(
        train_algo="SAC",
        train_side="predator",
        version="v2",
        opponent_info="MIX_prey_pool_v1",
        run_mode="prod"
    )
    print(f"Prod模式文件名: {filename}")
    
    print("\n✅ 文件命名测试通过！\n")


def test_banner():
    """测试横幅显示"""
    print("="*70)
    print("测试横幅显示")
    print("="*70)
    
    for mode in ["debug", "dryrun"]:  # prod需要确认，跳过
        config = get_mode_config(mode)
        print_mode_banner(mode, config)
    
    print("✅ 横幅显示测试通过！\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 配置系统测试")
    print("="*70 + "\n")
    
    try:
        test_config_loading()
        test_path_management()
        test_naming()
        test_banner()
        
        print("="*70)
        print("✅ 所有测试通过！配置系统工作正常")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}\n")
        import traceback
        traceback.print_exc()
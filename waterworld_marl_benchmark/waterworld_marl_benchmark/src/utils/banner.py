"""
运行模式横幅显示
提醒用户当前运行模式
"""

import sys
from typing import Dict, Any


def print_mode_banner(run_mode: str, config: Dict[str, Any]):
    """
    打印运行模式横幅
    
    Args:
        run_mode: 运行模式
        config: 模式配置
    """
    
    if run_mode == "debug":
        print("\n" + "="*70)
        print("🐛 DEBUG MODE - 调试模式")
        print("="*70)
        print(f"  训练步数: {config.get('total_timesteps', 'N/A')}")
        print(f"  并行环境: {config.get('n_envs', 'N/A')}")
        print(f"  保存模型: {'否' if not config.get('save_final_model', False) else '是'}")
        print(f"  输出目录: {config.get('output_base_dir', 'N/A')}")
        print(f"  TensorBoard: {'启用' if config.get('tensorboard_enabled', False) else '禁用'}")
        print("\n  ⚠️  此模式数据将被定期清理，不用于正式实验！")
        print("="*70 + "\n")
    
    elif run_mode == "dryrun":
        print("\n" + "="*70)
        print("🧪 DRYRUN MODE - 预演模式")
        print("="*70)
        print(f"  训练步数: {config.get('total_timesteps', 'N/A')}")
        print(f"  并行环境: {config.get('n_envs', 'N/A')}")
        print(f"  保存模型: 是（标记为DRYRUN）")
        print(f"  输出目录: {config.get('output_base_dir', 'N/A')}")
        print(f"  TensorBoard: {'启用' if config.get('tensorboard_enabled', False) else '禁用'}")
        print("\n  ℹ️  用于验证完整流程，数据保留最近 {config.get('max_runs', 3)} 次")
        print("="*70 + "\n")
    elif run_mode == "test":  # ✅ 新增
        print("\n" + "="*70)
        print("🧪 TEST MODE - 快速流程测试")
        print("="*70)
        print(f"  训练步数: {config.get('total_timesteps', 'N/A')}")
        print(f"  并行环境: {config.get('n_envs', 'N/A')}")
        print(f"  保存模型: 是（标记为TEST）")
        print(f"  输出目录: {config.get('output_base_dir', 'N/A')}")
        print(f"  TensorBoard: {'启用' if config.get('tensorboard_enabled', False) else '禁用'}")
        print("\n  ℹ️  用于测试完整训练流程，数据保留在 test_outputs/")
        print("="*70 + "\n")    
    else:  # production
        print("\n" + "="*70)
        print("✅ PRODUCTION MODE - 生产模式")
        print("="*70)
        print(f"  训练步数: {config.get('total_timesteps', 'N/A')}")
        print(f"  并行环境: {config.get('n_envs', 'N/A')}")
        print(f"  保存模型: 是")
        print(f"  输出目录: {config.get('output_base_dir', 'N/A')}")
        print(f"  TensorBoard: 启用")
        print("\n  🚨 所有数据将被永久保存，请确认配置无误！")
        print("="*70 + "\n")
        
        # 生产模式需要确认
        if config.get("require_confirmation", True):
            response = input("确认开始正式实验？(yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ 已取消")
                sys.exit(0)


def print_training_start(
    algo: str,
    side: str,
    version: str,
    opponent_info: str
):
    """
    打印训练开始信息
    
    Args:
        algo: 算法名称
        side: 训练方（predator/prey）
        version: 版本号
        opponent_info: 对手信息
    """
    print("\n" + "="*70)
    print(f"🚀 开始训练")
    print("="*70)
    print(f"  算法: {algo}")
    print(f"  角色: {side}")
    print(f"  版本: {version}")
    print(f"  对手: {opponent_info}")
    print("="*70 + "\n")


def print_training_complete(
    algo: str,
    side: str,
    total_steps: int,
    time_elapsed: float
):
    """
    打印训练完成信息
    
    Args:
        algo: 算法名称
        side: 训练方
        total_steps: 总训练步数
        time_elapsed: 训练耗时（秒）
    """
    hours = int(time_elapsed // 3600)
    minutes = int((time_elapsed % 3600) // 60)
    seconds = int(time_elapsed % 60)
    
    print("\n" + "="*70)
    print(f"✅ 训练完成！")
    print("="*70)
    print(f"  算法: {algo}")
    print(f"  角色: {side}")
    print(f"  总步数: {total_steps:,}")
    print(f"  耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print("="*70 + "\n")


def print_evaluation_start(n_episodes: int):
    """打印评估开始信息"""
    print("\n" + "-"*70)
    print(f"📊 开始评估 ({n_episodes} episodes)")
    print("-"*70)


def print_evaluation_results(metrics: Dict[str, Any]):
    """
    打印评估结果
    
    Args:
        metrics: 评估指标字典
    """
    print("\n" + "-"*70)
    print("📊 评估结果")
    print("-"*70)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("-"*70 + "\n")


def print_freeze_decision(
    algo: str,
    side: str,
    is_frozen: bool,
    reason: str = ""
):
    """
    打印冻结决策
    
    Args:
        algo: 算法名称
        side: 训练方
        is_frozen: 是否冻结
        reason: 原因说明
    """
    if is_frozen:
        print("\n" + "="*70)
        print(f"❄️  模型冻结：{algo}_{side}")
        print("="*70)
        print(f"  ✅ 达到冻结标准，已加入固定池")
        if reason:
            print(f"  原因: {reason}")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print(f"⚠️  模型未冻结：{algo}_{side}")
        print("="*70)
        print(f"  ❌ 未达到冻结标准")
        if reason:
            print(f"  原因: {reason}")
        print("="*70 + "\n")
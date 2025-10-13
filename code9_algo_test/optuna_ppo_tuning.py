"""
Optuna 超参数优化 - PPO 第一阶段
优化目标：训练智能体的平均奖励
优化参数：lr, eps_clip, gae_lambda, ent_coef, vf_coef
"""

import optuna
from optuna.trial import Trial
import argparse
import torch
import numpy as np
from datetime import datetime
import os
import json
from typing import Dict, Any

# 导入你的训练函数
from baselineonplicy_testoneprey import train_agent, get_args, get_parser


def create_trial_args(trial: Trial, base_args: argparse.Namespace):
    args = argparse.Namespace(**vars(base_args))
    
    # ✅ 降低学习率范围
    args.lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)  # 从 1e-3 降到 5e-4
    
    args.eps_clip = trial.suggest_float("eps_clip", 0.1, 0.3)
    args.gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
    args.ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)  # 从 1e-1 降到 1e-2
    
    # ✅ 降低 vf_coef 范围
    args.vf_coef = trial.suggest_float("vf_coef", 0.1, 0.5)  # 从 1.0 降到 0.5
    
    # === 固定其他参数以加速训练 ===
    
    # 减少训练轮数进行快速评估（第一阶段）
    args.epoch = 20  # 从 100 减少到 20
    args.step_per_epoch = 30000  # 从 50000 减少到 30000
    
    # 固定训练策略参数（第二阶段再优化）
    args.step_per_collect = 2000
    args.repeat_per_collect = 10
    args.batch_size = 64
    args.max_batchsize = 128
    
    # 固定网络结构（第三阶段再优化）
    args.hidden_sizes = [128, 128, 128, 128]
    
    # 固定连续动作参数
    args.log_std_init = -1.0
    args.log_std_min = -5.0
    args.log_std_max = 1.5
    
    # 为每个 trial 创建独立的日志目录
    args.logdir = os.path.join(
        "log", "optuna_trials", 
        f"trial_{trial.number:03d}"
    )
    
    return args


def objective(trial: Trial) -> float:
    """
    Optuna 优化目标函数
    
    Returns:
        float: 优化指标（最后 5 个 epoch 的平均测试奖励）
    """
    print(f"\n{'='*60}")
    print(f"Trial {trial.number} started")
    print(f"{'='*60}")
    
    # 获取基础参数
    base_args = get_args()
    
    # 从 trial 采样超参数
    trial_args = create_trial_args(trial, base_args)
    
    # 打印当前 trial 的超参数
    print("\nTrial hyperparameters:")
    print(f"  lr:         {trial_args.lr:.6f}")
    print(f"  eps_clip:   {trial_args.eps_clip:.3f}")
    print(f"  gae_lambda: {trial_args.gae_lambda:.3f}")
    print(f"  ent_coef:   {trial_args.ent_coef:.6f}")
    print(f"  vf_coef:    {trial_args.vf_coef:.3f}")
    
    try:
        # 运行训练
        result, policy = train_agent(trial_args)
        
        # ✅ 初始化默认值
        avg_reward = -1000.0
        last_n = 0
        
        # ✅ 调试：打印 result 的结构
        print(f"\n[DEBUG] Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"[DEBUG] Result keys: {result.keys()}")
            if "test_reward" in result:
                print(f"[DEBUG] test_reward type: {type(result['test_reward'])}")
                print(f"[DEBUG] test_reward length: {len(result['test_reward']) if hasattr(result['test_reward'], '__len__') else 'N/A'}")
                print(f"[DEBUG] test_reward sample: {result['test_reward'][:3] if hasattr(result['test_reward'], '__getitem__') else result['test_reward']}")
        
        # ✅ 提取评估指标 - 多种方法尝试
        if isinstance(result, dict):
            # 方法1: 尝试 test_reward (列表形式)
            if "test_reward" in result:
                test_rewards = result["test_reward"]
                if isinstance(test_rewards, (list, np.ndarray)) and len(test_rewards) > 0:
                    last_n = min(5, len(test_rewards))
                    avg_reward = float(np.mean(test_rewards[-last_n:]))
                    print(f"[INFO] Using test_reward (last {last_n} epochs): {avg_reward:.4f}")
            
            # 方法2: 尝试 rew (单个值或数组)
            if avg_reward == -1000.0 and "rew" in result:
                rew = result["rew"]
                if isinstance(rew, (float, int, np.number)):
                    avg_reward = float(rew)
                    print(f"[INFO] Using rew (scalar): {avg_reward:.4f}")
                elif isinstance(rew, (list, np.ndarray)) and len(rew) > 0:
                    avg_reward = float(np.mean(rew))
                    print(f"[INFO] Using rew (array mean): {avg_reward:.4f}")
            
            # 方法3: 尝试 rews (如果存在)
            if avg_reward == -1000.0 and "rews" in result:
                rews = result["rews"]
                if isinstance(rews, (list, np.ndarray)) and len(rews) > 0:
                    avg_reward = float(np.mean(rews))
                    print(f"[INFO] Using rews (array mean): {avg_reward:.4f}")
            
            # 方法4: 尝试 train_reward (备选)
            if avg_reward == -1000.0 and "train_reward" in result:
                train_rewards = result["train_reward"]
                if isinstance(train_rewards, (list, np.ndarray)) and len(train_rewards) > 0:
                    last_n = min(5, len(train_rewards))
                    avg_reward = float(np.mean(train_rewards[-last_n:]))
                    print(f"[INFO] Using train_reward (last {last_n} epochs): {avg_reward:.4f}")
        
        # ✅ 最终检查
        if avg_reward == -1000.0:
            print("[WARNING] No valid reward data found in result!")
            print(f"[WARNING] Available keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        print(f"\nTrial {trial.number} completed")
        print(f"Final average reward: {avg_reward:.4f}")
        
        # 报告中间结果（用于 pruning）
        trial.report(avg_reward, trial_args.epoch)
        
        # 清理 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return avg_reward
        
    except Exception as e:
        print(f"\nTrial {trial.number} failed with error:")
        print(f"  {str(e)}")
        
        # ✅ 打印完整的错误堆栈
        import traceback
        traceback.print_exc()
        
        # 清理 GPU 内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 返回一个很差的值，让 Optuna 知道这个配置不好
        return -1000.0

def run_optimization(
    n_trials: int = 50,
    timeout: int = None,
    n_jobs: int = 1,
    study_name: str = "ppo_phase1"
):
    """
    运行 Optuna 优化
    
    Args:
        n_trials: 试验次数
        timeout: 超时时间（秒）
        n_jobs: 并行任务数（注意 GPU 内存）
        study_name: 研究名称
    """
    
    # 创建输出目录
    output_dir = "optuna_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = os.path.join(output_dir, f"{study_name}_{timestamp}.db")
    
    # 创建或加载 study
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        direction="maximize",  # 最大化奖励
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(  # 使用中位数剪枝器
            n_startup_trials=5,  # 前 5 个 trial 不剪枝
            n_warmup_steps=5,    # 前 5 个 epoch 不剪枝
        )
    )
    
    print(f"\n{'='*60}")
    print(f"Starting Optuna optimization")
    print(f"{'='*60}")
    print(f"Study name: {study_name}")
    print(f"Database: {db_path}")
    print(f"Number of trials: {n_trials}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Timeout: {timeout}s" if timeout else "Timeout: None")
    print(f"{'='*60}\n")
    
    # 运行优化
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    # 输出最佳结果
    print(f"\n{'='*60}")
    print(f"Optimization completed!")
    print(f"{'='*60}")
    
    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Trial number: {best_trial.number}")
    print(f"  Value (avg reward): {best_trial.value:.4f}")
    print(f"\n  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # 保存最佳参数到 JSON
    best_params_path = os.path.join(
        output_dir, 
        f"{study_name}_{timestamp}_best_params.json"
    )
    with open(best_params_path, 'w') as f:
        json.dump({
            'trial_number': best_trial.number,
            'value': best_trial.value,
            'params': best_trial.params,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\nBest parameters saved to: {best_params_path}")
    
    # 输出 Top 5 trials
    print("\nTop 5 trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1000, reverse=True)[:5]
    for i, t in enumerate(top_trials, 1):
        print(f"\n{i}. Trial {t.number}:")
        print(f"   Value: {t.value:.4f}")
        print(f"   Params: {t.params}")
    
    # 保存完整的研究结果
    results_path = os.path.join(
        output_dir,
        f"{study_name}_{timestamp}_all_trials.json"
    )
    
    all_trials = []
    for t in study.trials:
        all_trials.append({
            'number': t.number,
            'value': t.value,
            'params': t.params,
            'state': str(t.state)
        })
    
    with open(results_path, 'w') as f:
        json.dump({
            'study_name': study_name,
            'timestamp': timestamp,
            'n_trials': len(study.trials),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'all_trials': all_trials
        }, f, indent=2)
    
    print(f"All trials saved to: {results_path}")
    
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna PPO Hyperparameter Tuning - Phase 1")
    
    parser.add_argument(
        "--n-trials", 
        type=int, 
        default=50,
        help="Number of trials to run (default: 50)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (default: None)"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, be careful with GPU memory)"
    )
    
    parser.add_argument(
        "--study-name",
        type=str,
        default="ppo_phase1",
        help="Name of the study (default: ppo_phase1)"
    )
    
    args = parser.parse_args()
    
    # 运行优化
    study = run_optimization(
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        study_name=args.study_name
    )
    
    print("\nOptimization finished! Check the 'optuna_results' directory for results.")
"""
Waterworld: Test Script
测试训练好的模型,不进行训练
"""

from train_selectedagent import (
    create_agent_configs,
    print_agent_configs,
    create_env,
    prepare_env_for_training,
    TrainedModelPolicy,
    RandomPolicy,
    RuleBasedPolicy
)
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os


def test_model(
    model_path: str,
    agent_configs: List,
    n_predators: int,
    n_preys: int,
    n_episodes: int = 20,
    render: bool = False,
    save_results: bool = True
):
    """
    测试训练好的模型
    
    Args:
        model_path: 训练好的模型路径
        agent_configs: Agent 配置列表
        n_predators: Predator 数量
        n_preys: Prey 数量
        n_episodes: 测试回合数
        render: 是否渲染（暂不支持）
        save_results: 是否保存测试结果
    """
    
    print("="*70)
    print("🧪 Waterworld Model Testing")
    print("="*70)
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    
    print(f"\n📦 Loading model: {model_path}")
    
    # 打印配置
    print_agent_configs(agent_configs)
    
    # 1. 创建环境
    raw_env = create_env(
        n_predators=n_predators,
        n_preys=n_preys,
        agent_configs=agent_configs
    )
    
    # 2. 准备环境
    env = prepare_env_for_training(raw_env, agent_configs)
    
    # 3. 加载模型
    model = PPO.load(model_path, device='cpu')
    print(f"✓ Model loaded successfully")
    
    # 4. 开始测试
    print("\n" + "="*70)
    print(f"🚀 Starting Testing ({n_episodes} episodes)")
    print("="*70)
    
    episode_rewards = []
    episode_lengths = []
    episode_metrics = {
        'hunting_rate': [],
        'escape_rate': [],
        'foraging_rate': []
    }
    
    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0
        ep_length = 0
        
        # 用于收集本episode的指标
        ep_infos = []
        
        while True:
            # 使用模型预测动作（确定性策略）
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            ep_reward += np.sum(reward)
            ep_length += 1
            ep_infos.extend(info)
            
            if np.any(done):
                break
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        
        # 提取性能指标
        _extract_metrics(ep_infos, episode_metrics)
        
        # 打印进度
        print(f"  Episode {ep+1:2d}/{n_episodes}: "
              f"Reward={ep_reward:7.2f}, Length={ep_length:4d}")
    
    # 5. 统计结果
    print("\n" + "="*70)
    print("📊 Test Results Summary")
    print("="*70)
    
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)
    
    print(f"\n🎯 Episode Rewards:")
    print(f"  Mean:   {np.mean(rewards_array):7.2f} ± {np.std(rewards_array):.2f}")
    print(f"  Median: {np.median(rewards_array):7.2f}")
    print(f"  Max:    {np.max(rewards_array):7.2f}")
    print(f"  Min:    {np.min(rewards_array):7.2f}")
    
    print(f"\n⏱️  Episode Lengths:")
    print(f"  Mean:   {np.mean(lengths_array):7.1f} ± {np.std(lengths_array):.1f}")
    print(f"  Median: {np.median(lengths_array):7.1f}")
    print(f"  Max:    {np.max(lengths_array):7.0f}")
    print(f"  Min:    {np.min(lengths_array):7.0f}")
    
    # 打印性能指标
    if episode_metrics['hunting_rate'] or episode_metrics['escape_rate'] or episode_metrics['foraging_rate']:
        print(f"\n📈 Performance Metrics:")
        for key, values in episode_metrics.items():
            if values:
                avg = np.mean(values)
                std = np.std(values)
                
                if 'hunting' in key:
                    emoji = "🎯"
                elif 'escape' in key:
                    emoji = "🏃"
                elif 'foraging' in key:
                    emoji = "🍎"
                else:
                    emoji = "📊"
                
                print(f"  {emoji} {key:15s}: {avg:.3f} ± {std:.3f}")
    
    # 6. 保存结果
    if save_results:
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'metrics': episode_metrics,
            'statistics': {
                'mean_reward': float(np.mean(rewards_array)),
                'std_reward': float(np.std(rewards_array)),
                'mean_length': float(np.mean(lengths_array)),
                'std_length': float(np.std(lengths_array))
            }
        }
        
        # 保存为numpy文件
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        results_file = f'test_results_{model_name}.npz'
        np.savez(results_file, **results)
        print(f"\n💾 Results saved: {results_file}")
        
        # 绘制测试结果图
        plot_test_results(episode_rewards, episode_lengths, model_name)
    
    env.close()
    
    print("\n" + "="*70)
    print("✅ Testing Complete!")
    print("="*70)
    
    return episode_rewards, episode_lengths, episode_metrics


def _extract_metrics(infos, episode_metrics):
    """从 infos 提取性能指标"""
    ep_metrics = {
        'hunting_rate': [],
        'escape_rate': [],
        'foraging_rate': []
    }
    
    for info in infos:
        if not isinstance(info, dict):
            continue
        
        pm = info.get('performance_metrics', {})
        if pm:
            if 'hunting_rate' in pm:
                ep_metrics['hunting_rate'].append(pm['hunting_rate'])
            if 'escape_rate' in pm:
                ep_metrics['escape_rate'].append(pm['escape_rate'])
            if 'foraging_rate' in pm:
                ep_metrics['foraging_rate'].append(pm['foraging_rate'])
    
    # 计算平均值并记录
    for key in ['hunting_rate', 'escape_rate', 'foraging_rate']:
        if ep_metrics[key]:
            avg = np.mean(ep_metrics[key])
            episode_metrics[key].append(avg)


def plot_test_results(episode_rewards, episode_lengths, model_name):
    """绘制测试结果"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    # 左图：奖励
    ax1 = axes[0]
    ax1.plot(episodes, episode_rewards, marker='o', linestyle='-', 
             color='steelblue', linewidth=2, markersize=6, label='Episode Reward')
    ax1.axhline(y=np.mean(episode_rewards), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax1.fill_between(episodes, 
                      np.mean(episode_rewards) - np.std(episode_rewards),
                      np.mean(episode_rewards) + np.std(episode_rewards),
                      alpha=0.2, color='red')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Test Episode Rewards', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：长度
    ax2 = axes[1]
    ax2.plot(episodes, episode_lengths, marker='s', linestyle='-', 
             color='forestgreen', linewidth=2, markersize=6, label='Episode Length')
    ax2.axhline(y=np.mean(episode_lengths), color='orange', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(episode_lengths):.1f}')
    ax2.fill_between(episodes,
                      np.mean(episode_lengths) - np.std(episode_lengths),
                      np.mean(episode_lengths) + np.std(episode_lengths),
                      alpha=0.2, color='orange')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Length (steps)', fontsize=12)
    ax2.set_title('Test Episode Lengths', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model: {model_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = f'test_results_{model_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Test plot saved: {save_path}")
    plt.close()


# ============================================================================
# 主函数：配置测试场景
# ============================================================================

def main():
    """主测试函数"""
    
    # ========================================
    # 配置测试参数
    # ========================================
    
    # 环境配置
    N_PREDATORS = 5
    N_PREYS = 10
    
    # 测试配置
    MODEL_PATH = 'predator_ppo_model.zip'  # 修改为你的模型路径
    N_TEST_EPISODES = 20  # 测试回合数
    
    # ========================================
    # Agent 配置
    # ========================================
    
    agent_configs = create_agent_configs(
        n_predators=N_PREDATORS,
        n_preys=N_PREYS,
        train_predators=[0, 1],  # 这两个使用训练好的模型
        train_preys=None,
        predator_policies={
            2: TrainedModelPolicy('predator_ppo_model.zip'),  # 如果有其他训练好的模型
            3: RandomPolicy(),
            4: RandomPolicy()
        },
        prey_policies={
            0: RandomPolicy(),
            1: RandomPolicy(),
            2: RandomPolicy(),
            3: RandomPolicy(),
            4: RandomPolicy(),
            5: RandomPolicy(),
            6: RandomPolicy(),
            7: RandomPolicy(),
            8: RandomPolicy(),
            9: RandomPolicy()
        }
    )
    
    # ========================================
    # 执行测试
    # ========================================
    
    try:
        episode_rewards, episode_lengths, metrics = test_model(
            model_path=MODEL_PATH,
            agent_configs=agent_configs,
            n_predators=N_PREDATORS,
            n_preys=N_PREYS,
            n_episodes=N_TEST_EPISODES,
            render=False,
            save_results=True
        )
        
        print("\n💡 Test completed successfully!")
        print(f"   - Tested {N_TEST_EPISODES} episodes")
        print(f"   - Average reward: {np.mean(episode_rewards):.2f}")
        print(f"   - Average length: {np.mean(episode_lengths):.1f}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please make sure the model file exists!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
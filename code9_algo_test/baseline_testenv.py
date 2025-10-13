# from pettingzoo.sisl import waterworld_v4
# import supersuit as ss
# from collections import defaultdict
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# import pandas as pd

# def run_waterworld_test(n_epochs=100, steps_per_epoch=10000):
#     """
#     运行waterworld环境测试
    
#     Args:
#         n_epochs: 运行的轮次数
#         steps_per_epoch: 每轮的步数
#     """
#     # 准备环境参数
#     agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"] * 40
    
#     # 存储所有epoch的结果
#     all_epoch_rewards = []
#     agent_types = set()
    
#     print(f"开始运行 {n_epochs} 个 epochs，每个 epoch {steps_per_epoch} 步...")
    
#     for epoch in tqdm(range(n_epochs), desc="Epochs"):
#         # 创建环境
#         env = waterworld_v4.env(
#             render_mode=None,  # 不渲染以加快速度
#             n_predators=4,
#             n_preys=4,
#             n_evaders=1,
#             n_obstacles=2,
#             obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
#             n_poisons=20,
#             agent_algorithms=agent_algos
#         )
        
#         # 黑死亡包装
#         env = ss.black_death_v3(env)
        
#         # 重置环境
#         obs = env.reset(seed=epoch)  # 每个epoch使用不同的seed
        
#         # 用于累加每个agent的本轮总reward
#         cumulative_rewards = defaultdict(float)
        
#         # 运行指定步数
#         step_count = 0
#         for agent in env.agent_iter():
#             if step_count >= steps_per_epoch:
#                 break
                
#             observation, reward, termination, truncation, info = env.last()
            
#             # 累加reward
#             if reward is not None:
#                 cumulative_rewards[agent] += reward
            
#             # 收集agent类型
#             agent_types.add(agent.split('_')[0])  # 获取agent类型前缀
            
#             # 选择动作
#             if termination or truncation:
#                 action = None
#             else:
#                 action = env.action_space(agent).sample()
            
#             env.step(action)
#             step_count += 1
        
#         # 保存这个epoch的结果
#         epoch_data = dict(cumulative_rewards)
#         epoch_data['epoch'] = epoch
#         all_epoch_rewards.append(epoch_data)
        
#         env.close()
    
#     return all_epoch_rewards, sorted(list(agent_types))

# def plot_rewards(all_epoch_rewards, agent_types):
#     """绘制奖励图表"""
    
#     # 转换数据格式用于绘图
#     df_data = []
#     for epoch_data in all_epoch_rewards:
#         epoch = epoch_data['epoch']
#         for agent, reward in epoch_data.items():
#             if agent != 'epoch':
#                 agent_type = agent.split('_')[0]
#                 df_data.append({
#                     'epoch': epoch,
#                     'agent': agent,
#                     'agent_type': agent_type,
#                     'reward': reward
#                 })
    
#     df = pd.DataFrame(df_data)
    
#     # 创建图表
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#     fig.suptitle('Waterworld环境奖励分析 (100 epochs, 10000 steps each)', fontsize=16)
    
#     # 1. 每个agent类型的平均奖励趋势
#     ax1 = axes[0, 0]
#     for agent_type in agent_types:
#         type_data = df[df['agent_type'] == agent_type]
#         avg_rewards = type_data.groupby('epoch')['reward'].mean()
#         ax1.plot(avg_rewards.index, avg_rewards.values, label=agent_type, alpha=0.8)
    
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Average Reward')
#     ax1.set_title('各Agent类型平均奖励趋势')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
    
#     # 2. 每个agent类型的奖励分布箱线图
#     ax2 = axes[0, 1]
#     sns.boxplot(data=df, x='agent_type', y='reward', ax=ax2)
#     ax2.set_title('各Agent类型奖励分布')
#     ax2.set_ylabel('Reward')
#     plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
#     # 3. 奖励直方图
#     ax3 = axes[1, 0]
#     for i, agent_type in enumerate(agent_types):
#         type_rewards = df[df['agent_type'] == agent_type]['reward']
#         ax3.hist(type_rewards, alpha=0.6, label=agent_type, bins=30)
    
#     ax3.set_xlabel('Reward')
#     ax3.set_ylabel('Frequency')
#     ax3.set_title('奖励分布直方图')
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)
    
#     # 4. 每个agent类型的累积奖励统计
#     ax4 = axes[1, 1]
#     summary_stats = df.groupby('agent_type')['reward'].agg(['mean', 'std', 'min', 'max'])
#     x_pos = np.arange(len(agent_types))
    
#     bars = ax4.bar(x_pos, summary_stats['mean'], yerr=summary_stats['std'], 
#                    capsize=5, alpha=0.7, color=plt.cm.Set3(np.arange(len(agent_types))))
    
#     ax4.set_xlabel('Agent Type')
#     ax4.set_ylabel('Mean Reward')
#     ax4.set_title('各Agent类型平均奖励 (带标准差)')
#     ax4.set_xticks(x_pos)
#     ax4.set_xticklabels(agent_types, rotation=45)
#     ax4.grid(True, alpha=0.3)
    
#     # 在柱子上显示数值
#     for i, (bar, mean_val, std_val) in enumerate(zip(bars, summary_stats['mean'], summary_stats['std'])):
#         ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.1, 
#                 f'{mean_val:.2f}', ha='center', va='bottom', fontsize=9)
    
#     plt.tight_layout()
#     plt.show()
    
#     # 打印统计摘要
#     print("\n=== 奖励统计摘要 ===")
#     print(summary_stats)
    
#     # 计算总体统计
#     total_episodes = len(all_epoch_rewards)
#     total_agents = len(df['agent'].unique())
#     print(f"\n总共运行了 {total_episodes} 个 epochs")
#     print(f"总共有 {total_agents} 个不同的 agents")
#     print(f"平均每个epoch的总奖励: {df.groupby('epoch')['reward'].sum().mean():.2f}")
    
#     return df

# def main():
#     """主函数"""
#     print("Waterworld环境奖励测试")
#     print("=" * 50)
    
#     # 运行测试
#     all_epoch_rewards, agent_types = run_waterworld_test(n_epochs=100, steps_per_epoch=10000)
    
#     # 绘制图表
#     df = plot_rewards(all_epoch_rewards, agent_types)
    
#     return all_epoch_rewards, df

# if __name__ == "__main__":
#     # 设置matplotlib中文字体支持（可选）
#     plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
#     plt.rcParams['axes.unicode_minus'] = False
    
#     # 运行主函数
#     results, df = main()

# diagnose_env_rewards.py
"""
环境奖励诊断工具 - 在训练前检测奖励分布是否合理
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from datetime import datetime

def diagnose_environment_rewards(env_fn, n_episodes=100, max_steps_per_episode=1000):
    """
    运行环境并收集奖励统计
    """
    
    env = env_fn()
    
    # 统计容器
    reward_stats = defaultdict(list)
    episode_returns = defaultdict(list)
    extreme_rewards = []
    
    print(f"Running {n_episodes} episodes to diagnose reward distribution...")
    print("=" * 60)
    
    for episode in range(n_episodes):
        env.reset(seed=episode)
        episode_rewards = defaultdict(list)
        
        step = 0
        
        # ✅ PettingZoo 使用 agent_iter() 遍历
        for agent in env.agent_iter(max_iter=max_steps_per_episode):
            observation, reward, termination, truncation, info = env.last()
            
            # 记录奖励（跳过第一步的None）
            if reward is not None:
                reward_stats[agent].append(reward)
                episode_rewards[agent].append(reward)
                
                # 检测极端奖励
                if abs(reward) > 100:
                    extreme_rewards.append({
                        'episode': episode,
                        'step': step,
                        'agent': agent,
                        'reward': float(reward),
                    })
            
            # 选择动作
            if termination or truncation:
                action = None
            else:
                # ✅ 正确的动作采样方式
                action = env.action_space(agent).sample()
            
            env.step(action)
            step += 1
        
        # 记录episode总回报
        for agent, rewards_list in episode_rewards.items():
            episode_returns[agent].append(sum(rewards_list))
        
        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{n_episodes} episodes")
        
        env.close()
    # 生成诊断报告
    print("\n" + "=" * 60)
    print("REWARD DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    for agent in sorted(reward_stats.keys()):
        rewards = np.array(reward_stats[agent])
        returns = np.array(episode_returns[agent])
        
        print(f"\n{agent}:")
        print(f"  Single-step rewards:")
        print(f"    Mean:   {rewards.mean():>10.4f}")
        print(f"    Std:    {rewards.std():>10.4f}")
        print(f"    Min:    {rewards.min():>10.4f}")
        print(f"    Max:    {rewards.max():>10.4f}")
        print(f"    Median: {np.median(rewards):>10.4f}")
        print(f"    99th percentile: {np.percentile(rewards, 99):>10.4f}")
        print(f"    1st percentile:  {np.percentile(rewards, 1):>10.4f}")
        
        print(f"  Episode returns:")
        print(f"    Mean:   {returns.mean():>10.4f}")
        print(f"    Std:    {returns.std():>10.4f}")
        print(f"    Min:    {returns.min():>10.4f}")
        print(f"    Max:    {returns.max():>10.4f}")
        
        # 检查问题
        issues = []
        if abs(rewards.max()) > 100:
            issues.append(f"MAX_REWARD_TOO_HIGH: {rewards.max():.2f}")
        if abs(rewards.min()) > 100:
            issues.append(f"MIN_REWARD_TOO_LOW: {rewards.min():.2f}")
        if abs(returns.max()) > 1000:
            issues.append(f"RETURN_TOO_HIGH: {returns.max():.2f}")
        if abs(returns.min()) > 1000:
            issues.append(f"RETURN_TOO_LOW: {returns.min():.2f}")
        if rewards.std() > 50:
            issues.append(f"HIGH_VARIANCE: std={rewards.std():.2f}")
        
        if issues:
            print(f"  ⚠️  ISSUES DETECTED:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  ✓ Looks healthy")
        
        results[agent] = {
            'reward_mean': float(rewards.mean()),
            'reward_std': float(rewards.std()),
            'reward_min': float(rewards.min()),
            'reward_max': float(rewards.max()),
            'return_mean': float(returns.mean()),
            'return_std': float(returns.std()),
            'return_min': float(returns.min()),
            'return_max': float(returns.max()),
            'issues': issues
        }
    
    # 极端奖励分析
    if extreme_rewards:
        print(f"\n{'='*60}")
        print(f"EXTREME REWARDS (|reward| > 100)")
        print(f"{'='*60}")
        print(f"Found {len(extreme_rewards)} extreme rewards:")
        
        # 按绝对值排序，显示前10个
        extreme_rewards.sort(key=lambda x: abs(x['reward']), reverse=True)
        for i, event in enumerate(extreme_rewards[:10]):
            print(f"{i+1}. Episode {event['episode']}, Step {event['step']}, "
                  f"{event['agent']}: reward={event['reward']:.2f}")
    
    # 建议
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    all_issues = []
    for agent_issues in results.values():
        all_issues.extend(agent_issues['issues'])
    
    if not all_issues:
        print("✓ Reward distribution looks healthy for RL training!")
    else:
        print("⚠️  Issues detected. Recommended fixes:")
        
        if any('TOO_HIGH' in issue or 'TOO_LOW' in issue for issue in all_issues):
            print("\n1. CLIP REWARDS in environment:")
            print("   Add to waterworld_base.py step():")
            print("   ```python")
            print("   for id in range(self.num_agents):")
            print("       self.behavior_rewards[id] = np.clip(self.behavior_rewards[id], -10, 10)")
            print("       self.collision_rewards[id] = np.clip(self.collision_rewards[id], -10, 10)")
            print("   ```")
        
        if any('HIGH_VARIANCE' in issue for issue in all_issues):
            print("\n2. ENABLE REWARD NORMALIZATION:")
            print("   Add --reward-normalization flag when training")
        
        if any('RETURN' in issue for issue in all_issues):
            print("\n3. REDUCE VF_COEF:")
            print("   Use --vf-coef 0.01 instead of default 0.5")
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"env_reward_diagnosis_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_episodes': n_episodes,
            'results': results,
            'extreme_rewards': extreme_rewards[:100],  # 保存前100个
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # 可选：绘制直方图
    try:
        plot_reward_distribution(reward_stats, episode_returns)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    return results


def plot_reward_distribution(reward_stats, episode_returns):
    """绘制奖励分布直方图"""
    import matplotlib.pyplot as plt
    
    n_agents = len(reward_stats)
    fig, axes = plt.subplots(n_agents, 2, figsize=(12, 4*n_agents))
    
    if n_agents == 1:
        axes = axes.reshape(1, -1)
    
    for i, (agent, rewards) in enumerate(sorted(reward_stats.items())):
        # 单步奖励分布
        axes[i, 0].hist(rewards, bins=50, edgecolor='black')
        axes[i, 0].set_title(f'{agent} - Single-step Rewards')
        axes[i, 0].set_xlabel('Reward')
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].axvline(0, color='r', linestyle='--', alpha=0.5)
        
        # Episode回报分布
        returns = episode_returns[agent]
        axes[i, 1].hist(returns, bins=30, edgecolor='black', color='orange')
        axes[i, 1].set_title(f'{agent} - Episode Returns')
        axes[i, 1].set_xlabel('Return')
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].axvline(0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'reward_distribution_{timestamp}.png', dpi=150)
    print(f"Plots saved to: reward_distribution_{timestamp}.png")
    plt.close()


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    # ✅ 不要用 get_env，直接创建原始环境
    import supersuit as ss
    from pettingzoo.sisl import waterworld_v4
    
    def create_raw_env():
        """创建未包装的原始环境"""
        env = waterworld_v4.env(
            render_mode=None,
            n_predators=2,
            n_preys=2,
            n_evaders=20,
            n_obstacles=2,
            obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
            n_poisons=20,
            agent_algorithms=["PPO", "PPO", "Random", "Random"]
        )
        env = ss.black_death_v3(env)
        return env
    
    results = diagnose_environment_rewards(
        env_fn=create_raw_env,
        n_episodes=100,
        max_steps_per_episode=500
    )
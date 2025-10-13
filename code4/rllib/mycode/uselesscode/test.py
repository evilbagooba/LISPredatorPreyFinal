"""
加载训练好的PettingZoo Waterworld模型并进行可视化评估
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.sisl import waterworld_v4
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env
import time


def load_and_evaluate_model():
    """加载训练好的模型并进行评估"""
    
    # 1. 设置checkpoint路径（根据您的训练日志修改）
    # 从日志中看到结果保存在: '/home/qrbao/ray_results/PPO_2025-06-23_19-21-45'
    checkpoint_dir = "/home/qrbao/ray_results/PPO_2025-06-23_19-21-45"
    
    # 查找最新的checkpoint
    checkpoint_path = None
    if os.path.exists(checkpoint_dir):
        # 查找checkpoint文件夹
        for item in os.listdir(checkpoint_dir):
            if item.startswith("checkpoint_"):
                checkpoint_path = os.path.join(checkpoint_dir, item)
                break
    
    if checkpoint_path is None:
        print("❌ 未找到checkpoint文件！")
        print(f"请检查路径: {checkpoint_dir}")
        return
    
    print(f"✅ 找到checkpoint: {checkpoint_path}")
    
    # 2. 重新注册环境（与训练时相同）
    register_env("env", lambda _: PettingZooEnv(waterworld_v4.env()))
    
    # 3. 重建配置（与训练时相同）
    config = (
        PPO.get_default_config()
        .environment("env")
        .multi_agent(
            policies={"p0"},
            policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
        )
        .training(
            model={"vf_share_layers": True},
            vf_loss_coeff=0.005,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={"p0": RLModuleSpec()},
            ),
        )
        # 评估时的特殊设置
        .env_runners(
            num_env_runners=0,  # 评估时不需要多个worker
            create_local_env_runner=True,
        )
        .debugging(
            log_level="INFO"
        )
    )
    
    # 4. 加载训练好的算法
    print("🔄 正在加载训练好的模型...")
    try:
        algorithm = PPO(config=config)
        algorithm.restore(checkpoint_path)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    return algorithm


def visualize_trained_agents(algorithm, num_episodes=3, render_mode="human"):
    """可视化训练好的智能体表现"""
    
    print(f"\n🎮 开始可视化评估 ({num_episodes} 集)...")
    
    # 创建环境用于可视化
    env = waterworld_v4.env(render_mode=render_mode)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"\n📊 Episode {episode + 1}/{num_episodes}")
        
        # 重置环境
        env.reset()
        
        episode_reward = {agent: 0 for agent in env.possible_agents}
        episode_length = 0
        
        # 运行一集
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            
            # 累计奖励
            if agent in episode_reward:
                episode_reward[agent] += reward
            
            if termination or truncation:
                # 智能体结束，选择None动作
                action = None
            else:
                # 使用训练好的策略选择动作
                try:
                    # 获取动作
                    action_dict = algorithm.compute_single_action(
                        observation=obs,
                        policy_id="p0"  # 使用训练时的策略ID
                    )
                    action = action_dict
                except Exception as e:
                    print(f"⚠️ 动作计算错误: {e}")
                    # 如果出错，使用随机动作
                    action = env.action_space(agent).sample()
            
            # 执行动作
            env.step(action)
            episode_length += 1
            
            # 可视化渲染
            if render_mode == "human":
                env.render()
                time.sleep(0.05)  # 控制播放速度
        
        # 记录本集结果
        total_reward = sum(episode_reward.values())
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        
        print(f"  📈 总奖励: {total_reward:.2f}")
        print(f"  📊 智能体奖励: {episode_reward}")
        print(f"  ⏱️ 集长度: {episode_length}")
    
    env.close()
    
    # 显示统计结果
    print(f"\n📊 评估统计结果:")
    print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  平均集长度: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"  最高奖励: {np.max(episode_rewards):.2f}")
    print(f"  最低奖励: {np.min(episode_rewards):.2f}")
    
    return episode_rewards, episode_lengths


def compare_with_random_policy(algorithm, num_episodes=5):
    """与随机策略进行对比评估"""
    
    print("\n🔄 正在进行对比评估（训练策略 vs 随机策略）...")
    
    env = waterworld_v4.env(render_mode=None)  # 不可视化以加快速度
    
    # 评估训练好的策略
    trained_rewards = []
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            total_reward += reward
            
            if termination or truncation:
                action = None
            else:
                try:
                    action = algorithm.compute_single_action(obs, policy_id="p0")
                except:
                    action = env.action_space(agent).sample()
            
            env.step(action)
        
        trained_rewards.append(total_reward)
    
    # 评估随机策略
    random_rewards = []
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            total_reward += reward
            
            if termination or truncation:
                action = None
            else:
                action = env.action_space(agent).sample()  # 随机动作
            
            env.step(action)
        
        random_rewards.append(total_reward)
    
    env.close()
    
    # 比较结果
    print(f"\n📊 对比结果:")
    print(f"  训练策略平均奖励: {np.mean(trained_rewards):.2f} ± {np.std(trained_rewards):.2f}")
    print(f"  随机策略平均奖励: {np.mean(random_rewards):.2f} ± {np.std(random_rewards):.2f}")
    improvement = np.mean(trained_rewards) - np.mean(random_rewards)
    print(f"  🚀 性能提升: {improvement:.2f}")
    
    return trained_rewards, random_rewards


def plot_training_progress(checkpoint_dir):
    """绘制训练过程中的性能变化"""
    
    # 这个函数需要从TensorBoard或其他日志中读取训练数据
    # 如果您有训练日志，可以在这里解析并绘制
    print("📈 训练进度图表功能待实现...")
    print("💡 建议使用TensorBoard查看详细训练进度:")
    print(f"   tensorboard --logdir {checkpoint_dir}")


def main():
    """主函数"""
    
    print("🎯 PettingZoo Waterworld 训练模型评估")
    print("=" * 50)
    
    # 1. 加载模型
    algorithm = load_and_evaluate_model()
    if algorithm is None:
        return
    
    # 2. 可视化评估
    print("\n" + "=" * 50)
    episode_rewards, episode_lengths = visualize_trained_agents(
        algorithm, 
        num_episodes=3,
        render_mode="human"  # 设置为"human"以显示可视化
    )
    
    # 3. 对比评估
    print("\n" + "=" * 50)
    trained_rewards, random_rewards = compare_with_random_policy(
        algorithm, 
        num_episodes=10
    )
    
    # 4. 提供建议
    print("\n" + "=" * 50)
    print("💡 进一步分析建议:")
    print("1. 使用 render_mode='human' 观看智能体实时表现")
    print("2. 调整环境参数测试模型的泛化能力")
    print("3. 尝试不同的智能体数量")
    print("4. 分析智能体的协作行为模式")
    
    # 清理资源
    algorithm.stop()
    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()
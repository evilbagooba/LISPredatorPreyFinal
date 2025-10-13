"""
Waterworld: Bidirectional Training System
支持 Prey 训练 vs Random Predators 和 Predator 训练 vs Fixed Prey
"""

from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import gymnasium as gym
from abc import ABC, abstractmethod
import os


# ============================================================================
# 策略接口：支持多种固定策略
# ============================================================================

class AgentPolicy(ABC):
    """固定 Agent 策略的抽象基类"""
    
    @abstractmethod
    def get_action(self, obs, agent_idx):
        """
        根据观察获取动作
        
        Args:
            obs: 观察值 (obs_dim,)
            agent_idx: Agent 索引
            
        Returns:
            action: 动作 (action_dim,)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置策略状态（如果需要）"""
        pass


class RandomPolicy(AgentPolicy):
    """随机策略"""
    
    def __init__(self, action_dim=2, low=-1.0, high=1.0):
        self.action_dim = action_dim
        self.low = low
        self.high = high
    
    def get_action(self, obs, agent_idx):
        return np.random.uniform(
            low=self.low,
            high=self.high,
            size=self.action_dim
        ).astype(np.float32)
    
    def reset(self):
        pass


class TrainedModelPolicy(AgentPolicy):
    """使用训练好的模型作为策略"""
    
    def __init__(self, model_path, device='cpu'):
        """
        Args:
            model_path: 训练好的模型路径 (.zip 文件)
            device: 'cpu' 或 'cuda'
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = PPO.load(model_path, device=device)
        print(f"  Loaded trained model from: {model_path}")
    
    def get_action(self, obs, agent_idx):
        # 使用确定性策略
        action, _ = self.model.predict(obs, deterministic=True)
        return action
    
    def reset(self):
        pass


class RuleBasedPolicy(AgentPolicy):
    """基于规则的策略（示例：可以扩展）"""
    
    def __init__(self):
        pass
    
    def get_action(self, obs, agent_idx):
        # 示例：简单的规则策略
        # 这里可以根据观察值实现复杂的规则
        # 比如：朝最近的目标移动等
        return np.array([0.5, 0.5], dtype=np.float32)
    
    def reset(self):
        pass


# ============================================================================
# 自定义 VecEnv：支持混合训练模式
# ============================================================================

class MixedAgentVecEnv(VecEnv):
    """
    支持两种训练模式的自定义 VecEnv：
    1. mode='train_prey': 训练 prey，predator 使用固定策略
    2. mode='train_predator': 训练 predator，prey 使用固定策略
    """

    def __init__(self, venv, n_predators, n_preys, mode='train_prey', fixed_policy=None):
        """
        Args:
            venv: 包装后的向量化环境
            n_predators: Predator 数量
            n_preys: Prey 数量
            mode: 'train_prey' 或 'train_predator'
            fixed_policy: AgentPolicy 实例，用于固定的 agents
        """
        self.venv = venv
        self.n_predators = n_predators
        self.n_preys = n_preys
        self.n_total_agents = n_predators + n_preys
        self.mode = mode
        
        # 设置固定策略
        if fixed_policy is None:
            self.fixed_policy = RandomPolicy()
            print(f"  Using default RandomPolicy for fixed agents")
        else:
            self.fixed_policy = fixed_policy
        
        # Agent 索引
        self.predator_indices = list(range(n_predators))
        self.prey_indices = list(range(n_predators, n_predators + n_preys))
        
        # 根据模式设置训练和固定的 agents
        if mode == 'train_prey':
            self.training_indices = self.prey_indices
            self.fixed_indices = self.predator_indices
            self.n_training = n_preys
            print(f"  Mode: Training {n_preys} Preys, Fixed {n_predators} Predators")
        elif mode == 'train_predator':
            self.training_indices = self.predator_indices
            self.fixed_indices = self.prey_indices
            self.n_training = n_predators
            print(f"  Mode: Training {n_predators} Predators, Fixed {n_preys} Preys")
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train_prey' or 'train_predator'")
        
        # 获取原始空间
        original_obs_space = venv.observation_space
        original_action_space = venv.action_space
        
        # 创建新的 VecEnv，num_envs = 训练 agents 的数量
        super().__init__(
            num_envs=self.n_training,
            observation_space=original_obs_space,
            action_space=original_action_space
        )
        
        # 缓存最新的观察值（用于固定策略）
        self.latest_obs = None

    def reset(self):
        """重置环境，返回训练 agents 的观察"""
        obs = self.venv.reset()
        self.latest_obs = obs
        self.fixed_policy.reset()
        training_obs = obs[self.training_indices]
        return training_obs

    def step_async(self, actions):
        """
        组合训练 agents 的动作和固定 agents 的动作
        
        Args:
            actions: shape (n_training, action_dim) - 训练 agents 的动作
        """
        # 生成固定 agents 的动作
        fixed_actions = np.zeros((len(self.fixed_indices), 2), dtype=np.float32)
        for i, agent_idx in enumerate(self.fixed_indices):
            # 获取该 agent 的观察
            obs = self.latest_obs[agent_idx] if self.latest_obs is not None else None
            fixed_actions[i] = self.fixed_policy.get_action(obs, agent_idx)
        
        # 组合所有动作：按照 agent 顺序
        full_actions = np.zeros((self.n_total_agents, 2), dtype=np.float32)
        
        if self.mode == 'train_prey':
            # Predators (fixed) 在前，Preys (training) 在后
            full_actions[self.predator_indices] = fixed_actions
            full_actions[self.prey_indices] = actions
        else:  # train_predator
            # Predators (training) 在前，Preys (fixed) 在后
            full_actions[self.predator_indices] = actions
            full_actions[self.prey_indices] = fixed_actions
        
        # 传递给底层环境
        self.venv.step_async(full_actions)

    def step_wait(self):
        """获取环境结果，提取训练 agents 的数据"""
        obs, rewards, dones, infos = self.venv.step_wait()
        
        # 缓存观察值
        self.latest_obs = obs
        
        # 提取训练 agents 的数据
        training_obs = obs[self.training_indices]
        training_rewards = rewards[self.training_indices]
        training_dones = dones[self.training_indices]
        training_infos = [infos[i] for i in self.training_indices]
        
        return training_obs, training_rewards, training_dones, training_infos

    def close(self):
        """关闭底层环境"""
        return self.venv.close()

    def get_attr(self, attr_name, indices=None):
        """获取底层环境属性"""
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        """设置底层环境属性"""
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """调用底层环境方法"""
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        """检查环境是否被包装"""
        return self.venv.env_is_wrapped(wrapper_class, indices)


# ============================================================================
# 训练监控回调
# ============================================================================

class TrainingMonitorCallback(BaseCallback):
    """监控训练过程"""

    def __init__(self, agent_type='Prey', check_freq=1000, verbose=1):
        """
        Args:
            agent_type: 'Prey' 或 'Predator'
        """
        super().__init__(verbose)
        self.agent_type = agent_type
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_ep_reward = 0
        self.current_ep_length = 0

    def _on_step(self):
        # 累加当前步的所有训练 agents 的奖励
        reward_sum = np.sum(self.locals['rewards'])
        self.current_ep_reward += reward_sum
        self.current_ep_length += 1

        # 检查是否有 agent 完成 episode
        if np.any(self.locals['dones']):
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)

            if len(self.episode_rewards) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                print(f"\n[{self.agent_type} Training] Episode {len(self.episode_rewards)}:")
                print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
                print(f"  Avg Length: {np.mean(self.episode_lengths[-10:]):.0f}")
                print(f"  Max Reward: {np.max(recent_rewards):.2f}")

            self.current_ep_reward = 0
            self.current_ep_length = 0

        return True


# ============================================================================
# 训练曲线绘制
# ============================================================================

def plot_training_curve(episode_rewards, agent_type='Prey', save_path=None):
    """
    绘制训练曲线
    
    Args:
        episode_rewards: Episode 奖励列表
        agent_type: 'Prey' 或 'Predator'
        save_path: 保存路径
    """
    if save_path is None:
        save_path = f'training_curve_{agent_type.lower()}.png'
    
    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards)

    plt.figure(figsize=(12, 6))

    # 原始奖励
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Rewards')

    # 移动平均
    if len(rewards) >= 10:
        window_size = min(10, len(rewards))
        smoothed = uniform_filter1d(rewards, size=window_size, mode='nearest')
        plt.plot(episodes, smoothed, color='red', linewidth=2, 
                label=f'Moving Average (window={window_size})')

    # 趋势线
    if len(rewards) >= 50:
        window_size = min(50, len(rewards))
        trend = uniform_filter1d(rewards, size=window_size, mode='nearest')
        plt.plot(episodes, trend, color='green', linewidth=2, linestyle='--', 
                label=f'Trend (window={window_size})')

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(f'Episode Reward ({agent_type} Only)', fontsize=12)
    plt.title(f'PPO Training: {agent_type} Agents', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # 统计信息
    stats_text = f'Episodes: {len(rewards)}\n'
    stats_text += f'Mean: {np.mean(rewards):.2f}\n'
    stats_text += f'Std: {np.std(rewards):.2f}\n'
    stats_text += f'Max: {np.max(rewards):.2f}\n'
    stats_text += f'Min: {np.min(rewards):.2f}'

    n = len(rewards)
    if n >= 10:
        early = rewards[:max(1, n//10)]
        late = rewards[-max(1, n//10):]
        improvement = np.mean(late) - np.mean(early)
        stats_text += f'\nImprovement: {improvement:+.2f}'

    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curve saved to: {save_path}")
    plt.close()


# ============================================================================
# 环境创建和准备
# ============================================================================

def create_mixed_env(n_predators=3, n_preys=5, mode='train_prey'):
    """创建混合环境"""
    print("\n" + "="*60)
    print(f"Creating Mixed Environment")
    print("="*60)

    total_agents = n_predators + n_preys
    
    # 设置算法标签（仅用于环境初始化）
    if mode == 'train_prey':
        agent_algos = ["Fixed"] * n_predators + ["PPO"] * n_preys
    else:  # train_predator
        agent_algos = ["PPO"] * n_predators + ["Fixed"] * n_preys

    env = waterworld_v4.parallel_env(
        render_mode=None,
        n_predators=n_predators,
        n_preys=n_preys,
        n_evaders=180,
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=10,
        agent_algorithms=agent_algos,
        max_cycles=3000,
        static_food=True,
        static_poison=True,
    )

    print(f"Environment Configuration:")
    print(f"  Training Mode: {mode}")
    print(f"  Predators: {n_predators}")
    print(f"  Preys: {n_preys}")
    print(f"  Total Agents: {total_agents}")
    print(f"  All Agents: {env.possible_agents}")
    print(f"  Food: 180 (static)")
    print(f"  Poison: 10 (static)")
    print(f"  Observation Dim: {env.observation_space('prey_0').shape}")
    print(f"  Action Dim: {env.action_space('prey_0').shape}")

    return env, n_predators, n_preys


def prepare_env_for_training(env, n_predators, n_preys, mode='train_prey', 
                            fixed_policy=None):
    """
    准备训练环境
    
    Args:
        env: 原始环境
        n_predators: Predator 数量
        n_preys: Prey 数量
        mode: 'train_prey' 或 'train_predator'
        fixed_policy: 固定 agents 使用的策略（AgentPolicy 实例）
    """
    print("\nConverting environment format...")

    # 标准转换
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')

    print("  Standard conversion complete")
    print(f"  num_envs after conversion: {env.num_envs}")

    # 应用自定义混合环境包装器
    env = MixedAgentVecEnv(
        env, 
        n_predators=n_predators, 
        n_preys=n_preys,
        mode=mode,
        fixed_policy=fixed_policy
    )
    
    print(f"  Applied MixedAgentVecEnv wrapper")
    print(f"    - Final num_envs: {env.num_envs}")

    # 添加监控
    env = VecMonitor(env)
    print("  Environment preparation complete")

    return env


# ============================================================================
# 训练和评估
# ============================================================================

def train_ppo(env, agent_type='Prey', total_timesteps=1000000):
    """使用 PPO 训练"""
    print("\n" + "="*60)
    print(f"Starting PPO Training ({agent_type} Only)")
    print("="*60)
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device='cpu'
    )

    callback = TrainingMonitorCallback(agent_type=agent_type, check_freq=1000)

    print(f"\nStarting training...")
    print(f"Note: {agent_type} agents are learning")
    print(f"      Other agents are executing fixed policy")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    print("\nTraining complete!")

    # 显示训练统计
    if callback.episode_rewards:
        rewards = np.array(callback.episode_rewards)
        print("\n" + "="*60)
        print(f"Training Statistics ({agent_type} Only)")
        print("="*60)
        print(f"Total Episodes: {len(rewards)}")
        print(f"Mean Reward: {np.mean(rewards):.2f}")
        print(f"Max Reward: {np.max(rewards):.2f}")
        print(f"Min Reward: {np.min(rewards):.2f}")

        n = len(rewards)
        early = rewards[:max(1, n//10)]
        late = rewards[-max(1, n//10):]
        improvement = np.mean(late) - np.mean(early)

        print(f"\nLearning Analysis:")
        print(f"  Early Mean: {np.mean(early):.2f}")
        print(f"  Late Mean: {np.mean(late):.2f}")
        print(f"  Improvement: {improvement:+.2f}")

        if improvement > 5:
            print("  Conclusion: Effective Learning")
        elif improvement > -5:
            print("  Conclusion: Limited Learning")
        else:
            print("  Conclusion: No Effective Learning")

    return model, callback


def evaluate_model(model, env, agent_type='Prey', n_episodes=10):
    """评估训练好的模型"""
    print("\n" + "="*60)
    print(f"Evaluating {agent_type} Model ({n_episodes} episodes)")
    print("="*60)

    episode_rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0
        ep_length = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += np.sum(reward)
            ep_length += 1

            if np.any(done):
                break

        episode_rewards.append(ep_reward)
        print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Length={ep_length}")

    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Max Reward: {np.max(episode_rewards):.2f}")

    return episode_rewards


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*60)
    print("Waterworld: Bidirectional Training System")
    print("="*60)

    # ========================================
    # 配置区域：可以轻松切换训练模式
    # ========================================
    
    # 选择训练模式：'train_prey' 或 'train_predator'
    TRAINING_MODE = 'train_predator'  # 修改这里来切换模式
    
    # 环境配置
    N_PREDATORS = 3
    N_PREYS = 15
    TOTAL_TIMESTEPS = 150000
    
    # 固定策略配置
    # 选项1: 使用随机策略
    fixed_policy = RandomPolicy()
    
    # 选项2: 使用训练好的模型（取消注释来使用）
    # fixed_policy = TrainedModelPolicy('prey_ppo_model.zip', device='cpu')
    
    # 选项3: 使用规则策略（取消注释来使用）
    # fixed_policy = RuleBasedPolicy()
    
    # ========================================
    
    # 根据模式设置文件名和 agent 类型
    if TRAINING_MODE == 'train_prey':
        agent_type = 'Prey'
        model_filename = 'prey_ppo_model'
        curve_filename = 'training_curve_prey.png'
    else:  # train_predator
        agent_type = 'Predator'
        model_filename = 'predator_ppo_model'
        curve_filename = 'training_curve_predator.png'
    
    print(f"\n🎯 Selected Mode: Training {agent_type} Agents")
    print(f"📦 Fixed Policy: {fixed_policy.__class__.__name__}")
    
    # 1. 创建环境
    raw_env, n_predators, n_preys = create_mixed_env(
        n_predators=N_PREDATORS,
        n_preys=N_PREYS,
        mode=TRAINING_MODE
    )

    # 2. 准备训练环境
    env = prepare_env_for_training(
        raw_env, 
        n_predators, 
        n_preys, 
        mode=TRAINING_MODE,
        fixed_policy=fixed_policy
    )

    # 3. 训练
    model, callback = train_ppo(env, agent_type=agent_type, total_timesteps=TOTAL_TIMESTEPS)

    # 4. 绘制训练曲线
    if callback.episode_rewards:
        print("\n" + "="*60)
        print("Generating Training Curve")
        print("="*60)
        plot_training_curve(
            callback.episode_rewards,
            agent_type=agent_type,
            save_path=curve_filename
        )

    # 5. 评估
    episode_rewards = evaluate_model(model, env, agent_type=agent_type, n_episodes=10)

    # 6. 保存模型
    model.save(model_filename)
    print(f"\n💾 Model saved: {model_filename}.zip")

    # 7. 最终总结
    print("\n" + "="*60)
    print("Training Complete Summary")
    print("="*60)

    if callback.episode_rewards:
        rewards = np.array(callback.episode_rewards)
        eval_rewards = np.array(episode_rewards)

        print(f"Training Mode: {TRAINING_MODE}")
        print(f"  Training Agents: {agent_type}")
        print(f"  Fixed Policy: {fixed_policy.__class__.__name__}")

        print(f"\nDuring Training:")
        print(f"  Mean Reward: {np.mean(rewards):.2f}")
        print(f"  Max Reward: {np.max(rewards):.2f}")

        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {np.mean(eval_rewards):.2f}")
        print(f"  Max Reward: {np.max(eval_rewards):.2f}")

        if np.mean(eval_rewards) > np.mean(rewards[:len(rewards)//10]):
            print(f"\n✓ {agent_type} training is effective")
        else:
            print(f"\n✗ {agent_type} training shows limited effectiveness")

    env.close()
    print("\n🎉 Training complete!")
    print(f"📁 Files generated:")
    print(f"  - {model_filename}.zip")
    print(f"  - {curve_filename}")


if __name__ == "__main__":
    main()
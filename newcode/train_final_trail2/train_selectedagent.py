"""
Waterworld: Flexible Training System
支持任意 agent 组合的训练配置
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
from typing import List, Dict, Optional, Union
import os


# ============================================================================
# 策略接口：支持多种固定策略
# ============================================================================

class AgentPolicy(ABC):
    """固定 Agent 策略的抽象基类"""
    
    @abstractmethod
    def get_action(self, obs):
        """
        根据观察获取动作
        
        Args:
            obs: 观察值 (obs_dim,)
            
        Returns:
            action: 动作 (action_dim,)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """重置策略状态（如果需要）"""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class RandomPolicy(AgentPolicy):
    """随机策略"""
    
    def __init__(self, action_dim=2, low=-1.0, high=1.0):
        self.action_dim = action_dim
        self.low = low
        self.high = high
    
    def get_action(self, obs):
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
        self.model_path = model_path
        print(f"    Loaded model: {model_path}")
    
    def get_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
    
    def reset(self):
        pass
    
    def __repr__(self):
        return f"TrainedModelPolicy('{os.path.basename(self.model_path)}')"


class RuleBasedPolicy(AgentPolicy):
    """基于规则的策略（可扩展）"""
    
    def __init__(self, rule_type='stay'):
        """
        Args:
            rule_type: 规则类型
                - 'stay': 保持静止
                - 'forward': 向前移动
                - 'circle': 圆周运动
        """
        self.rule_type = rule_type
        self.step_count = 0
    
    def get_action(self, obs):
        self.step_count += 1
        
        if self.rule_type == 'stay':
            return np.array([0.0, 0.0], dtype=np.float32)
        
        elif self.rule_type == 'forward':
            return np.array([1.0, 0.0], dtype=np.float32)
        
        elif self.rule_type == 'circle':
            angle = self.step_count * 0.1
            return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        
        else:
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def reset(self):
        self.step_count = 0
    
    def __repr__(self):
        return f"RuleBasedPolicy('{self.rule_type}')"


# ============================================================================
# Agent 配置系统
# ============================================================================

class AgentConfig:
    """Agent 配置类"""
    
    def __init__(self, agent_idx, agent_type, agent_name, role, policy=None):
        """
        Args:
            agent_idx: Agent 在环境中的全局索引
            agent_type: 'predator' 或 'prey'
            agent_name: Agent 名称，例如 'predator_0'
            role: 'training' 或 'fixed'
            policy: AgentPolicy 实例 (role='fixed' 时必须提供)
        """
        self.agent_idx = agent_idx
        self.agent_type = agent_type
        self.agent_name = agent_name
        self.role = role
        self.policy = policy
        
        # 验证
        if role not in ['training', 'fixed']:
            raise ValueError(f"role must be 'training' or 'fixed', got: {role}")
        
        if role == 'fixed' and policy is None:
            raise ValueError(f"policy must be provided when role='fixed' for agent {agent_name}")
        
        if role == 'training' and policy is not None:
            raise ValueError(f"policy should be None when role='training' for agent {agent_name}")
    
    def __repr__(self):
        if self.role == 'training':
            return f"AgentConfig({self.agent_name}, role=training)"
        else:
            return f"AgentConfig({self.agent_name}, role=fixed, policy={self.policy})"


def create_agent_configs(
    n_predators: int,
    n_preys: int,
    train_predators: Optional[List[int]] = None,
    train_preys: Optional[List[int]] = None,
    predator_policies: Optional[Union[AgentPolicy, Dict[int, AgentPolicy]]] = None,
    prey_policies: Optional[Union[AgentPolicy, Dict[int, AgentPolicy]]] = None
) -> List[AgentConfig]:
    """
    创建 Agent 配置列表
    
    Args:
        n_predators: Predator 总数
        n_preys: Prey 总数
        train_predators: 要训练的 predator 索引列表（0-based），None 表示全部固定
        train_preys: 要训练的 prey 索引列表（0-based），None 表示全部固定
        predator_policies: Predator 的固定策略
            - AgentPolicy: 所有固定 predators 使用相同策略
            - Dict[int, AgentPolicy]: 每个 predator 使用不同策略 {predator_idx: policy}
        prey_policies: Prey 的固定策略（同上）
    
    Returns:
        List[AgentConfig]: Agent 配置列表
        
    Examples:
        # 训练 predator 0 和 2，其他全部随机
        configs = create_agent_configs(
            n_predators=3, n_preys=5,
            train_predators=[0, 2],
            predator_policies=RandomPolicy(),
            prey_policies=RandomPolicy()
        )
        
        # 训练 prey 1 和 3，每个 predator 使用不同策略
        configs = create_agent_configs(
            n_predators=3, n_preys=5,
            train_preys=[1, 3],
            predator_policies={
                0: RandomPolicy(),
                1: TrainedModelPolicy('model1.zip'),
                2: RuleBasedPolicy('circle')
            },
            prey_policies=RandomPolicy()
        )
    """
    configs = []
    
    # 默认值
    if train_predators is None:
        train_predators = []
    if train_preys is None:
        train_preys = []
    
    # 验证索引范围
    if any(i >= n_predators or i < 0 for i in train_predators):
        raise ValueError(f"train_predators indices must be in [0, {n_predators-1}]")
    if any(i >= n_preys or i < 0 for i in train_preys):
        raise ValueError(f"train_preys indices must be in [0, {n_preys-1}]")
    
    # 配置 Predators（索引 0 到 n_predators-1）
    for pred_idx in range(n_predators):
        agent_idx = pred_idx
        agent_name = f'predator_{pred_idx}'
        
        if pred_idx in train_predators:
            # 训练的 predator
            config = AgentConfig(
                agent_idx=agent_idx,
                agent_type='predator',
                agent_name=agent_name,
                role='training',
                policy=None
            )
        else:
            # 固定的 predator
            if predator_policies is None:
                raise ValueError(f"predator_policies must be provided for fixed predator {pred_idx}")
            
            # 获取该 predator 的策略
            if isinstance(predator_policies, dict):
                if pred_idx not in predator_policies:
                    raise ValueError(f"No policy provided for predator {pred_idx} in predator_policies dict")
                policy = predator_policies[pred_idx]
            else:
                policy = predator_policies
            
            config = AgentConfig(
                agent_idx=agent_idx,
                agent_type='predator',
                agent_name=agent_name,
                role='fixed',
                policy=policy
            )
        
        configs.append(config)
    
    # 配置 Preys（索引 n_predators 到 n_predators+n_preys-1）
    for prey_idx in range(n_preys):
        agent_idx = n_predators + prey_idx
        agent_name = f'prey_{prey_idx}'
        
        if prey_idx in train_preys:
            # 训练的 prey
            config = AgentConfig(
                agent_idx=agent_idx,
                agent_type='prey',
                agent_name=agent_name,
                role='training',
                policy=None
            )
        else:
            # 固定的 prey
            if prey_policies is None:
                raise ValueError(f"prey_policies must be provided for fixed prey {prey_idx}")
            
            # 获取该 prey 的策略
            if isinstance(prey_policies, dict):
                if prey_idx not in prey_policies:
                    raise ValueError(f"No policy provided for prey {prey_idx} in prey_policies dict")
                policy = prey_policies[prey_idx]
            else:
                policy = prey_policies
            
            config = AgentConfig(
                agent_idx=agent_idx,
                agent_type='prey',
                agent_name=agent_name,
                role='fixed',
                policy=policy
            )
        
        configs.append(config)
    
    return configs


def print_agent_configs(configs: List[AgentConfig]):
    """打印 Agent 配置信息"""
    print("\n" + "="*70)
    print("Agent Configuration")
    print("="*70)
    
    training_agents = [c for c in configs if c.role == 'training']
    fixed_agents = [c for c in configs if c.role == 'fixed']
    
    print(f"\n📊 Summary:")
    print(f"  Total Agents: {len(configs)}")
    print(f"  Training Agents: {len(training_agents)}")
    print(f"  Fixed Agents: {len(fixed_agents)}")
    
    if training_agents:
        print(f"\n🎯 Training Agents ({len(training_agents)}):")
        for config in training_agents:
            print(f"  - {config.agent_name} (idx={config.agent_idx})")
    
    if fixed_agents:
        print(f"\n🔒 Fixed Agents ({len(fixed_agents)}):")
        for config in fixed_agents:
            print(f"  - {config.agent_name} (idx={config.agent_idx}): {config.policy}")
    
    print("="*70)


# ============================================================================
# 灵活的 VecEnv 包装器
# ============================================================================

class FlexibleMixedAgentVecEnv(VecEnv):
    """
    支持任意 agent 组合训练的自定义 VecEnv
    """

    def __init__(self, venv, agent_configs: List[AgentConfig]):
        """
        Args:
            venv: 包装后的向量化环境
            agent_configs: Agent 配置列表
        """
        self.venv = venv
        self.agent_configs = agent_configs
        self.n_total_agents = len(agent_configs)
        
        # 分离训练和固定的 agents
        self.training_configs = [c for c in agent_configs if c.role == 'training']
        self.fixed_configs = [c for c in agent_configs if c.role == 'fixed']
        
        self.training_indices = [c.agent_idx for c in self.training_configs]
        self.fixed_indices = [c.agent_idx for c in self.fixed_configs]
        
        self.n_training = len(self.training_indices)
        self.n_fixed = len(self.fixed_indices)
        
        if self.n_training == 0:
            raise ValueError("Must have at least one training agent")
        
        # 创建索引到配置的映射
        self.idx_to_config = {c.agent_idx: c for c in agent_configs}
        
        # 获取原始空间
        original_obs_space = venv.observation_space
        original_action_space = venv.action_space
        
        # 创建新的 VecEnv，num_envs = 训练 agents 的数量
        super().__init__(
            num_envs=self.n_training,
            observation_space=original_obs_space,
            action_space=original_action_space
        )
        
        # 缓存最新的观察值
        self.latest_obs = None
        
        # 统计信息
        print(f"\n  FlexibleMixedAgentVecEnv initialized:")
        print(f"    - Total agents: {self.n_total_agents}")
        print(f"    - Training agents: {self.n_training}")
        print(f"    - Fixed agents: {self.n_fixed}")

    def reset(self):
        """重置环境"""
        obs = self.venv.reset()
        self.latest_obs = obs
        
        # 重置所有固定策略
        for config in self.fixed_configs:
            config.policy.reset()
        
        # 返回训练 agents 的观察
        training_obs = obs[self.training_indices]
        return training_obs

    def step_async(self, actions):
        """
        组合训练 agents 的动作和固定 agents 的动作
        
        Args:
            actions: shape (n_training, action_dim) - 训练 agents 的动作
        """
        # 生成固定 agents 的动作
        fixed_actions = np.zeros((self.n_fixed, 2), dtype=np.float32)
        for i, config in enumerate(self.fixed_configs):
            agent_idx = config.agent_idx
            obs = self.latest_obs[agent_idx] if self.latest_obs is not None else None
            fixed_actions[i] = config.policy.get_action(obs)
        
        # 组合所有动作
        full_actions = np.zeros((self.n_total_agents, 2), dtype=np.float32)
        
        # 填充训练 agents 的动作
        for i, agent_idx in enumerate(self.training_indices):
            full_actions[agent_idx] = actions[i]
        
        # 填充固定 agents 的动作
        for i, agent_idx in enumerate(self.fixed_indices):
            full_actions[agent_idx] = fixed_actions[i]
        
        # 传递给底层环境
        self.venv.step_async(full_actions)

    def step_wait(self):
        """获取环境结果"""
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
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self.venv.env_is_wrapped(wrapper_class, indices)


# ============================================================================
# 训练监控回调
# ============================================================================

class TrainingMonitorCallback(BaseCallback):
    """监控训练过程 + 性能指标"""

    def __init__(self, training_agent_names, check_freq=1000, verbose=1):
        """
        Args:
            training_agent_names: 训练 agents 的名称列表
        """
        super().__init__(verbose)
        self.training_agent_names = training_agent_names
        self.check_freq = check_freq
        
        # 原有的奖励统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_ep_reward = 0
        self.current_ep_length = 0
        
        # ✅ 新增：性能指标统计
        self.episode_metrics = {
            'hunting_rate': [],
            'escape_rate': [],
            'foraging_rate': []
        }

    def _on_step(self):
        reward_sum = np.sum(self.locals['rewards'])
        self.current_ep_reward += reward_sum
        self.current_ep_length += 1

        if np.any(self.locals['dones']):
            # 记录奖励
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)
            
            # ✅ 提取性能指标
            infos = self.locals.get('infos', [])
            self._extract_performance_metrics(infos)

            if len(self.episode_rewards) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                print(f"\n[Training] Episode {len(self.episode_rewards)}:")
                print(f"  Agents: {', '.join(self.training_agent_names)}")
                print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
                print(f"  Avg Length: {np.mean(self.episode_lengths[-10:]):.0f}")
                print(f"  Max Reward: {np.max(recent_rewards):.2f}")
                
                # ✅ 打印性能指标
                self._print_performance_metrics()

            self.current_ep_reward = 0
            self.current_ep_length = 0

        return True
    
    def _extract_performance_metrics(self, infos):
        """✅ 从 infos 提取性能指标"""
        if not infos:
            return
        
        # 聚合本episode的指标
        ep_metrics = {
            'hunting_rate': [],
            'escape_rate': [],
            'foraging_rate': []
        }
        
        for info in infos:
            if not isinstance(info, dict):
                continue
            
            # 检查是否有 performance_metrics
            pm = info.get('performance_metrics', {})
            agent_name = info.get('agent_name', None)  # 可能没有这个字段
            
            if pm:
                # 收集所有训练agent的指标
                # 注意：在SB3的VecEnv中，infos可能只包含训练agent的数据
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
                self.episode_metrics[key].append(avg)
    
    def _print_performance_metrics(self):
        """✅ 打印最近10个episode的性能指标"""
        if not self.episode_metrics['hunting_rate']:
            return
        
        recent_n = min(10, len(self.episode_metrics['hunting_rate']))
        
        print(f"\n  📊 Performance Metrics (last {recent_n} episodes):")
        
        for key, values in self.episode_metrics.items():
            if values:
                recent = values[-recent_n:]
                avg = np.mean(recent)
                std = np.std(recent)
                
                # 根据指标类型选择emoji
                if 'hunting' in key:
                    emoji = "🎯"
                elif 'escape' in key:
                    emoji = "🏃"
                elif 'foraging' in key:
                    emoji = "🍎"
                else:
                    emoji = "📈"
                
                print(f"    {emoji} {key}: {avg:.3f} ± {std:.3f}")

# ============================================================================
# 训练曲线绘制
# ============================================================================

def plot_training_curve(episode_rewards, training_info, save_path='training_curve.png'):
    """
    绘制训练曲线
    
    Args:
        episode_rewards: Episode 奖励列表
        training_info: 训练信息字符串
        save_path: 保存路径
    """
    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards)

    plt.figure(figsize=(12, 6))

    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Rewards')

    if len(rewards) >= 10:
        window_size = min(10, len(rewards))
        smoothed = uniform_filter1d(rewards, size=window_size, mode='nearest')
        plt.plot(episodes, smoothed, color='red', linewidth=2, 
                label=f'Moving Average (window={window_size})')

    if len(rewards) >= 50:
        window_size = min(50, len(rewards))
        trend = uniform_filter1d(rewards, size=window_size, mode='nearest')
        plt.plot(episodes, trend, color='green', linewidth=2, linestyle='--', 
                label=f'Trend (window={window_size})')

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title(f'PPO Training: {training_info}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

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
    print(f"\n📈 Training curve saved to: {save_path}")
    plt.close()


# ============================================================================
# 环境创建和准备
# ============================================================================

def create_env(n_predators, n_preys, agent_configs):
    """创建环境"""
    print("\n" + "="*70)
    print("Creating Waterworld Environment")
    print("="*70)

    total_agents = n_predators + n_preys
    
    # 构建 agent_algorithms 列表
    agent_algos = []
    for config in agent_configs:
        if config.role == 'training':
            agent_algos.append("PPO")
        else:
            agent_algos.append("Fixed")

    env = waterworld_v4.parallel_env(
        render_mode=None,
        n_predators=n_predators,
        n_preys=n_preys,
        n_evaders=90,
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=10,
        agent_algorithms=agent_algos,
        max_cycles=3000,
        static_food=True,
        static_poison=True,
    )

    print(f"\nEnvironment Details:")
    print(f"  Predators: {n_predators}")
    print(f"  Preys: {n_preys}")
    print(f"  Total Agents: {total_agents}")
    print(f"  Food: 180 (static)")
    print(f"  Poison: 10 (static)")
    print(f"  Observation Dim: {env.observation_space('prey_0').shape}")
    print(f"  Action Dim: {env.action_space('prey_0').shape}")

    return env


def prepare_env_for_training(env, agent_configs):
    """准备训练环境"""
    print("\n" + "="*70)
    print("Converting Environment Format")
    print("="*70)

    # 标准转换
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')

    print("  ✓ Standard conversion complete")
    print(f"  ✓ num_envs: {env.num_envs}")

    # 应用灵活的混合环境包装器
    env = FlexibleMixedAgentVecEnv(env, agent_configs)

    # 添加监控
    env = VecMonitor(env)
    print("  ✓ Environment preparation complete")

    return env


# ============================================================================
# 训练和评估
# ============================================================================

def train_ppo(env, agent_configs, total_timesteps=1000000):
    """使用 PPO 训练"""
    training_configs = [c for c in agent_configs if c.role == 'training']
    training_names = [c.agent_name for c in training_configs]
    
    print("\n" + "="*70)
    print("Starting PPO Training")
    print("="*70)
    print(f"Training Agents: {', '.join(training_names)}")
    print(f"Total Timesteps: {total_timesteps:,}")
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

    callback = TrainingMonitorCallback(training_agent_names=training_names, check_freq=1000)

    print("\n🚀 Starting training...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    print("\n✓ Training complete!")

    # 显示训练统计
    if callback.episode_rewards:
        rewards = np.array(callback.episode_rewards)
        print("\n" + "="*70)
        print("Training Statistics")
        print("="*70)
        print(f"Total Episodes: {len(rewards)}")
        print(f"Mean Reward: {np.mean(rewards):.2f}")
        print(f"Max Reward: {np.max(rewards):.2f}")
        print(f"Min Reward: {np.min(rewards):.2f}")

        n = len(rewards)
        if n >= 10:
            early = rewards[:max(1, n//10)]
            late = rewards[-max(1, n//10):]
            improvement = np.mean(late) - np.mean(early)

            print(f"\nLearning Analysis:")
            print(f"  Early Mean (first 10%): {np.mean(early):.2f}")
            print(f"  Late Mean (last 10%): {np.mean(late):.2f}")
            print(f"  Improvement: {improvement:+.2f}")

            if improvement > 5:
                print("  Conclusion: ✓ Effective Learning")
            elif improvement > -5:
                print("  Conclusion: ~ Limited Learning")
            else:
                print("  Conclusion: ✗ No Effective Learning")

    return model, callback


def evaluate_model(model, env, n_episodes=10):
    """评估训练好的模型"""
    print("\n" + "="*70)
    print(f"Evaluating Model ({n_episodes} episodes)")
    print("="*70)

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
    print(f"  Min Reward: {np.min(episode_rewards):.2f}")

    return episode_rewards


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*70)
    print("Waterworld: Flexible Training System")
    print("="*70)

    # ========================================
    # 配置区域：灵活配置训练场景
    # ========================================
    
    # 环境配置
    N_PREDATORS = 5
    N_PREYS = 10
    TOTAL_TIMESTEPS = 10000000
    
    # ========================================
    # 场景 1: 训练部分 Predators，其他全部随机
    # ========================================
    """
    agent_configs = create_agent_configs(
        n_predators=N_PREDATORS,
        n_preys=N_PREYS,
        train_predators=[0, 2, 4],  # 训练 predator 0, 2, 4
        train_preys=None,  # 不训练任何 prey
        predator_policies=RandomPolicy(),  # 固定的 predators 使用随机策略
        prey_policies=RandomPolicy()  # 所有 preys 使用随机策略
    )
    """
    
    # ========================================
    # 场景 2: 训练部分 Preys，其他全部随机
    # ========================================
    # agent_configs = create_agent_configs(
    #     n_predators=N_PREDATORS,
    #     n_preys=N_PREYS,
    #     train_predators=[1, 2],  # 训练 predator 1, 2
    #     train_preys=None,  # 训练 prey 1, 3, 5, 7
    #     predator_policies=RandomPolicy(),  # 所有 predators 使用随机策略
    #     prey_policies=RandomPolicy()  # 固定的 preys 使用随机策略
    # )
    
    # ========================================
    # 场景 3: 同时训练部分 Predators 和 Preys
    # ========================================
    """
    agent_configs = create_agent_configs(
        n_predators=N_PREDATORS,
        n_preys=N_PREYS,
        train_predators=[0, 1],  # 训练 predator 0, 1
        train_preys=[2, 4, 6],  # 训练 prey 2, 4, 6
        predator_policies=RandomPolicy(),
        prey_policies=RandomPolicy()
    )
    """
    
    # ========================================
    # 场景 4: 每个固定 agent 使用不同策略
    # ========================================
    """
    agent_configs = create_agent_configs(
        n_predators=N_PREDATORS,
        n_preys=N_PREYS,
        train_predators=[0],  # 只训练 predator 0
        train_preys=[0, 1],  # 训练 prey 0, 1
        predator_policies={
            # 为每个固定的 predator 指定不同策略
            1: RandomPolicy(),
            2: RuleBasedPolicy('circle'),
            3: RuleBasedPolicy('forward'),
            4: RandomPolicy()
        },
        prey_policies={
            # 为每个固定的 prey 指定不同策略
            2: RandomPolicy(),
            3: RuleBasedPolicy('stay'),
            4: RandomPolicy(),
            5: RuleBasedPolicy('circle'),
            6: RandomPolicy(),
            7: RandomPolicy(),
            8: RuleBasedPolicy('forward'),
            9: RandomPolicy()
        }
    )
    """
    
    # ========================================
    # 场景 5: 使用训练好的模型作为固定策略（需要先有模型）
    # ========================================

    agent_configs = create_agent_configs(
        n_predators=N_PREDATORS,
        n_preys=N_PREYS,
        train_predators=[0, 1],
        train_preys=None,
        predator_policies={
            2: TrainedModelPolicy('predator_ppo_model.zip'),
            3: RandomPolicy(),
            4: RandomPolicy()
        },
        prey_policies={
            # 部分 prey 使用训练好的模型，部分随机
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
    
    # 打印配置信息
    print_agent_configs(agent_configs)
    
    # 生成训练信息字符串（用于文件名和图表）
    training_configs = [c for c in agent_configs if c.role == 'training']
    training_predators = [c for c in training_configs if c.agent_type == 'predator']
    training_preys = [c for c in training_configs if c.agent_type == 'prey']
    
    training_info_parts = []
    if training_predators:
        pred_names = [c.agent_name for c in training_predators]
        training_info_parts.append(f"Predators[{','.join([n.split('_')[1] for n in pred_names])}]")
    if training_preys:
        prey_names = [c.agent_name for c in training_preys]
        training_info_parts.append(f"Preys[{','.join([n.split('_')[1] for n in prey_names])}]")
    
    training_info = "_".join(training_info_parts)
    model_filename = f'model_{training_info}'
    curve_filename = f'training_curve_{training_info}.png'
    
    print(f"\n📝 Files will be saved as:")
    print(f"  - Model: {model_filename}.zip")
    print(f"  - Curve: {curve_filename}")
    
    # 1. 创建环境
    raw_env = create_env(
        n_predators=N_PREDATORS,
        n_preys=N_PREYS,
        agent_configs=agent_configs
    )

    # 2. 准备训练环境
    env = prepare_env_for_training(raw_env, agent_configs)

    # 3. 训练
    model, callback = train_ppo(env, agent_configs, total_timesteps=TOTAL_TIMESTEPS)

    # 4. 绘制训练曲线
    if callback.episode_rewards:
        print("\n" + "="*70)
        print("Generating Training Curve")
        print("="*70)
        plot_training_curve(
            callback.episode_rewards,
            training_info=training_info.replace('_', ' + '),
            save_path=curve_filename
        )

    # 5. 评估
    episode_rewards = evaluate_model(model, env, n_episodes=10)

    # 6. 保存模型
    model.save(model_filename)
    print(f"\n💾 Model saved: {model_filename}.zip")

    # 7. 最终总结
    print("\n" + "="*70)
    print("Training Complete Summary")
    print("="*70)

    if callback.episode_rewards:
        rewards = np.array(callback.episode_rewards)
        eval_rewards = np.array(episode_rewards)

        print(f"\n🎯 Training Configuration:")
        if training_predators:
            print(f"  Training Predators: {[c.agent_name for c in training_predators]}")
        if training_preys:
            print(f"  Training Preys: {[c.agent_name for c in training_preys]}")

        print(f"\n📊 Training Performance:")
        print(f"  Mean Reward: {np.mean(rewards):.2f}")
        print(f"  Max Reward: {np.max(rewards):.2f}")
        print(f"  Std Reward: {np.std(rewards):.2f}")

        print(f"\n🧪 Evaluation Performance:")
        print(f"  Mean Reward: {np.mean(eval_rewards):.2f}")
        print(f"  Max Reward: {np.max(eval_rewards):.2f}")
        print(f"  Std Reward: {np.std(eval_rewards):.2f}")

        n = len(rewards)
        if n >= 10:
            early = rewards[:max(1, n//10)]
            late = rewards[-max(1, n//10):]
            improvement = np.mean(late) - np.mean(early)
            
            print(f"\n📈 Learning Progress:")
            print(f"  Early Mean (first 10%): {np.mean(early):.2f}")
            print(f"  Late Mean (last 10%): {np.mean(late):.2f}")
            print(f"  Improvement: {improvement:+.2f}")
            
            if improvement > 5:
                print(f"  Status: ✓ Training agents learned effectively")
            elif improvement > -5:
                print(f"  Status: ~ Limited learning observed")
            else:
                print(f"  Status: ✗ No significant learning")

    env.close()
    
    print("\n" + "="*70)
    print("🎉 All Done!")
    print("="*70)
    print(f"\n📁 Generated Files:")
    print(f"  ✓ {model_filename}.zip")
    print(f"  ✓ {curve_filename}")
    print("\n💡 Tips:")
    print("  - Modify agent_configs in main() to try different training scenarios")
    print("  - Use saved models with TrainedModelPolicy() for hierarchical training")
    print("  - Create custom policies by extending AgentPolicy class")


if __name__ == "__main__":
    main()
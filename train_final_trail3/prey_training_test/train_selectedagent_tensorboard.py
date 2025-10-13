"""
Waterworld: Flexible Training System
支持任意 agent 组合的训练配置 + TensorBoard集成
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
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


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
# 训练监控回调 + TensorBoard集成
# ============================================================================

class TrainingMonitorCallback(BaseCallback):
    """
    监控训练过程 + 性能指标 + TensorBoard日志
    
    记录指标：
    - Individual Level: 每个训练agent的独立数据
    - Average Level: 所有训练agent的平均数据
    """

    def __init__(self, training_agent_names, log_dir=None, check_freq=1000, verbose=1):
        """
        Args:
            training_agent_names: 训练 agents 的名称列表
            log_dir: TensorBoard日志目录（None表示不使用TensorBoard）
            check_freq: 打印频率
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.training_agent_names = training_agent_names
        self.check_freq = check_freq
        
        # === 全局统计（保持向后兼容） ===
        self.episode_rewards = []  # 所有agent的总奖励
        self.episode_lengths = []
        
        # === 每个Agent的独立数据 ===
        self.agent_episode_data = {}
        for agent_name in training_agent_names:
            self.agent_episode_data[agent_name] = {
                'cumulative_reward': 0.0,      # 累积奖励
                'survival_time': 0,            # 存活步数
                'is_dead': False,              # 是否已死亡
                'final_metrics': None,         # 最终性能指标
            }
        
        # === 当前Episode的步数 ===
        self.current_ep_length = 0
        
        # === TensorBoard Writer ===
        self.writer = None
        if log_dir:
            self.writer = SummaryWriter(log_dir)
            print(f"\n  📊 TensorBoard logging enabled")
            print(f"     Log directory: {log_dir}")
            print(f"     Run: tensorboard --logdir={log_dir}")

    def _on_step(self):
        """每个step的回调"""
        rewards = self.locals['rewards']  # shape: (n_training,)
        dones = self.locals['dones']      # shape: (n_training,)
        infos = self.locals.get('infos', [])
        
        self.current_ep_length += 1
        
        # === 为每个训练agent累积数据 ===
        for i, agent_name in enumerate(self.training_agent_names):
            agent_data = self.agent_episode_data[agent_name]
            
            # 累积奖励（包括死亡后的0）
            agent_data['cumulative_reward'] += float(rewards[i])
            
            # 检测首次死亡
            if dones[i] and not agent_data['is_dead']:
                agent_data['is_dead'] = True
                agent_data['survival_time'] = self.current_ep_length
                
                # 立即保存死亡时的performance_metrics（防止被清零）
                if i < len(infos) and isinstance(infos[i], dict):
                    pm = infos[i].get('performance_metrics')
                    if pm:
                        agent_data['final_metrics'] = pm.copy()
        
        # === 检测Episode结束（所有agent都done） ===
        if np.all(dones):
            self._on_episode_end(infos)
        
        return True
    
    def _on_episode_end(self, infos):
        """当Episode结束时的处理"""
        episode_num = len(self.episode_rewards) + 1
        
        # === 收集所有agent的数据 ===
        all_returns = []
        all_hunting_rates = []
        all_escape_rates = []
        all_foraging_rates = []
        all_survival_times = []
        
        for i, agent_name in enumerate(self.training_agent_names):
            agent_data = self.agent_episode_data[agent_name]
            
            # 1. Episode Return
            ep_return = agent_data['cumulative_reward']
            all_returns.append(ep_return)
            
            if self.writer:
                self.writer.add_scalar(
                    f'Individual/{agent_name}/episode_return',
                    ep_return,
                    episode_num
                )
            
            # 2. Survival Time
            survival_time = agent_data['survival_time'] if agent_data['survival_time'] > 0 else self.current_ep_length
            all_survival_times.append(survival_time)
            
            if self.writer:
                self.writer.add_scalar(
                    f'Individual/{agent_name}/survival_time',
                    survival_time,
                    episode_num
                )
            
            # 3. Performance Metrics
            # 优先使用死亡时保存的，否则从当前info读取
            metrics = agent_data['final_metrics']
            if metrics is None and i < len(infos) and isinstance(infos[i], dict):
                metrics = infos[i].get('performance_metrics', {})
            
            if metrics:
                hunting_rate = metrics.get('hunting_rate', 0.0)
                escape_rate = metrics.get('escape_rate', 0.0)
                foraging_rate = metrics.get('foraging_rate', 0.0)
                
                all_hunting_rates.append(hunting_rate)
                all_escape_rates.append(escape_rate)
                all_foraging_rates.append(foraging_rate)
                
                if self.writer:
                    self.writer.add_scalar(
                        f'Individual/{agent_name}/hunting_rate',
                        hunting_rate,
                        episode_num
                    )
                    self.writer.add_scalar(
                        f'Individual/{agent_name}/escape_rate',
                        escape_rate,
                        episode_num
                    )
                    self.writer.add_scalar(
                        f'Individual/{agent_name}/foraging_rate',
                        foraging_rate,
                        episode_num
                    )
        
        # === 记录平均值 ===
        if self.writer:
            if all_returns:
                self.writer.add_scalar('Average/episode_return', np.mean(all_returns), episode_num)
            if all_hunting_rates:
                self.writer.add_scalar('Average/hunting_rate', np.mean(all_hunting_rates), episode_num)
            if all_escape_rates:
                self.writer.add_scalar('Average/escape_rate', np.mean(all_escape_rates), episode_num)
            if all_foraging_rates:
                self.writer.add_scalar('Average/foraging_rate', np.mean(all_foraging_rates), episode_num)
            if all_survival_times:
                self.writer.add_scalar('Average/survival_time', np.mean(all_survival_times), episode_num)
        
        # === 记录全局统计（向后兼容） ===
        total_return = sum(all_returns)
        self.episode_rewards.append(total_return)
        self.episode_lengths.append(self.current_ep_length)
        
        # === 打印进度（每10个episode） ===
        if episode_num % 10 == 0:
            self._print_progress(episode_num, all_returns, all_hunting_rates, all_escape_rates, all_survival_times)
        
        # === 重置所有agent的数据 ===
        for agent_name in self.training_agent_names:
            self.agent_episode_data[agent_name] = {
                'cumulative_reward': 0.0,
                'survival_time': 0,
                'is_dead': False,
                'final_metrics': None
            }
        
        self.current_ep_length = 0
    
    def _print_progress(self, episode_num, returns, hunting_rates, escape_rates, survival_times):
        """打印训练进度"""
        recent_n = min(10, episode_num)
        recent_rewards = self.episode_rewards[-recent_n:]
        
        print(f"\n[Training] Episode {episode_num}:")
        print(f"  Agents: {', '.join(self.training_agent_names)}")
        print(f"  Avg Total Reward: {np.mean(recent_rewards):.2f}")
        print(f"  Avg Episode Length: {np.mean(self.episode_lengths[-recent_n:]):.0f}")
        
        # 打印当前episode的individual数据
        print(f"\n  📊 Current Episode Metrics:")
        for i, agent_name in enumerate(self.training_agent_names):
            print(f"    {agent_name}:")
            if i < len(returns):
                print(f"      Return: {returns[i]:.2f}")
            if i < len(hunting_rates):
                print(f"      🎯 Hunting: {hunting_rates[i]:.3f}")
            if i < len(escape_rates):
                print(f"      🏃 Escape: {escape_rates[i]:.3f}")
            if i < len(survival_times):
                print(f"      ⏱️  Survival: {survival_times[i]} steps")
        
        # 打印平均值
        if returns:
            print(f"\n  📈 Average Across Agents:")
            print(f"      Return: {np.mean(returns):.2f}")
            if hunting_rates:
                print(f"      🎯 Hunting: {np.mean(hunting_rates):.3f}")
            if escape_rates:
                print(f"      🏃 Escape: {np.mean(escape_rates):.3f}")
            if survival_times:
                print(f"      ⏱️  Survival: {np.mean(survival_times):.0f} steps")
    
    def on_training_end(self):
        """训练结束时的清理"""
        if self.writer:
            self.writer.close()
            print("\n  ✓ TensorBoard writer closed")


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
        n_evaders=1,
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        sensor_range=0.5,  # 增加传感器范围
        n_poisons=1,
        agent_algorithms=agent_algos,
        max_cycles=1000,
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

def train_ppo(env, agent_configs, total_timesteps=1000000, log_dir=None):
    """
    使用 PPO 训练
    
    Args:
        env: 训练环境
        agent_configs: Agent配置列表
        total_timesteps: 总训练步数
        log_dir: TensorBoard日志目录（None表示不记录）
    """
    training_configs = [c for c in agent_configs if c.role == 'training']
    training_names = [c.agent_name for c in training_configs]
    
    print("\n" + "="*70)
    print("Starting PPO Training")
    print("="*70)
    print(f"Training Agents: {', '.join(training_names)}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if log_dir:
        print(f"TensorBoard Log: {log_dir}")

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

    # 创建带TensorBoard的callback
    callback = TrainingMonitorCallback(
        training_agent_names=training_names,
        log_dir=log_dir,
        check_freq=1000
    )

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
    print("Waterworld: Flexible Training System + TensorBoard")
    print("="*70)

    # ========================================
    # 配置区域：灵活配置训练场景
    # ========================================
    
    # 环境配置
    N_PREDATORS = 3
    N_PREYS = 30
    TOTAL_TIMESTEPS = 100000
    
    # ========================================
    # 场景 2: 训练部分 Predators
    # ========================================
    agent_configs = create_agent_configs(
        n_predators=N_PREDATORS,
        n_preys=N_PREYS,
        train_predators=[1, 2],  # 训练 predator 1, 2
        train_preys=None,
        predator_policies=RandomPolicy(),
        prey_policies=RandomPolicy()
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
    
    # ✅ 生成TensorBoard日志目录（结构化）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", "waterworld", training_info, f"run_{timestamp}")
    
    model_filename = f'model_{training_info}'
    curve_filename = f'training_curve_{training_info}.png'
    
    print(f"\n📝 Files will be saved as:")
    print(f"  - Model: {model_filename}.zip")
    print(f"  - Curve: {curve_filename}")
    print(f"  - TensorBoard: {log_dir}")
    
    # 1. 创建环境
    raw_env = create_env(
        n_predators=N_PREDATORS,
        n_preys=N_PREYS,
        agent_configs=agent_configs
    )

    # 2. 准备训练环境
    env = prepare_env_for_training(raw_env, agent_configs)

    # 3. 训练（带TensorBoard）
    model, callback = train_ppo(
        env, 
        agent_configs, 
        total_timesteps=TOTAL_TIMESTEPS,
        log_dir=log_dir  # ✅ 传入日志目录
    )

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
    print(f"  ✓ TensorBoard logs: {log_dir}")
    print("\n💡 View TensorBoard:")
    print(f"  tensorboard --logdir={os.path.join('logs', 'waterworld')}")
    print("\n💡 Tips:")
    print("  - Modify agent_configs in main() to try different training scenarios")
    print("  - Use saved models with TrainedModelPolicy() for hierarchical training")
    print("  - Create custom policies by extending AgentPolicy class")


if __name__ == "__main__":
    main()
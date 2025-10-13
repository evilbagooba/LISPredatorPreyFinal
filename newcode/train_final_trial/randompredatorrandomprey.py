"""
Waterworld Dual-List Multi-Agent Training System
支持 Predator 和 Prey 独立配置的多智能体训练框架
✅ 只记录训练agents的reward数据
"""

from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import os
from datetime import datetime
from collections import deque


# ============================================================================
# Algorithm Configuration System
# ============================================================================

class AlgorithmConfig(ABC):
    """Base class for algorithm configurations"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_model_class(self):
        """Return the algorithm model class"""
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return hyperparameters for the algorithm"""
        pass
    
    def get_color(self) -> str:
        """Return color for plotting"""
        return 'blue'


class PPOConfig(AlgorithmConfig):
    def __init__(self):
        super().__init__("PPO")
    
    def get_model_class(self):
        return PPO
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'learning_rate': 5e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.98,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0001,
            'vf_coef': 0.4,
            'max_grad_norm': 0.5,
            'verbose': 1,
        }
    
    def get_color(self) -> str:
        return 'blue'


class SACConfig(AlgorithmConfig):
    def __init__(self):
        super().__init__("SAC")
    
    def get_model_class(self):
        return SAC
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'learning_rate': 3e-4,
            'buffer_size': 1000000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',
            'verbose': 1,
        }
    
    def get_color(self) -> str:
        return 'red'


class TD3Config(AlgorithmConfig):
    def __init__(self):
        super().__init__("TD3")
    
    def get_model_class(self):
        return TD3
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'learning_rate': 1e-3,
            'buffer_size': 1000000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'verbose': 1,
        }
    
    def get_color(self) -> str:
        return 'green'


class A2CConfig(AlgorithmConfig):
    def __init__(self):
        super().__init__("A2C")
    
    def get_model_class(self):
        return A2C
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'learning_rate': 7e-4,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1,
        }
    
    def get_color(self) -> str:
        return 'orange'


ALGORITHM_REGISTRY = {
    'ppo': PPOConfig,
    'sac': SACConfig,
    'td3': TD3Config,
    'a2c': A2CConfig,
}


def get_algorithm_config(algo_name: str) -> AlgorithmConfig:
    """Get algorithm configuration by name"""
    algo_name = algo_name.lower()
    if algo_name not in ALGORITHM_REGISTRY:
        available = ', '.join(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm '{algo_name}'. Available: {available}")
    return ALGORITHM_REGISTRY[algo_name]()


# ============================================================================
# Agent Name Parser
# ============================================================================

class AgentNameParser:
    """解析agent名称: algorithm_role_version_mode"""
    
    @staticmethod
    def parse(name: str) -> Dict[str, Any]:
        """
        解析agent名称
        支持格式:
        - 'ppo_prey_v1_train'
        - 'ppo_predator_v0_exe'
        - 'random_prey_exe'
        """
        parts = name.split('_')
        
        if len(parts) == 3:  # random_prey_exe
            algorithm, role, mode = parts
            return {
                'algorithm': algorithm.lower(),
                'role': role.lower(),
                'version': None,
                'mode': mode.lower()
            }
        elif len(parts) == 4:  # ppo_prey_v1_train
            algorithm, role, version, mode = parts
            return {
                'algorithm': algorithm.lower(),
                'role': role.lower(),
                'version': version.lower(),
                'mode': mode.lower()
            }
        else:
            raise ValueError(
                f"Invalid agent name: '{name}'\n"
                f"Expected: 'algorithm_role_version_mode' or 'random_role_mode'"
            )
    
    @staticmethod
    def validate(name: str) -> Tuple[bool, str]:
        """验证agent名称格式"""
        try:
            parsed = AgentNameParser.parse(name)
            
            valid_algorithms = ['ppo', 'sac', 'td3', 'a2c', 'random']
            if parsed['algorithm'] not in valid_algorithms:
                return False, f"Invalid algorithm '{parsed['algorithm']}'"
            
            valid_roles = ['prey', 'predator']
            if parsed['role'] not in valid_roles:
                return False, f"Invalid role '{parsed['role']}'"
            
            valid_modes = ['train', 'exe']
            if parsed['mode'] not in valid_modes:
                return False, f"Invalid mode '{parsed['mode']}'"
            
            return True, "Valid"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def get_model_path(name: str, base_dir: str = 'models') -> Optional[str]:
        """生成模型文件路径"""
        parsed = AgentNameParser.parse(name)
        
        if parsed['mode'] == 'train' or parsed['algorithm'] == 'random':
            return None
        
        filename = f"{parsed['algorithm']}_{parsed['role']}_{parsed['version']}.zip"
        return os.path.join(base_dir, filename)


# ============================================================================
# Dual-List Configuration Manager
# ============================================================================

class DualListConfigManager:
    """双列表配置管理器 - 分别管理 Predator 和 Prey"""
    
    def __init__(
        self, 
        predator_configs: List[Tuple[int, str]],
        prey_configs: List[Tuple[int, str]],
        model_base_dir: str = 'models'
    ):
        """
        Args:
            predator_configs: [(count, name), ...] for predators
            prey_configs: [(count, name), ...] for preys
        """
        self.model_base_dir = model_base_dir
        
        # 解析和验证配置
        self.predator_configs = self._parse_configs(predator_configs, 'predator')
        self.prey_configs = self._parse_configs(prey_configs, 'prey')
        
        self.n_predators = len(self.predator_configs)
        self.n_preys = len(self.prey_configs)
        self.n_total = self.n_predators + self.n_preys
        
        # 识别训练配置
        self._identify_training_config()
        
        # 验证配置合法性
        self._validate_training_config()
    
    def _parse_configs(self, configs: List[Tuple[int, str]], expected_role: str) -> List[Dict]:
        """解析配置列表"""
        parsed_list = []
        
        for count, name in configs:
            # 验证名称
            is_valid, error_msg = AgentNameParser.validate(name)
            if not is_valid:
                raise ValueError(f"Invalid config '{name}': {error_msg}")
            
            # 解析
            parsed = AgentNameParser.parse(name)
            
            # 验证角色匹配
            if parsed['role'] != expected_role:
                raise ValueError(
                    f"Role mismatch: config '{name}' has role '{parsed['role']}', "
                    f"expected '{expected_role}'"
                )
            
            # 展开到每个agent
            for _ in range(count):
                parsed_list.append({
                    'name': name,
                    'algorithm': parsed['algorithm'],
                    'role': parsed['role'],
                    'version': parsed['version'],
                    'mode': parsed['mode'],
                })
        
        return parsed_list
    
    def _identify_training_config(self):
        """识别训练配置"""
        self.training_role = None
        self.training_algorithm = None
        self.training_version = None
        self.training_indices = []
        
        # 检查 predator 训练
        predator_train = [i for i, cfg in enumerate(self.predator_configs) if cfg['mode'] == 'train']
        
        # 检查 prey 训练
        prey_train = [i for i, cfg in enumerate(self.prey_configs) if cfg['mode'] == 'train']
        
        if predator_train:
            self.training_role = 'predator'
            # ✅ 修复：predator在环境中排在prey之后，需要加偏移
            self.training_indices = [i + self.n_preys for i in predator_train]
            self.training_algorithm = self.predator_configs[predator_train[0]]['algorithm']
            self.training_version = self.predator_configs[predator_train[0]]['version']
        
        if prey_train:
            self.training_role = 'prey'
            # ✅ 修复：prey在环境中排在最前面，不需要偏移
            self.training_indices = prey_train
            self.training_algorithm = self.prey_configs[prey_train[0]]['algorithm']
            self.training_version = self.prey_configs[prey_train[0]]['version']
    def _validate_training_config(self):
        """验证训练配置"""
        # 检查是否有训练agent
        predator_train = any(cfg['mode'] == 'train' for cfg in self.predator_configs)
        prey_train = any(cfg['mode'] == 'train' for cfg in self.prey_configs)
        
        # ✅ 允许全部都是执行模式（无训练）
        if not predator_train and not prey_train:
            print("\n⚠️  Warning: No training agents configured. Running in EXECUTION-ONLY mode.")
            return
        
        # 不支持同时训练两种角色
        if predator_train and prey_train:
            raise ValueError(
                "Cannot train both predator and prey simultaneously! "
                "Only one role can be in training mode."
            )
        
        # 验证训练角色只有一种算法
        if predator_train:
            train_algos = {cfg['algorithm'] for cfg in self.predator_configs if cfg['mode'] == 'train'}
            if len(train_algos) > 1:
                raise ValueError(
                    f"Multiple training algorithms in predator: {train_algos}. "
                    f"Only one algorithm can be trained at a time."
                )
        
        if prey_train:
            train_algos = {cfg['algorithm'] for cfg in self.prey_configs if cfg['mode'] == 'train'}
            if len(train_algos) > 1:
                raise ValueError(
                    f"Multiple training algorithms in prey: {train_algos}. "
                    f"Only one algorithm can be trained at a time."
                )
    
    def get_agent_config(self, agent_id: int) -> Dict[str, Any]:
        """获取指定agent的配置（全局索引）"""
        if agent_id < self.n_predators:
            return self.predator_configs[agent_id].copy()
        else:
            return self.prey_configs[agent_id - self.n_predators].copy()
    
    def is_training_agent(self, agent_id: int) -> bool:
        """判断agent是否在训练"""
        return agent_id in self.training_indices
    
    def get_training_indices(self) -> List[int]:
        """获取训练agent的全局索引"""
        return self.training_indices.copy()
    
    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "="*70)
        print("Dual-List Agent Configuration")
        print("="*70)
        
        # Prey 配置（显示在前，因为环境中prey在前）
        print("\n🐰 PREY Configuration:")
        print("-" * 70)
        self._print_role_summary(self.prey_configs, 'prey', 0)  # ✅ prey在环境索引0开始
        
        # Predator 配置
        print("\n🦁 PREDATOR Configuration:")
        print("-" * 70)
        self._print_role_summary(self.predator_configs, 'predator', self.n_preys)  # ✅ predator从n_preys开始
    def _print_role_summary(self, configs: List[Dict], role: str, index_offset: int):
        """打印单个角色的配置摘要"""
        if not configs:
            print(f"  No {role}s configured")
            return
        
        # 按名称分组
        name_groups = {}
        for i, cfg in enumerate(configs):
            name = cfg['name']
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(i + index_offset)
        
        # 打印每组
        for name, indices in name_groups.items():
            parsed = AgentNameParser.parse(name)
            
            print(f"\n  📋 {name}")
            print(f"     Count: {len(indices)}")
            print(f"     Global Indices: [{indices[0]}..{indices[-1]}]")
            print(f"     Algorithm: {parsed['algorithm'].upper()}")
            if parsed['version']:
                print(f"     Version: {parsed['version']}")
            
            mode_str = '🔥 TRAINING' if parsed['mode'] == 'train' else '▶️  EXECUTION'
            print(f"     Mode: {mode_str}")
            
            # 如果是执行模式且需要模型
            if parsed['mode'] == 'exe' and parsed['algorithm'] != 'random':
                model_path = AgentNameParser.get_model_path(name, self.model_base_dir)
                exists = os.path.exists(model_path) if model_path else False
                status = "✓" if exists else "✗ (not found)"
                print(f"     Model: {model_path} {status}")


# ============================================================================
# Custom VecMonitor for Training Agents Only
# ============================================================================

class TrainingAgentVecMonitor(VecEnvWrapper):
    """
    自定义VecMonitor：只记录训练agents的统计信息
    
    关键：过滤掉执行agents（如random agents）的reward
    """
    
    def __init__(self, venv, training_indices: List[int], filename=None):
        """
        Args:
            venv: 向量化环境
            training_indices: 训练agents的环境索引列表
            filename: 可选的日志文件路径
        """
        VecEnvWrapper.__init__(self, venv)
        
        self.training_indices = set(training_indices)
        self.n_training_agents = len(training_indices)
        
        # Episode统计（关键：需要用于SB3的统计系统）
        self.episode_returns = None
        self.episode_lengths = None
        self.episode_count = 0
        
        # 👇 关键：SB3需要这些属性
        self.episode_reward_buffer = deque(maxlen=100)
        self.episode_length_buffer = deque(maxlen=100)
        
        self.filename = filename
        
        print(f"\n🎯 TrainingAgentVecMonitor Initialized")
        print(f"   Total environments: {self.num_envs}")
        print(f"   Training agents: {self.n_training_agents}")
        print(f"   Training indices: {sorted(list(self.training_indices))[:10]}...")
        print(f"   Execution agents (ignored): {self.num_envs - self.n_training_agents}")
    
    def reset(self):
        obs = self.venv.reset()
        self.episode_returns = np.zeros(self.num_envs)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return obs
    
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        
        # 确保infos是列表
        if not isinstance(infos, (list, tuple)):
            infos = [infos] * self.num_envs
        else:
            infos = list(infos)
        
        # 确保每个info是字典
        for i in range(len(infos)):
            if not isinstance(infos[i], dict):
                infos[i] = {}
        
        # 累积reward和length
        self.episode_returns += rewards
        self.episode_lengths += 1
        
        # 处理完成的episode
        for i in range(len(dones)):
            if dones[i]:
                # 只为训练agents记录episode信息
                if i in self.training_indices:
                    ep_return = float(self.episode_returns[i])
                    ep_length = int(self.episode_lengths[i])
                    
                    # 添加到buffer（SB3会使用这些）
                    self.episode_reward_buffer.append(ep_return)
                    self.episode_length_buffer.append(ep_length)
                    self.episode_count += 1
                    
                    # 👇 关键：添加到infos，SB3的logger会读取这个
                    ep_info = {
                        'r': ep_return,
                        'l': ep_length,
                        't': self.episode_count * ep_length
                    }
                    
                    # 确保'episode'键存在
                    infos[i]['episode'] = ep_info
                    
                    # 调试输出
                    if self.episode_count <= 10:
                        print(f"   Episode {self.episode_count} (training agent {i}):")
                        print(f"     Reward: {ep_return:.2f}, Length: {ep_length}")
                
                # 重置所有agents的计数器
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        
        return obs, rewards, dones, infos
    
    # 👇 关键：添加这些方法供SB3使用
    def get_episode_rewards(self):
        """返回episode reward buffer（SB3会调用）"""
        return list(self.episode_reward_buffer)
    
    def get_episode_lengths(self):
        """返回episode length buffer（SB3会调用）"""
        return list(self.episode_length_buffer)
    
    def get_episode_times(self):
        """返回episode times（SB3可能需要）"""
        return []
# ============================================================================
# TensorBoard Helper Functions
# ============================================================================

def create_tensorboard_log_dir(algo_name: str, role: str, base_dir: str = "./tensorboard_logs") -> str:
    """创建TensorBoard日志目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, f"{algo_name.lower()}_{role}", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("TensorBoard Configuration")
    print(f"{'='*60}")
    print(f"Log Directory: {log_dir}")
    print(f"\nTo view TensorBoard, run:")
    print(f"  tensorboard --logdir={base_dir}")
    print(f"\nThen open: http://localhost:6006")
    print(f"{'='*60}\n")
    
    return log_dir


# ============================================================================
# Training Components
# ============================================================================

class TrainingMonitorCallback(BaseCallback):
    """
    Monitor training process with custom TensorBoard logging
    
    ✅ 现在只记录训练agents的数据（通过TrainingAgentVecMonitor过滤）
    """
    
    def __init__(self, config_manager: 'DualListConfigManager', check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.config_manager = config_manager
        
        # Episode统计
        self.episode_rewards = []
        self.episode_lengths = []
        
        # 训练配置
        self.n_training_agents = len(config_manager.get_training_indices())
        role = config_manager.training_role
        algo = config_manager.training_algorithm
        version = config_manager.training_version or 'v1'
        self.tag_prefix = f"{role}/{algo}_{version}"
        
        print(f"\n📊 TrainingMonitorCallback Initialized")
        print(f"   Tag Prefix: {self.tag_prefix}")
        print(f"   Training Agents: {self.n_training_agents}")
        print(f"   ✅ Only training agents' rewards will be recorded")
        
    def _on_step(self):
        """每个step调用"""
        # 从infos中提取episode信息
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info:
                # Episode结束（已被TrainingAgentVecMonitor过滤，只包含训练agents）
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                
                # 每10个episode打印进度
                if len(self.episode_rewards) % 10 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    print(f"\n📈 Training Episode {len(self.episode_rewards)}:")
                    print(f"   Avg Reward (training agents only): {np.mean(recent_rewards):.2f}")
                    print(f"   Avg Length: {np.mean(self.episode_lengths[-10:]):.0f}")
                    print(f"   Max Reward: {np.max(recent_rewards):.2f}")
                    print(f"   Min Reward: {np.min(recent_rewards):.2f}")
        
        return True
    
    def _on_rollout_end(self):
        """在每个rollout结束时记录到TensorBoard"""
        if hasattr(self.logger, 'name_to_value'):
            # 获取平均reward（现在只包含训练agents）
            ep_rew_mean = self.logger.name_to_value.get('rollout/ep_rew_mean', None)
            
            if ep_rew_mean is not None:
                # 记录训练agents的平均reward
                self.logger.record(
                    f"{self.tag_prefix}/training_agents_reward",
                    ep_rew_mean
                )
                
                # 记录训练agent数量
                self.logger.record(
                    f"{self.tag_prefix}/n_training_agents",
                    self.n_training_agents
                )
                
                # 记录最近的统计
                if len(self.episode_rewards) >= 10:
                    self.logger.record(
                        f"{self.tag_prefix}/reward_mean_10ep",
                        np.mean(self.episode_rewards[-10:])
                    )
                
                if len(self.episode_rewards) > 0:
                    self.logger.record(
                        f"{self.tag_prefix}/reward_std",
                        np.std(self.episode_rewards)
                    )
                    self.logger.record(
                        f"{self.tag_prefix}/reward_max",
                        np.max(self.episode_rewards)
                    )
                    self.logger.record(
                        f"{self.tag_prefix}/reward_min",
                        np.min(self.episode_rewards)
                    )
        
        return True


def plot_training_curve(episode_rewards, algo_name, role, save_path=None):
    """Plot episode rewards"""
    if save_path is None:
        save_path = f'training_curve_{algo_name.lower()}_{role}.png'
    
    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards)
    
    try:
        config = get_algorithm_config(algo_name)
        color = config.get_color()
    except:
        color = 'blue'
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, alpha=0.3, color=color, label='Raw Rewards')
    
    if len(rewards) >= 10:
        window_size = min(10, len(rewards))
        smoothed = uniform_filter1d(rewards, size=window_size, mode='nearest')
        plt.plot(episodes, smoothed, color='red', linewidth=2, 
                label=f'Moving Average (window={window_size})')
    
    if len(rewards) >= 50:
        window_size = min(50, len(rewards))
        trend = uniform_filter1d(rewards, size=window_size, mode='nearest')
        plt.plot(episodes, trend, color='green', linewidth=2, 
                linestyle='--', label=f'Trend (window={window_size})')
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title(f'{algo_name} ({role.capitalize()}) Training Progress', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Statistics
    stats_text = f'Algorithm: {algo_name}\n'
    stats_text += f'Role: {role.capitalize()}\n'
    stats_text += f'Episodes: {len(rewards)}\n'
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
    print(f"\n📊 Training curve saved to: {save_path}")
    plt.close()


# ============================================================================
# Environment Setup
# ============================================================================

def create_waterworld_env(config_manager: DualListConfigManager):
    """创建 Waterworld 环境"""
    print("\n" + "="*60)
    print("Creating Waterworld Environment")
    print("="*60)
    
    # 为 PettingZoo 生成 agent_algorithms 列表
    # 格式: [prey算法...] + [predator算法...]
    agent_algos = []
    
    # Prey algorithms
    for cfg in config_manager.prey_configs:
        algo = cfg['algorithm'].upper() if cfg['algorithm'] != 'random' else 'Random'
        agent_algos.append(algo)
    
    # Predator algorithms
    for cfg in config_manager.predator_configs:
        algo = cfg['algorithm'].upper() if cfg['algorithm'] != 'random' else 'Random'
        agent_algos.append(algo)
    
    env = waterworld_v4.parallel_env(
        render_mode=None,
        n_predators=config_manager.n_predators,
        n_preys=config_manager.n_preys,
        n_evaders=60,
        n_obstacles=2,
        thrust_penalty=0,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=10,
        agent_algorithms=agent_algos,
        max_cycles=1000,
        static_food=True,
        static_poison=True,
    )
    
    print(f"Environment Created:")
    print(f"  Predators: {config_manager.n_predators}")
    print(f"  Preys: {config_manager.n_preys}")
    print(f"  Total Agents: {config_manager.n_total}")
    print(f"  Agent Algorithms: {agent_algos[:5]}..." if len(agent_algos) > 5 else f"  Agent Algorithms: {agent_algos}")
    
    return env


def prepare_env_for_training(env, config_manager: DualListConfigManager):
    """准备训练环境（修改版：只监控训练agents）"""
    print("\nConverting environment format...")
    
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    
    # 👇 使用自定义的TrainingAgentVecMonitor
    # 只记录训练agents的统计信息
    training_indices = config_manager.get_training_indices()
    env = TrainingAgentVecMonitor(env, training_indices=training_indices)
    
    print("  Environment conversion complete")
    print(f"  ✅ Monitoring ONLY {len(training_indices)} training agents")
    
    return env


# ============================================================================
# Main Training Function
# ============================================================================

def main(
    # 👇 核心配置：两个列表
    predator_configs=[
        (2, 'random_predator_exe'),
    ],
    prey_configs=[
        (30, 'ppo_prey_v1_train'),
        (20, 'random_prey_exe'),
    ],
    
    # 训练参数
    total_timesteps=10000000,
    use_tensorboard=True,
    model_base_dir='models',
    
    # 👇 执行模式参数
    execution_mode=False,  # 如果为True，只运行环境不训练
    n_episodes=100,        # 执行模式下运行的回合数
):
    """
    双列表多智能体训练主函数
    
    Args:
        predator_configs: [(count, name), ...] Predator配置
        prey_configs: [(count, name), ...] Prey配置
        total_timesteps: 训练总步数（环境交互次数）
        use_tensorboard: 是否启用TensorBoard
        model_base_dir: 模型保存目录
        execution_mode: 是否为纯执行模式（不训练，只运行）
        n_episodes: 执行模式下运行的回合数
    """
    print("="*70)
    print("Waterworld Dual-List Multi-Agent System")
    print("="*70)
    
    # 1. 创建配置管理器
    config_manager = DualListConfigManager(
        predator_configs=predator_configs,
        prey_configs=prey_configs,
        model_base_dir=model_base_dir
    )
    config_manager.print_summary()
    
    # 2. 检查是否有训练agent
    has_training = config_manager.training_role is not None
    
    if not has_training and not execution_mode:
        print("\n" + "="*70)
        print("⚠️  No Training Agents Detected")
        print("="*70)
        print("You have two options:")
        print("  1. Add training agents to your configs")
        print("  2. Set execution_mode=True to run in execution-only mode")
        print("\nExample for execution-only mode:")
        print("  main(")
        print("      predator_configs=[(2, 'random_predator_exe')],")
        print("      prey_configs=[(50, 'random_prey_exe')],")
        print("      execution_mode=True,")
        print("      n_episodes=100")
        print("  )")
        return
    
    # 3. 创建环境
    raw_env = create_waterworld_env(config_manager)
    
    # ============================================================
    # 分支1：执行模式（无训练）
    # ============================================================
    if not has_training or execution_mode:
        print("\n" + "="*70)
        print("🎮 EXECUTION-ONLY MODE")
        print("="*70)
        print(f"Running {n_episodes} episodes...")
        
        # TensorBoard设置
        tensorboard_log = None
        writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_dir = os.path.join("./tensorboard_logs", "execution_mode", timestamp)
                os.makedirs(log_dir, exist_ok=True)
                
                writer = SummaryWriter(log_dir=log_dir)
                
                print(f"\n{'='*60}")
                print("TensorBoard Configuration (Execution Mode)")
                print(f"{'='*60}")
                print(f"Log Directory: {log_dir}")
                print(f"\nTo view TensorBoard, run:")
                print(f"  tensorboard --logdir=./tensorboard_logs")
                print(f"\nThen open: http://localhost:6006")
                print(f"{'='*60}\n")
            except ImportError:
                print("\n⚠️  TensorBoard not available. Install with: pip install tensorboard")
                use_tensorboard = False
        
        # 👇 分别记录predator和prey的数据
        episode_rewards_all = []
        episode_rewards_predator = []
        episode_rewards_prey = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            observations, infos = raw_env.reset()
            ep_reward_all = 0
            ep_reward_predator = 0
            ep_reward_prey = 0
            ep_length = 0
            episode_done = False
            
            # 获取当前环境中的agent列表
            current_agents = list(raw_env.agents)
            
            # 识别predator和prey agents（基于环境命名规则）
            predator_agents = [a for a in current_agents if 'predator' in a]
            prey_agents = [a for a in current_agents if 'prey' in a]
            
            while not episode_done:
                # 为每个agent生成随机动作
                actions = {}
                for agent in raw_env.agents:
                    actions[agent] = raw_env.action_space(agent).sample()
                
                # 环境step
                observations, rewards, terminations, truncations, infos = raw_env.step(actions)
                
                # 分别累计不同角色的奖励
                if rewards:
                    # 所有agents的平均
                    ep_reward_all += np.mean(list(rewards.values()))
                    
                    # Predator的平均
                    predator_rewards = [r for agent, r in rewards.items() if agent in predator_agents]
                    if predator_rewards:
                        ep_reward_predator += np.mean(predator_rewards)
                    
                    # Prey的平均
                    prey_rewards = [r for agent, r in rewards.items() if agent in prey_agents]
                    if prey_rewards:
                        ep_reward_prey += np.mean(prey_rewards)
                
                ep_length += 1
                
                # 检查是否结束
                episode_done = len(raw_env.agents) == 0 or all(terminations.values()) or all(truncations.values())
            
            # 记录数据
            episode_rewards_all.append(ep_reward_all)
            episode_rewards_predator.append(ep_reward_predator)
            episode_rewards_prey.append(ep_reward_prey)
            episode_lengths.append(ep_length)
            
            # 记录到TensorBoard
            if use_tensorboard and writer is not None:
                # 总体reward
                writer.add_scalar('execution/all/episode_reward', ep_reward_all, ep)
                
                # Predator reward
                writer.add_scalar('execution/predator/episode_reward', ep_reward_predator, ep)
                
                # Prey reward
                writer.add_scalar('execution/prey/episode_reward', ep_reward_prey, ep)
                
                # Episode length
                writer.add_scalar('execution/episode_length', ep_length, ep)
                
                # 移动平均
                if len(episode_rewards_all) >= 10:
                    writer.add_scalar('execution/all/reward_mean_10ep', 
                                    np.mean(episode_rewards_all[-10:]), ep)
                    writer.add_scalar('execution/predator/reward_mean_10ep', 
                                    np.mean(episode_rewards_predator[-10:]), ep)
                    writer.add_scalar('execution/prey/reward_mean_10ep', 
                                    np.mean(episode_rewards_prey[-10:]), ep)
                
                # 累计平均
                writer.add_scalar('execution/all/reward_mean_all', 
                                np.mean(episode_rewards_all), ep)
                writer.add_scalar('execution/predator/reward_mean_all', 
                                np.mean(episode_rewards_predator), ep)
                writer.add_scalar('execution/prey/reward_mean_all', 
                                np.mean(episode_rewards_prey), ep)
            
            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{n_episodes}:")
                print(f"  All Agents:  {np.mean(episode_rewards_all[-10:]):.2f}")
                print(f"  Predator:    {np.mean(episode_rewards_predator[-10:]):.2f}")
                print(f"  Prey:        {np.mean(episode_rewards_prey[-10:]):.2f}")
                print(f"  Length:      {np.mean(episode_lengths[-10:]):.0f}")
        
        # 最终统计
        if use_tensorboard and writer is not None:
            # 汇总统计
            writer.add_scalar('execution/final/all_mean_reward', np.mean(episode_rewards_all), n_episodes)
            writer.add_scalar('execution/final/predator_mean_reward', np.mean(episode_rewards_predator), n_episodes)
            writer.add_scalar('execution/final/prey_mean_reward', np.mean(episode_rewards_prey), n_episodes)
            writer.add_scalar('execution/final/mean_length', np.mean(episode_lengths), n_episodes)
            
            # 文本摘要
            summary_text = f"""
            ## Execution Mode Summary
            
            **Configuration:**
            - Episodes: {n_episodes}
            - Predators: {config_manager.n_predators}
            - Preys: {config_manager.n_preys}
            
            **Results:**
            - All Agents: {np.mean(episode_rewards_all):.2f} ± {np.std(episode_rewards_all):.2f}
            - Predator: {np.mean(episode_rewards_predator):.2f} ± {np.std(episode_rewards_predator):.2f}
            - Prey: {np.mean(episode_rewards_prey):.2f} ± {np.std(episode_rewards_prey):.2f}
            - Mean Length: {np.mean(episode_lengths):.0f}
            """
            writer.add_text('execution/summary', summary_text, 0)
            
            writer.close()
            print(f"\n📊 TensorBoard logs saved to: {log_dir}")
        
        # 打印统计
        print("\n" + "="*70)
        print("Execution Statistics")
        print("="*70)
        print(f"Episodes: {n_episodes}")
        print(f"\nAll Agents:")
        print(f"  Mean Reward: {np.mean(episode_rewards_all):.2f} ± {np.std(episode_rewards_all):.2f}")
        print(f"\nPredator:")
        print(f"  Mean Reward: {np.mean(episode_rewards_predator):.2f} ± {np.std(episode_rewards_predator):.2f}")
        print(f"\nPrey:")
        print(f"  Mean Reward: {np.mean(episode_rewards_prey):.2f} ± {np.std(episode_rewards_prey):.2f}")
        print(f"\nMean Length: {np.mean(episode_lengths):.0f}")
        
        # 绘制曲线
        if episode_rewards_all:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_training_curve(episode_rewards_all, 'Random', 'all_agents', 
                              save_path=f'execution_all_{timestamp_str}.png')
            plot_training_curve(episode_rewards_predator, 'Random', 'predator', 
                              save_path=f'execution_predator_{timestamp_str}.png')
            plot_training_curve(episode_rewards_prey, 'Random', 'prey', 
                              save_path=f'execution_prey_{timestamp_str}.png')
        
        raw_env.close()
        print("\n✅ Execution Complete!")
        return
    
    # ============================================================
    # 分支2：训练模式 - 需要向量化环境
    # ============================================================
    env = prepare_env_for_training(raw_env, config_manager)  # 👈 添加config_manager参数
    
    # 4. 获取训练算法配置
    algo_config = get_algorithm_config(config_manager.training_algorithm)
    
    # 5. 设置TensorBoard
    tensorboard_log = None
    if use_tensorboard:
        tensorboard_log = create_tensorboard_log_dir(
            algo_config.name, 
            config_manager.training_role
        )
    
    # 6. 创建模型
    print("\n" + "="*60)
    print(f"Creating {algo_config.name} Model")
    print("="*60)
    
    ModelClass = algo_config.get_model_class()
    hyperparams = algo_config.get_hyperparameters()
    
    model = ModelClass(
        "MlpPolicy",
        env,
        **hyperparams,
        tensorboard_log=tensorboard_log,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Model: {algo_config.name}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Training Role: {config_manager.training_role.upper()}")
    print(f"Training Agents: {len(config_manager.training_indices)}")
    
    # 7. 训练
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    n_training_agents = len(config_manager.training_indices)
    
    print(f"\n⚠️  Training Step Explanation:")
    print(f"  Number of training agents: {n_training_agents}")
    print(f"  Total timesteps setting: {total_timesteps:,}")
    print(f"  → Each agent will collect ~{total_timesteps:,} samples")
    print(f"  → Total samples collected: ~{total_timesteps * n_training_agents:,}")
    print(f"  → Environment steps: ~{total_timesteps:,}")
    print(f"\n  Note: With parameter sharing, all {n_training_agents} training agents")
    print(f"        share the same policy and learn from each other's experiences.")
    
    callback = TrainingMonitorCallback(config_manager=config_manager, check_freq=1000)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name=f"{algo_config.name}_{config_manager.training_role}"
    )
    
    print("\n✅ Training Complete!")
    
    # 8. 保存模型
    os.makedirs(model_base_dir, exist_ok=True)
    
    version = config_manager.training_version or 'v1'
    model_filename = f"{config_manager.training_algorithm}_{config_manager.training_role}_{version}"
    model_path = os.path.join(model_base_dir, model_filename)
    
    model.save(model_path)
    print(f"\n💾 Model saved: {model_path}.zip")
    
    # 9. 绘制训练曲线
    if callback.episode_rewards:
        plot_training_curve(
            callback.episode_rewards, 
            algo_config.name,
            config_manager.training_role
        )
    
    # 10. 统计信息
    if callback.episode_rewards:
        rewards = np.array(callback.episode_rewards)
        print("\n" + "="*70)
        print("Training Statistics (Training Agents Only)")  # 👈 修改标题
        print("="*70)
        print(f"✅ Data Source: ONLY {n_training_agents} training agents")
        print(f"   (Execution agents' rewards are NOT included)")
        print(f"\nTotal Episodes: {len(rewards)}")
        print(f"Mean Reward: {np.mean(rewards):.2f}")
        print(f"Std Reward: {np.std(rewards):.2f}")
        print(f"Max Reward: {np.max(rewards):.2f}")
        print(f"Min Reward: {np.min(rewards):.2f}")
        
        n = len(rewards)
        if n >= 10:
            early = rewards[:max(1, n//10)]
            late = rewards[-max(1, n//10):]
            improvement = np.mean(late) - np.mean(early)
            
            print(f"\nLearning Progress:")
            print(f"  Early Mean: {np.mean(early):.2f}")
            print(f"  Late Mean: {np.mean(late):.2f}")
            print(f"  Improvement: {improvement:+.2f}")
            
            if improvement > 5:
                print("  ✓ Effective Learning")
            elif improvement > -5:
                print("  ~ Limited Learning")
            else:
                print("  ✗ No Effective Learning")
    
    env.close()
    print("\n" + "="*70)
    print("🎉 Training Pipeline Complete!")
    print("="*70)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    """
    ========================================
    使用方式：修改下面两个列表即可
    ========================================
    """
    
    # ============================================================
    # 场景1：训练 Predator，Prey 随机
    # ============================================================
    predator_configs = [
        (3, 'random_predator_exe'),  # 3个随机捕食者
    ]
    
    prey_configs = [
        (3, 'random_prey_exe'),   # 3个随机猎物
        (3, 'random_prey_exe'),     # 3个随机猎物
    ]

    
    # ============================================================
    # 场景2：训练 Prey，Predator 随机
    # ============================================================
    # predator_configs = [
    #     (2, 'random_predator_exe'),  # 2个随机捕食者
    # ]
    # 
    # prey_configs = [
    #     (30, 'ppo_prey_v1_train'),   # 30个训练PPO
    #     (20, 'random_prey_exe'),     # 20个随机
    # ]
    
    # ============================================================
    # 场景3：全部random执行（无训练，只运行环境）
    # ============================================================
    # predator_configs = [
    #     (2, 'random_predator_exe'),
    # ]
    # 
    # prey_configs = [
    #     (50, 'random_prey_exe'),
    # ]
    # 
    # main(
    #     predator_configs=predator_configs,
    #     prey_configs=prey_configs,
    #     execution_mode=True,
    #     n_episodes=100,
    #     use_tensorboard=True,
    # )
    
    # ============================================================
    # 运行训练
    # ============================================================
    main(
        predator_configs=predator_configs,
        prey_configs=prey_configs,
        total_timesteps=1000000,
        use_tensorboard=True,
        execution_mode=True,
        model_base_dir='models',
    )
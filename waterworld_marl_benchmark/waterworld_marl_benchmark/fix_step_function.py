"""
修复 step 函数中的智能体访问问题
"""

from pathlib import Path

environment_content = '''"""
环境创建与管理
"""

from typing import Dict, Any, Optional, List
from pettingzoo.sisl import waterworld_v4
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class WaterworldEnvManager:
    """Waterworld环境管理器"""
    
    def __init__(self, env_config: Dict[str, Any]):
        """
        初始化环境管理器
        
        Args:
            env_config: 环境配置字典
        """
        self.config = env_config['environment']
        self.env = None
    
    def create_env(self, render_mode: Optional[str] = None):
        """
        创建Waterworld环境
        
        Args:
            render_mode: 渲染模式（None/"rgb_array"/"human"）
        
        Returns:
            PettingZoo环境实例
        """
        if render_mode is None:
            render_mode = self.config.get('render_mode', None)
        
        self.env = waterworld_v4.parallel_env(
            n_predators=self.config.get('n_predators', 5),
            n_preys=self.config.get('n_preys', 10),
            n_evaders=self.config.get('n_evaders', 90),
            n_poisons=self.config.get('n_poisons', 10),
            n_obstacles=self.config.get('n_obstacles', 2),
            obstacle_coord=self.config.get('obstacle_coord', [[0.2, 0.2], [0.8, 0.2]]),
            
            predator_speed=self.config.get('predator_speed', 0.06),
            prey_speed=self.config.get('prey_speed', 0.001),
            
            sensor_range=self.config.get('sensor_range', 0.8),
            thrust_penalty=self.config.get('thrust_penalty', 0.0),
            
            max_cycles=self.config.get('max_cycles', 3000),
            
            render_mode=render_mode
        )
        
        return self.env
    
    def get_observation_space(self, agent_type: str) -> gym.Space:
        """
        获取观察空间
        
        Args:
            agent_type: 智能体类型（predator/prey）
        
        Returns:
            观察空间
        """
        if self.env is None:
            self.create_env()
        
        for agent in self.env.possible_agents:
            if agent_type in agent:
                return self.env.observation_space(agent)
        
        raise ValueError(f"未找到类型为 {agent_type} 的智能体")
    
    def get_action_space(self, agent_type: str) -> gym.Space:
        """
        获取动作空间
        
        Args:
            agent_type: 智能体类型（predator/prey）
        
        Returns:
            动作空间
        """
        if self.env is None:
            self.create_env()
        
        for agent in self.env.possible_agents:
            if agent_type in agent:
                return self.env.action_space(agent)
        
        raise ValueError(f"未找到类型为 {agent_type} 的智能体")
    
    def get_agents_by_type(self, agent_type: str) -> List[str]:
        """
        获取指定类型的所有智能体ID
        
        Args:
            agent_type: 智能体类型（predator/prey）
        
        Returns:
            智能体ID列表
        """
        if self.env is None:
            self.create_env()
        
        return [agent for agent in self.env.possible_agents if agent_type in agent]
    
    def reset(self, seed: Optional[int] = None):
        """重置环境"""
        if self.env is None:
            self.create_env()
        
        return self.env.reset(seed=seed)
    
    def close(self):
        """关闭环境"""
        if self.env is not None:
            self.env.close()
            self.env = None


class SingleAgentWrapper(gym.Env):
    """
    将PettingZoo多智能体环境包装为Gym单智能体环境
    用于训练单个智能体（其他智能体使用固定策略）
    """
    
    def __init__(
        self,
        env,
        train_agent_id: str,
        opponent_policies: Dict[str, Any]
    ):
        """
        初始化包装器
        
        Args:
            env: PettingZoo环境
            train_agent_id: 要训练的智能体ID
            opponent_policies: 对手策略字典 {agent_id: policy}
        """
        super().__init__()
        
        self.env = env
        self.train_agent_id = train_agent_id
        self.opponent_policies = opponent_policies
        
        # 设置观察和动作空间
        self.observation_space = env.observation_space(train_agent_id)
        self.action_space = env.action_space(train_agent_id)
        
        self.agents = env.possible_agents
        self._last_observations = {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        observations, infos = self.env.reset(seed=seed)
        
        # 保存观察
        self._last_observations = observations
        
        # 返回训练智能体的观察
        obs = observations.get(self.train_agent_id, np.zeros(self.observation_space.shape))
        info = infos.get(self.train_agent_id, {})
        
        return obs, info
    
    def step(self, action):
        """执行一步"""
        # 收集所有智能体的动作
        actions = {}
        
        # 只为当前活跃的智能体收集动作
        for agent_id in self.env.agents:
            if agent_id == self.train_agent_id:
                # 训练智能体使用传入的动作
                actions[agent_id] = action
            elif agent_id in self.opponent_policies:
                # 对手使用固定策略
                policy = self.opponent_policies[agent_id]
                
                # 获取该智能体的观察
                if agent_id in self._last_observations:
                    obs = self._last_observations[agent_id]
                else:
                    # 如果没有观察，使用零向量
                    obs = np.zeros(self.env.observation_space(agent_id).shape)
                
                try:
                    opponent_action, _ = policy.predict(obs, deterministic=False)
                    actions[agent_id] = opponent_action
                except Exception as e:
                    # 如果预测失败，使用随机动作
                    actions[agent_id] = self.env.action_space(agent_id).sample()
        
        # 执行环境步进
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # 保存新的观察
        self._last_observations = observations
        
        # 返回训练智能体的信息
        obs = observations.get(self.train_agent_id, np.zeros(self.observation_space.shape))
        reward = rewards.get(self.train_agent_id, 0.0)
        terminated = terminations.get(self.train_agent_id, False)
        truncated = truncations.get(self.train_agent_id, False)
        info = infos.get(self.train_agent_id, {})
        
        # 如果所有智能体都结束了，标记为 terminated
        if len(self.env.agents) == 0:
            terminated = True
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """渲染环境"""
        return self.env.render()
    
    def close(self):
        """关闭环境"""
        self.env.close()


def create_training_env(
    env_config: Dict[str, Any],
    train_side: str,
    opponent_policies: Dict[str, Any],
    n_envs: int = 1
):
    """
    创建训练环境
    
    Args:
        env_config: 环境配置
        train_side: 训练方（predator/prey）
        opponent_policies: 对手策略字典
        n_envs: 并行环境数量
    
    Returns:
        训练环境（单环境或DummyVecEnv）
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        """创建单个环境的工厂函数"""
        # 创建环境管理器
        env_manager = WaterworldEnvManager(env_config)
        pz_env = env_manager.create_env()
        
        # 获取训练方的第一个智能体ID
        train_agents = [agent for agent in pz_env.possible_agents if train_side in agent]
        if not train_agents:
            raise ValueError(f"未找到 {train_side} 类型的智能体")
        
        train_agent_id = train_agents[0]  # 使用第一个智能体
        
        # 包装为单智能体环境
        wrapped_env = SingleAgentWrapper(
            env=pz_env,
            train_agent_id=train_agent_id,
            opponent_policies=opponent_policies
        )
        
        return wrapped_env
    
    # 创建向量化环境
    if n_envs == 1:
        return make_env()
    else:
        env_fns = [make_env for _ in range(n_envs)]
        return DummyVecEnv(env_fns)
'''

# 写入文件
file_path = Path('src/core/environment.py')
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(environment_content)

print("✅ 已修复 src/core/environment.py - 修复了 step 函数中的智能体访问问题")
print("\n现在请重新运行测试：")
print("python test_training_system.py")
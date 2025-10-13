"""
使用 SuperSuit 正确包装环境（参考成功代码）
"""

from pathlib import Path

environment_content = '''"""
环境创建与管理 - 使用 SuperSuit 方案
"""

from typing import Dict, Any, Optional, List
from pettingzoo.sisl import waterworld_v4
import gymnasium as gym
import numpy as np
import supersuit as ss
from stable_baselines3.common.vec_env import VecEnv


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


class MixedAgentVecEnv(VecEnv):
    """
    混合智能体向量化环境
    基于参考代码的成功实现
    """
    
    def __init__(self, venv, train_agent_indices: List[int], opponent_policies: Dict[str, Any]):
        """
        初始化
        
        Args:
            venv: SuperSuit转换后的向量化环境
            train_agent_indices: 训练智能体的索引列表
            opponent_policies: 对手策略字典 {agent_name: policy}
        """
        self.venv = venv
        self.train_indices = train_agent_indices
        self.opponent_policies = opponent_policies
        
        self.n_training = len(train_agent_indices)
        self.n_total = venv.num_envs
        
        # 获取训练智能体的空间
        super().__init__(
            num_envs=self.n_training,
            observation_space=venv.observation_space,
            action_space=venv.action_space
        )
        
        self.latest_obs = None
        
        print(f"  MixedAgentVecEnv: training {self.n_training}/{self.n_total} agents")
    
    def reset(self):
        """重置环境"""
        obs = self.venv.reset()
        self.latest_obs = obs
        
        # 重置对手策略
        for policy in self.opponent_policies.values():
            if hasattr(policy, 'reset'):
                policy.reset()
        
        # 返回训练智能体的观察
        return obs[self.train_indices]
    
    def step_async(self, actions):
        """异步步进"""
        # 构建完整动作数组
        full_actions = np.zeros((self.n_total, self.action_space.shape[0]), dtype=np.float32)
        
        # 填充训练智能体的动作
        for i, train_idx in enumerate(self.train_indices):
            full_actions[train_idx] = actions[i]
        
        # 填充对手智能体的动作
        agent_names = list(self.opponent_policies.keys())
        for i, agent_name in enumerate(agent_names):
            if i not in self.train_indices:
                policy = self.opponent_policies[agent_name]
                obs = self.latest_obs[i] if self.latest_obs is not None else None
                if obs is not None:
                    full_actions[i] = policy.predict(obs, deterministic=False)[0]
                else:
                    full_actions[i] = self.action_space.sample()
        
        self.venv.step_async(full_actions)
    
    def step_wait(self):
        """等待步进结果"""
        obs, rewards, dones, infos = self.venv.step_wait()
        self.latest_obs = obs
        
        # 提取训练智能体的数据
        train_obs = obs[self.train_indices]
        train_rewards = rewards[self.train_indices]
        train_dones = dones[self.train_indices]
        train_infos = [infos[i] for i in self.train_indices]
        
        return train_obs, train_rewards, train_dones, train_infos
    
    def close(self):
        """关闭环境"""
        return self.venv.close()
    
    def get_attr(self, attr_name, indices=None):
        return self.venv.get_attr(attr_name, indices)
    
    def set_attr(self, attr_name, value, indices=None):
        return self.venv.set_attr(attr_name, value, indices)
    
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)


# 简化占位符
class SingleAgentWrapper:
    pass


def create_training_env(
    env_config: Dict[str, Any],
    train_side: str,
    opponent_policies: Dict[str, Any],
    n_envs: int = 1
):
    """
    创建训练环境 - 使用 SuperSuit 方案
    
    Args:
        env_config: 环境配置
        train_side: 训练方（predator/prey）
        opponent_policies: 对手策略字典
        n_envs: 并行环境数量（目前仅支持1）
    
    Returns:
        训练环境
    """
    # 1. 创建基础环境
    env_manager = WaterworldEnvManager(env_config)
    pz_env = env_manager.create_env()
    
    # 2. 使用 SuperSuit 转换（参考成功代码）
    # black_death: 死亡的智能体返回零观察和零奖励
    env = ss.black_death_v3(pz_env)
    
    # 转换为向量化环境
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # 拼接环境（这里 n_envs 总是1）
    env = ss.concat_vec_envs_v1(
        env, 
        num_vec_envs=1, 
        num_cpus=1, 
        base_class='stable_baselines3'
    )
    
    # 3. 确定训练智能体
    all_agents = pz_env.possible_agents
    train_agents = [agent for agent in all_agents if train_side in agent]
    
    if not train_agents:
        raise ValueError(f"没有找到 {train_side} 类型的智能体")
    
    # 使用第一个作为训练智能体
    train_agent_name = train_agents[0]
    train_agent_idx = all_agents.index(train_agent_name)
    
    print(f"  训练智能体: {train_agent_name} (index={train_agent_idx})")
    
    # 4. 应用混合智能体包装器
    env = MixedAgentVecEnv(
        venv=env,
        train_agent_indices=[train_agent_idx],
        opponent_policies=opponent_policies
    )
    
    return env
'''

# 写入文件
file_path = Path('src/core/environment.py')
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(environment_content)

print("✅ 已修复 src/core/environment.py")
print("   - 使用 SuperSuit 进行环境转换（参考成功代码）")
print("   - 应用 black_death_v3 处理死亡智能体")
print("   - 使用标准的 pettingzoo_env_to_vec_env_v1 转换")
print("\n⚠️  请确保已安装 SuperSuit:")
print("   pip install supersuit")
print("\n现在请重新运行测试：")
print("python test_training_system.py")
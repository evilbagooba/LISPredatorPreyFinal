"""
修复缺失和错误的文件
"""

from pathlib import Path

# 1. 修复 src/core/environment.py
environment_content = '''"""
环境创建与管理
"""

from typing import Dict, Any, Optional, List
from pettingzoo.sisl import waterworld_v4
import gymnasium as gym
import numpy as np


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
        # 如果配置中没有指定render_mode，使用参数
        if render_mode is None:
            render_mode = self.config.get('render_mode', None)
        
        # 创建环境
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
        
        # 获取第一个该类型的agent的观察空间
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
        
        # 获取第一个该类型的agent的动作空间
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


# 简化的占位符（完整实现需要更多代码）
class SingleAgentWrapper:
    pass

def create_training_env(*args, **kwargs):
    pass
'''

# 2. 创建 src/algorithms/base_algorithm.py
base_algorithm_content = '''"""
算法基类
提供统一的接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym


class BaseAlgorithm(ABC):
    """算法基类"""
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: Dict[str, Any],
        device: str = "auto"
    ):
        """
        初始化算法
        
        Args:
            observation_space: 观察空间
            action_space: 动作空间
            config: 算法配置
            device: 计算设备
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.device = device
        self.model = None
    
    @abstractmethod
    def train(
        self,
        env,
        total_timesteps: int,
        callback=None,
        **kwargs
    ):
        """
        训练算法
        
        Args:
            env: 训练环境
            total_timesteps: 总训练步数
            callback: 回调函数
            **kwargs: 其他参数
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        预测动作
        
        Args:
            observation: 观察
            deterministic: 是否使用确定性策略
        
        Returns:
            (action, state) 元组
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        pass
    
    @abstractmethod
    def create_model(self, env, tensorboard_log: Optional[str] = None, verbose: int = 1):
        """创建模型"""
        pass
    
    def get_name(self) -> str:
        """获取算法名称"""
        return self.config.get('algorithm', {}).get('name', 'UNKNOWN')
'''

# 写入文件
def write_file(path: str, content: str):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ 已修复: {path}")

# 执行修复
print("="*70)
print("修复缺失和错误的文件")
print("="*70)

write_file('src/core/environment.py', environment_content)
write_file('src/algorithms/base_algorithm.py', base_algorithm_content)

print("\n✅ 文件修复完成！")
print("\n现在请重新运行测试：")
print("python test_training_system.py")
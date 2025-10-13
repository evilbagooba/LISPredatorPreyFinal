"""
智能体管理
加载和管理训练好的智能体
"""

from pathlib import Path
from typing import Dict, Any, Optional
import gymnasium as gym

from src.algorithms import create_algorithm
from src.utils.config_loader import get_algo_config


class AgentManager:
    """智能体管理器"""
    
    @staticmethod
    def load_agent(
        model_path: Path,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "auto"
    ):
        """
        加载训练好的智能体
        
        Args:
            model_path: 模型文件路径
            observation_space: 观察空间
            action_space: 动作空间
            device: 计算设备
        
        Returns:
            加载的算法实例
        """
        # 从文件名解析算法名称
        filename = model_path.stem
        parts = filename.split('_')
        
        # 移除可能的前缀（DEBUG_, DRYRUN_）
        if parts[0] in ['DEBUG', 'DRYRUN']:
            parts = parts[1:]
        
        algo_name = parts[0]
        
        # 加载算法配置
        algo_config = get_algo_config(algo_name)
        
        # 创建算法实例
        algorithm = create_algorithm(
            algo_name=algo_name,
            observation_space=observation_space,
            action_space=action_space,
            config=algo_config,
            device=device
        )
        
        # 加载模型权重
        algorithm.load(str(model_path))
        
        return algorithm
    
    @staticmethod
    def create_random_agent(
        observation_space: gym.Space,
        action_space: gym.Space
    ):
        """
        创建随机智能体
        
        Args:
            observation_space: 观察空间
            action_space: 动作空间
        
        Returns:
            随机策略实例
        """
        algo_config = get_algo_config('RANDOM')
        
        algorithm = create_algorithm(
            algo_name='RANDOM',
            observation_space=observation_space,
            action_space=action_space,
            config=algo_config,
            device='cpu'
        )
        
        algorithm.create_model(None)
        
        return algorithm
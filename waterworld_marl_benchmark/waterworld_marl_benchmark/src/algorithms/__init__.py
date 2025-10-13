"""
算法模块初始化
提供算法工厂函数
"""

from typing import Dict, Any
import gymnasium as gym

from .base_algorithm import BaseAlgorithm
from .ppo_wrapper import PPOWrapper
from .a2c_wrapper import A2CWrapper
from .sac_wrapper import SACWrapper
from .td3_wrapper import TD3Wrapper
from .random_policy import RandomPolicy


# 算法映射
ALGORITHM_MAP = {
    'PPO': PPOWrapper,
    'A2C': A2CWrapper,
    'SAC': SACWrapper,
    'TD3': TD3Wrapper,
    'RANDOM': RandomPolicy
}


def create_algorithm(
    algo_name: str,
    observation_space: gym.Space,
    action_space: gym.Space,
    config: Dict[str, Any],
    device: str = "auto"
) -> BaseAlgorithm:
    """
    创建算法实例
    
    Args:
        algo_name: 算法名称（PPO/A2C/SAC/TD3/RANDOM）
        observation_space: 观察空间
        action_space: 动作空间
        config: 算法配置
        device: 计算设备
    
    Returns:
        算法实例
    """
    algo_name = algo_name.upper()
    
    if algo_name not in ALGORITHM_MAP:
        raise ValueError(
            f"未知的算法: {algo_name}. "
            f"支持的算法: {list(ALGORITHM_MAP.keys())}"
        )
    
    algo_class = ALGORITHM_MAP[algo_name]
    return algo_class(observation_space, action_space, config, device)


__all__ = [
    'BaseAlgorithm',
    'PPOWrapper',
    'A2CWrapper',
    'SACWrapper',
    'TD3Wrapper',
    'RandomPolicy',
    'create_algorithm',
    'ALGORITHM_MAP'
]
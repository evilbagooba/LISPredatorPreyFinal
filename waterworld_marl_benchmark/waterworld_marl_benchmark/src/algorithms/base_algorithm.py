"""
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

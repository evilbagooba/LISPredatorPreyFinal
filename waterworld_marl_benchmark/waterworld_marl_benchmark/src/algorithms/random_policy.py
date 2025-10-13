"""
Random策略（Baseline）
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym

from .base_algorithm import BaseAlgorithm


class RandomPolicy(BaseAlgorithm):
    """随机策略（Baseline）"""
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        super().__init__(observation_space, action_space, config, device)
        self.model = self  # Random策略自己就是模型
    
    def train(self, env, total_timesteps: int, callback=None, **kwargs):
        """Random策略不需要训练"""
        print("⚠️  Random策略不需要训练，跳过训练步骤")
        pass
    
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,  # ← 新增参数
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机采样动作
        
        Args:
            observation: 观察（未使用）
            state: RNN 状态（Random策略不使用）
            episode_start: Episode开始标志（Random策略不使用）
            deterministic: 是否确定性（未使用）
        
        Returns:
            (action, state) 元组
        """
        action = self.action_space.sample()
        return action, None
    
    def save(self, path: str):
        """Random策略不需要保存"""
        print(f"ℹ️  Random策略不需要保存模型文件: {path}")
        pass
    
    def load(self, path: str):
        """Random策略不需要加载"""
        print(f"ℹ️  Random策略不需要加载模型文件: {path}")
        return self
    
    def create_model(self, env, tensorboard_log: Optional[str] = None, verbose: int = 1):
        """Random策略不需要创建模型"""
        self.model = self
        return self

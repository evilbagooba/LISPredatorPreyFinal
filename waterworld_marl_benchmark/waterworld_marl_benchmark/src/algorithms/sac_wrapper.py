"""
SAC算法封装
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import torch

from .base_algorithm import BaseAlgorithm


class SACWrapper(BaseAlgorithm):
    """SAC算法包装器"""
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: Dict[str, Any],
        device: str = "auto"
    ):
        super().__init__(observation_space, action_space, config, device)
        
        hyperparams = config.get('hyperparameters', {})
        policy_kwargs = hyperparams.get('policy_kwargs', {}).copy()
        
        if 'activation_fn' in policy_kwargs:
            activation_str = policy_kwargs['activation_fn']
            if activation_str == "torch.nn.ReLU":
                policy_kwargs['activation_fn'] = torch.nn.ReLU
            elif activation_str == "torch.nn.Tanh":
                policy_kwargs['activation_fn'] = torch.nn.Tanh
        
        self.hyperparams = {
            'policy': hyperparams.get('policy', 'MlpPolicy'),
            'learning_rate': hyperparams.get('learning_rate', 3e-4),
            'buffer_size': hyperparams.get('buffer_size', 1000000),
            'learning_starts': hyperparams.get('learning_starts', 10000),
            'batch_size': hyperparams.get('batch_size', 256),
            'tau': hyperparams.get('tau', 0.005),
            'gamma': hyperparams.get('gamma', 0.99),
            'train_freq': hyperparams.get('train_freq', 1),
            'gradient_steps': hyperparams.get('gradient_steps', 1),
            'ent_coef': hyperparams.get('ent_coef', 'auto'),
            'target_entropy': hyperparams.get('target_entropy', 'auto'),
            'use_sde': hyperparams.get('use_sde', False),
            'sde_sample_freq': hyperparams.get('sde_sample_freq', -1),
            'policy_kwargs': policy_kwargs,
            'device': device,
            'seed': config.get('seed', None)
        }
    
    def create_model(self, env, tensorboard_log: Optional[str] = None, verbose: int = 1):
        """创建SAC模型"""
        self.model = SAC(
            env=env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **self.hyperparams
        )
        return self.model
    
    def train(self, env, total_timesteps: int, callback=None, **kwargs):
        """训练SAC"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            **kwargs
        )
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """预测动作"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未初始化")
        
        self.model.save(path)
    
    def load(self, path: str):
        """加载模型"""
        self.model = SAC.load(path, device=self.device)
        return self.model

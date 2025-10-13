"""
PPO算法封装
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import torch

from .base_algorithm import BaseAlgorithm


class PPOWrapper(BaseAlgorithm):
    """PPO算法包装器"""
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: Dict[str, Any],
        device: str = "auto"
    ):
        super().__init__(observation_space, action_space, config, device)
        
        # 提取超参数
        hyperparams = config.get('hyperparameters', {})
        
        # 处理policy_kwargs
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
            'n_steps': hyperparams.get('n_steps', 2048),
            'batch_size': hyperparams.get('batch_size', 64),
            'n_epochs': hyperparams.get('n_epochs', 10),
            'gamma': hyperparams.get('gamma', 0.99),
            'gae_lambda': hyperparams.get('gae_lambda', 0.95),
            'clip_range': hyperparams.get('clip_range', 0.2),
            'clip_range_vf': hyperparams.get('clip_range_vf', None),
            'ent_coef': hyperparams.get('ent_coef', 0.01),
            'vf_coef': hyperparams.get('vf_coef', 0.5),
            'max_grad_norm': hyperparams.get('max_grad_norm', 0.5),
            'normalize_advantage': hyperparams.get('normalize_advantage', True),
            'target_kl': hyperparams.get('target_kl', None),
            'policy_kwargs': policy_kwargs,
            'device': device,
            'seed': config.get('seed', None)
        }
    
    def create_model(self, env, tensorboard_log: Optional[str] = None, verbose: int = 1):
        """创建PPO模型 - 关键修复：直接传入env，让SB3自动获取空间"""
        self.model = PPO(
            env=env,  # SB3会从env自动获取observation_space和action_space
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **self.hyperparams
        )
        return self.model
    
    def train(self, env, total_timesteps: int, callback=None, **kwargs):
        """训练PPO"""
        if self.model is None:
            raise ValueError("模型未初始化，请先调用 create_model()")
        
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
        self.model = PPO.load(path, device=self.device)
        return self.model

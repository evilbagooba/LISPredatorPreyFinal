"""
修复算法包装器 - 正确处理 observation_space 和 action_space
"""

from pathlib import Path

# ============================================================================
# PPO Wrapper
# ============================================================================
ppo_content = '''"""
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
'''

# ============================================================================
# A2C Wrapper
# ============================================================================
a2c_content = '''"""
A2C算法封装
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym
from stable_baselines3 import A2C
import torch

from .base_algorithm import BaseAlgorithm


class A2CWrapper(BaseAlgorithm):
    """A2C算法包装器"""
    
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
            'learning_rate': hyperparams.get('learning_rate', 7e-4),
            'n_steps': hyperparams.get('n_steps', 5),
            'gamma': hyperparams.get('gamma', 0.99),
            'gae_lambda': hyperparams.get('gae_lambda', 1.0),
            'ent_coef': hyperparams.get('ent_coef', 0.01),
            'vf_coef': hyperparams.get('vf_coef', 0.25),
            'max_grad_norm': hyperparams.get('max_grad_norm', 0.5),
            'rms_prop_eps': hyperparams.get('rms_prop_eps', 1e-5),
            'normalize_advantage': hyperparams.get('normalize_advantage', False),
            'policy_kwargs': policy_kwargs,
            'device': device,
            'seed': config.get('seed', None)
        }
    
    def create_model(self, env, tensorboard_log: Optional[str] = None, verbose: int = 1):
        """创建A2C模型"""
        self.model = A2C(
            env=env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **self.hyperparams
        )
        return self.model
    
    def train(self, env, total_timesteps: int, callback=None, **kwargs):
        """训练A2C"""
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
        self.model = A2C.load(path, device=self.device)
        return self.model
'''

# ============================================================================
# SAC Wrapper
# ============================================================================
sac_content = '''"""
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
'''

# ============================================================================
# TD3 Wrapper
# ============================================================================
td3_content = '''"""
TD3算法封装
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import torch

from .base_algorithm import BaseAlgorithm


class TD3Wrapper(BaseAlgorithm):
    """TD3算法包装器"""
    
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
        
        # 处理动作噪声
        action_noise = None
        if hyperparams.get('action_noise') == 'normal':
            noise_kwargs = hyperparams.get('action_noise_kwargs', {})
            n_actions = action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions) + noise_kwargs.get('mean', 0.0),
                sigma=np.ones(n_actions) * noise_kwargs.get('sigma', 0.1)
            )
        
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
            'policy_delay': hyperparams.get('policy_delay', 2),
            'target_policy_noise': hyperparams.get('target_policy_noise', 0.2),
            'target_noise_clip': hyperparams.get('target_noise_clip', 0.5),
            'action_noise': action_noise,
            'policy_kwargs': policy_kwargs,
            'device': device,
            'seed': config.get('seed', None)
        }
    
    def create_model(self, env, tensorboard_log: Optional[str] = None, verbose: int = 1):
        """创建TD3模型"""
        self.model = TD3(
            env=env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            **self.hyperparams
        )
        return self.model
    
    def train(self, env, total_timesteps: int, callback=None, **kwargs):
        """训练TD3"""
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
        self.model = TD3.load(path, device=self.device)
        return self.model
'''

# 写入文件
def write_file(path: str, content: str):
    file_path = Path(path)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ 已修复: {path}")

print("="*70)
print("修复算法包装器")
print("="*70)

write_file('src/algorithms/ppo_wrapper.py', ppo_content)
write_file('src/algorithms/a2c_wrapper.py', a2c_content)
write_file('src/algorithms/sac_wrapper.py', sac_content)
write_file('src/algorithms/td3_wrapper.py', td3_content)

print("\n✅ 所有算法包装器已修复！")
print("\n现在请重新运行测试：")
print("python test_training_system.py")
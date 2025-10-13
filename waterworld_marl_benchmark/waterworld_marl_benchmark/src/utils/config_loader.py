"""
配置加载工具
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_root: str = "configs"):
        self.config_root = Path(config_root)
        
        # 缓存已加载的配置
        self._cache = {}
    
    def load_yaml(self, config_path: str) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            config_path: 相对于config_root的路径，或绝对路径
        
        Returns:
            配置字典
        """
        # 检查缓存
        if config_path in self._cache:
            return self._cache[config_path]
        
        # 构建完整路径
        if os.path.isabs(config_path):
            full_path = Path(config_path)
        else:
            full_path = self.config_root / config_path
        
        # 加载YAML
        if not full_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {full_path}")
        
        with open(full_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 缓存
        self._cache[config_path] = config
        
        return config
    
    def load_run_mode_config(self, mode: str) -> Dict[str, Any]:
        """加载运行模式配置"""
        config = self.load_yaml("run_modes.yaml")
        
        if mode not in config:
            raise ValueError(f"未知的运行模式: {mode}. 支持的模式: {list(config.keys())}")
        
        return config[mode]
    
    def load_environment_config(self, env_name: str = "waterworld_standard") -> Dict[str, Any]:
        """加载环境配置"""
        return self.load_yaml(f"environments/{env_name}.yaml")
    
    def load_algorithm_config(self, algo_name: str) -> Dict[str, Any]:
        """加载算法配置"""
        return self.load_yaml(f"algorithms/{algo_name.lower()}.yaml")
    
    def load_training_config(self, stage_name: str) -> Dict[str, Any]:
        """加载训练阶段配置"""
        return self.load_yaml(f"training/{stage_name}.yaml")
    
    def get_freeze_criteria(self, side: str) -> Dict[str, Any]:
        """获取冻结条件"""
        config = self.load_yaml("run_modes.yaml")
        return config['freeze_criteria'][side]
    
    def get_fixed_pool_config(self) -> Dict[str, Any]:
        """获取固定池配置"""
        config = self.load_yaml("run_modes.yaml")
        return config['fixed_pool']
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并多个配置（后面的覆盖前面的）
        
        Args:
            *configs: 多个配置字典
        
        Returns:
            合并后的配置
        """
        merged = {}
        
        for config in configs:
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


# 全局配置加载器实例
config_loader = ConfigLoader()


# 便捷函数
def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置的便捷函数"""
    return config_loader.load_yaml(config_path)


def get_mode_config(mode: str) -> Dict[str, Any]:
    """获取运行模式配置"""
    return config_loader.load_run_mode_config(mode)


def get_env_config(env_name: str = "waterworld_standard") -> Dict[str, Any]:
    """获取环境配置"""
    return config_loader.load_environment_config(env_name)


def get_algo_config(algo_name: str) -> Dict[str, Any]:
    """获取算法配置"""
    return config_loader.load_algorithm_config(algo_name)


def get_training_config(stage_name: str) -> Dict[str, Any]:
    """获取训练配置"""
    return config_loader.load_training_config(stage_name)




# from src.utils.config_loader import get_mode_config, get_env_config, get_algo_config

# # 加载运行模式配置
# debug_config = get_mode_config("debug")
# print(f"Debug模式训练步数: {debug_config['total_timesteps']}")

# # 加载环境配置
# env_config = get_env_config("waterworld_standard")
# print(f"Predator数量: {env_config['environment']['n_predators']}")

# # 加载算法配置
# ppo_config = get_algo_config("PPO")
# print(f"PPO学习率: {ppo_config['hyperparameters']['learning_rate']}")

# # 合并配置
# from src.utils.config_loader import config_loader

# final_config = config_loader.merge_configs(
#     debug_config,
#     {"total_timesteps": 5000}  # 覆盖默认值
# )
# print(f"最终训练步数: {final_config['total_timesteps']}")
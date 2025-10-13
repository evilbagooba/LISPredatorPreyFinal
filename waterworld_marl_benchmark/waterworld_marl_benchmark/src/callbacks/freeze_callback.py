"""
冻结条件检查回调
检查模型是否达到冻结标准，可以加入固定池
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class FreezeCallback(BaseCallback):
    """冻结条件检查回调"""
    
    def __init__(
        self,
        eval_callback: BaseCallback,
        train_side: str,
        freeze_criteria: Dict[str, Any],
        on_freeze: Optional[Callable] = None,
        verbose: int = 1
    ):
        """
        初始化回调
        
        Args:
            eval_callback: 评估回调（用于获取评估结果）
            train_side: 训练方（predator/prey）
            freeze_criteria: 冻结条件
            on_freeze: 达到冻结条件时的回调函数
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.train_side = train_side
        self.freeze_criteria = freeze_criteria
        self.on_freeze = on_freeze
        
        # 冻结状态
        self.is_frozen = False
        self.freeze_timestep = None
    
    def _on_step(self) -> bool:
        """每步调用"""
        # 只在评估后检查
        if not hasattr(self.eval_callback, 'last_mean_reward'):
            return True
        
        # 如果已经冻结，不再检查
        if self.is_frozen:
            return True
        
        # 检查是否达到冻结条件
        if self._check_freeze_criteria():
            self._freeze_model()
        
        return True
    
    def _check_freeze_criteria(self) -> bool:
        """
        检查是否达到冻结条件
        
        Returns:
            是否达到冻结条件
        """
        # 获取评估结果
        last_mean_reward = self.eval_callback.get_last_mean_reward()
        
        # 获取对应角色的冻结标准
        criteria = self.freeze_criteria
        
        # 检查最低奖励
        min_reward = criteria.get('min_avg_reward', -np.inf)
        if last_mean_reward < min_reward:
            if self.verbose > 1:
                print(f"  ❌ 奖励不足: {last_mean_reward:.2f} < {min_reward:.2f}")
            return False
        
        # 检查角色特定指标
        if self.train_side == "predator":
            # Predator需要检查捕获率
            min_catch_rate = criteria.get('min_catch_rate', 0.0)
            # 这里需要从评估结果中获取catch_rate
            # 暂时简化处理
            if self.verbose > 1:
                print(f"  ✓ 奖励达标: {last_mean_reward:.2f} >= {min_reward:.2f}")
        
        elif self.train_side == "prey":
            # Prey需要检查生存率
            min_survival_rate = criteria.get('min_survival_rate', 0.0)
            # 这里需要从评估结果中获取survival_rate
            # 暂时简化处理
            if self.verbose > 1:
                print(f"  ✓ 奖励达标: {last_mean_reward:.2f} >= {min_reward:.2f}")
        
        # 检查评估episode数
        min_episodes = criteria.get('min_episodes', 0)
        n_eval_episodes = self.eval_callback.n_eval_episodes
        if n_eval_episodes < min_episodes:
            if self.verbose > 1:
                print(f"  ❌ 评估episode不足: {n_eval_episodes} < {min_episodes}")
            return False
        
        return True
    
    def _freeze_model(self):
        """冻结模型"""
        self.is_frozen = True
        self.freeze_timestep = self.n_calls
        
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"❄️  模型达到冻结条件")
            print(f"{'='*70}")
            print(f"  训练步数: {self.freeze_timestep}")
            print(f"  平均奖励: {self.eval_callback.get_last_mean_reward():.2f}")
            print(f"{'='*70}\n")
        
        # 调用自定义回调
        if self.on_freeze:
            self.on_freeze(self.model, self.freeze_timestep)
    
    def is_model_frozen(self) -> bool:
        """模型是否已冻结"""
        return self.is_frozen
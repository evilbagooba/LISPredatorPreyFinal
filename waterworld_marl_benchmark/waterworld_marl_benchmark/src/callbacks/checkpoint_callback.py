"""
检查点保存回调
定期保存训练检查点
"""

import os
from pathlib import Path
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """检查点保存回调"""
    
    def __init__(
        self,
        save_freq: int,
        save_path: Path,
        name_prefix: str = "checkpoint",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0
    ):
        """
        初始化回调
        
        Args:
            save_freq: 保存频率（步数）
            save_path: 保存路径
            name_prefix: 文件名前缀
            save_replay_buffer: 是否保存replay buffer（SAC/TD3）
            save_vecnormalize: 是否保存VecNormalize统计
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        
        # 创建保存目录
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        """每步调用"""
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            self._save_checkpoint()
        
        return True
    
    def _save_checkpoint(self):
        """保存检查点"""
        # 构建文件名
        checkpoint_name = f"{self.name_prefix}_step_{self.n_calls}.zip"
        checkpoint_path = self.save_path / checkpoint_name
        
        # 保存模型
        self.model.save(checkpoint_path)
        
        if self.verbose > 0:
            print(f"💾 保存检查点: {checkpoint_path}")
        
        # 保存replay buffer（如果适用）
        if self.save_replay_buffer and hasattr(self.model, 'replay_buffer'):
            if self.model.replay_buffer is not None:
                buffer_path = self.save_path / f"{self.name_prefix}_replay_buffer_step_{self.n_calls}.pkl"
                self.model.save_replay_buffer(buffer_path)
                
                if self.verbose > 0:
                    print(f"💾 保存replay buffer: {buffer_path}")
        
        # 保存VecNormalize统计（如果适用）
        if self.save_vecnormalize:
            from stable_baselines3.common.vec_env import VecNormalize
            if isinstance(self.training_env, VecNormalize):
                vecnorm_path = self.save_path / f"{self.name_prefix}_vecnormalize_step_{self.n_calls}.pkl"
                self.training_env.save(vecnorm_path)
                
                if self.verbose > 0:
                    print(f"💾 保存VecNormalize: {vecnorm_path}")
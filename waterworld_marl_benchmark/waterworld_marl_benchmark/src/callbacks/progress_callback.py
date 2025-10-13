"""
进度显示回调
显示训练进度条和统计信息
"""

from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


class ProgressBarCallback(BaseCallback):
    """进度条回调"""
    
    def __init__(
        self,
        total_timesteps: int,
        verbose: int = 1
    ):
        """
        初始化回调
        
        Args:
            total_timesteps: 总训练步数
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar: Optional[tqdm] = None
    
    def _on_training_start(self) -> None:
        """训练开始时创建进度条"""
        if self.verbose > 0:
            self.pbar = tqdm(
                total=self.total_timesteps,
                desc="训练进度",
                unit="步"
            )
    
    def _on_step(self) -> bool:
        """每步更新进度条"""
        if self.pbar:
            self.pbar.update(1)
            
            # 更新进度条后缀信息
            if hasattr(self, 'locals') and 'infos' in self.locals:
                for info in self.locals['infos']:
                    if 'episode' in info:
                        ep_reward = info['episode']['r']
                        ep_length = info['episode']['l']
                        self.pbar.set_postfix({
                            'reward': f'{ep_reward:.2f}',
                            'length': f'{ep_length:.0f}'
                        })
                        break
        
        return True
    
    def _on_training_end(self) -> None:
        """训练结束时关闭进度条"""
        if self.pbar:
            self.pbar.close()
            self.pbar = None
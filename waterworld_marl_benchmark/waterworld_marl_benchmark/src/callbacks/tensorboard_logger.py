"""
自定义TensorBoard日志回调
记录多智能体特定的指标
"""

from typing import Dict, Any, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MultiAgentTensorBoardCallback(BaseCallback):
    """多智能体TensorBoard日志回调"""
    
    def __init__(
        self,
        train_side: str,
        verbose: int = 0
    ):
        """
        初始化回调
        
        Args:
            train_side: 训练方（predator/prey）
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.train_side = train_side
        
        # 统计信息
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_counts = 0
        
        # 角色特定指标
        self.catch_rates = []  # Predator
        self.survival_rates = []  # Prey
    
    def _on_step(self) -> bool:
        """每步调用"""
        # 检查是否有episode结束
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                # 记录基本指标
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_counts += 1
                
                # 记录到TensorBoard
                self.logger.record('rollout/ep_rew_mean', episode_reward)
                self.logger.record('rollout/ep_len_mean', episode_length)
                
                # 角色特定指标
                if self.train_side == "predator":
                    # Predator指标：捕获率
                    if 'catch_rate' in info:
                        catch_rate = info['catch_rate']
                        self.catch_rates.append(catch_rate)
                        self.logger.record('metrics/catch_rate', catch_rate)
                    
                    if 'first_catch_time' in info:
                        self.logger.record('metrics/first_catch_time', info['first_catch_time'])
                    
                    if 'energy_efficiency' in info:
                        self.logger.record('metrics/energy_efficiency', info['energy_efficiency'])
                
                elif self.train_side == "prey":
                    # Prey指标：生存率
                    if 'survival_rate' in info:
                        survival_rate = info['survival_rate']
                        self.survival_rates.append(survival_rate)
                        self.logger.record('metrics/survival_rate', survival_rate)
                    
                    if 'avg_lifespan' in info:
                        self.logger.record('metrics/avg_lifespan', info['avg_lifespan'])
                    
                    if 'escape_success' in info:
                        self.logger.record('metrics/escape_success', info['escape_success'])
                
                # 对战级指标
                if 'reward_gap' in info:
                    self.logger.record('metrics/reward_gap', info['reward_gap'])
                
                if 'balance_score' in info:
                    self.logger.record('metrics/balance_score', info['balance_score'])
        
        return True
    
    def _on_training_end(self) -> None:
        """训练结束时调用"""
        if self.episode_counts > 0:
            # 记录汇总统计
            self.logger.record('summary/total_episodes', self.episode_counts)
            self.logger.record('summary/mean_episode_reward', np.mean(self.episode_rewards))
            self.logger.record('summary/mean_episode_length', np.mean(self.episode_lengths))
            
            if self.train_side == "predator" and self.catch_rates:
                self.logger.record('summary/mean_catch_rate', np.mean(self.catch_rates))
            
            if self.train_side == "prey" and self.survival_rates:
                self.logger.record('summary/mean_survival_rate', np.mean(self.survival_rates))
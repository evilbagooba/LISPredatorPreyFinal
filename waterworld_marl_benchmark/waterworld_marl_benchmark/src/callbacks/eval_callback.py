"""
评估回调
定期评估模型性能
"""

import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class EvalCallback(BaseCallback):
    """评估回调"""
    
    def __init__(
        self,
        eval_env,
        train_side: str,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        best_model_save_path: Optional[Path] = None,
        log_path: Optional[Path] = None
    ):
        """
        初始化回调
        
        Args:
            eval_env: 评估环境
            train_side: 训练方（predator/prey）
            eval_freq: 评估频率（步数）
            n_eval_episodes: 评估episode数
            deterministic: 是否使用确定性策略
            render: 是否渲染
            verbose: 详细程度
            best_model_save_path: 最佳模型保存路径
            log_path: 日志保存路径
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.train_side = train_side
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        
        # 最佳性能跟踪
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        
        # 评估历史
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_length = []
        
        # 创建保存目录
        if self.best_model_save_path:
            self.best_model_save_path = Path(self.best_model_save_path)
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
    
    def _on_step(self) -> bool:
        """每步调用"""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._evaluate()
        
        return True
    
    def _evaluate(self):
        """执行评估"""
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"📊 评估 (步数: {self.n_calls})")
            print(f"{'='*70}")
        
        # 评估模型
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True
        )
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        self.last_mean_reward = mean_reward
        
        # 记录结果
        self.evaluations_timesteps.append(self.n_calls)
        self.evaluations_results.append(episode_rewards)
        self.evaluations_length.append(episode_lengths)
        
        # 记录到TensorBoard
        self.logger.record('eval/mean_reward', mean_reward)
        self.logger.record('eval/std_reward', std_reward)
        self.logger.record('eval/mean_ep_length', mean_length)
        
        if self.verbose > 0:
            print(f"  平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"  平均长度: {mean_length:.0f}")
        
        # 保存最佳模型
        if mean_reward > self.best_mean_reward:
            if self.verbose > 0:
                print(f"  🎉 新的最佳模型! (旧: {self.best_mean_reward:.2f}, 新: {mean_reward:.2f})")
            
            self.best_mean_reward = mean_reward
            
            if self.best_model_save_path:
                best_model_path = self.best_model_save_path / "best_model.zip"
                self.model.save(best_model_path)
                
                if self.verbose > 0:
                    print(f"  💾 保存最佳模型: {best_model_path}")
        
        if self.verbose > 0:
            print(f"{'='*70}\n")
    
    def get_best_mean_reward(self) -> float:
        """获取最佳平均奖励"""
        return self.best_mean_reward
    
    def get_last_mean_reward(self) -> float:
        """获取最近一次平均奖励"""
        return self.last_mean_reward
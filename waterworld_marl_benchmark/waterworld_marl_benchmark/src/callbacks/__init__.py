"""
回调模块初始化
"""

from stable_baselines3.common.callbacks import CallbackList

from .tensorboard_logger import MultiAgentTensorBoardCallback
from .checkpoint_callback import CheckpointCallback
from .eval_callback import EvalCallback
from .freeze_callback import FreezeCallback
from .progress_callback import ProgressBarCallback


def create_callbacks(
    train_side: str,
    checkpoint_path,
    eval_env,
    config: dict,
    on_freeze=None
) -> CallbackList:
    """
    创建回调列表
    
    Args:
        train_side: 训练方
        checkpoint_path: 检查点保存路径
        eval_env: 评估环境
        config: 配置字典
        on_freeze: 冻结回调函数
    
    Returns:
        回调列表
    """
    callbacks = []
    
    # 1. TensorBoard日志
    tb_callback = MultiAgentTensorBoardCallback(
        train_side=train_side,
        verbose=config.get('verbose', 1)
    )
    callbacks.append(tb_callback)
    
    # 2. 检查点保存
    if config.get('save_checkpoints', True):
        checkpoint_freq = config.get('checkpoint_freq', 100000)
        if checkpoint_freq > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_path,
                name_prefix="checkpoint",
                verbose=config.get('verbose', 1)
            )
            callbacks.append(checkpoint_callback)
    
    # 3. 评估
    if config.get('eval_freq', -1) > 0:
        eval_callback = EvalCallback(
            eval_env=eval_env,
            train_side=train_side,
            eval_freq=config.get('eval_freq', 10000),
            n_eval_episodes=config.get('n_eval_episodes', 10),
            deterministic=config.get('deterministic_eval', True),
            verbose=config.get('verbose', 1),
            best_model_save_path=checkpoint_path
        )
        callbacks.append(eval_callback)
        
        # 4. 冻结条件检查
        if config.get('check_freeze', False):
            freeze_criteria = config.get('freeze_criteria', {})
            freeze_callback = FreezeCallback(
                eval_callback=eval_callback,
                train_side=train_side,
                freeze_criteria=freeze_criteria,
                on_freeze=on_freeze,
                verbose=config.get('verbose', 1)
            )
            callbacks.append(freeze_callback)
    
    # 5. 进度条
    if config.get('show_progress', True):
        progress_callback = ProgressBarCallback(
            total_timesteps=config.get('total_timesteps', 1000000),
            verbose=config.get('verbose', 1)
        )
        callbacks.append(progress_callback)
    
    return CallbackList(callbacks)


__all__ = [
    'MultiAgentTensorBoardCallback',
    'CheckpointCallback',
    'EvalCallback',
    'FreezeCallback',
    'ProgressBarCallback',
    'create_callbacks'
]
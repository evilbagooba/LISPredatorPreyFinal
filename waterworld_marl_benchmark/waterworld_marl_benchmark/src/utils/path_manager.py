"""
路径管理工具
根据运行模式和实验配置生成正确的输出路径
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class PathManager:
    """路径管理器"""
    
    def __init__(self, run_mode: str, experiment_name: str):
        """
        初始化路径管理器
        
        Args:
            run_mode: 运行模式 (debug/dryrun/prod)
            experiment_name: 实验名称
        """
        self.run_mode = run_mode
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 根据模式设置基础目录
        if run_mode == "debug":
            self.base_dir = Path("debug_outputs/current")
        elif run_mode == "dryrun":
            self.base_dir = Path(f"dryrun_outputs/run_{self.timestamp}")
        else:  # prod
            self.base_dir = Path("outputs")
        
        # 创建基础目录
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_dir(self, stage_name: Optional[str] = None) -> Path:
        """
        获取模型保存目录
        
        Args:
            stage_name: 训练阶段名称（如 stage1.1_prey_warmup）
        
        Returns:
            模型目录路径
        """
        if stage_name:
            path = self.base_dir / "saved_models" / stage_name
        else:
            path = self.base_dir / "saved_models" / self.experiment_name
        
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_checkpoint_dir(self, stage_name: Optional[str] = None) -> Path:
        """获取检查点目录"""
        if stage_name:
            path = self.base_dir / "checkpoints" / stage_name / self.experiment_name
        else:
            path = self.base_dir / "checkpoints" / self.experiment_name
        
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_tensorboard_dir(self, stage_name: Optional[str] = None) -> Path:
        """获取TensorBoard日志目录"""
        if stage_name:
            # 同一 stage 的所有实验共享一个目录
            path = self.base_dir / "tensorboard_logs" / stage_name
        else:
            path = self.base_dir / "tensorboard_logs" / self.experiment_name
        
        path.mkdir(parents=True, exist_ok=True)
        return path
    def get_experiment_dir(self, stage_name: Optional[str] = None) -> Path:
        """获取实验记录目录"""
        if stage_name:
            path = self.base_dir / "experiments" / stage_name
        else:
            path = self.base_dir / "experiments" / self.experiment_name
        
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_fixed_pool_dir(self, pool_name: str) -> Path:
        """
        获取固定池目录
        
        Args:
            pool_name: 池名称（如 prey_pool_v1）
        """
        # 固定池只在正式输出目录
        path = Path("outputs/fixed_pools") / pool_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_evaluation_dir(self) -> Path:
        """获取评估结果目录"""
        path = Path("outputs/evaluation_results")
        path.mkdir(parents=True, exist_ok=True)
        return path


def create_path_manager(run_mode: str, experiment_name: str) -> PathManager:
    """创建路径管理器的便捷函数"""
    return PathManager(run_mode, experiment_name)
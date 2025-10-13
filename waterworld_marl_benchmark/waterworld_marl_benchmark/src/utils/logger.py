"""
日志工具
提供统一的日志接口
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class TrainingLogger:
    """训练日志器"""
    
    def __init__(
        self,
        name: str,
        log_file: Optional[Path] = None,
        log_level: str = "INFO",
        console_output: bool = True
    ):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            log_file: 日志文件路径（None则不写文件）
            log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
            console_output: 是否输出到控制台
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除已有的handlers
        self.logger.handlers = []
        
        # 日志格式
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台输出
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件输出
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str):
        """Debug级别日志"""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """Info级别日志"""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Warning级别日志"""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Error级别日志"""
        self.logger.error(msg)
    
    def critical(self, msg: str):
        """Critical级别日志"""
        self.logger.critical(msg)
    
    def log_config(self, config: dict, title: str = "Configuration"):
        """记录配置信息"""
        self.info(f"\n{'='*70}")
        self.info(f"{title}")
        self.info(f"{'='*70}")
        
        for key, value in config.items():
            if isinstance(value, dict):
                self.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.info(f"  {sub_key}: {sub_value}")
            else:
                self.info(f"{key}: {value}")
        
        self.info(f"{'='*70}\n")
    
    def log_banner(self, text: str, char: str = "="):
        """打印横幅"""
        banner = char * 70
        self.info(f"\n{banner}")
        self.info(text)
        self.info(f"{banner}\n")


def create_logger(
    name: str,
    log_dir: Optional[Path] = None,
    log_level: str = "INFO"
) -> TrainingLogger:
    """
    创建日志器的便捷函数
    
    Args:
        name: 日志器名称
        log_dir: 日志目录（None则不写文件）
        log_level: 日志级别
    
    Returns:
        日志器实例
    """
    if log_dir:
        log_file = log_dir / f"{name}.log"
    else:
        log_file = None
    
    return TrainingLogger(name, log_file, log_level)
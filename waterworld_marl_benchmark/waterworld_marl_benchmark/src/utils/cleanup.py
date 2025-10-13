"""
清理工具
管理调试和预演数据的自动清理
"""

import shutil
from pathlib import Path
from datetime import datetime
from typing import List
import os


class OutputCleaner:
    """输出清理器"""
    
    @staticmethod
    def cleanup_debug_on_start(config: dict):
        """启动时清理调试目录"""
        if not config.get("clear_on_start", False):
            return
        
        current_dir = Path("debug_outputs/current")
        
        if current_dir.exists():
            print(f"🗑️  清空调试目录: {current_dir}")
            shutil.rmtree(current_dir)
        
        current_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def archive_debug_run():
        """归档当前调试会话"""
        current_dir = Path("debug_outputs/current")
        
        if not current_dir.exists() or not any(current_dir.iterdir()):
            print("ℹ️  调试目录为空，跳过归档")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = Path(f"debug_outputs/archive/debug_{timestamp}")
        
        print(f"📦 归档调试数据到: {archive_dir}")
        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(current_dir), str(archive_dir))
        
        # 清理旧归档
        OutputCleaner.cleanup_old_debug_archives()
    
    @staticmethod
    def cleanup_old_debug_archives(max_archives: int = 5):
        """删除超出限制的旧调试归档"""
        archive_dir = Path("debug_outputs/archive")
        
        if not archive_dir.exists():
            return
        
        # 获取所有归档目录
        archives = sorted(
            [d for d in archive_dir.iterdir() if d.is_dir() and d.name.startswith("debug_")],
            key=lambda x: x.name,
            reverse=True
        )
        
        # 删除超出的
        for old_archive in archives[max_archives:]:
            print(f"🗑️  删除旧调试归档: {old_archive.name}")
            shutil.rmtree(old_archive)
    
    @staticmethod
    def cleanup_old_dryruns(max_runs: int = 3):
        """清理旧的预演数据"""
        dryrun_dir = Path("dryrun_outputs")
        
        if not dryrun_dir.exists():
            return
        
        # 获取所有运行目录
        runs = sorted(
            [d for d in dryrun_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda x: x.name,
            reverse=True
        )
        
        # 删除超出的
        for old_run in runs[max_runs:]:
            print(f"🗑️  删除旧预演数据: {old_run.name}")
            shutil.rmtree(old_run)
    
    @staticmethod
    def get_directory_size(path: Path) -> float:
        """
        获取目录大小（MB）
        
        Args:
            path: 目录路径
        
        Returns:
            大小（MB）
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        return total_size / (1024 * 1024)  # 转换为MB
    
    @staticmethod
    def print_storage_summary():
        """打印存储空间使用摘要"""
        print("\n" + "="*70)
        print("💾 存储空间使用摘要")
        print("="*70)
        
        directories = {
            "outputs": Path("outputs"),
            "dryrun_outputs": Path("dryrun_outputs"),
            "debug_outputs": Path("debug_outputs")
        }
        
        for name, path in directories.items():
            if path.exists():
                size = OutputCleaner.get_directory_size(path)
                print(f"{name:20s}: {size:>10.2f} MB")
            else:
                print(f"{name:20s}: {'不存在':>10s}")
        
        print("="*70 + "\n")


# 便捷函数
def cleanup_debug(config: dict):
    """清理调试数据"""
    OutputCleaner.cleanup_debug_on_start(config)


def archive_debug():
    """归档调试数据"""
    OutputCleaner.archive_debug_run()


def cleanup_dryrun(max_runs: int = 3):
    """清理预演数据"""
    OutputCleaner.cleanup_old_dryruns(max_runs)
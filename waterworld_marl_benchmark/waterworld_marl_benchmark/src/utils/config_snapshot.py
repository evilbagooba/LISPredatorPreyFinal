"""
配置快照工具
保存每次训练的完整配置，确保可复现
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def make_json_serializable(obj):
    """将对象转换为JSON可序列化的格式"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    elif isinstance(obj, type):
        # 将类型对象转换为字符串
        return f"{obj.__module__}.{obj.__name__}"
    elif hasattr(obj, '__class__') and obj.__class__.__module__ not in ['builtins', '__builtin__']:
        # 复杂对象转为字符串表示
        return str(obj)
    else:
        return obj


class ConfigSnapshot:
    """配置快照管理器"""
    
    @staticmethod
    def save_snapshot(config: Dict[str, Any], save_dir: Path, filename: str):
        save_dir.mkdir(parents=True, exist_ok=True)
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'config': make_json_serializable(config)
        }
        
        # 保存为YAML
        yaml_path = save_dir / f"{filename}.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True)
        
        # 尝试保存为JSON
        json_path = save_dir / f"{filename}.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
        except TypeError as e:
            print(f"⚠️  JSON序列化失败，仅保存YAML: {e}")
        
        print(f"💾 配置快照已保存: {yaml_path}")
    
    @staticmethod
    def load_snapshot(snapshot_path: Path) -> Dict[str, Any]:
        if snapshot_path.suffix == '.yaml':
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                snapshot = yaml.safe_load(f)
        elif snapshot_path.suffix == '.json':
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                snapshot = json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {snapshot_path.suffix}")
        
        return snapshot.get('config', {})
    
    @staticmethod
    def save_training_summary(summary: Dict[str, Any], save_dir: Path, filename: str):
        save_dir.mkdir(parents=True, exist_ok=True)
        
        summary['saved_at'] = datetime.now().isoformat()
        serializable_summary = make_json_serializable(summary)
        
        json_path = save_dir / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📝 训练摘要已保存: {json_path}")


def save_config_snapshot(config: Dict, save_dir: Path, name: str):
    ConfigSnapshot.save_snapshot(config, save_dir, name)


def save_training_summary(summary: Dict, save_dir: Path, name: str):
    ConfigSnapshot.save_training_summary(summary, save_dir, name)

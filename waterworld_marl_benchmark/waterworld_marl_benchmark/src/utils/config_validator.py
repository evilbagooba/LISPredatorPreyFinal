"""
配置验证工具
防止配置错误导致训练失败
"""

import sys
from typing import Dict, Any, List
from src.utils.config_loader import config_loader


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_run_mode(self, mode: str, config: Dict[str, Any]) -> bool:
        """
        验证运行模式配置
        
        Args:
            mode: 运行模式 (debug/dryrun/prod)
            config: 配置字典
        
        Returns:
            是否通过验证
        """
        self.errors = []
        self.warnings = []
        
        # 加载验证规则
        validation_rules = config_loader.load_yaml("run_modes.yaml").get("validation", {})
        
        if mode == "prod":
            self._validate_production_mode(config, validation_rules.get("prod", {}))
        elif mode == "debug":
            self._validate_debug_mode(config, validation_rules.get("debug", {}))
        
        return len(self.errors) == 0
    
    def _validate_production_mode(self, config: Dict, rules: Dict):
        """验证生产模式"""
        
        # 检查训练步数
        min_timesteps = rules.get("min_timesteps", 500000)
        if config.get("total_timesteps", 0) < min_timesteps:
            self.errors.append(
                f"⚠️  生产模式训练步数过低: {config['total_timesteps']} "
                f"(建议至少 {min_timesteps} 步)"
            )
        
        # 检查实验名称
        if "experiment_name" in config:
            exp_name = config["experiment_name"].lower()
            forbidden = rules.get("forbidden_keywords", [])
            
            for keyword in forbidden:
                if keyword in exp_name:
                    self.warnings.append(
                        f"⚠️  实验名称 '{config['experiment_name']}' 包含 '{keyword}' "
                        "字样，确认这是正式实验吗？"
                    )
        
        # 检查是否启用了保存
        if not config.get("save_final_model", True):
            self.warnings.append(
                "⚠️  生产模式未启用模型保存，这可能不是你想要的！"
            )
    
    def _validate_debug_mode(self, config: Dict, rules: Dict):
        """验证调试模式"""
        
        # 检查训练步数是否过多
        max_timesteps = rules.get("max_timesteps", 10000)
        if config.get("total_timesteps", 0) > max_timesteps:
            self.warnings.append(
                f"💡 调试模式训练步数较多 ({config['total_timesteps']})，"
                "可能耗时较长，考虑降低步数？"
            )
    
    def validate_environment_config(self, config: Dict[str, Any]) -> bool:
        """验证环境配置"""
        self.errors = []
        self.warnings = []
        
        env_config = config.get("environment", {})
        
        # 必需字段
        required_fields = [
            "n_predators", "n_preys", "max_cycles",
            "predator_speed", "prey_speed"
        ]
        
        for field in required_fields:
            if field not in env_config:
                self.errors.append(f"❌ 环境配置缺少必需字段: {field}")
        
        # 合理性检查
        if env_config.get("n_predators", 0) > env_config.get("n_preys", 0):
            self.warnings.append(
                "⚠️  Predator数量多于Prey，可能导致训练不平衡"
            )
        
        if env_config.get("predator_speed", 0) < env_config.get("prey_speed", 0):
            self.warnings.append(
                "⚠️  Predator速度慢于Prey，可能难以捕获"
            )
        
        return len(self.errors) == 0
    
    def validate_algorithm_config(self, algo_name: str, config: Dict[str, Any]) -> bool:
        """验证算法配置"""
        self.errors = []
        self.warnings = []
        
        hyperparams = config.get("hyperparameters", {})
        
        # 检查学习率
        lr = hyperparams.get("learning_rate")
        if lr is not None:
            if lr > 0.01:
                self.warnings.append(
                    f"⚠️  {algo_name} 学习率较高 ({lr})，可能导致训练不稳定"
                )
            elif lr < 1e-6:
                self.warnings.append(
                    f"⚠️  {algo_name} 学习率过低 ({lr})，可能学习缓慢"
                )
        
        return len(self.errors) == 0
    
    def validate_training_config(self, stage_config: Dict[str, Any]) -> bool:
        """验证训练阶段配置"""
        self.errors = []
        self.warnings = []
        
        # 检查对手配置
        opponent = stage_config.get("opponent", {})
        if not opponent:
            self.errors.append("❌ 缺少对手配置")
        else:
            opp_type = opponent.get("type")
            if opp_type == "mixed_pool":
                pool_path = opponent.get("pool_path")
                if not pool_path:
                    self.errors.append("❌ mixed_pool模式需要指定pool_path")
        
        # 检查训练算法列表
        algos = stage_config.get("algorithms_to_train", [])
        if not algos:
            self.errors.append("❌ 未指定要训练的算法")
        
        return len(self.errors) == 0
    
    def print_results(self):
        """打印验证结果"""
        if self.errors:
            print("\n❌ 配置错误:")
            for err in self.errors:
                print(f"  {err}")
        
        if self.warnings:
            print("\n⚠️  配置警告:")
            for warn in self.warnings:
                print(f"  {warn}")
    
    def require_confirmation(self) -> bool:
        """
        如果有警告，要求用户确认
        
        Returns:
            用户是否确认继续
        """
        if not self.warnings:
            return True
        
        self.print_results()
        response = input("\n继续吗？(yes/no): ").strip().lower()
        return response in ['yes', 'y']


# 全局验证器实例
validator = ConfigValidator()


def validate_config(mode: str, config: Dict[str, Any]) -> bool:
    """
    验证配置的便捷函数
    
    Args:
        mode: 运行模式
        config: 配置字典
    
    Returns:
        是否通过验证
    """
    if not validator.validate_run_mode(mode, config):
        validator.print_results()
        return False
    
    return True
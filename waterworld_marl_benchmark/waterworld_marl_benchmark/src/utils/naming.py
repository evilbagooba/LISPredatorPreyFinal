"""
文件命名规范工具
确保所有文件名都包含必要信息且格式一致
"""

from datetime import datetime
from typing import Optional


class FileNaming:
    """文件命名工具"""
    
    @staticmethod
    def generate_model_filename(
        train_algo: str,
        train_side: str,
        version: str,
        opponent_info: str,
        run_mode: str,
        extension: str = "zip"
    ) -> str:
        """
        生成模型文件名
        
        Args:
            train_algo: 训练算法（如 PPO）
            train_side: 训练方（predator/prey）
            version: 版本号（如 v1）
            opponent_info: 对手信息（如 RANDOM_pred 或 MIX_pool_v1）
            run_mode: 运行模式（debug/dryrun/prod）
            extension: 文件扩展名
        
        Returns:
            文件名
        
        Example:
            PPO_prey_v1_vs_RANDOM_pred_20251013_143022.zip
            DRYRUN_SAC_pred_v2_vs_MIX_pool_v1_20251013_151033.zip
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 添加模式前缀
        if run_mode == "debug":
            prefix = "DEBUG_"
        elif run_mode == "dryrun":
            prefix = "DRYRUN_"
        else:
            prefix = ""
        
        # 构建文件名
        filename = (
            f"{prefix}{train_algo}_{train_side}_{version}"
            f"_vs_{opponent_info}_{timestamp}.{extension}"
        )
        
        return filename
    
    @staticmethod
    def generate_checkpoint_filename(
        train_algo: str,
        train_side: str,
        step: int,
        extension: str = "zip"
    ) -> str:
        """
        生成检查点文件名
        
        Args:
            train_algo: 训练算法
            train_side: 训练方
            step: 训练步数
            extension: 文件扩展名
        
        Returns:
            文件名
        
        Example:
            PPO_prey_step_500000.zip
        """
        return f"{train_algo}_{train_side}_step_{step}.{extension}"
    
    @staticmethod
    def generate_config_filename(
        train_algo: str,
        train_side: str,
        version: str,
        extension: str = "yaml"
    ) -> str:
        """
        生成配置快照文件名
        
        Example:
            PPO_prey_v1_config.yaml
        """
        return f"{train_algo}_{train_side}_{version}_config.{extension}"
    
    @staticmethod
    def generate_log_filename(
        train_algo: str,
        train_side: str,
        version: str,
        extension: str = "log"
    ) -> str:
        """
        生成日志文件名
        
        Example:
            PPO_prey_v1_training.log
        """
        return f"{train_algo}_{train_side}_{version}_training.{extension}"
    
    @staticmethod
    def generate_summary_filename(
        train_algo: str,
        train_side: str,
        version: str,
        extension: str = "json"
    ) -> str:
        """
        生成训练摘要文件名
        
        Example:
            PPO_prey_v1_summary.json
        """
        return f"{train_algo}_{train_side}_{version}_summary.{extension}"
    
    @staticmethod
    def parse_model_filename(filename: str) -> dict:
        """
        解析模型文件名，提取信息
        
        Args:
            filename: 模型文件名
        
        Returns:
            包含解析信息的字典
        
        Example:
            Input: "PPO_prey_v1_vs_RANDOM_pred_20251013_143022.zip"
            Output: {
                'algo': 'PPO',
                'side': 'prey',
                'version': 'v1',
                'opponent': 'RANDOM_pred',
                'timestamp': '20251013_143022',
                'is_debug': False,
                'is_dryrun': False
            }
        """
        # 去除扩展名
        name = filename.rsplit('.', 1)[0]
        
        # 检查模式前缀
        is_debug = name.startswith("DEBUG_")
        is_dryrun = name.startswith("DRYRUN_")
        
        if is_debug:
            name = name[6:]  # 去除 "DEBUG_"
        elif is_dryrun:
            name = name[7:]  # 去除 "DRYRUN_"
        
        # 分割字段
        parts = name.split('_')
        
        # 基本解析（假设格式正确）
        if len(parts) >= 6:
            return {
                'algo': parts[0],
                'side': parts[1],
                'version': parts[2],
                'opponent': '_'.join(parts[4:-2]),  # vs之后到时间戳之前
                'timestamp': '_'.join(parts[-2:]),
                'is_debug': is_debug,
                'is_dryrun': is_dryrun
            }
        else:
            return {}
    
    @staticmethod
    def format_opponent_info(opponent_config: dict) -> str:
        """
        根据对手配置生成对手信息字符串
        
        Args:
            opponent_config: 对手配置字典
        
        Returns:
            对手信息字符串
        
        Example:
            {"type": "algorithm", "algorithm": "RANDOM", "side": "predator"}
            -> "RANDOM_pred"
            
            {"type": "mixed_pool", "pool_path": "outputs/fixed_pools/prey_pool_v1"}
            -> "MIX_prey_pool_v1"
        """
        opp_type = opponent_config.get("type", "unknown")
        
        if opp_type == "algorithm":
            algo = opponent_config.get("algorithm", "UNKNOWN")
            side = opponent_config.get("side", "unknown")
            side_abbr = "pred" if side == "predator" else "prey"
            return f"{algo}_{side_abbr}"
        
        elif opp_type == "fixed_model":
            # 从路径提取模型名
            path = opponent_config.get("path", "")
            model_name = path.split('/')[-1].replace('.zip', '')
            return model_name
        
        elif opp_type == "mixed_pool":
            # 从池路径提取池名
            pool_path = opponent_config.get("pool_path", "")
            pool_name = pool_path.split('/')[-1]
            return f"MIX_{pool_name}"
        
        else:
            return "UNKNOWN"


# 全局命名工具实例
naming = FileNaming()


# 便捷函数
def generate_model_name(
    algo: str, side: str, version: str,
    opponent: dict, mode: str
) -> str:
    """生成模型文件名的便捷函数"""
    opponent_info = naming.format_opponent_info(opponent)
    return naming.generate_model_filename(algo, side, version, opponent_info, mode)
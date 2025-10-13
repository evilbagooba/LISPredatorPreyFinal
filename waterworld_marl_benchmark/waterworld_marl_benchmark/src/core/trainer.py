"""
核心训练器
整合所有模块，提供统一的训练接口
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from src.core.environment import WaterworldEnvManager, create_training_env
from src.core.opponent_pool import create_opponent_policies
from src.algorithms import create_algorithm
from src.callbacks import create_callbacks
from src.utils.config_loader import get_mode_config, get_env_config, get_algo_config
from src.utils.path_manager import PathManager
from src.utils.naming import FileNaming
from src.utils.logger import create_logger
from src.utils.banner import print_mode_banner, print_training_start, print_training_complete
from src.utils.config_snapshot import save_config_snapshot, save_training_summary
from src.utils.config_validator import validator
from src.utils.cleanup import cleanup_debug


class MultiAgentTrainer:
    """多智能体训练器"""
    
    def __init__(
        self,
        # 核心配置
        train_side: str,
        train_algo: str,
        opponent_config: Dict[str, Any],
        
        # 实验元数据
        experiment_name: str,
        stage_name: str,
        generation: int = 0,
        version: str = "v1",
        
        # 运行模式
        run_mode: str = "prod",
        
        # 环境配置
        env_config: Optional[Dict[str, Any]] = None,
        
        # 算法配置
        algo_config: Optional[Dict[str, Any]] = None,
        
        # 训练配置（覆盖模式默认值）
        total_timesteps: Optional[int] = None,
        n_envs: Optional[int] = None,
        eval_freq: Optional[int] = None,
        checkpoint_freq: Optional[int] = None,
        n_eval_episodes: Optional[int] = None,
        
        # 其他
        device: str = "auto",
        seed: Optional[int] = None,
        notes: str = ""
    ):
        """
        初始化训练器
        
        Args:
            train_side: 训练方（predator/prey）
            train_algo: 训练算法（PPO/A2C/SAC/TD3）
            opponent_config: 对手配置
            experiment_name: 实验名称
            stage_name: 训练阶段名称（stage1.1_prey_warmup等）
            generation: 代数
            version: 版本号
            run_mode: 运行模式（debug/dryrun/prod）
            env_config: 环境配置（None则使用默认）
            algo_config: 算法配置（None则使用默认）
            total_timesteps: 总训练步数（None则使用模式默认）
            n_envs: 并行环境数（None则使用模式默认）
            eval_freq: 评估频率（None则使用模式默认）
            checkpoint_freq: 检查点频率（None则使用模式默认）
            n_eval_episodes: 评估episode数（None则使用模式默认）
            device: 计算设备
            seed: 随机种子
            notes: 实验备注
        """
        
        # =====================================================================
        # 1. 基本配置
        # =====================================================================
        self.train_side = train_side
        self.train_algo = train_algo.upper()
        self.opponent_config = opponent_config
        self.experiment_name = experiment_name
        self.stage_name = stage_name
        self.generation = generation
        self.version = version
        self.run_mode = run_mode
        self.device = device
        self.seed = seed
        self.notes = notes
        
        # =====================================================================
        # 2. 加载配置
        # =====================================================================
        
        # 加载运行模式配置
        self.mode_config = get_mode_config(run_mode)
        
        # 加载环境配置
        if env_config is None:
            env_config = get_env_config("waterworld_standard")
        self.env_config = env_config
        
        # 加载算法配置
        if algo_config is None:
            algo_config = get_algo_config(self.train_algo)
        self.algo_config = algo_config
        
        # 合并训练配置（用户指定 > 模式默认）
        self.training_config = {
            'total_timesteps': total_timesteps or self.mode_config.get('total_timesteps', 1000000),
            'n_envs': n_envs or self.mode_config.get('n_envs', 1),
            'eval_freq': eval_freq if eval_freq is not None else self.mode_config.get('eval_freq', 10000),
            'checkpoint_freq': checkpoint_freq if checkpoint_freq is not None else self.mode_config.get('checkpoint_freq', 100000),
            'n_eval_episodes': n_eval_episodes or self.mode_config.get('n_eval_episodes', 10),
            'save_checkpoints': self.mode_config.get('save_checkpoints', True),
            'save_final_model': self.mode_config.get('save_final_model', True),
            'tensorboard_enabled': self.mode_config.get('tensorboard_enabled', True),
            'verbose': self.mode_config.get('verbose', 1),
            'deterministic_eval': self.mode_config.get('deterministic_eval', True),
            'show_progress': True,
            'check_freeze': False  # 在训练后手动检查
        }
        
        # =====================================================================
        # 3. 验证配置
        # =====================================================================
        full_config = {
            'run_mode': run_mode,
            'train_side': train_side,
            'train_algo': train_algo,
            'experiment_name': experiment_name,
            **self.training_config
        }
        
        if not validator.validate_run_mode(run_mode, full_config):
            validator.print_results()
            raise ValueError("配置验证失败")
        # test 模式不需要确认
        if run_mode not in ['debug', 'test']:  # ✅ 添加 'test'
            if not validator.require_confirmation():
                raise KeyboardInterrupt("用户取消训练")
        if not validator.require_confirmation():
            raise KeyboardInterrupt("用户取消训练")
        
        # =====================================================================
        # 4. 路径管理
        # =====================================================================
        self.path_manager = PathManager(run_mode, experiment_name)
        
        # 各种输出路径
        self.model_dir = self.path_manager.get_model_dir(stage_name)
        self.checkpoint_dir = self.path_manager.get_checkpoint_dir(stage_name)
        self.tensorboard_dir = self.path_manager.get_tensorboard_dir(stage_name)
        self.experiment_dir = self.path_manager.get_experiment_dir(stage_name)
        
        # =====================================================================
        # 5. 日志系统
        # =====================================================================
        self.naming = FileNaming()
        log_filename = self.naming.generate_log_filename(
            train_algo=self.train_algo,
            train_side=self.train_side,
            version=self.version
        )
        
        self.logger = create_logger(
            name=f"{self.train_algo}_{self.train_side}",
            log_dir=self.experiment_dir,
            log_level=self.mode_config.get('log_level', 'INFO')
        )
        
        # =====================================================================
        # 6. 环境和模型（延迟初始化）
        # =====================================================================
        self.env_manager = None
        self.train_env = None
        self.eval_env = None
        self.algorithm = None
        self.opponent_policies = None
        
        # =====================================================================
        # 7. 训练统计
        # =====================================================================
        self.training_start_time = None
        self.training_end_time = None
        self.total_training_time = None
        self.final_model_path = None
        
        # =====================================================================
        # 8. 打印横幅
        # =====================================================================
        print_mode_banner(run_mode, self.mode_config)
        
        # =====================================================================
        # 9. 清理调试数据（如果需要）
        # =====================================================================
        if run_mode == "debug":
            cleanup_debug(self.mode_config)
        
        # =====================================================================
        # 10. 保存配置快照
        # =====================================================================
        self._save_config_snapshot()
    
    def _save_config_snapshot(self):
        """保存配置快照"""
        snapshot = {
            'run_mode': self.run_mode,
            'train_side': self.train_side,
            'train_algo': self.train_algo,
            'opponent_config': self.opponent_config,
            'experiment_name': self.experiment_name,
            'stage_name': self.stage_name,
            'generation': self.generation,
            'version': self.version,
            'device': self.device,
            'seed': self.seed,
            'notes': self.notes,
            'env_config': self.env_config,
            'algo_config': self.algo_config,
            'training_config': self.training_config,
            'mode_config': self.mode_config
        }
        
        config_filename = self.naming.generate_config_filename(
            train_algo=self.train_algo,
            train_side=self.train_side,
            version=self.version
        )
        
        save_config_snapshot(
            config=snapshot,
            save_dir=self.experiment_dir,
            name=config_filename.replace('.yaml', '')
        )
    
    def setup(self):
        """
        设置训练环境和模型
        在训练前必须调用此方法
        """
        self.logger.log_banner("🔧 设置训练环境", "=")
        
        # =====================================================================
        # 1. 创建环境管理器
        # =====================================================================
        self.logger.info("创建环境管理器...")
        self.env_manager = WaterworldEnvManager(self.env_config)
        
        # =====================================================================
        # 2. 创建对手策略
        # =====================================================================
        self.logger.info(f"创建对手策略 (类型: {self.opponent_config.get('type')})...")
        self.opponent_policies = create_opponent_policies(
            opponent_config=self.opponent_config,
            env_manager=self.env_manager,
            device=self.device
        )
        
        self.logger.info(f"  ✓ 创建了 {len(self.opponent_policies)} 个对手策略")
        
        # =====================================================================
        # 3. 创建训练环境
        # =====================================================================
        self.logger.info(f"创建训练环境 (并行数: {self.training_config['n_envs']})...")
        self.train_env = create_training_env(
            env_config=self.env_config,
            train_side=self.train_side,
            opponent_policies=self.opponent_policies,
            n_envs=self.training_config['n_envs']
        )
        
        # =====================================================================
        # 4. 创建评估环境
        # =====================================================================
        if self.training_config['eval_freq'] > 0:
            self.logger.info("创建评估环境...")
            self.eval_env = create_training_env(
                env_config=self.env_config,
                train_side=self.train_side,
                opponent_policies=self.opponent_policies,
                n_envs=1  # 评估用单环境
            )
        
        # =====================================================================
        # 5. 创建算法
        # =====================================================================
        self.logger.info(f"创建算法: {self.train_algo}...")
        
        # 获取空间信息
        obs_space = self.env_manager.get_observation_space(self.train_side)
        action_space = self.env_manager.get_action_space(self.train_side)
        
        # 创建算法实例
        self.algorithm = create_algorithm(
            algo_name=self.train_algo,
            observation_space=obs_space,
            action_space=action_space,
            config=self.algo_config,
            device=self.device
        )
        
        # 创建模型
        tensorboard_log = str(self.tensorboard_dir) if self.training_config['tensorboard_enabled'] else None
        
        self.algorithm.create_model(
            env=self.train_env,
            tensorboard_log=tensorboard_log,
            verbose=self.training_config['verbose']
        )
        
        self.logger.info("  ✓ 算法创建完成")
        
        # =====================================================================
        # 6. 记录配置
        # =====================================================================
        self.logger.log_config({
            '训练方': self.train_side,
            '训练算法': self.train_algo,
            '版本': self.version,
            '总步数': self.training_config['total_timesteps'],
            '并行环境': self.training_config['n_envs'],
            '评估频率': self.training_config['eval_freq'],
            '检查点频率': self.training_config['checkpoint_freq'],
            '设备': self.device,
            '随机种子': self.seed
        }, title="训练配置")
        
        self.logger.log_banner("✅ 环境设置完成", "=")
    
    def train(self):
        """执行训练"""
        
        # =====================================================================
        # 1. 检查是否已设置
        # =====================================================================
        if self.algorithm is None:
            raise RuntimeError("请先调用 setup() 方法设置环境和模型")
        
        # =====================================================================
        # 2. 打印训练开始信息
        # =====================================================================
        opponent_info = self.naming.format_opponent_info(self.opponent_config)
        print_training_start(
            algo=self.train_algo,
            side=self.train_side,
            version=self.version,
            opponent_info=opponent_info
        )
        
        self.logger.log_banner(f"🚀 开始训练 {self.train_algo}_{self.train_side}_{self.version}", "=")
        
        # =====================================================================
        # 3. 创建回调
        # =====================================================================
        callbacks = create_callbacks(
            train_side=self.train_side,
            checkpoint_path=self.checkpoint_dir,
            eval_env=self.eval_env,
            config=self.training_config,
            on_freeze=None  # 可以添加冻结回调
        )
        
        # =====================================================================
        # 4. 执行训练
        # =====================================================================
        self.training_start_time = time.time()
        
        try:
            self.algorithm.train(
                env=self.train_env,
                total_timesteps=self.training_config['total_timesteps'],
                callback=callbacks
            )
            
            self.training_end_time = time.time()
            self.total_training_time = self.training_end_time - self.training_start_time
            
        except KeyboardInterrupt:
            self.logger.warning("\n⚠️  训练被用户中断")
            self.training_end_time = time.time()
            self.total_training_time = self.training_end_time - self.training_start_time
            raise
        
        except Exception as e:
            self.logger.error(f"\n❌ 训练过程中发生错误: {e}")
            raise
        
        # =====================================================================
        # 5. 打印训练完成信息
        # =====================================================================
        print_training_complete(
            algo=self.train_algo,
            side=self.train_side,
            total_steps=self.training_config['total_timesteps'],
            time_elapsed=self.total_training_time
        )
        
        self.logger.log_banner("✅ 训练完成", "=")
    
    def save_model(self, save_to_pool: bool = False, pool_name: Optional[str] = None):
        """
        保存最终模型
        
        Args:
            save_to_pool: 是否保存到固定池
            pool_name: 池名称（如 prey_pool_v1）
        """
        
        if not self.training_config['save_final_model']:
            self.logger.info("配置禁用了模型保存，跳过")
            return
        
        self.logger.log_banner("💾 保存模型", "-")
        
        # =====================================================================
        # 1. 保存到模型目录
        # =====================================================================
        opponent_info = self.naming.format_opponent_info(self.opponent_config)
        
        model_filename = self.naming.generate_model_filename(
            train_algo=self.train_algo,
            train_side=self.train_side,
            version=self.version,
            opponent_info=opponent_info,
            run_mode=self.run_mode
        )
        
        model_path = self.model_dir / model_filename
        self.algorithm.save(str(model_path))
        self.final_model_path = model_path
        
        self.logger.info(f"✓ 模型已保存: {model_path}")
        
        # =====================================================================
        # 2. 保存到固定池（如果需要）
        # =====================================================================
        if save_to_pool and pool_name:
            pool_dir = self.path_manager.get_fixed_pool_dir(pool_name)
            
            pool_model_filename = f"{self.train_algo}_{self.train_side}_{self.version}.zip"
            pool_model_path = pool_dir / pool_model_filename
            
            # 复制模型到池
            import shutil
            shutil.copy(str(model_path), str(pool_model_path))
            
            self.logger.info(f"✓ 模型已加入固定池: {pool_model_path}")
            
            # 更新池的metadata
            self._update_pool_metadata(pool_dir, pool_model_filename)
        
        self.logger.log_banner("", "-")
    
    def _update_pool_metadata(self, pool_dir: Path, model_filename: str):
        """更新固定池的metadata"""
        metadata_path = pool_dir / "metadata.json"
        
        # 加载现有metadata
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                'pool_name': pool_dir.name,
                'created_at': datetime.now().isoformat(),
                'models': []
            }
        
        # 添加新模型信息
        model_info = {
            'name': model_filename.replace('.zip', ''),
            'path': model_filename,
            'algorithm': self.train_algo,
            'training_steps': self.training_config['total_timesteps'],
            'trained_against': self.naming.format_opponent_info(self.opponent_config),
            'added_at': datetime.now().isoformat(),
            'eval_metrics': {}  # 可以在评估后填充
        }
        
        # 检查是否已存在
        existing = [m for m in metadata['models'] if m['name'] == model_info['name']]
        if existing:
            # 更新现有条目
            idx = metadata['models'].index(existing[0])
            metadata['models'][idx] = model_info
        else:
            # 添加新条目
            metadata['models'].append(model_info)
        
        # 保存metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def evaluate(self, n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            n_episodes: 评估episode数（None则使用配置默认值）
        
        Returns:
            评估结果字典
        """
        from stable_baselines3.common.evaluation import evaluate_policy
        
        if self.eval_env is None:
            self.logger.warning("评估环境未创建，无法评估")
            return {}
        
        n_episodes = n_episodes or self.training_config['n_eval_episodes']
        
        self.logger.log_banner(f"📊 评估模型 ({n_episodes} episodes)", "-")
        
        episode_rewards, episode_lengths = evaluate_policy(
            self.algorithm.model,
            self.eval_env,
            n_eval_episodes=n_episodes,
            deterministic=self.training_config['deterministic_eval'],
            return_episode_rewards=True
        )
        
        results = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'std_length': float(np.std(episode_lengths)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'n_episodes': n_episodes
        }
        
        self.logger.log_config(results, title="评估结果")
        self.logger.log_banner("", "-")
        
        return results
    
    def check_freeze_criteria(self, freeze_criteria: Dict[str, Any]) -> bool:
        """
        检查是否达到冻结条件
        
        Args:
            freeze_criteria: 冻结条件字典
        
        Returns:
            是否达到冻结条件
        """
        # 先评估模型
        eval_results = self.evaluate()
        
        if not eval_results:
            return False
        
        # 检查最低奖励
        min_reward = freeze_criteria.get('min_avg_reward', -np.inf)
        if eval_results['mean_reward'] < min_reward:
            self.logger.info(f"❌ 未达到最低奖励: {eval_results['mean_reward']:.2f} < {min_reward:.2f}")
            return False
        
        # 角色特定检查（简化版，完整版需要在回调中实现）
        if self.train_side == "predator":
            min_catch_rate = freeze_criteria.get('min_catch_rate', 0.0)
            # 这里需要从评估中获取catch_rate，暂时用奖励代替
            self.logger.info(f"✓ 达到最低奖励: {eval_results['mean_reward']:.2f} >= {min_reward:.2f}")
        
        elif self.train_side == "prey":
            min_survival_rate = freeze_criteria.get('min_survival_rate', 0.0)
            # 这里需要从评估中获取survival_rate，暂时用奖励代替
            self.logger.info(f"✓ 达到最低奖励: {eval_results['mean_reward']:.2f} >= {min_reward:.2f}")
        
        return True
    
    def save_training_summary(self):
        """保存训练摘要"""
        summary = {
            'experiment_name': self.experiment_name,
            'stage_name': self.stage_name,
            'train_side': self.train_side,
            'train_algo': self.train_algo,
            'version': self.version,
            'generation': self.generation,
            'run_mode': self.run_mode,
            'opponent': self.naming.format_opponent_info(self.opponent_config),
            'training_config': self.training_config,
            'training_time_seconds': self.total_training_time,
            'final_model_path': str(self.final_model_path) if self.final_model_path else None,
            'notes': self.notes
        }
        
        summary_filename = self.naming.generate_summary_filename(
            train_algo=self.train_algo,
            train_side=self.train_side,
            version=self.version
        )
        
        save_training_summary(
            summary=summary,
            save_dir=self.experiment_dir,
            name=summary_filename.replace('.json', '')
        )
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("清理资源...")
        
        if self.train_env:
            self.train_env.close()
        
        if self.eval_env:
            self.eval_env.close()
        
        if self.env_manager:
            self.env_manager.close()
        
        self.logger.info("✓ 资源清理完成")
    
    def run(
        self,
        save_to_pool: bool = False,
        pool_name: Optional[str] = None,
        check_freeze: bool = False,
        freeze_criteria: Optional[Dict[str, Any]] = None
    ):
        """
        完整训练流程（setup → train → evaluate → save）
        
        Args:
            save_to_pool: 是否保存到固定池
            pool_name: 池名称
            check_freeze: 是否检查冻结条件
            freeze_criteria: 冻结条件
        """
        try:
            # 1. 设置
            self.setup()
            
            # 2. 训练
            self.train()
            
            # 3. 最终评估
            final_eval = self.evaluate()
            
            # 4. 检查冻结条件
            can_freeze = True
            if check_freeze and freeze_criteria:
                can_freeze = self.check_freeze_criteria(freeze_criteria)
                
                if can_freeze:
                    self.logger.log_banner("❄️  模型达到冻结标准", "=")
                else:
                    self.logger.log_banner("⚠️  模型未达到冻结标准", "=")
            
            # 5. 保存模型
            should_save_to_pool = save_to_pool and (not check_freeze or can_freeze)
            self.save_model(save_to_pool=should_save_to_pool, pool_name=pool_name)
            
            # 6. 保存训练摘要
            self.save_training_summary()
            
            return final_eval
        
        finally:
            # 7. 清理
            self.cleanup()


# 导入numpy（前面忘记了）
import numpy as np
"""
对手池管理
负责固定对手的加载、采样和维护
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import gymnasium as gym  # ← 添加这一行！
from src.algorithms import create_algorithm
from src.utils.config_loader import get_algo_config


class OpponentPool:
    """对手池管理器"""
    
    def __init__(self, pool_dir: Optional[Path] = None):
        """
        初始化对手池
        
        Args:
            pool_dir: 池目录路径（如果为None则创建空池）
        """
        self.pool_dir = pool_dir
        self.opponents = []  # 对手列表：[{name, path, policy, metadata}, ...]
        self.metadata = {}
        
        if pool_dir and pool_dir.exists():
            self.load_pool()
    
    def load_pool(self):
        """从目录加载对手池"""
        if not self.pool_dir.exists():
            print(f"⚠️  对手池目录不存在: {self.pool_dir}")
            return
        
        # 加载元数据
        metadata_path = self.pool_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        
        # 加载所有模型
        for model_file in self.pool_dir.glob("*.zip"):
            opponent_info = self._load_opponent(model_file)
            if opponent_info:
                self.opponents.append(opponent_info)
        
        print(f"✅ 从 {self.pool_dir} 加载了 {len(self.opponents)} 个对手")
    
    def _load_opponent(self, model_path: Path) -> Optional[Dict[str, Any]]:
        """
        加载单个对手
        
        Args:
            model_path: 模型文件路径
        
        Returns:
            对手信息字典，加载失败返回None
        """
        try:
            # 从文件名解析信息
            filename = model_path.stem  # 去掉.zip
            
            # 假设文件名格式: ALGO_side_version
            parts = filename.split('_')
            if len(parts) < 3:
                print(f"⚠️  无法解析文件名: {filename}")
                return None
            
            algo_name = parts[0]
            side = parts[1]
            version = parts[2]
            
            # 从metadata中获取详细信息
            model_metadata = {}
            for model_info in self.metadata.get('models', []):
                if model_info.get('name') == filename:
                    model_metadata = model_info
                    break
            
            # 加载算法配置
            try:
                algo_config = get_algo_config(algo_name)
            except:
                print(f"⚠️  无法加载算法配置: {algo_name}")
                return None
            
            # 创建算法实例（暂不加载模型，延迟加载）
            opponent_info = {
                'name': filename,
                'algo': algo_name,
                'side': side,
                'version': version,
                'path': model_path,
                'policy': None,  # 延迟加载
                'config': algo_config,
                'metadata': model_metadata
            }
            
            return opponent_info
        
        except Exception as e:
            print(f"❌ 加载对手失败 {model_path}: {e}")
            return None
    
    def get_opponent_policy(
        self, 
        opponent_info: Dict[str, Any], 
        device: str = "auto",
        observation_space: gym.Space = None,
        action_space: gym.Space = None
    ):
        """
        获取对手策略（延迟加载）
        
        Args:
            opponent_info: 对手信息
            device: 计算设备
        
        Returns:
            加载的策略
        """
        # 如果已经加载过，直接返回
        if opponent_info['policy'] is not None:
            return opponent_info['policy']
        
        # ✅ 创建算法实例（使用传入的空间信息）
        algo = create_algorithm(
            algo_name=opponent_info['algo'],
            observation_space=observation_space,  # 使用传入的参数
            action_space=action_space,             # 使用传入的参数
            config=opponent_info['config'],
            device=device
        )
        
        # 加载模型权重
        algo.load(str(opponent_info['path']))
        
        # 缓存
        opponent_info['policy'] = algo
        
        return algo
    
    def sample_opponents(
        self,
        n: int,
        strategy: str = "uniform",
        exclude: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        从池中采样对手
        
        Args:
            n: 采样数量
            strategy: 采样策略（uniform/weighted）
            exclude: 要排除的对手名称列表
        
        Returns:
            对手信息列表
        """
        if not self.opponents:
            print("⚠️  对手池为空，无法采样")
            return []
        
        # 过滤要排除的对手
        available = self.opponents
        if exclude:
            available = [opp for opp in self.opponents if opp['name'] not in exclude]
        
        if not available:
            print("⚠️  没有可用的对手")
            return []
        
        # 采样
        if strategy == "uniform":
            # 均匀采样（可重复）
            sampled = random.choices(available, k=min(n, len(available)))
        
        elif strategy == "weighted":
            # 按性能加权采样（性能越好权重越高）
            weights = []
            for opp in available:
                # 从metadata中获取性能指标
                perf = opp['metadata'].get('eval_metrics', {})
                # 使用平均奖励作为权重
                weight = perf.get('avg_reward', 1.0) + 5.0  # +5确保权重为正
                weights.append(max(weight, 0.1))
            
            sampled = random.choices(available, weights=weights, k=min(n, len(available)))
        
        else:
            raise ValueError(f"未知的采样策略: {strategy}")
        
        return sampled
    
    def add_opponent(
        self,
        model_path: Path,
        metadata: Dict[str, Any]
    ):
        """
        添加新对手到池中
        
        Args:
            model_path: 模型文件路径
            metadata: 对手元数据
        """
        opponent_info = self._load_opponent(model_path)
        if opponent_info:
            opponent_info['metadata'] = metadata
            self.opponents.append(opponent_info)
            
            # 更新池的metadata文件
            self._update_metadata()
            
            print(f"✅ 添加对手到池: {opponent_info['name']}")
    
    def _update_metadata(self):
        """更新池的metadata文件"""
        if not self.pool_dir:
            return
        
        self.metadata['models'] = []
        for opp in self.opponents:
            self.metadata['models'].append({
                'name': opp['name'],
                'algo': opp['algo'],
                'side': opp['side'],
                'version': opp['version'],
                'path': str(opp['path']),
                'metadata': opp['metadata']
            })
        
        metadata_path = self.pool_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def size(self) -> int:
        """获取池大小"""
        return len(self.opponents)
    
    def get_all_opponents(self) -> List[Dict[str, Any]]:
        """获取所有对手"""
        return self.opponents.copy()
    
    def clear(self):
        """清空对手池（仅清空内存，不删除文件）"""
        self.opponents = []


class MixedOpponentSampler:
    """混合对手采样器（固定池 + 随机）"""
    
    def __init__(
        self,
        fixed_pool: OpponentPool,
        random_policy_creator,
        fixed_ratio: float = 0.7,
        sampling_strategy: str = "uniform"
    ):
        """
        初始化混合采样器
        
        Args:
            fixed_pool: 固定对手池
            random_policy_creator: 创建随机策略的函数
            fixed_ratio: 固定对手占比
            sampling_strategy: 采样策略
        """
        self.fixed_pool = fixed_pool
        self.random_policy_creator = random_policy_creator
        self.fixed_ratio = fixed_ratio
        self.sampling_strategy = sampling_strategy
    
    def sample(self, n: int) -> List[Dict[str, Any]]:
        """
        采样混合对手
        
        Args:
            n: 总对手数量
        
        Returns:
            对手列表（包含固定对手和随机对手）
        """
        n_fixed = int(n * self.fixed_ratio)
        n_random = n - n_fixed
        
        opponents = []
        
        # 从固定池采样
        if n_fixed > 0 and self.fixed_pool.size() > 0:
            fixed_opponents = self.fixed_pool.sample_opponents(
                n=n_fixed,
                strategy=self.sampling_strategy
            )
            opponents.extend(fixed_opponents)
        
        # 添加随机对手
        for i in range(n_random):
            random_opp = {
                'name': f'RANDOM_{i}',
                'algo': 'RANDOM',
                'policy': self.random_policy_creator(),
                'is_random': True
            }
            opponents.append(random_opp)
        
        return opponents


def create_opponent_policies(
    opponent_config: Dict[str, Any],
    env_manager,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    根据配置创建对手策略
    
    Args:
        opponent_config: 对手配置
        env_manager: 环境管理器（用于获取空间信息）
        device: 计算设备
    
    Returns:
        对手策略字典 {agent_id: policy}
    """
    opp_type = opponent_config.get('type', 'algorithm')
    opp_side = opponent_config.get('side', 'predator')
    
    # 获取对手智能体列表
    opponent_agents = env_manager.get_agents_by_type(opp_side)
    
    # 获取空间信息
    obs_space = env_manager.get_observation_space(opp_side)
    action_space = env_manager.get_action_space(opp_side)
    
    policies = {}
    
    if opp_type == "algorithm":
        # 使用指定算法（通常是RANDOM）
        algo_name = opponent_config.get('algorithm', 'RANDOM')
        algo_config = get_algo_config(algo_name)
        
        # 为所有对手创建相同的策略
        policy = create_algorithm(
            algo_name=algo_name,
            observation_space=obs_space,
            action_space=action_space,
            config=algo_config,
            device=device
        )
        
        # 如果是RANDOM，需要创建模型
        if algo_name == 'RANDOM':
            policy.create_model(None)
        
        for agent in opponent_agents:
            policies[agent] = policy
    
    elif opp_type == "fixed_model":
        # 加载固定模型
        model_path = opponent_config.get('path')
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"对手模型不存在: {model_path}")
        
        # 从路径解析算法名称
        filename = Path(model_path).stem
        algo_name = filename.split('_')[0]
        algo_config = get_algo_config(algo_name)
        
        policy = create_algorithm(
            algo_name=algo_name,
            observation_space=obs_space,
            action_space=action_space,
            config=algo_config,
            device=device
        )
        policy.load(model_path)
        
        for agent in opponent_agents:
            policies[agent] = policy
    
    elif opp_type == "mixed_pool":
        # 从混合池采样
        pool_path = Path(opponent_config.get('pool_path', ''))
        
        if not pool_path.exists():
            print(f"⚠️  对手池不存在: {pool_path}，使用RANDOM策略")
            # 回退到RANDOM
            algo_config = get_algo_config('RANDOM')
            policy = create_algorithm(
                algo_name='RANDOM',
                observation_space=obs_space,
                action_space=action_space,
                config=algo_config,
                device=device
            )
            policy.create_model(None)
            
            for agent in opponent_agents:
                policies[agent] = policy
        else:
            # 加载池
            pool = OpponentPool(pool_path)
            
            # 创建混合采样器
            def create_random():
                algo_config = get_algo_config('RANDOM')
                policy = create_algorithm(
                    algo_name='RANDOM',
                    observation_space=obs_space,
                    action_space=action_space,
                    config=algo_config,
                    device=device
                )
                policy.create_model(None)
                return policy
            
            mix_strategy = opponent_config.get('mix_strategy', {})
            sampler = MixedOpponentSampler(
                fixed_pool=pool,
                random_policy_creator=create_random,
                fixed_ratio=mix_strategy.get('fixed_ratio', 0.7),
                sampling_strategy=mix_strategy.get('sampling', 'uniform')
            )
            
            # 采样对手
            n_opponents = len(opponent_agents)
            sampled_opponents = sampler.sample(n_opponents)
            
            # 分配给智能体
            for i, agent in enumerate(opponent_agents):
                opp = sampled_opponents[i % len(sampled_opponents)]
                
                if opp.get('is_random', False):
                    # 随机对手
                    policies[agent] = opp['policy']
                else:
                    # ✅ 固定池对手（传入空间信息）
                    loaded_policy = pool.get_opponent_policy(
                        opp, 
                        device,
                        obs_space,
                        action_space
                    )
                    policies[agent] = loaded_policy
    
    else:
        raise ValueError(f"未知的对手类型: {opp_type}")
    
    return policies
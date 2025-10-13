"""
指标计算器
计算交叉评估中的各种性能指标
"""

import numpy as np
from typing import Dict, List, Any


class MetricsCalculator:
    """评估指标计算器"""
    
    @staticmethod
    def compute_episode_metrics(episode_data: Dict) -> Dict[str, float]:
        """
        计算单个episode的指标
        
        Args:
            episode_data: {
                'predator_rewards': [...],
                'prey_rewards': [...],
                'predator_dones': [...],
                'prey_dones': [...],
                'episode_length': int
            }
        
        Returns:
            指标字典
        """
        metrics = {}
        
        # Predator指标
        pred_rewards = np.array(episode_data['predator_rewards'])
        pred_dones = np.array(episode_data['predator_dones'])
        
        metrics['pred_total_reward'] = float(np.sum(pred_rewards))
        metrics['pred_avg_reward'] = float(np.mean(pred_rewards))
        
        # 捕获事件（假设prey死亡时predator获得正奖励）
        prey_dones = np.array(episode_data['prey_dones'])
        metrics['n_catches'] = int(np.sum(prey_dones))
        
        # Prey指标
        prey_rewards = np.array(episode_data['prey_rewards'])
        metrics['prey_total_reward'] = float(np.sum(prey_rewards))
        metrics['prey_avg_reward'] = float(np.mean(prey_rewards))
        
        # Episode级指标
        metrics['episode_length'] = episode_data['episode_length']
        metrics['reward_gap'] = metrics['pred_total_reward'] - metrics['prey_total_reward']
        
        return metrics
    
    @staticmethod
    def aggregate_metrics(episode_metrics_list: List[Dict]) -> Dict[str, Any]:
        """
        聚合多个episode的指标
        
        Args:
            episode_metrics_list: 多个episode的指标列表
        
        Returns:
            聚合后的指标
        """
        n_episodes = len(episode_metrics_list)
        
        # 提取各指标的数组
        pred_rewards = [m['pred_avg_reward'] for m in episode_metrics_list]
        prey_rewards = [m['prey_avg_reward'] for m in episode_metrics_list]
        n_catches = [m['n_catches'] for m in episode_metrics_list]
        episode_lengths = [m['episode_length'] for m in episode_metrics_list]
        
        # 聚合统计
        aggregated = {
            'n_episodes': n_episodes,
            
            # Predator指标
            'pred_avg_reward': float(np.mean(pred_rewards)),
            'pred_std_reward': float(np.std(pred_rewards)),
            'pred_min_reward': float(np.min(pred_rewards)),
            'pred_max_reward': float(np.max(pred_rewards)),
            
            # Prey指标
            'prey_avg_reward': float(np.mean(prey_rewards)),
            'prey_std_reward': float(np.std(prey_rewards)),
            'prey_min_reward': float(np.min(prey_rewards)),
            'prey_max_reward': float(np.max(prey_rewards)),
            
            # 对战指标
            'catch_rate': float(np.mean(n_catches)) / 10.0,  # 假设10个prey
            'avg_catches': float(np.mean(n_catches)),
            'std_catches': float(np.std(n_catches)),
            
            'survival_rate': 1.0 - (float(np.mean(n_catches)) / 10.0),
            
            'avg_episode_length': float(np.mean(episode_lengths)),
            'std_episode_length': float(np.std(episode_lengths)),
            
            # 平衡度指标
            'reward_gap': float(np.mean([m['reward_gap'] for m in episode_metrics_list])),
            'balance_score': MetricsCalculator._compute_balance_score(n_catches)
        }
        
        # 额外指标
        aggregated['energy_efficiency'] = (
            aggregated['pred_avg_reward'] / (aggregated['avg_episode_length'] + 1e-6)
        )
        
        aggregated['escape_success'] = aggregated['survival_rate']
        
        # 首次捕获时间（简化版，假设第一次捕获发生在episode中间）
        aggregated['first_catch_time'] = aggregated['avg_episode_length'] / 2.0
        
        # Prey平均寿命
        aggregated['prey_avg_lifespan'] = aggregated['avg_episode_length'] * aggregated['survival_rate']
        
        return aggregated
    
    @staticmethod
    def _compute_balance_score(n_catches_list: List[int]) -> float:
        """
        计算平衡度分数
        
        0.5 = 完美平衡（catch_rate = 50%）
        0.0 = 极度不平衡
        
        Args:
            n_catches_list: 每个episode的捕获数
        
        Returns:
            平衡度分数 [0, 1]
        """
        avg_catches = np.mean(n_catches_list)
        catch_rate = avg_catches / 10.0  # 假设10个prey
        
        # 距离0.5越近，平衡度越高
        deviation = abs(catch_rate - 0.5)
        balance_score = 1.0 - (deviation * 2.0)  # 归一化到[0, 1]
        
        return float(max(0.0, balance_score))
    
    @staticmethod
    def compute_adaptability_scores(results_matrix: Dict) -> Dict[str, Dict]:
        """
        计算所有算法的自适应性得分
        
        Args:
            results_matrix: {
                'PPO': {'PPO': {...}, 'A2C': {...}, ...},
                'A2C': {...},
                ...
            }
        
        Returns:
            自适应性得分字典
        """
        adaptability_scores = {}
        
        for pred_algo in results_matrix.keys():
            if pred_algo == 'RANDOM':
                continue
            
            # In-Distribution性能（对角线）
            in_dist_perf = results_matrix[pred_algo][pred_algo]['catch_rate']
            
            # Out-of-Distribution性能（非对角线，排除RANDOM）
            ood_perfs = []
            for prey_algo in results_matrix[pred_algo].keys():
                if prey_algo != pred_algo and prey_algo != 'RANDOM':
                    ood_perfs.append(results_matrix[pred_algo][prey_algo]['catch_rate'])
            
            ood_avg = float(np.mean(ood_perfs))
            ood_std = float(np.std(ood_perfs))
            
            # 自适应性得分 = OOD保持率
            adaptability = ood_avg / (in_dist_perf + 1e-6)
            
            adaptability_scores[pred_algo] = {
                'algorithm': pred_algo,
                'in_dist_performance': float(in_dist_perf),
                'ood_avg_performance': ood_avg,
                'ood_std': ood_std,
                'adaptability_score': float(adaptability),
                'performance_drop': float(in_dist_perf - ood_avg),
                'ood_performances': ood_perfs  # 原始数据
            }
        
        return adaptability_scores
    
    @staticmethod
    def compute_ranking(adaptability_scores: Dict) -> List[Dict]:
        """
        根据自适应性得分排名
        
        Returns:
            排序后的列表
        """
        scores_list = list(adaptability_scores.values())
        scores_list.sort(key=lambda x: x['adaptability_score'], reverse=True)
        
        # 添加排名
        for i, scores in enumerate(scores_list):
            scores['rank'] = i + 1
        
        return scores_list
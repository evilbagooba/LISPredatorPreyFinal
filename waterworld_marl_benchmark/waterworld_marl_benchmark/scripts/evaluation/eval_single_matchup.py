"""
单次对战评估
评估一对 Predator vs Prey 的性能
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.environment import WaterworldEnvManager, create_training_env
from src.core.opponent_pool import create_opponent_policies
from src.core.agent_manager import AgentManager
from src.utils.config_loader import get_env_config


def evaluate_single_matchup(
    predator_model_path: Path,
    prey_model_path: Path,
    predator_algo: str,
    prey_algo: str,
    env_config_name: str = "waterworld_fast",
    n_episodes: int = 20,
    deterministic: bool = True,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    评估单次Predator vs Prey对战
    
    Args:
        predator_model_path: Predator模型路径
        prey_model_path: Prey模型路径
        predator_algo: Predator算法名称
        prey_algo: Prey算法名称
        env_config_name: 环境配置名称
        n_episodes: 评估episode数
        deterministic: 是否使用确定性策略
        verbose: 详细程度
    
    Returns:
        评估指标字典
    """
    
    if verbose > 0:
        print(f"\n{'='*70}")
        print(f"Evaluating: {predator_algo}_pred vs {prey_algo}_prey")
        print(f"{'='*70}")
        # ✅ 修复：处理 None 的情况
        pred_model_name = predator_model_path.name if predator_model_path else "RANDOM (no model)"
        prey_model_name = prey_model_path.name if prey_model_path else "RANDOM (no model)"
        print(f"  Predator Model: {pred_model_name}")
        print(f"  Prey Model:     {prey_model_name}")
        print(f"  Episodes:       {n_episodes}")
    
    # 1. 加载环境配置
    env_config = get_env_config(env_config_name)
    env_manager = WaterworldEnvManager(env_config)
    env_manager.create_env()
    
    # 2. 获取空间信息
    pred_obs_space = env_manager.get_observation_space('predator')
    pred_action_space = env_manager.get_action_space('predator')
    
    prey_obs_space = env_manager.get_observation_space('prey')
    prey_action_space = env_manager.get_action_space('prey')
    
    # 3. 加载模型
    try:
        if predator_algo == 'RANDOM':
            predator_model = AgentManager.create_random_agent(
                pred_obs_space, pred_action_space
            )
        else:
            predator_model = AgentManager.load_agent(
                predator_model_path, pred_obs_space, pred_action_space
            )
        
        if prey_algo == 'RANDOM':
            prey_model = AgentManager.create_random_agent(
                prey_obs_space, prey_action_space
            )
        else:
            prey_model = AgentManager.load_agent(
                prey_model_path, prey_obs_space, prey_action_space
            )
    
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 4. 创建评估环境（单环境，便于精确控制）
    # 注意：这里我们需要一个特殊的评估环境，双方都用固定策略
    
    # 5. 运行评估
    episode_results = []
    
    for ep in range(n_episodes):
        obs, info = env_manager.reset(seed=42 + ep)
        done = False
        step = 0
        
        ep_data = {
            'predator_rewards': [],
            'prey_rewards': [],
            'predator_dones': [],
            'prey_dones': [],
            'episode_length': 0
        }
        
        while not done and step < 500:  # 最多500步
            # 获取所有智能体的动作
            actions = {}
            
            for agent_id in env_manager.env.agents:
                if 'predator' in agent_id:
                    obs_agent = obs.get(agent_id, np.zeros(pred_obs_space.shape))
                    action, _ = predator_model.predict(obs_agent, deterministic=deterministic)
                    actions[agent_id] = action
                
                elif 'prey' in agent_id:
                    obs_agent = obs.get(agent_id, np.zeros(prey_obs_space.shape))
                    action, _ = prey_model.predict(obs_agent, deterministic=deterministic)
                    actions[agent_id] = action
            
            # 执行环境步进
            obs, rewards, terminations, truncations, infos = env_manager.env.step(actions)
            
            # 记录数据
            pred_rewards = [rewards.get(a, 0) for a in env_manager.env.possible_agents if 'predator' in a]
            prey_rewards = [rewards.get(a, 0) for a in env_manager.env.possible_agents if 'prey' in a]
            
            ep_data['predator_rewards'].extend(pred_rewards)
            ep_data['prey_rewards'].extend(prey_rewards)
            
            # 记录死亡事件
            pred_dones = [terminations.get(a, False) for a in env_manager.env.possible_agents if 'predator' in a]
            prey_dones = [terminations.get(a, False) for a in env_manager.env.possible_agents if 'prey' in a]
            
            ep_data['predator_dones'].extend(pred_dones)
            ep_data['prey_dones'].extend(prey_dones)
            
            step += 1
            
            # 检查是否所有智能体都结束
            if len(env_manager.env.agents) == 0:
                done = True
        
        ep_data['episode_length'] = step
        episode_results.append(ep_data)
        
        if verbose > 1:
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"Length={step}, "
                  f"PredReward={np.mean(ep_data['predator_rewards']):.2f}, "
                  f"PreyReward={np.mean(ep_data['prey_rewards']):.2f}")
    
    # 6. 计算聚合指标
    from metrics_calculator import MetricsCalculator
    
    episode_metrics = [
        MetricsCalculator.compute_episode_metrics(ep_data)
        for ep_data in episode_results
    ]
    
    aggregated_metrics = MetricsCalculator.aggregate_metrics(episode_metrics)
    
    # 7. 添加元数据
    aggregated_metrics['predator_algo'] = predator_algo
    aggregated_metrics['prey_algo'] = prey_algo
    aggregated_metrics['is_ood'] = (predator_algo != prey_algo) and (prey_algo != 'RANDOM')
    
    # 8. 清理
    env_manager.close()
    
    if verbose > 0:
        print(f"\n  Results:")
        print(f"    Catch Rate:     {aggregated_metrics['catch_rate']:.3f}")
        print(f"    Survival Rate:  {aggregated_metrics['survival_rate']:.3f}")
        print(f"    Pred Reward:    {aggregated_metrics['pred_avg_reward']:+.2f}")
        print(f"    Prey Reward:    {aggregated_metrics['prey_avg_reward']:+.2f}")
        print(f"    Balance Score:  {aggregated_metrics['balance_score']:.3f}")
        print(f"    Is OOD:         {aggregated_metrics['is_ood']}")
        print(f"{'='*70}\n")
    
    return aggregated_metrics


if __name__ == "__main__":
    # 测试单次评估
    import argparse
    
    parser = argparse.ArgumentParser(description='单次对战评估')
    parser.add_argument('--pred-model', type=str, required=True, help='Predator模型路径')
    parser.add_argument('--prey-model', type=str, required=True, help='Prey模型路径')
    parser.add_argument('--pred-algo', type=str, required=True, help='Predator算法')
    parser.add_argument('--prey-algo', type=str, required=True, help='Prey算法')
    parser.add_argument('--n-episodes', type=int, default=20, help='评估episode数')
    
    args = parser.parse_args()
    
    result = evaluate_single_matchup(
        predator_model_path=Path(args.pred_model),
        prey_model_path=Path(args.prey_model),
        predator_algo=args.pred_algo,
        prey_algo=args.prey_algo,
        n_episodes=args.n_episodes
    )
    
    print("\n最终结果:")
    for key, value in result.items():
        print(f"  {key}: {value}")
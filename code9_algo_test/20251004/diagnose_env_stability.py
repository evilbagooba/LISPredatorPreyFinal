# diagnose_env_stability.py
"""
环境稳定性和数值健康诊断工具
检测环境在长时间运行下是否会崩溃、产生NaN、或状态不一致
"""

import numpy as np
import sys
from collections import defaultdict
from datetime import datetime
import json
import traceback

def diagnose_environment_stability(env_fn, n_episodes=50, max_steps=2000, stress_test=True):
    """
    全面的环境稳定性测试
    
    Args:
        env_fn: 创建环境的函数
        n_episodes: 测试episode数
        max_steps: 每个episode最大步数
        stress_test: 是否进行压力测试
    """
    
    print("=" * 70)
    print("ENVIRONMENT STABILITY DIAGNOSIS")
    print("=" * 70)
    
    issues = []
    stats = {
        'crashes': [],
        'nan_observations': [],
        'nan_rewards': [],
        'inf_values': [],
        'state_errors': [],
        'extreme_values': [],
    }
    
    # ========== 测试1: 基础稳定性 ==========
    print("\n[Test 1/4] Basic Stability Test")
    print("-" * 70)
    
    env = env_fn()
    
    for episode in range(n_episodes):
        try:
            env.reset(seed=episode)
            
            step = 0
            for agent in env.agent_iter(max_iter=max_steps):
                obs, reward, term, trunc, info = env.last()
                
                # 检查NaN
                if obs is not None:
                    if isinstance(obs, dict):
                        for k, v in obs.items():
                            if np.any(np.isnan(v)):
                                stats['nan_observations'].append({
                                    'episode': episode,
                                    'step': step,
                                    'agent': agent,
                                    'key': k
                                })
                    elif np.any(np.isnan(obs)):
                        stats['nan_observations'].append({
                            'episode': episode,
                            'step': step,
                            'agent': agent
                        })
                
                # 检查reward
                if reward is not None:
                    if np.isnan(reward):
                        stats['nan_rewards'].append({
                            'episode': episode,
                            'step': step,
                            'agent': agent,
                            'reward': reward
                        })
                    elif np.isinf(reward):
                        stats['inf_values'].append({
                            'episode': episode,
                            'step': step,
                            'agent': agent,
                            'value': float(reward),
                            'type': 'reward'
                        })
                    elif abs(reward) > 10000:
                        stats['extreme_values'].append({
                            'episode': episode,
                            'step': step,
                            'agent': agent,
                            'value': float(reward),
                            'type': 'reward'
                        })
                
                # 执行动作
                if term or trunc:
                    action = None
                else:
                    action = env.action_space(agent).sample()
                
                env.step(action)
                step += 1
            
            env.close()
            
            if (episode + 1) % 10 == 0:
                print(f"  ✓ Completed {episode + 1}/{n_episodes} episodes")
                
        except Exception as e:
            stats['crashes'].append({
                'episode': episode,
                'step': step,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            print(f"  ✗ Episode {episode} crashed at step {step}: {e}")
            env = env_fn()  # 重新创建环境
    
    # ========== 测试2: 连续长时间运行 ==========
    print(f"\n[Test 2/4] Continuous Long Run Test ({max_steps * 3} steps)")
    print("-" * 70)
    
    env = env_fn()
    try:
        env.reset(seed=999)
        
        total_steps = max_steps * 3
        for i, agent in enumerate(env.agent_iter(max_iter=total_steps)):
            obs, reward, term, trunc, info = env.last()
            
            if term or trunc:
                action = None
            else:
                action = env.action_space(agent).sample()
            
            env.step(action)
            
            if (i + 1) % 1000 == 0:
                print(f"  ✓ Step {i + 1}/{total_steps}")
        
        env.close()
        print("  ✓ Long run completed successfully")
    except Exception as e:
        stats['crashes'].append({
            'test': 'long_run',
            'step': i,
            'error': str(e),
            'traceback': traceback.format_exc()
        })
        print(f"  ✗ Long run crashed at step {i}: {e}")
    
    # ========== 测试3: 快速重置测试 ==========
    print(f"\n[Test 3/4] Rapid Reset Test (100 quick resets)")
    print("-" * 70)
    
    env = env_fn()
    try:
        for i in range(100):
            env.reset(seed=i)
            # 只执行几步就重置
            for _ in range(10):
                for agent in env.agent_iter(max_iter=10):
                    obs, reward, term, trunc, info = env.last()
                    if term or trunc:
                        break
                    action = env.action_space(agent).sample()
                    env.step(action)
                break  # 只做一轮
            
            if (i + 1) % 20 == 0:
                print(f"  ✓ Reset {i + 1}/100")
        
        env.close()
        print("  ✓ Rapid reset test passed")
    except Exception as e:
        stats['state_errors'].append({
            'test': 'rapid_reset',
            'iteration': i,
            'error': str(e)
        })
        print(f"  ✗ Rapid reset failed at iteration {i}: {e}")
    
    # ========== 测试4: 压力测试（可选）==========
    if stress_test:
        print(f"\n[Test 4/4] Stress Test (parallel episodes simulation)")
        print("-" * 70)
        
        try:
            # 模拟多环境并行
            envs = [env_fn() for _ in range(5)]
            
            for episode in range(20):
                for env_idx, env in enumerate(envs):
                    env.reset(seed=episode * 5 + env_idx)
                    
                    for agent in env.agent_iter(max_iter=500):
                        obs, reward, term, trunc, info = env.last()
                        if term or trunc:
                            action = None
                        else:
                            action = env.action_space(agent).sample()
                        env.step(action)
                
                if (episode + 1) % 5 == 0:
                    print(f"  ✓ Parallel episode {episode + 1}/20")
            
            for env in envs:
                env.close()
            
            print("  ✓ Stress test passed")
        except Exception as e:
            stats['crashes'].append({
                'test': 'stress',
                'error': str(e)
            })
            print(f"  ✗ Stress test failed: {e}")
    
    # ========== 生成报告 ==========
    print("\n" + "=" * 70)
    print("DIAGNOSIS RESULTS")
    print("=" * 70)
    
    # 统计问题
    total_issues = sum([
        len(stats['crashes']),
        len(stats['nan_observations']),
        len(stats['nan_rewards']),
        len(stats['inf_values']),
        len(stats['state_errors']),
        len(stats['extreme_values'])
    ])
    
    if total_issues == 0:
        print("\n✓ ENVIRONMENT IS HEALTHY")
        print("  No crashes, NaN values, or state errors detected.")
        print("  Environment is safe for training.")
    else:
        print(f"\n⚠️  FOUND {total_issues} ISSUES")
        
        if stats['crashes']:
            print(f"\n🔥 CRASHES: {len(stats['crashes'])}")
            for i, crash in enumerate(stats['crashes'][:3]):
                print(f"  {i+1}. {crash.get('test', 'Episode ' + str(crash.get('episode', '?')))}")
                print(f"     Error: {crash['error']}")
        
        if stats['nan_observations']:
            print(f"\n⚠️  NaN OBSERVATIONS: {len(stats['nan_observations'])}")
            print("  Environment is producing NaN in observations!")
            print(f"  First occurrence: Episode {stats['nan_observations'][0]['episode']}, "
                  f"Step {stats['nan_observations'][0]['step']}")
        
        if stats['nan_rewards']:
            print(f"\n⚠️  NaN REWARDS: {len(stats['nan_rewards'])}")
            print("  Environment is producing NaN rewards!")
        
        if stats['inf_values']:
            print(f"\n⚠️  INF VALUES: {len(stats['inf_values'])}")
        
        if stats['extreme_values']:
            print(f"\n⚠️  EXTREME VALUES: {len(stats['extreme_values'])}")
            extreme = stats['extreme_values'][0]
            print(f"  Example: reward={extreme['value']} at episode {extreme['episode']}")
    
    # 保存详细报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"env_stability_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # 建议
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if stats['crashes']:
        print("\n1. FIX CRASHES:")
        print("   Your environment has critical bugs that cause crashes.")
        print("   Check the traceback in the detailed report.")
        print("   Common issues:")
        print("   - Body already removed (your current error)")
        print("   - Thread safety issues")
        print("   - State management bugs")
    
    if stats['nan_observations'] or stats['nan_rewards']:
        print("\n2. FIX NaN VALUES:")
        print("   Add checks in your environment:")
        print("   ```python")
        print("   reward = np.nan_to_num(reward, nan=0.0)")
        print("   observation = np.nan_to_num(observation, nan=0.0)")
        print("   ```")
    
    if stats['extreme_values']:
        print("\n3. CLIP EXTREME VALUES:")
        print("   Already discussed - clip rewards to [-10, 10]")
    
    if not total_issues:
        print("\n✓ Environment is stable and ready for training.")
        print("  The training crashes are likely due to:")
        print("  - Network instability (Actor/Critic)")
        print("  - Gradient explosion")
        print("  - Not environment bugs")
    
    return stats


if __name__ == "__main__":
    sys.path.append('.')
    
    import supersuit as ss
    from pettingzoo.sisl import waterworld_v4
    
    def create_test_env():
        """创建测试环境"""
        env = waterworld_v4.env(
            render_mode=None,
            n_predators=2,
            n_preys=2,
            n_evaders=20,
            n_obstacles=2,
            obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
            n_poisons=20,
            agent_algorithms=["PPO", "PPO", "Random", "Random"]
        )
        env = ss.black_death_v3(env)
        return env
    
    print("Starting comprehensive environment stability diagnosis...")
    print("This will take a few minutes...\n")
    
    results = diagnose_environment_stability(
        env_fn=create_test_env,
        n_episodes=50,
        max_steps=2000,
        stress_test=True
    )
    
    print("\n✓ Diagnosis complete!")
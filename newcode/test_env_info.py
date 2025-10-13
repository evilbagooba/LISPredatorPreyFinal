"""
测试 waterworld_v4 环境是否返回性能指标
"""

from pettingzoo.sisl import waterworld_v4
import numpy as np

def test_environment_info():
    """测试环境返回的info结构"""
    
    # 创建环境
    env = waterworld_v4.parallel_env(
        n_predators=2,
        n_preys=2,
        n_evaders=20,
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=10,
        agent_algorithms=["PPO", "PPO", "PPO", "PPO"],
        max_cycles=1000,
    )
    
    print("="*60)
    print("Testing Waterworld Environment Info")
    print("="*60)
    
    # 重置环境
    observations, infos = env.reset()
    
    print("\n1. After reset:")
    print(f"   Agents: {env.agents}")
    print(f"   Observation keys: {observations.keys()}")
    print(f"   Info keys: {infos.keys()}")
    
    # 检查初始info内容
    if infos:
        first_agent = list(infos.keys())[0]
        print(f"\n   Info for {first_agent}:")
        print(f"   Type: {type(infos[first_agent])}")
        print(f"   Content: {infos[first_agent]}")
    
    # 运行一个episode
    step_count = 0
    done = False
    
    while not done and step_count < 100:
        # 随机动作
        actions = {agent: env.action_space(agent).sample() 
                   for agent in env.agents}
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        step_count += 1
        done = any(terminations.values()) or any(truncations.values())
    
    print(f"\n2. After {step_count} steps:")
    print(f"   Terminations: {terminations}")
    print(f"   Truncations: {truncations}")
    
    # 检查step后的info
    if infos:
        print(f"\n3. Detailed Info Check:")
        for agent_name in list(infos.keys())[:2]:  # 检查前2个agent
            print(f"\n   Agent: {agent_name}")
            info = infos[agent_name]
            
            # 打印info的类型和内容
            print(f"   Type: {type(info)}")
            
            if isinstance(info, dict):
                print(f"   Keys: {info.keys()}")
                for key, value in info.items():
                    print(f"     - {key}: {value}")
            else:
                print(f"   Content: {info}")
            
            # 特别检查是否有性能指标相关的字段
            potential_keys = [
                'hunting_rate', 'escape_rate', 'foraging_rate',
                'performance', 'metrics', 'stats',
                'hunt_success', 'escape_success',
                'captures', 'escapes'
            ]
            
            print(f"\n   Checking for performance metrics:")
            if isinstance(info, dict):
                for key in potential_keys:
                    if key in info:
                        print(f"   ✅ Found: {key} = {info[key]}")
            
            # 检查是否有嵌套对象
            if isinstance(info, dict):
                for key, value in info.items():
                    if hasattr(value, '__dict__'):
                        print(f"\n   Object in '{key}':")
                        print(f"   Attributes: {dir(value)}")
    
    env.close()
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_environment_info()
import numpy as np
import supersuit as ss
from pettingzoo.sisl import waterworld_v4
from tianshou.env.pettingzoo_env import PettingZooEnv

def check_env_data():
    print("检查环境数据问题...")
    
    # 创建环境 - 与你的原代码完全相同
    agent_algos = ["SAC", "SAC", "SAC", "SAC"] * 50
    env = waterworld_v4.env(
        render_mode=None,
        n_predators=0,
        n_preys=2,
        n_evaders=50,
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=40,
        agent_algorithms=agent_algos
    )
    
    print("1. 原始环境测试:")
    obs = env.reset()  # PettingZoo格式
    print(f"   智能体: {env.agents}")
    print(f"   观测类型: {type(obs)}")
    
    if isinstance(obs, dict):
        print("   观测内容:")
        for agent, agent_obs in obs.items():
            print(f"     {agent}: {type(agent_obs)} - 形状: {getattr(agent_obs, 'shape', 'N/A')}")
            if isinstance(agent_obs, str):
                print(f"       ❌ 发现字符串: '{agent_obs}'")
    
    # 添加black_death包装器
    print("\n2. 添加black_death_v3包装器后:")
    env = ss.black_death_v3(env)
    obs = env.reset()  # PettingZoo格式
    print(f"   智能体: {env.agents}")
    print(f"   观测类型: {type(obs)}")
    
    if isinstance(obs, dict):
        print("   观测内容:")
        for agent, agent_obs in obs.items():
            print(f"     {agent}: {type(agent_obs)} - 形状: {getattr(agent_obs, 'shape', 'N/A')}")
            if isinstance(agent_obs, str):
                print(f"       ❌ 发现字符串: '{agent_obs}'")
            elif hasattr(agent_obs, 'dtype'):
                print(f"       数据类型: {agent_obs.dtype}")
    
    # 添加PettingZoo包装器
    print("\n3. 添加PettingZooEnv包装器后:")
    env = PettingZooEnv(env)
    
    # Tianshou包装器可能使用不同格式
    try:
        obs, info = env.reset()  # Tianshou可能返回tuple
    except:
        obs = env.reset()  # 或者只返回obs
        
    print(f"   智能体: {env.agents}")
    print(f"   观测类型: {type(obs)}")
    
    if isinstance(obs, dict):
        print("   观测内容:")
        for agent, agent_obs in obs.items():
            print(f"     {agent}: {type(agent_obs)}")
            if isinstance(agent_obs, str):
                print(f"       ❌ 发现字符串: '{agent_obs}'")
            elif hasattr(agent_obs, 'shape'):
                print(f"       形状: {agent_obs.shape}, 数据类型: {agent_obs.dtype}")
                # 检查是否有NaN
                if hasattr(agent_obs, 'shape') and len(agent_obs.shape) > 0:
                    has_nan = np.any(np.isnan(agent_obs))
                    print(f"       包含NaN: {has_nan}")
    
    print("\n4. 测试一步执行:")
    try:
        # 生成随机动作
        actions = {}
        for agent in env.agents:
            actions[agent] = np.random.uniform(-1, 1, 2).astype(np.float32)
        
        obs, rewards, terminations, truncations, info = env.step(actions)
        
        print(f"   执行成功")
        print(f"   返回类型: obs={type(obs)}, rewards={type(rewards)}")
        print(f"   terminations类型: {type(terminations)}")
        print(f"   truncations类型: {type(truncations)}")
        
        # 检查观测数据
        if isinstance(obs, dict):
            print("   步骤后观测:")
            for agent, agent_obs in obs.items():
                if isinstance(agent_obs, str):
                    print(f"     ❌ {agent}: 字符串 '{agent_obs}'")
                else:
                    print(f"     ✓ {agent}: 正常数值数据")
        
    except Exception as e:
        print(f"   ❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()

def test_without_black_death():
    print("\n" + "="*50)
    print("测试不使用black_death_v3的环境:")
    
    agent_algos = ["SAC", "SAC", "SAC", "SAC"] * 50
    env = waterworld_v4.env(
        render_mode=None,
        n_predators=0,
        n_preys=2,
        n_evaders=50,
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=40,
        agent_algorithms=agent_algos
    )
    
    # 直接用PettingZooEnv包装
    env = PettingZooEnv(env)
    
    print("1. 重置测试:")
    try:
        obs, info = env.reset()  # 尝试Tianshou格式
    except:
        obs = env.reset()  # 回退到PettingZoo格式
        
    print(f"   智能体: {env.agents}")
    
    string_found = False
    if isinstance(obs, dict):
        for agent, agent_obs in obs.items():
            if isinstance(agent_obs, str):
                print(f"   ❌ {agent}: 发现字符串 '{agent_obs}'")
                string_found = True
            else:
                print(f"   ✓ {agent}: 正常数据，形状 {agent_obs.shape}")
    
    if not string_found:
        print("   ✓ 未发现字符串数据")
    
    print("\n2. 步骤测试:")
    try:
        actions = {}
        for agent in env.agents:
            actions[agent] = np.random.uniform(-1, 1, 2).astype(np.float32)
        
        obs, rewards, terminations, truncations, info = env.step(actions)
        print("   ✓ 步骤执行成功")
        
        # 检查返回的数据类型
        print(f"   terminations: {type(terminations)} - {terminations}")
        print(f"   truncations: {type(truncations)} - {truncations}")
        
    except Exception as e:
        print(f"   ❌ 步骤执行失败: {e}")

if __name__ == "__main__":
    print("环境数据问题检测")
    print("="*50)
    
    # 检查完整包装链
    check_env_data()
    
    # 检查无black_death的情况
    test_without_black_death()
    
    print("\n总结:")
    print("如果看到字符串数据，那就是环境包装的问题")
    print("如果没有字符串数据，那问题可能在SAC的数据处理上")
# from pettingzoo.sisl import waterworld_v4
from pettingzoo.test import api_test
# import numpy as np
# agent_algos = ["PPO", "PPO", "DQN", "DQN","PPO", "PPO", "DQN", "DQN","DQN", "DQN"]
# algo_name_to_id = {name: idx for idx, name in enumerate(sorted(set(agent_algos)))}

# import supersuit as ss

# # env = waterworld_v4.env(render_mode="human",n_predators=5,n_preys=5,n_evaders=5,n_obstacles=1,n_poisons=1,agent_algorithms=agent_algos)
# # env.reset(seed=42)
# # api_test(env, num_cycles=1000, verbose_progress=True)
# # for agent in env.agent_iter():
# #     observation, reward, termination, truncation, info = env.last()
# #     print(reward)


# #     if termination or truncation:
# #         action = None
# #     else:
# #         # this is where you would insert your policy
# #         action = env.action_space(agent).sample()

# #     env.step(action)
# # env.close()

# # 调试环境结构
# from collections import defaultdict
# import supersuit as ss
# from pettingzoo.sisl import waterworld_v4

# # 1. 准备环境
# agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"] * 4
# env = waterworld_v4.env(
#     render_mode="human",
#     n_predators=5,
#     n_preys=15,
#     n_evaders=1,
#     n_obstacles=2,
#     obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
#     n_poisons=20,
#     agent_algorithms=agent_algos
# )
# env = ss.black_death_v3(env)
# obs = env.reset(seed=42)

# # 2. 初始化计数器
# violation_count = defaultdict(int)

# # 3. 迭代
# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
    
#     # —— 检查 reward 阈值 —— #
#     if reward is not None and reward < -10000:
#         violation_count[agent] += 1
#         if violation_count[agent] >= 2:
#             raise AssertionError(
#                 f"❗️ Agent `{agent}` 已第 {violation_count[agent]} 次获得 reward < -10000，触发错误！"
#             )
    
#     # 选择动作
#     if termination or truncation:
#         action = None
#     else:
#         action = env.action_space(agent).sample()
    
#     env.step(action)

# env.close()








# api_test(env, num_cycles=1000, verbose_progress=True)
#         






from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from collections import defaultdict

# 准备环境参数
agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"] * 40
env = waterworld_v4.env(
    render_mode="human",
    n_predators=4,
    n_preys=4,
    n_evaders=1,
    n_obstacles=2,
    obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
    n_poisons=20,
    agent_algorithms=agent_algos
)
# 黑死亡包装，会把死亡的 agent 移除
env = ss.black_death_v3(env)

# 只执行一个回合
print("\n=== Starting episode ===")
# 重置环境
obs = env.reset(seed=42)

# 用于累加每个 agent 的本轮总 reward
cumulative_rewards = defaultdict(float)

# 按 pettingzoo 规范的 agent_iter 刷新一轮
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    # 累加 reward（注意 reward 可能为 None）
    if reward is not None:
        cumulative_rewards[agent] += reward

    # 选择动作
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()

    env.step(action)

        # 每次执行后立即打印当前 agent 的累计 reward
    # if agent.startswith('predator') and 'current_health' in info:
    #     print(f"Agent {agent} health: {info['current_health']}")
    # print(observation)
    # print(info)

env.close()




# api_test(env, num_cycles=1000, verbose_progress=True)

#         












# from pettingzoo.sisl import waterworld_v4

# # 创建环境
# env = waterworld_v4.env(
#     n_predators=2,
#     n_preys=3,
#     agent_algorithms=["aggressive", "cooperative", "evasive", "smart", "default"],
#     max_cycles=100
# )

# env.reset()

# print("=== 开始测试 Info 功能 ===")

# # 运行几个步骤来测试
# for step in range(10):
#     print(f"\n--- Step {step + 1} ---")
    
#     for agent in env.agent_iter():
#         observation, reward, termination, truncation, info = env.last()
        
#         if termination or truncation:
#             action = None
#         else:
#             action = env.action_space(agent).sample()  # 随机动作
        
#         env.step(action)
        
#         # 打印 info 内容
#         if info:  # 只有在有 info 时才打印
#             print(f"Agent {agent}:")
#             print(f"  Health: {info.get('current_health', 'N/A')}")
#             print(f"  Alive: {info.get('is_alive', 'N/A')}")
#             print(f"  Frame: {info.get('current_frame', 'N/A')}")
#             print(f"  Type: {info.get('agent_type', 'N/A')}")
#             print(f"  Algorithm: {info.get('algorithm', 'N/A')}")
#             print(f"  Food caught: {info.get('food_caught', 'N/A')}")
#             print(f"  Poison contacted: {info.get('poison_contacted', 'N/A')}")
#             print(f"  Death cause: {info.get('death_cause', 'N/A')}")

# env.close()
# print("\n=== 测试完成 ===")

# # 外部分析函数示例
# def analyze_agent_performance(info_history):
#     """
#     分析智能体性能的示例函数
#     info_history: 每个步骤收集的 info 字典列表
#     """
#     performance_summary = {}
    
#     for step_infos in info_history:
#         for agent, info in step_infos.items():
#             if agent not in performance_summary:
#                 performance_summary[agent] = {
#                     'survival_frames': 0,
#                     'food_caught_count': 0,
#                     'poison_contacts': 0,
#                     'predator_catches': 0,
#                     'health_changes': [],
#                     'algorithm': info.get('algorithm', 'unknown'),
#                     'agent_type': info.get('agent_type', 'unknown')
#                 }
            
#             # 更新统计
#             if info.get('is_alive', False):
#                 performance_summary[agent]['survival_frames'] = info.get('current_frame', 0)
            
#             if info.get('food_caught', False):
#                 performance_summary[agent]['food_caught_count'] += 1
                
#             if info.get('poison_contacted', False):
#                 performance_summary[agent]['poison_contacts'] += 1
                
#             if info.get('predator_catch', False):
#                 performance_summary[agent]['predator_catches'] += 1
                
#             performance_summary[agent]['health_changes'].append(info.get('current_health', 0))
    
#     return performance_summary

# print("\n=== 性能分析函数示例已定义 ===")
# print("使用方法：")
# print("1. 收集每步的 info 数据到 info_history 列表中")
# print("2. 调用 analyze_agent_performance(info_history) 获取性能摘要")
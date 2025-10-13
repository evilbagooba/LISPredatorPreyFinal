from pettingzoo.sisl import waterworld_v4
from pettingzoo.test import api_test
import numpy as np
agent_algos = ["PPO", "PPO", "DQN", "DQN","PPO", "PPO", "DQN", "DQN","DQN", "DQN"]
algo_name_to_id = {name: idx for idx, name in enumerate(sorted(set(agent_algos)))}

import supersuit as ss

env = waterworld_v4.env(render_mode="human",n_predators=5,n_preys=5,n_evaders=5,n_obstacles=1,n_poisons=1,agent_algorithms=agent_algos)
env.reset(seed=42)
# api_test(env, num_cycles=1000, verbose_progress=True)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(reward)


    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()

# 调试环境结构

# from pettingzoo.sisl import waterworld_v4

# agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"]

# env = waterworld_v4.env(
#     render_mode="human",
#     n_predators=2,
#     n_preys=3,
#     n_evaders=5,
#     n_obstacles=1,
#     n_poisons=1,
#     agent_algorithms=agent_algos
# )
# env.reset(seed=42)
# from pettingzoo.test import api_test
# api_test(env, num_cycles=1000, verbose_progress=True)
# 确保环境能正常运行
# from pettingzoo.sisl import waterworld_v4
# from pettingzoo.utils.conversions import aec_to_parallel
# env = waterworld_v4.env(
#     render_mode="human",
#     n_predators=5,
#     n_preys=5,
#     n_evaders=1,
#     n_obstacles=2,
#     obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
#     n_poisons=20,
#     agent_algorithms=agent_algos
# )
# env = aec_to_parallel(env)
# env = ss.black_death_v3(env)
# obs = env.reset(seed=42)

# print("Reset obs:", obs)

# for agent in env.agent_iter():
#     observation, reward, termination, truncation, info = env.last()
#     assert isinstance(termination, bool), "terminated from last is not True or False"
#     assert isinstance(truncation, bool), "terminated from last is not True or False"
#     # print(f"Agent: {agent}, Reward: {reward}, Termination: {termination}, Truncation: {truncation}, Info: {info}")
#     print(type(termination))

#     # print(observation)
#     if termination or truncation:
#         action = None  # 或者使用 None
#     else:
#         # this is where you would insert your policy
#         action = env.action_space(agent).sample()


#     env.step(action)
# env.close()
# api_test(env, num_cycles=1000, verbose_progress=True)
# #         
# from pettingzoo.test import parallel_api_test
# parallel_api_test(env, num_cycles=10)

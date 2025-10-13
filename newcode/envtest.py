from pettingzoo.sisl import waterworld_v4

env = waterworld_v4.env(
    n_predators=5,
    n_preys=30,
    n_evaders=10,
    n_poisons=10,
    n_coop=1,  # 单个agent即可吃到食物
    static_food=True,
    render_mode="human"
)

env.reset()

for agent in env.agent_iter(max_iter=1000):
    observation, reward, termination, truncation, info = env.last()
    
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    
    env.step(action)
    
    # 检查奖励是否正常
    if reward > 0:
        print(f"{agent}: reward={reward:.2f}, info={info.get('performance_metrics', {})}")

env.close()
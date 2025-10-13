from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from collections import defaultdict
import imageio
import numpy as np

# 准备环境参数
agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"] * 40
env = waterworld_v4.env(
    render_mode="rgb_array",  # 改为 rgb_array 以获取帧数据
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

# 视频设置
output_video_path = "waterworld_episode.mp4"
fps = 15  # 视频帧率

# 用于存储视频帧
frames = []

print("\n=== Starting episode and recording ===")
# 重置环境
obs = env.reset(seed=42)

# 用于累加每个 agent 的本轮总 reward
cumulative_rewards = defaultdict(float)

# 按 pettingzoo 规范的 agent_iter 刷新一轮
step_count = 0
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
    
    # 每一步渲染并保存帧
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    step_count += 1
    
    # 每100步打印一次进度
    if step_count % 100 == 0:
        print(f"Recorded {step_count} steps, {len(frames)} frames")

env.close()

# 保存视频
print(f"\n=== Saving video to {output_video_path} ===")
print(f"Total frames captured: {len(frames)}")

if frames:
    # 使用 imageio 保存视频
    imageio.mimsave(
        output_video_path, 
        frames, 
        fps=fps,
        codec='libx264',  # 使用 H.264 编码
        quality=8  # 质量设置 (1-10, 10最高)
    )
    print(f"✓ Video saved successfully!")
    print(f"  Duration: {len(frames)/fps:.2f} seconds")
    print(f"  FPS: {fps}")
else:
    print("⚠ No frames captured!")

# 打印累积奖励摘要
print("\n=== Cumulative Rewards Summary ===")
for agent, total_reward in sorted(cumulative_rewards.items()):
    print(f"{agent}: {total_reward:.2f}")
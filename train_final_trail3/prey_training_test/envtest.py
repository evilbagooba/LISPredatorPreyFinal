"""
Simple Waterworld Execution Script
Prey uses trained PPO model, Predator is random.
The environment will be recorded as a video.
"""

from pettingzoo.sisl import waterworld_v4
from stable_baselines3 import PPO
import numpy as np
import os

def run_video_episode(model_path="models/ppo_prey_v1.zip",
                      n_predators=2,
                      n_preys=5,
                      video_path="waterworld_demo.mp4",
                      max_cycles=1000):

    print("🎬 Loading environment...")
    env = waterworld_v4.parallel_env(
        render_mode="rgb_array",
        n_predators=n_predators,
        n_preys=n_preys,
        n_evaders=20,
        n_obstacles=2,
        n_poisons=10,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        thrust_penalty=0,
        static_food=True,
        static_poison=True,
    )

    # 加载PPO模型
    print(f"🎯 Loading PPO model from: {model_path}")
    model = PPO.load(model_path, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

    print(f"🎥 Recording video to: {video_path}")
    frames = []
    obs, infos = env.reset()
    done = False
    step_count = 0

    while not done and step_count < max_cycles:
        step_count += 1
        actions = {}
        for agent in env.agents:
            if "prey" in agent:
                action, _ = model.predict(obs[agent], deterministic=True)
            else:
                action = env.action_space(agent).sample()
            actions[agent] = action

        obs, rewards, terminations, truncations, infos = env.step(actions)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        done = all(terminations.values()) or all(truncations.values()) or len(env.agents) == 0

    # 保存视频
    import imageio
    print(f"💾 Saving {len(frames)} frames to {video_path} ...")
    imageio.mimsave(video_path, frames, fps=15)
    print("✅ Video saved successfully!")

    env.close()

if __name__ == "__main__":
    run_video_episode(
        model_path="prey_ppo_with_random_predators.zip",   # 你训练好的Prey模型
        n_predators=5,                         # 捕食者数量
        n_preys=8,                             # 猎物数量
        video_path="waterworld_prey_vs_random.mp4",
        max_cycles=1000                        # 每回合最大步数
    )

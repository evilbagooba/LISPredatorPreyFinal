# from pettingzoo.sisl import waterworld_v4
# from pettingzoo.test import api_test
# import numpy as np
# agent_algos = ["PPO", "PPO", "DQN", "DQN","PPO", "PPO", "DQN", "DQN","DQN", "DQN","PPO", "PPO", "DQN", "DQN","PPO", "PPO", "DQN", "DQN","DQN", "DQN"]
# algo_name_to_id = {name: idx for idx, name in enumerate(sorted(set(agent_algos)))}
# import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel
# # # env = waterworld_v4.env(render_mode="human",n_predators=5,n_preys=5,n_evaders=5,n_obstacles=1,n_poisons=1,agent_algorithms=agent_algos)
# # # env.reset(seed=42)
# # # api_test(env, num_cycles=1000, verbose_progress=True)
# # # for agent in env.agent_iter():
# # #     observation, reward, termination, truncation, info = env.last()
# # #     print(reward)


# # #     if termination or truncation:
# # #         action = None
# # #     else:
# # #         # this is where you would insert your policy
# # #         action = env.action_space(agent).sample()

# # #     env.step(action)
# # # env.close()

# # # 调试环境结构

# # # from pettingzoo.sisl import waterworld_v4

# # # agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"]

# # # env = waterworld_v4.env(
# # #     render_mode="human",
# # #     n_predators=2,
# # #     n_preys=3,
# # #     n_evaders=5,
# # #     n_obstacles=1,
# # #     n_poisons=1,
# # #     agent_algorithms=agent_algos
# # # )
# # # env.reset(seed=42)
# # # from pettingzoo.test import api_test
# # # api_test(env, num_cycles=1000, verbose_progress=True)
# # # 确保环境能正常运行
# # from pettingzoo.sisl import waterworld_v4

# # # env = waterworld_v4.env(
# # #     render_mode="human",
# # #     n_predators=10,
# # #     n_preys=10,
# # #     n_evaders=1,
# # #     n_obstacles=2,
# # #     obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
# # #     n_poisons=20,
# # #     agent_algorithms=agent_algos
# # # )
# # # env = aec_to_parallel(env)  
# # # env = ss.black_death_v3(env)
# # # env = ss.pettingzoo_env_to_vec_env_v1(env)
# # # env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")


# # # # obs = env.reset(seed=42)
# # # observations = env.reset()

# # # # print("Reset obs:", observations)

# # # while env.num_agents:
# # #     # this is where you would insert your policy
# # #     actions = {agent: env.action_space(agent).sample() for agent in env.agents}

# # #     observations, rewards, terminations, truncations, infos = env.step(actions)
# # # env.close()
# # # # api_test(env, num_cycles=1000, verbose_progress=True)
# # #         


# # from __future__ import annotations
agent_algos = ["PPO", "PPO", "DQN", "DQN","PPO", "PPO", "DQN", "DQN","DQN", "DQN"]

# from __future__ import annotations

import glob
import os
import time
import cv2

import supersuit as ss
# from stable_baselines3 import PPO
# from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy as MlpPolicy

from pettingzoo.sisl import waterworld_v4


def train_butterfly_supersuit(
    env, steps: int = 10, seed: int | None = 0, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    # env = env_fn.parallel_env(**env_kwargs)

    env.reset()

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    # Note: Waterworld's observation space is discrete (242,) so we use an MLP policy rather than CNN
    model = SAC(
        MlpPolicy,
        env,
        verbose=3,
        learning_rate=1e-3,
        batch_size=256,
    )

    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent

    # 创建文件夹用于保存视频
    video_dir = "rendered_videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    # 录制视频的设置
    video_filename = os.path.join(video_dir, "gameplay_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 设置视频编码方式
    video_writer = None  # 视频写入器，将在第一次渲染时初始化
    env = waterworld_v4.env(
        render_mode='rgb_array',
        n_predators=5,
        n_preys=5,
        n_evaders=1,
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=20,
        agent_algorithms=agent_algos
    )
    # env = aec_to_parallel(env)  
    env = ss.black_death_v3(env)
    env_kwargs = {}

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob("/home/qrbao/Documents/code4/rllib/mycode/waterworld_v4_20250719-121839.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = SAC.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            env.step(act)
            # 录制视频
            frame = env.render()  # 获取渲染帧
            if video_writer is None:
                # 初始化视频写入器，使用渲染帧的尺寸
                height, width, _ = frame.shape
                video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (width, height))

            # 写入帧到视频文件
            video_writer.write(frame)
    # 完成游戏后关闭视频写入器
    if video_writer:
        video_writer.release()
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    env_fn = waterworld_v4.env(
        render_mode="human",
        n_predators=5,
        n_preys=5,
        n_evaders=1,
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=20,
        agent_algorithms=agent_algos
    )
    env_fn = aec_to_parallel(env_fn)  
    env_fn = ss.black_death_v3(env_fn)
    env_kwargs = {}

    # Train a model (takes ~3 minutes on GPU)
    # train_butterfly_supersuit(env_fn, steps=196, seed=0, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Watch 2 games
    eval(env_fn, num_games=2, render_mode="human", **env_kwargs)
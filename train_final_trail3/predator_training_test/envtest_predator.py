"""
Waterworld Predator Demo Script
Predator uses trained PPO model, Prey is random.
The environment will be recorded as a video.
"""

from pettingzoo.sisl import waterworld_v4
from stable_baselines3 import PPO
import numpy as np
import os


def run_predator_video_episode(
    model_path="predator_ppo_model.zip",  # Predator 模型路径
    n_predators=5,
    n_preys=10,
    video_path="waterworld_predator_demo.mp4",
    max_cycles=3000
):
    """
    运行 Predator 视频录制
    
    Args:
        model_path: 训练好的 Predator 模型路径
        n_predators: Predator 数量
        n_preys: Prey 数量
        video_path: 视频保存路径
        max_cycles: 最大步数
    """
    
    print("="*70)
    print("🎬 Waterworld Predator Demo")
    print("="*70)
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found: {model_path}")
    
    print(f"\n📦 Configuration:")
    print(f"  - Predators: {n_predators} (using trained PPO model)")
    print(f"  - Preys: {n_preys} (random policy)")
    print(f"  - Max steps: {max_cycles}")
    print(f"  - Video output: {video_path}")
    
    # 创建环境
    print(f"\n🌊 Creating Waterworld environment...")
    env = waterworld_v4.parallel_env(
        render_mode="rgb_array",
        n_predators=n_predators,
        n_preys=n_preys,
        n_evaders=1,
        n_obstacles=2,
        n_poisons=1,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        thrust_penalty=0,
        predator_speed=0.06,    # Predator 速度
        prey_speed=0.001,        # Prey 速度（更快）
        sensor_range=0.8,  # 增加传感器范围
        static_food=True,
        static_poison=True,
        max_cycles=max_cycles,
    )
    
    # 加载 PPO 模型
    print(f"\n🎯 Loading Predator PPO model...")
    print(f"   Path: {model_path}")
    
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    model = PPO.load(model_path, device=device)
    print(f"   ✓ Model loaded on device: {device}")
    
    # 运行一个 episode 并录制
    print(f"\n🎥 Recording episode...")
    frames = []
    obs, infos = env.reset()
    done = False
    step_count = 0
    
    total_predator_reward = 0
    total_prey_reward = 0
    
    while not done and step_count < max_cycles:
        step_count += 1
        actions = {}
        
        # 为每个 agent 选择动作
        for agent in env.agents:
            if "predator" in agent:
                # Predator 使用训练好的模型
                action, _ = model.predict(obs[agent], deterministic=True)
            else:  # prey
                # Prey 使用随机策略
                action = env.action_space(agent).sample()
            
            actions[agent] = action
        
        # 执行动作
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # 统计奖励
        for agent, reward in rewards.items():
            if "predator" in agent:
                total_predator_reward += reward
            else:
                total_prey_reward += reward
        
        # 渲染帧
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # 检查是否结束
        done = all(terminations.values()) or all(truncations.values()) or len(env.agents) == 0
        
        # 打印进度
        if step_count % 100 == 0:
            print(f"   Step {step_count}/{max_cycles} - Active agents: {len(env.agents)}")
    
    print(f"\n📊 Episode Summary:")
    print(f"   Total steps: {step_count}")
    print(f"   Frames captured: {len(frames)}")
    print(f"   Total Predator reward: {total_predator_reward:.2f}")
    print(f"   Total Prey reward: {total_prey_reward:.2f}")
    print(f"   Final agents remaining: {len(env.agents)}")
    
    # 保存视频
    if len(frames) > 0:
        import imageio
        print(f"\n💾 Saving video...")
        print(f"   Output: {video_path}")
        print(f"   Frames: {len(frames)}")
        print(f"   FPS: 30")
        
        imageio.mimsave(video_path, frames, fps=30)
        print(f"   ✅ Video saved successfully!")
        
        # 计算文件大小
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        print(f"   📦 File size: {file_size:.2f} MB")
    else:
        print("   ⚠️  No frames captured!")
    
    env.close()
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)


def run_multiple_episodes(
    model_path="predator_ppo_model.zip",
    n_predators=5,
    n_preys=10,
    n_episodes=3,
    output_dir="videos",
    max_cycles=3000
):
    """
    运行多个 episode 并分别录制视频
    
    Args:
        model_path: Predator 模型路径
        n_predators: Predator 数量
        n_preys: Prey 数量
        n_episodes: 录制的 episode 数量
        output_dir: 视频输出目录
        max_cycles: 每个 episode 的最大步数
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print(f"🎬 Recording {n_episodes} Episodes")
    print("="*70)
    
    for ep in range(1, n_episodes + 1):
        print(f"\n{'='*70}")
        print(f"Episode {ep}/{n_episodes}")
        print(f"{'='*70}")
        
        video_path = os.path.join(output_dir, f"predator_demo_ep{ep}.mp4")
        
        run_predator_video_episode(
            model_path=model_path,
            n_predators=n_predators,
            n_preys=n_preys,
            video_path=video_path,
            max_cycles=max_cycles
        )
    
    print(f"\n{'='*70}")
    print(f"✅ All {n_episodes} episodes recorded!")
    print(f"📁 Videos saved in: {output_dir}/")
    print(f"{'='*70}")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    
    # ========================================
    # 选项 1: 录制单个视频
    # ========================================
    
    run_predator_video_episode(
        model_path="predator_ppo_model.zip",  # 你的 Predator 模型
        n_predators=5,                       # Predator 数量
        n_preys=10,                             # Prey 数量
        video_path="predator_ppo_model.mp4",
        max_cycles=3000                         # 最大步数
    )
    
    # ========================================
    # 选项 2: 录制多个视频（可选）
    # ========================================
    """
    run_multiple_episodes(
        model_path="predator_ppo_model.zip",
        n_predators=5,
        n_preys=10,
        n_episodes=3,                           # 录制3个视频
        output_dir="predator_demos",            # 输出目录
        max_cycles=3000
    )
    """
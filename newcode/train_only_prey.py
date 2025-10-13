"""
Waterworld Prey-Only Environment PPO Training with Visualization
"""

from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

class TrainingMonitorCallback(BaseCallback):
    """Monitor training process"""
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_ep_reward = 0
        self.current_ep_length = 0
        
    def _on_step(self):
        self.current_ep_reward += self.locals['rewards'][0]
        self.current_ep_length += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)
            
            if len(self.episode_rewards) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                print(f"\nEpisode {len(self.episode_rewards)}:")
                print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
                print(f"  Avg Length: {np.mean(self.episode_lengths[-10:]):.0f}")
                print(f"  Max Reward: {np.max(recent_rewards):.2f}")
            
            self.current_ep_reward = 0
            self.current_ep_length = 0
        
        return True

def plot_training_curve(episode_rewards, save_path='training_curve.png'):
    """
    Plot episode rewards with smoothed curve
    
    Args:
        episode_rewards: List of episode rewards
        save_path: Path to save the figure
    """
    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    # Calculate and plot moving average (window size = 10)
    if len(rewards) >= 10:
        window_size = min(10, len(rewards))
        smoothed = uniform_filter1d(rewards, size=window_size, mode='nearest')
        plt.plot(episodes, smoothed, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
    
    # Calculate and plot trend line
    if len(rewards) >= 50:
        window_size = min(50, len(rewards))
        trend = uniform_filter1d(rewards, size=window_size, mode='nearest')
        plt.plot(episodes, trend, color='green', linewidth=2, linestyle='--', label=f'Trend (window={window_size})')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    # Labels and title
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title('PPO Training Progress: Episode Rewards', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = f'Episodes: {len(rewards)}\n'
    stats_text += f'Mean: {np.mean(rewards):.2f}\n'
    stats_text += f'Std: {np.std(rewards):.2f}\n'
    stats_text += f'Max: {np.max(rewards):.2f}\n'
    stats_text += f'Min: {np.min(rewards):.2f}'
    
    # Calculate improvement
    n = len(rewards)
    if n >= 10:
        early = rewards[:max(1, n//10)]
        late = rewards[-max(1, n//10):]
        improvement = np.mean(late) - np.mean(early)
        stats_text += f'\nImprovement: {improvement:+.2f}'
    
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curve saved to: {save_path}")
    plt.close()

def create_prey_only_env():
    """Create prey-only environment"""
    print("\n" + "="*60)
    print("Creating Prey-Only Training Environment")
    print("="*60)
    
    agent_algos = ["PPO"] * 5
    
    env = waterworld_v4.parallel_env(
        render_mode=None,
        n_predators=0,
        n_preys=5,
        n_evaders=200,
        n_obstacles=2,
        food_reward=20,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=10,
        agent_algorithms=agent_algos,
        max_cycles=500,
        static_food=True,
        static_poison=True,
    )
    
    print(f"Environment Configuration:")
    print(f"  Agents: {env.possible_agents}")
    print(f"  Observation Dim: {env.observation_space('prey_0').shape}")
    print(f"  Action Dim: {env.action_space('prey_0').shape}")
    
    return env

def prepare_env_for_training(env):
    """Prepare environment for training"""
    print("\nConverting environment format...")
    
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    env = VecMonitor(env)
    
    print("  Environment conversion complete")
    return env

def train_ppo(env, total_timesteps=1000000):
    """Train using PPO"""
    print("\n" + "="*60)
    print("Starting PPO Training")
    print("="*60)
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device='auto'
    )
    
    callback = TrainingMonitorCallback(check_freq=1000)
    
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    print("\nTraining complete!")
    
    # Display training statistics
    if callback.episode_rewards:
        rewards = np.array(callback.episode_rewards)
        print("\n" + "="*60)
        print("Training Statistics")
        print("="*60)
        print(f"Total Episodes: {len(rewards)}")
        print(f"Mean Reward: {np.mean(rewards):.2f}")
        print(f"Max Reward: {np.max(rewards):.2f}")
        print(f"Min Reward: {np.min(rewards):.2f}")
        
        n = len(rewards)
        early = rewards[:max(1, n//10)]
        late = rewards[-max(1, n//10):]
        improvement = np.mean(late) - np.mean(early)
        
        print(f"\nLearning Analysis:")
        print(f"  Early Mean: {np.mean(early):.2f}")
        print(f"  Late Mean: {np.mean(late):.2f}")
        print(f"  Improvement: {improvement:+.2f}")
        
        if improvement > 5:
            print("  Conclusion: Effective Learning")
        elif improvement > -5:
            print("  Conclusion: Limited Learning")
        else:
            print("  Conclusion: No Effective Learning")
    
    return model, callback

def evaluate_model(model, env, n_episodes=10):
    """Evaluate trained model"""
    print("\n" + "="*60)
    print(f"Evaluating Model ({n_episodes} episodes)")
    print("="*60)
    
    episode_rewards = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0
        ep_length = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            ep_length += 1
            
            if done[0]:
                break
        
        episode_rewards.append(ep_reward)
        print(f"  Episode {ep+1}: Reward={ep_reward:.2f}, Length={ep_length}")
    
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Max Reward: {np.max(episode_rewards):.2f}")
    
    return episode_rewards

def main():
    """Main function"""
    print("="*60)
    print("Waterworld Prey-Only PPO Training")
    print("="*60)
    
    # 1. Create environment
    raw_env = create_prey_only_env()
    
    # 2. Prepare training environment
    env = prepare_env_for_training(raw_env)
    
    # 3. Train
    model, callback = train_ppo(env, total_timesteps=1000000)
    
    # 4. Plot training curve
    if callback.episode_rewards:
        print("\n" + "="*60)
        print("Generating Training Curve")
        print("="*60)
        plot_training_curve(callback.episode_rewards, save_path='training_curve.png')
    
    # 5. Evaluate
    episode_rewards = evaluate_model(model, env, n_episodes=10)
    
    # 6. Save model
    model_path = "prey_ppo_model"
    model.save(model_path)
    print(f"\nModel saved: {model_path}.zip")
    
    # 7. Final summary
    print("\n" + "="*60)
    print("Training Complete Summary")
    print("="*60)
    
    if callback.episode_rewards:
        rewards = np.array(callback.episode_rewards)
        eval_rewards = np.array(episode_rewards)
        
        print(f"During Training:")
        print(f"  Mean Reward: {np.mean(rewards):.2f}")
        print(f"  Max Reward: {np.max(rewards):.2f}")
        
        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {np.mean(eval_rewards):.2f}")
        print(f"  Max Reward: {np.max(eval_rewards):.2f}")
        
        if np.mean(eval_rewards) > np.mean(rewards[:len(rewards)//10]):
            print("\nEnvironment training is effective")
        else:
            print("\nEnvironment training shows limited effectiveness")
    
    env.close()
    print("\nTraining test complete")
    print(f"Check 'training_curve.png' for visualization")

if __name__ == "__main__":
    main()
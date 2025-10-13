"""
Waterworld: Train Prey with Random Predator Agents
Training prey agents while predators execute random policy
"""

from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import gymnasium as gym


class MixedAgentVecEnv(VecEnv):
    """
    Custom VecEnv wrapper that:
    - Allows prey agents to be trained by RL algorithm
    - Forces predator agents to take random actions
    - Only exposes prey observations/rewards to the training algorithm

    Key insight: After pettingzoo_env_to_vec_env_v1 + concat_vec_envs_v1,
    we have num_envs environments, each representing one agent.
    Observations shape: (num_envs, obs_dim)
    Actions shape: (num_envs, action_dim)
    """

    def __init__(self, venv, n_predators, n_preys):
        """
        Args:
            venv: The wrapped vectorized environment
            n_predators: Number of predator agents (use random policy)
            n_preys: Number of prey agents (to be trained)
        """
        self.venv = venv
        self.n_predators = n_predators
        self.n_preys = n_preys
        self.n_total_agents = n_predators + n_preys

        # Indices for prey agents (they come after predators in the agent list)
        self.prey_indices = list(range(n_predators, n_predators + n_preys))
        self.predator_indices = list(range(n_predators))

        # Get original spaces (they are per-agent spaces)
        original_obs_space = venv.observation_space
        original_action_space = venv.action_space

        # The underlying venv has num_envs = n_total_agents
        # We create a new VecEnv with num_envs = n_preys (only prey agents)
        super().__init__(
            num_envs=n_preys,
            observation_space=original_obs_space,  # Same per-agent space
            action_space=original_action_space      # Same per-agent space
        )

    def reset(self):
        """Reset environment and return only prey observations"""
        obs = self.venv.reset()
        prey_obs = obs[self.prey_indices]  # Shape: (n_preys, obs_dim)
        return prey_obs

    def step_async(self, actions):
        """
        Combine prey actions (from policy) with random predator actions

        Args:
            actions: shape (n_preys, action_dim) - actions for prey agents only
        """
        # Generate random actions for predators
        predator_actions = np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(self.n_predators, 2)  # (n_predators, 2)
        ).astype(np.float32)

        # Combine: predator actions first, then prey actions
        # Full actions shape: (n_total_agents, 2)
        full_actions = np.zeros((self.n_total_agents, 2), dtype=np.float32)
        full_actions[self.predator_indices] = predator_actions
        full_actions[self.prey_indices] = actions

        # Pass to underlying environment
        self.venv.step_async(full_actions)

    def step_wait(self):
        """
        Get results from environment and extract only prey rewards/dones/infos
        """
        obs, rewards, dones, infos = self.venv.step_wait()

        # Extract prey-specific data
        # obs shape: (n_total_agents, obs_dim) -> (n_preys, obs_dim)
        prey_obs = obs[self.prey_indices]

        # rewards shape: (n_total_agents,) -> (n_preys,)
        prey_rewards = rewards[self.prey_indices]

        # dones shape: (n_total_agents,) -> (n_preys,)
        prey_dones = dones[self.prey_indices]

        # infos is a list of dicts
        prey_infos = [infos[i] for i in self.prey_indices]

        return prey_obs, prey_rewards, prey_dones, prey_infos

    def close(self):
        """Close underlying environment"""
        return self.venv.close()

    def get_attr(self, attr_name, indices=None):
        """Get attribute from underlying environment"""
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute in underlying environment"""
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call method on underlying environment"""
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environment is wrapped with a given wrapper"""
        return self.venv.env_is_wrapped(wrapper_class, indices)


class PreyTrainingMonitorCallback(BaseCallback):
    """Monitor training process - tracks only prey rewards"""

    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_ep_reward = 0
        self.current_ep_length = 0

    def _on_step(self):
        # Sum rewards from all prey agents in this step
        reward_sum = np.sum(self.locals['rewards'])
        self.current_ep_reward += reward_sum
        self.current_ep_length += 1

        # Check if episode is done (any prey is done)
        if np.any(self.locals['dones']):
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)

            if len(self.episode_rewards) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                print(f"\n[Prey Training] Episode {len(self.episode_rewards)}:")
                print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
                print(f"  Avg Length: {np.mean(self.episode_lengths[-10:]):.0f}")
                print(f"  Max Reward: {np.max(recent_rewards):.2f}")

            self.current_ep_reward = 0
            self.current_ep_length = 0

        return True


def plot_training_curve(episode_rewards, save_path='training_curve_prey_with_predators.png'):
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
    plt.ylabel('Episode Reward (Prey Only)', fontsize=12)
    plt.title('PPO Training: Prey vs Random Predators', fontsize=14, fontweight='bold')
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


def create_mixed_env(n_predators=3, n_preys=5):
    """Create environment with both predators and prey"""
    print("\n" + "="*60)
    print("Creating Mixed Environment (Prey Training + Random Predators)")
    print("="*60)

    total_agents = n_predators + n_preys
    agent_algos = ["Random"] * n_predators + ["PPO"] * n_preys

    env = waterworld_v4.parallel_env(
        render_mode=None,
        n_predators=n_predators,
        n_preys=n_preys,
        n_evaders=90,
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=10,
        agent_algorithms=agent_algos,
        max_cycles=3000,
        static_food=True,
        static_poison=True,
    )

    print(f"Environment Configuration:")
    print(f"  Predators (Random): {n_predators}")
    print(f"  Preys (Training): {n_preys}")
    print(f"  Total Agents: {total_agents}")
    print(f"  All Agents: {env.possible_agents}")
    print(f"  Food: 180 (static)")
    print(f"  Poison: 10 (static)")
    print(f"  Observation Dim: {env.observation_space('prey_0').shape}")
    print(f"  Action Dim: {env.action_space('prey_0').shape}")

    return env, n_predators, n_preys


def prepare_env_for_training(env, n_predators, n_preys):
    """Prepare environment for training with custom wrapper"""
    print("\nConverting environment format...")

    # Standard conversions
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')

    print("  Standard conversion complete")
    print(f"  num_envs after conversion: {env.num_envs}")
    print(f"  obs_space: {env.observation_space}")
    print(f"  action_space: {env.action_space}")

    # Wrap with custom mixed agent environment
    env = MixedAgentVecEnv(env, n_predators=n_predators, n_preys=n_preys)
    print(f"  Applied MixedAgentVecEnv wrapper")
    print(f"    - Training agents: {n_preys} preys")
    print(f"    - Random agents: {n_predators} predators")
    print(f"    - Final num_envs: {env.num_envs}")

    # Add monitor
    env = VecMonitor(env)
    print("  Environment preparation complete")

    return env


def train_ppo(env, total_timesteps=1000000):
    """Train using PPO"""
    print("\n" + "="*60)
    print("Starting PPO Training (Prey Only)")
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
        device='cpu'  # Use CPU for MLP policy as recommended
    )

    callback = PreyTrainingMonitorCallback(check_freq=1000)

    print("\nStarting training...")
    print("Note: Predators are executing random policy")
    print("      Only prey agents are learning")

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
        print("Training Statistics (Prey Only)")
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
            ep_reward += np.sum(reward)  # Sum over all prey
            ep_length += 1

            if np.any(done):
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
    print("Waterworld: Train Prey with Random Predators")
    print("="*60)

    # Configuration
    N_PREDATORS = 3
    N_PREYS = 15
    TOTAL_TIMESTEPS = 10000000

    # 1. Create environment
    raw_env, n_predators, n_preys = create_mixed_env(
        n_predators=N_PREDATORS,
        n_preys=N_PREYS
    )

    # 2. Prepare training environment
    env = prepare_env_for_training(raw_env, n_predators, n_preys)

    # 3. Train
    model, callback = train_ppo(env, total_timesteps=TOTAL_TIMESTEPS)

    # 4. Plot training curve
    if callback.episode_rewards:
        print("\n" + "="*60)
        print("Generating Training Curve")
        print("="*60)
        plot_training_curve(
            callback.episode_rewards,
            save_path='training_curve_prey_with_random_predators.png'
        )

    # 5. Evaluate
    episode_rewards = evaluate_model(model, env, n_episodes=10)

    # 6. Save model
    model_path = "prey_ppo_with_random_predators"
    model.save(model_path)
    print(f"\nModel saved: {model_path}.zip")

    # 7. Final summary
    print("\n" + "="*60)
    print("Training Complete Summary")
    print("="*60)

    if callback.episode_rewards:
        rewards = np.array(callback.episode_rewards)
        eval_rewards = np.array(episode_rewards)

        print(f"Environment Setup:")
        print(f"  Predators (Random): {N_PREDATORS}")
        print(f"  Preys (Trained): {N_PREYS}")

        print(f"\nDuring Training:")
        print(f"  Mean Reward: {np.mean(rewards):.2f}")
        print(f"  Max Reward: {np.max(rewards):.2f}")

        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {np.mean(eval_rewards):.2f}")
        print(f"  Max Reward: {np.max(eval_rewards):.2f}")

        if np.mean(eval_rewards) > np.mean(rewards[:len(rewards)//10]):
            print("\n✓ Prey training is effective in predator environment")
        else:
            print("\n✗ Prey training shows limited effectiveness")

    env.close()
    print("\nTraining complete!")
    print(f"Files generated:")
    print(f"  - {model_path}.zip")
    print(f"  - training_curve_prey_with_random_predators.png")


if __name__ == "__main__":
    main()

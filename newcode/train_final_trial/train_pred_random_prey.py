"""
Waterworld Multi-Algorithm Training Framework
Supports: PPO, SAC, TD3, A2C with TensorBoard integration
"""

from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import os
from datetime import datetime


# ============================================================================
# Algorithm Configuration System
# ============================================================================

class AlgorithmConfig(ABC):
    """Base class for algorithm configurations"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_model_class(self):
        """Return the algorithm model class"""
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return hyperparameters for the algorithm"""
        pass
    
    def get_color(self) -> str:
        """Return color for plotting"""
        return 'blue'


class PPOConfig(AlgorithmConfig):
    """PPO Algorithm Configuration"""
    
    def __init__(self):
        super().__init__("PPO")
    
    def get_model_class(self):
        return PPO
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'learning_rate': 5e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.98,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0001,
            'vf_coef': 0.4,
            'max_grad_norm': 0.5,
            'verbose': 1,
        }
    
    def get_color(self) -> str:
        return 'blue'


class SACConfig(AlgorithmConfig):
    """SAC Algorithm Configuration"""
    
    def __init__(self):
        super().__init__("SAC")
    
    def get_model_class(self):
        return SAC
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'learning_rate': 3e-4,
            'buffer_size': 1000000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',  # Auto-tune temperature
            'verbose': 1,
        }
    
    def get_color(self) -> str:
        return 'red'


class TD3Config(AlgorithmConfig):
    """TD3 Algorithm Configuration"""
    
    def __init__(self):
        super().__init__("TD3")
    
    def get_model_class(self):
        return TD3
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'learning_rate': 1e-3,
            'buffer_size': 1000000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'verbose': 1,
        }
    
    def get_color(self) -> str:
        return 'green'


class A2CConfig(AlgorithmConfig):
    """A2C Algorithm Configuration"""
    
    def __init__(self):
        super().__init__("A2C")
    
    def get_model_class(self):
        return A2C
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            'learning_rate': 7e-4,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1,
        }
    
    def get_color(self) -> str:
        return 'orange'


# Algorithm Registry
ALGORITHM_REGISTRY = {
    'ppo': PPOConfig,
    'sac': SACConfig,
    'td3': TD3Config,
    'a2c': A2CConfig,
}


def get_algorithm_config(algo_name: str) -> AlgorithmConfig:
    """Get algorithm configuration by name"""
    algo_name = algo_name.lower()
    if algo_name not in ALGORITHM_REGISTRY:
        available = ', '.join(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm '{algo_name}'. Available: {available}")
    return ALGORITHM_REGISTRY[algo_name]()


# ============================================================================
# TensorBoard Helper Functions
# ============================================================================

def create_tensorboard_log_dir(algo_name: str, base_dir: str = "./tensorboard_logs") -> str:
    """
    Create organized TensorBoard log directory
    
    Structure: ./tensorboard_logs/{algorithm}/{timestamp}/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_dir, algo_name.lower(), timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("TensorBoard Configuration")
    print(f"{'='*60}")
    print(f"Log Directory: {log_dir}")
    print(f"\nTo view TensorBoard, run:")
    print(f"  tensorboard --logdir={base_dir}")
    print(f"\nThen open: http://localhost:6006")
    print(f"{'='*60}\n")
    
    return log_dir


def print_tensorboard_instructions(log_dir: str):
    """Print instructions for launching TensorBoard"""
    print(f"\n{'='*60}")
    print("🎯 TensorBoard Instructions")
    print(f"{'='*60}")
    print("\n1. Open a new terminal")
    print(f"2. Run: tensorboard --logdir={log_dir}")
    print("3. Open browser: http://localhost:6006")
    print("\n💡 Tip: You can compare multiple runs by using:")
    print(f"   tensorboard --logdir=./tensorboard_logs")
    print(f"{'='*60}\n")


# ============================================================================
# Training Components
# ============================================================================

class TrainingMonitorCallback(BaseCallback):
    """Monitor training process - simplified for TensorBoard compatibility"""
    
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


def plot_training_curve(episode_rewards, algo_name, save_path=None):
    """Plot episode rewards with smoothed curve"""
    if save_path is None:
        save_path = f'training_curve_{algo_name.lower()}.png'
    
    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards)
    
    # Get algorithm color
    try:
        config = get_algorithm_config(algo_name)
        color = config.get_color()
    except:
        color = 'blue'
    
    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.plot(episodes, rewards, alpha=0.3, color=color, label='Raw Rewards')
    
    # Moving average
    if len(rewards) >= 10:
        window_size = min(10, len(rewards))
        smoothed = uniform_filter1d(rewards, size=window_size, mode='nearest')
        plt.plot(episodes, smoothed, color='red', linewidth=2, 
                label=f'Moving Average (window={window_size})')
    
    # Trend line
    if len(rewards) >= 50:
        window_size = min(50, len(rewards))
        trend = uniform_filter1d(rewards, size=window_size, mode='nearest')
        plt.plot(episodes, trend, color='green', linewidth=2, 
                linestyle='--', label=f'Trend (window={window_size})')
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title(f'{algo_name} Training Progress: Episode Rewards', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Statistics box
    stats_text = f'Algorithm: {algo_name}\n'
    stats_text += f'Episodes: {len(rewards)}\n'
    stats_text += f'Mean: {np.mean(rewards):.2f}\n'
    stats_text += f'Std: {np.std(rewards):.2f}\n'
    stats_text += f'Max: {np.max(rewards):.2f}\n'
    stats_text += f'Min: {np.min(rewards):.2f}'
    
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


# ============================================================================
# Environment Setup
# ============================================================================

def create_waterworld_env(n_predators=2, n_preys=5, algo_name="PPO"):
    """Create Waterworld environment"""
    print("\n" + "="*60)
    print(f"Creating Training Environment ({algo_name})")
    print("="*60)
    
    agent_algos = [algo_name.upper()] * n_preys + ["Random"] * n_predators
    
    env = waterworld_v4.parallel_env(
        render_mode=None,
        n_predators=n_predators,
        n_preys=n_preys,
        n_evaders=120,
        n_obstacles=2,
        thrust_penalty=0,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=10,
        agent_algorithms=agent_algos,
        max_cycles=5000,
        static_food=True,
        static_poison=True,
    )
    
    print(f"Environment Configuration:")
    print(f"  Algorithm: {algo_name}")
    print(f"  Predators: {n_predators} (Random)")
    print(f"  Prey: {n_preys} ({algo_name})")
    print(f"  All Agents: {env.possible_agents}")
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


# ============================================================================
# Training Functions
# ============================================================================

def train_algorithm(env, algo_config: AlgorithmConfig, total_timesteps=1000000, 
                   tensorboard_log=None):
    """
    Train using specified algorithm with TensorBoard support
    
    Args:
        env: Training environment
        algo_config: Algorithm configuration
        total_timesteps: Total training timesteps
        tensorboard_log: TensorBoard log directory (if None, TensorBoard is disabled)
    """
    print("\n" + "="*60)
    print(f"Starting {algo_config.name} Training")
    print("="*60)
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"TensorBoard: {'Enabled ✓' if tensorboard_log else 'Disabled ✗'}")
    
    # Get algorithm class and hyperparameters
    ModelClass = algo_config.get_model_class()
    hyperparams = algo_config.get_hyperparameters()
    
    # Create model with TensorBoard support
    model = ModelClass(
        "MlpPolicy",
        env,
        **hyperparams,
        tensorboard_log=tensorboard_log,  # KEY: Enable TensorBoard logging
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\n{algo_config.name} Hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    callback = TrainingMonitorCallback(check_freq=1000)
    
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name=algo_config.name  # Set run name in TensorBoard
    )
    
    print("\nTraining complete!")
    
    # Display statistics
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
            print("  Conclusion: Effective Learning ✓")
        elif improvement > -5:
            print("  Conclusion: Limited Learning ~")
        else:
            print("  Conclusion: No Effective Learning ✗")
    
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


# ============================================================================
# Main Function
# ============================================================================

def main(algorithm='sac', total_timesteps=10000000, use_tensorboard=True):
    """
    Main training function
    
    Args:
        algorithm: Algorithm to use ('ppo', 'sac', 'td3', 'a2c')
        total_timesteps: Total training timesteps
        use_tensorboard: Enable TensorBoard logging (default: True)
    """
    print("="*60)
    print(f"Waterworld Training: {algorithm.upper()}")
    print("="*60)
    
    # Get algorithm configuration
    algo_config = get_algorithm_config(algorithm)
    
    # Setup TensorBoard logging
    tensorboard_log = None
    if use_tensorboard:
        tensorboard_log = create_tensorboard_log_dir(algo_config.name)
    
    # 1. Create environment
    raw_env = create_waterworld_env(
        n_predators=2, 
        n_preys=50, 
        algo_name=algo_config.name
    )
    
    # 2. Prepare training environment
    env = prepare_env_for_training(raw_env)
    
    # 3. Train with TensorBoard
    model, callback = train_algorithm(
        env, 
        algo_config, 
        total_timesteps,
        tensorboard_log=tensorboard_log
    )
    
    # 4. Plot training curve
    if callback.episode_rewards:
        print("\n" + "="*60)
        print("Generating Training Curve")
        print("="*60)
        plot_training_curve(
            callback.episode_rewards, 
            algo_config.name,
            save_path=f'training_curve_{algorithm.lower()}.png'
        )
    
    # 5. Evaluate
    episode_rewards = evaluate_model(model, env, n_episodes=10)
    
    # 6. Save model
    model_path = f"prey_{algorithm.lower()}_model"
    model.save(model_path)
    print(f"\nModel saved: {model_path}.zip")
    
    # 7. Final summary
    print("\n" + "="*60)
    print("Training Complete Summary")
    print("="*60)
    
    if callback.episode_rewards:
        rewards = np.array(callback.episode_rewards)
        eval_rewards = np.array(episode_rewards)
        
        print(f"Algorithm: {algo_config.name}")
        print(f"\nDuring Training:")
        print(f"  Mean Reward: {np.mean(rewards):.2f}")
        print(f"  Max Reward: {np.max(rewards):.2f}")
        
        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {np.mean(eval_rewards):.2f}")
        print(f"  Max Reward: {np.max(eval_rewards):.2f}")
        
        if np.mean(eval_rewards) > np.mean(rewards[:len(rewards)//10]):
            print("\n✓ Training is effective")
        else:
            print("\n✗ Training shows limited effectiveness")
    
    # Print TensorBoard instructions
    if use_tensorboard and tensorboard_log:
        print_tensorboard_instructions(os.path.dirname(tensorboard_log))
    
    env.close()
    print(f"\nTraining complete! Check 'training_curve_{algorithm.lower()}.png'")


if __name__ == "__main__":
    # ========================================
    # 使用方式：修改这里选择算法和配置
    # ========================================
    
    # 选择算法: 'ppo', 'sac', 'td3', 'a2c'
    ALGORITHM = 'ppo'  # <-- 在这里切换算法
    
    # 训练步数
    TOTAL_TIMESTEPS = 100000000
    
    # 是否启用 TensorBoard (默认启用)
    USE_TENSORBOARD = True
    
    # 运行训练
    main(
        algorithm=ALGORITHM, 
        total_timesteps=TOTAL_TIMESTEPS,
        use_tensorboard=USE_TENSORBOARD
    )
    
    # ========================================
    # TensorBoard 使用说明：
    # ========================================
    # 1. 训练开始后，会自动创建日志目录
    # 2. 打开新终端，运行：tensorboard --logdir=./tensorboard_logs
    # 3. 浏览器访问：http://localhost:6006
    # 4. 可以同时查看多个算法的训练曲线进行对比
    # ========================================
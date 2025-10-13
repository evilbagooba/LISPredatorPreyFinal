"""
Waterworld Multi-Algorithm Training Framework - GPU ACCELERATED VERSION
专注于GPU优化，避免多进程复杂性
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


# ============================================================================
# GPU 加速配置
# ============================================================================

class AccelerationConfig:
    """GPU加速配置 - 简单稳定的方案"""
    
    # GPU配置
    GPU_ID = 0  # 使用哪块GPU (0-3)
    USE_CUDA = True
    
    @classmethod
    def print_config(cls):
        print("\n" + "="*60)
        print("🚀 GPU加速配置")
        print("="*60)
        print(f"GPU: CUDA:{cls.GPU_ID}")
        print(f"策略: 增大Batch Size + 优化超参数")
        print(f"优势: 充分利用GPU算力，避免多进程开销")
        print("="*60)


# ============================================================================
# 优化的算法配置 - 关键是增大batch size
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
    def get_hyperparameters(self, accelerated: bool = True) -> Dict[str, Any]:
        """Return hyperparameters for the algorithm"""
        pass
    
    def get_color(self) -> str:
        """Return color for plotting"""
        return 'blue'


class PPOConfig(AlgorithmConfig):
    """PPO Algorithm Configuration - GPU OPTIMIZED"""
    
    def __init__(self):
        super().__init__("PPO")
    
    def get_model_class(self):
        return PPO
    
    def get_hyperparameters(self, accelerated: bool = True) -> Dict[str, Any]:
        if accelerated:
            # GPU优化版：大幅增加batch size
            return {
                'learning_rate': 5e-4,
                'n_steps': 8192,        # 2048 -> 8192 (4x)
                'batch_size': 512,      # 64 -> 512 (8x，充分利用GPU)
                'n_epochs': 10,
                'gamma': 0.98,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0001,
                'vf_coef': 0.4,
                'max_grad_norm': 0.5,
                'verbose': 1,
            }
        else:
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
    """SAC Algorithm Configuration - GPU OPTIMIZED"""
    
    def __init__(self):
        super().__init__("SAC")
    
    def get_model_class(self):
        return SAC
    
    def get_hyperparameters(self, accelerated: bool = True) -> Dict[str, Any]:
        if accelerated:
            # GPU优化版：大幅增加batch size和gradient steps
            return {
                'learning_rate': 3e-4,
                'buffer_size': 1000000,
                'learning_starts': 1000,
                'batch_size': 1024,     # 256 -> 1024 (4x，充分利用GPU)
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 4,    # 1 -> 4 (每步更多更新)
                'ent_coef': 'auto',
                'verbose': 1,
            }
        else:
            return {
                'learning_rate': 3e-4,
                'buffer_size': 1000000,
                'learning_starts': 1000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'ent_coef': 'auto',
                'verbose': 1,
            }
    
    def get_color(self) -> str:
        return 'red'


class TD3Config(AlgorithmConfig):
    """TD3 Algorithm Configuration - GPU OPTIMIZED"""
    
    def __init__(self):
        super().__init__("TD3")
    
    def get_model_class(self):
        return TD3
    
    def get_hyperparameters(self, accelerated: bool = True) -> Dict[str, Any]:
        if accelerated:
            return {
                'learning_rate': 1e-3,
                'buffer_size': 1000000,
                'learning_starts': 1000,
                'batch_size': 1024,     # 256 -> 1024
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 4,    # 1 -> 4
                'policy_delay': 2,
                'target_policy_noise': 0.2,
                'target_noise_clip': 0.5,
                'verbose': 1,
            }
        else:
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
    """A2C Algorithm Configuration - GPU OPTIMIZED"""
    
    def __init__(self):
        super().__init__("A2C")
    
    def get_model_class(self):
        return A2C
    
    def get_hyperparameters(self, accelerated: bool = True) -> Dict[str, Any]:
        if accelerated:
            return {
                'learning_rate': 7e-4,
                'n_steps': 20,          # 5 -> 20
                'gamma': 0.99,
                'gae_lambda': 1.0,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'verbose': 1,
            }
        else:
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
# Training Components
# ============================================================================

class TrainingMonitorCallback(BaseCallback):
    """Monitor training process with detailed timing"""
    
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_ep_reward = 0
        self.current_ep_length = 0
        self.start_time = None
        self.last_report_time = None
        
    def _on_training_start(self):
        import time
        self.start_time = time.time()
        self.last_report_time = self.start_time
        
    def _on_step(self):
        self.current_ep_reward += self.locals['rewards'][0]
        self.current_ep_length += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)
            
            if len(self.episode_rewards) % 10 == 0:
                import time
                current_time = time.time()
                elapsed = current_time - self.start_time
                fps = self.num_timesteps / elapsed if elapsed > 0 else 0
                
                recent_rewards = self.episode_rewards[-10:]
                print(f"\nEpisode {len(self.episode_rewards)}:")
                print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
                print(f"  Avg Length: {np.mean(self.episode_lengths[-10:]):.0f}")
                print(f"  Max Reward: {np.max(recent_rewards):.2f}")
                print(f"  FPS: {fps:.1f} | Elapsed: {elapsed/60:.1f}min")
                
                # GPU utilization info
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / 1e9
                    gpu_mem_max = torch.cuda.max_memory_allocated() / 1e9
                    print(f"  GPU Mem: {gpu_mem:.2f}GB / Peak: {gpu_mem_max:.2f}GB")
            
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
    plt.title(f'{algo_name} Training Progress - GPU ACCELERATED', 
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
# Environment Setup - SIMPLE VERSION
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
        n_evaders=40,
        n_obstacles=2,
        thrust_penalty=0,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=10,
        agent_algorithms=agent_algos,
        max_cycles=50000,
        static_food=True,
        static_poison=True,
    )
    
    print(f"Environment Configuration:")
    print(f"  Algorithm: {algo_name}")
    print(f"  Predators: {n_predators} (Random)")
    print(f"  Prey: {n_preys} ({algo_name})")
    print(f"  All Agents: {len(env.possible_agents)} total")
    print(f"  Observation Dim: {env.observation_space('prey_0').shape}")
    print(f"  Action Dim: {env.action_space('prey_0').shape}")
    
    return env


def prepare_env_for_training(env):
    """Prepare environment for training - SIMPLE & STABLE"""
    print("\nPreparing environment...")
    
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    env = VecMonitor(env)
    
    print("  ✓ Environment ready")
    return env


# ============================================================================
# Training Functions - GPU OPTIMIZED
# ============================================================================

def setup_gpu(gpu_id=0):
    """Setup GPU environment"""
    if torch.cuda.is_available():
        # Set CUDA device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        torch.cuda.set_device(0)
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        
        print(f"\n🎮 GPU Setup:")
        print(f"  Using GPU {gpu_id}: {torch.cuda.get_device_name(0)}")
        print(f"  Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  cuDNN Benchmark: Enabled")
    else:
        print("\n⚠️  No CUDA available, using CPU")


def train_algorithm(env, algo_config: AlgorithmConfig, total_timesteps=1000000, 
                   accelerated=True):
    """Train using specified algorithm - GPU ACCELERATED"""
    print("\n" + "="*60)
    print(f"Starting {algo_config.name} Training {'🚀 GPU ACCELERATED' if accelerated else ''}")
    print("="*60)
    print(f"Total Timesteps: {total_timesteps:,}")
    
    # Get algorithm class and hyperparameters
    ModelClass = algo_config.get_model_class()
    hyperparams = algo_config.get_hyperparameters(accelerated=accelerated)
    
    # Setup device
    device = 'cuda' if (AccelerationConfig.USE_CUDA and torch.cuda.is_available()) else 'cpu'
    print(f"Device: {device.upper()}")
    
    # Create model
    model = ModelClass(
        "MlpPolicy",
        env,
        **hyperparams,
        device=device
    )
    
    print(f"\n{algo_config.name} Hyperparameters {'(GPU OPTIMIZED)' if accelerated else ''}:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    if accelerated:
        print("\n💡 GPU Optimization Strategy:")
        print(f"  • Batch Size: {hyperparams.get('batch_size', 'N/A')} (充分利用GPU)")
        if 'gradient_steps' in hyperparams:
            print(f"  • Gradient Steps: {hyperparams['gradient_steps']} (更快收敛)")
        if 'n_steps' in hyperparams:
            print(f"  • N Steps: {hyperparams['n_steps']} (减少环境交互开销)")
    
    callback = TrainingMonitorCallback(check_freq=1000)
    
    print("\nStarting training...")
    import time
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"⏱️  Total Time: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.1f} minutes)")
    print(f"⚡ Throughput: {total_timesteps/elapsed_time:.1f} steps/sec")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU Stats:")
        print(f"  Peak Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        print(f"  Current Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
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
        if n >= 10:
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
# Main Function - GPU ACCELERATED
# ============================================================================

def main(algorithm='sac', total_timesteps=10000000, accelerated=True,
         gpu_id=0, n_predators=2, n_preys=50):
    """
    Main training function - GPU ACCELERATED VERSION
    
    Args:
        algorithm: Algorithm to use ('ppo', 'sac', 'td3', 'a2c')
        total_timesteps: Total training timesteps
        accelerated: Use GPU-optimized configuration
        gpu_id: Which GPU to use (0-3)
        n_predators: Number of predators
        n_preys: Number of prey
    """
    print("="*60)
    print(f"🚀 Waterworld Training: {algorithm.upper()}")
    print("="*60)
    
    # Update acceleration config
    AccelerationConfig.GPU_ID = gpu_id
    AccelerationConfig.print_config()
    
    # Setup GPU
    setup_gpu(gpu_id)
    
    # Get algorithm configuration
    algo_config = get_algorithm_config(algorithm)
    
    # 1. Create environment
    raw_env = create_waterworld_env(
        n_predators=n_predators, 
        n_preys=n_preys, 
        algo_name=algo_config.name
    )
    
    # 2. Prepare training environment
    env = prepare_env_for_training(raw_env)
    
    # 3. Train
    model, callback = train_algorithm(
        env, algo_config, total_timesteps, accelerated=accelerated
    )
    
    # 4. Plot training curve
    if callback.episode_rewards:
        print("\n" + "="*60)
        print("Generating Training Curve")
        print("="*60)
        plot_training_curve(
            callback.episode_rewards, 
            algo_config.name,
            save_path=f'training_curve_{algorithm.lower()}_gpu_accelerated.png'
        )
    
    # 5. Evaluate
    episode_rewards = evaluate_model(model, env, n_episodes=10)
    
    # 6. Save model
    model_path = f"prey_{algorithm.lower()}_model_gpu_accelerated"
    model.save(model_path)
    print(f"\n💾 Model saved: {model_path}.zip")
    
    # 7. Final summary
    print("\n" + "="*60)
    print("🎉 Training Complete Summary")
    print("="*60)
    
    if callback.episode_rewards:
        rewards = np.array(callback.episode_rewards)
        eval_rewards = np.array(episode_rewards)
        
        print(f"Algorithm: {algo_config.name}")
        print(f"Configuration: {'GPU ACCELERATED 🚀' if accelerated else 'Standard'}")
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
    
    env.close()
    print(f"\nAll done! Check 'training_curve_{algorithm.lower()}_gpu_accelerated.png'")


if __name__ == "__main__":
    # ========================================
    # 🚀 GPU加速配置 - 简单稳定
    # ========================================
    
    # 选择算法
    ALGORITHM = 'sac'  # 'ppo', 'sac', 'td3', 'a2c'
    
    # 训练步数
    TOTAL_TIMESTEPS = 10000000
    
    # GPU选择
    GPU_ID = 0  # 使用GPU 0-3
    
    # 环境配置
    N_PREDATORS = 2
    N_PREYS = 50
    
    # 运行训练
    main(
        algorithm=ALGORITHM, 
        total_timesteps=TOTAL_TIMESTEPS,
        accelerated=True,  # 使用GPU优化配置
        gpu_id=GPU_ID,
        n_predators=N_PREDATORS,
        n_preys=N_PREYS
    )
    
    # ========================================
    # 💡 优化说明：
    # ========================================
    # 这个版本通过以下方式加速：
    # 1. 大幅增加batch size (256 -> 1024)
    #    → 充分利用GPU并行计算能力
    # 2. 增加gradient steps (1 -> 4)
    #    → 每次采样后更多训练更新
    # 3. 增加n_steps (PPO: 2048 -> 8192)
    #    → 减少环境交互开销
    # 
    # 预期效果：
    # - GPU利用率: 显著提升 (30-40% -> 70-80%)
    # - 训练效率: 提升30-50%
    # - 总时间: 可能减少30-40%
    # 
    # 优势: 简单稳定，无多进程复杂性
    # ========================================
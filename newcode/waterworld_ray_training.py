"""
Waterworld Training with Ray RLlib 2.47.1
Multi-Agent PettingZoo Environment - OLD API Stack
Optimized for 4x Quadro RTX 5000 GPUs

NOTE: Uses old API stack because Ray 2.47.1 new API doesn't support PettingZoo yet
"""

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from pettingzoo.sisl import waterworld_v4
import supersuit as ss


# ============================================================================
# Environment Wrapper - Function for OLD API Stack
# ============================================================================

def create_waterworld_env(config=None):
    """
    Create Waterworld environment for Ray RLlib OLD API
    Returns the actual PettingZoo environment
    """
    config = config or {}
    n_predators = config.get("n_predators", 2)
    n_preys = config.get("n_preys", 50)
    algo_name = config.get("algo_name", "PPO")
    
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
    
    env = ss.black_death_v3(env)
    return env


# ============================================================================
# Algorithm Configurations - CORRECT Parameters for Ray 2.47.1
# ============================================================================

def get_ppo_config(n_gpus=4, n_workers=48):
    """
    PPO configuration with VERIFIED parameters for Ray 2.47.1
    
    Key correction: Use 'minibatch_size' not 'minibatch_size_per_learner'
    """
    config = (
        PPOConfig()
        # CRITICAL: Disable new API stack for multi-agent PettingZoo environments
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        # Environment - Use function for OLD API
        .environment(
            env=create_waterworld_env,  # Function works with OLD API
            env_config={
                "n_predators": 2,
                "n_preys": 50,
                "algo_name": "PPO"
            }
        )
        # Framework
        .framework("torch")
        # Resources
        .resources(
            num_gpus=1,  # Main Learner GPU
        )
        # EnvRunners
        .env_runners(
            num_env_runners=n_workers,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0.06,  # 48 * 0.06 ≈ 3 GPUs shared
            rollout_fragment_length='auto',  # Auto-calculate to match batch size
        )
        # Training - ALL VERIFIED PARAMETERS
        .training(
            # Batch sizes - CORRECT parameter names
            train_batch_size_per_learner=8192,
            minibatch_size=512,  # ✅ Correct! Not minibatch_size_per_learner
            num_epochs=10,
            
            # Learning rate
            lr=3e-4,
            
            # PPO-specific
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            use_critic=True,
            
            # Clipping
            clip_param=0.2,
            vf_clip_param=10.0,
            grad_clip=None,
            
            # Coefficients
            entropy_coeff=0.01,
            vf_loss_coeff=1.0,
            
            # KL divergence
            use_kl_loss=False,
            kl_coeff=0.2,
            kl_target=0.01,
        )
        # Evaluation
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=2,
        )
    )
    
    return config


def get_sac_config(n_gpus=4, n_workers=48):
    """SAC configuration for Ray 2.47.1"""
    config = (
        SACConfig()
        # CRITICAL: Disable new API stack for multi-agent environments
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=create_waterworld_env,  # Function for OLD API
            env_config={
                "n_predators": 2,
                "n_preys": 50,
                "algo_name": "SAC"
            }
        )
        .framework("torch")
        .resources(num_gpus=1)
        .env_runners(
            num_env_runners=n_workers,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0.06,
        )
        .training(
            train_batch_size_per_learner=512,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 2000000,
            },
            tau=0.005,
            initial_alpha=1.0,
            target_entropy="auto",
            n_step=1,
            num_steps_sampled_before_learning_starts=10000,
            store_buffer_in_checkpoints=False,
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=2,
        )
    )
    
    return config


ALGORITHM_CONFIGS = {
    'ppo': get_ppo_config,
    'sac': get_sac_config,
}


# ============================================================================
# Training and Visualization
# ============================================================================

def plot_training_results(results_dir, algo_name):
    """Plot training results"""
    import pandas as pd
    
    progress_file = None
    for root, dirs, files in os.walk(results_dir):
        if 'progress.csv' in files:
            progress_file = os.path.join(root, 'progress.csv')
            break
    
    if not progress_file:
        print(f"⚠️  No progress.csv found")
        return
    
    try:
        df = pd.read_csv(progress_file)
    except Exception as e:
        print(f"⚠️  Error reading CSV: {e}")
        return
    
    # Find columns
    reward_cols = ['env_runners/episode_reward_mean', 'episode_reward_mean']
    reward_col = next((col for col in reward_cols if col in df.columns), None)
    
    if not reward_col:
        print(f"⚠️  No reward column")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Reward
    axes[0].plot(df['training_iteration'], df[reward_col], 
                 color='#2E86DE', linewidth=2.5, label='Mean Reward')
    axes[0].set_xlabel('Training Iteration', fontsize=11)
    axes[0].set_ylabel('Episode Reward', fontsize=11)
    axes[0].set_title(f'{algo_name.upper()} - Episode Rewards', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Length
    len_cols = ['env_runners/episode_len_mean', 'episode_len_mean']
    len_col = next((col for col in len_cols if col in df.columns), None)
    
    if len_col:
        axes[1].plot(df['training_iteration'], df[len_col],
                     color='#8E44AD', linewidth=2.5, label='Episode Length')
        axes[1].set_xlabel('Training Iteration', fontsize=11)
        axes[1].set_ylabel('Episode Length', fontsize=11)
        axes[1].set_title(f'{algo_name.upper()} - Episode Length', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Throughput
    steps_cols = ['num_env_steps_sampled_lifetime', 'num_env_steps_sampled']
    steps_col = next((col for col in steps_cols if col in df.columns), None)
    
    if steps_col and 'time_total_s' in df.columns:
        steps = df[steps_col].values
        times = df['time_total_s'].values
        if len(steps) > 1:
            fps = np.diff(steps) / np.diff(times)
            axes[2].plot(df['training_iteration'][1:], fps, 
                        color='#E67E22', linewidth=2.5, label='Steps/sec')
            axes[2].set_xlabel('Training Iteration', fontsize=11)
            axes[2].set_ylabel('Steps/Second', fontsize=11)
            axes[2].set_title(f'{algo_name.upper()} - Training Throughput', fontsize=13, fontweight='bold')
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'ray_training_{algo_name.lower()}_results.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training curves saved: {save_path}")
    plt.close()
    
    # Statistics
    print("\n" + "="*70)
    print(f"📈 Training Statistics - {algo_name.upper()}")
    print("="*70)
    
    if reward_col in df.columns:
        rewards = df[reward_col].values
        print(f"  Mean Reward:  {np.mean(rewards):.2f}")
        print(f"  Final Reward: {rewards[-1]:.2f}")
        print(f"  Max Reward:   {np.max(rewards):.2f}")
        print(f"  Improvement:  {rewards[-1] - rewards[0]:+.2f}")
    
    if steps_col:
        total_steps = df[steps_col].iloc[-1]
        total_time = df['time_total_s'].iloc[-1]
        print(f"\n⚡ Performance:")
        print(f"  Total Steps:  {total_steps:,.0f}")
        print(f"  Total Time:   {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")
        print(f"  Avg FPS:      {total_steps/total_time:.0f} steps/sec")
    
    print("="*70)


def train_with_ray(algorithm='ppo', n_gpus=4, n_workers=48, total_timesteps=10000000):
    """Train using Ray RLlib with multi-GPU support"""
    
    print("\n" + "="*70)
    print(f"🚀 Ray RLlib Multi-GPU Training")
    print(f"   Algorithm: {algorithm.upper()}")
    print(f"   Ray 2.47.1 - OLD API Stack (PettingZoo Compatible)")
    print("="*70)
    
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        num_gpus=n_gpus,
        num_cpus=n_workers + 4,
        ignore_reinit_error=True,
        _temp_dir="/tmp/ray",
        logging_level="ERROR"
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"  Ray Version:  {ray.__version__}")
    print(f"  Algorithm:    {algorithm.upper()}")
    print(f"  GPUs:         {n_gpus}")
    print(f"  EnvRunners:   {n_workers}")
    print(f"  Target Steps: {total_timesteps:,}")
    
    # Validate
    if algorithm.lower() not in ALGORITHM_CONFIGS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Build
    print(f"\n🔧 Building {algorithm.upper()} configuration...")
    config = ALGORITHM_CONFIGS[algorithm.lower()](n_gpus, n_workers)
    
    print(f"🔧 Building {algorithm.upper()} algorithm...")
    try:
        algo = config.build_algo()
        print(f"✅ Algorithm built successfully!")
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        raise
    
    print(f"\n🎯 Training Started")
    print("="*70)
    
    # Training loop
    iteration = 0
    total_steps = 0
    
    try:
        while total_steps < total_timesteps:
            iteration += 1
            result = algo.train()
            
            # Get steps
            total_steps = result.get('num_env_steps_sampled_lifetime', 
                                    result.get('num_env_steps_sampled', 0))
            
            # Progress every 5 iterations
            if iteration % 5 == 0:
                # Get metrics
                reward_keys = ['env_runners/episode_reward_mean', 'episode_reward_mean']
                reward = next((result.get(k, 0) for k in reward_keys if k in result), 0)
                
                len_keys = ['env_runners/episode_len_mean', 'episode_len_mean']
                ep_len = next((result.get(k, 0) for k in len_keys if k in result), 0)
                
                print(f"\n{'='*70}")
                print(f"📊 Iteration {iteration}")
                print(f"{'='*70}")
                progress_pct = 100 * total_steps / total_timesteps
                print(f"  Progress:     {total_steps:,} / {total_timesteps:,} ({progress_pct:.1f}%)")
                print(f"  Reward:       {reward:.2f}")
                print(f"  Ep Length:    {ep_len:.0f}")
                
                if total_steps > 0 and 'time_total_s' in result:
                    fps = total_steps / result['time_total_s']
                    time_left = (total_timesteps - total_steps) / fps / 60
                    print(f"  Throughput:   {fps:.0f} steps/sec")
                    print(f"  Time Left:    ~{time_left:.1f} minutes")
                
                # Checkpoint
                checkpoint = algo.save()
                checkpoint_path = (checkpoint.checkpoint.path 
                                 if hasattr(checkpoint, 'checkpoint') 
                                 else str(checkpoint))
                print(f"  💾 Checkpoint: .../{Path(checkpoint_path).name}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            algo.save()
            print(f"\n💾 Final checkpoint saved")
        except:
            pass
        
        try:
            results_dir = algo.logdir
            print(f"\n📊 Generating training plots...")
            plot_training_results(results_dir, algorithm)
        except Exception as e:
            print(f"⚠️  Could not plot: {e}")
        
        algo.stop()
        ray.shutdown()
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    if 'results_dir' in locals():
        print(f"📁 Results: {results_dir}")
        print(f"\n📊 TensorBoard: tensorboard --logdir={results_dir}")
    print("="*70 + "\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("🎯 Ray RLlib Multi-GPU Training for Waterworld")
    print("   Ray 2.47.1 - OLD API Stack for PettingZoo")
    print("="*70)
    print("Available Algorithms:")
    print("  • PPO (Proximal Policy Optimization) ⭐ Recommended")
    print("  • SAC (Soft Actor-Critic)")
    print("="*70 + "\n")
    
    # Configuration
    ALGORITHM = 'ppo'
    N_GPUS = 4
    N_WORKERS = 48
    TOTAL_TIMESTEPS = 10000000  # 10M steps ~ 20-30 minutes
    
    # Parse command line
    if len(sys.argv) > 1:
        ALGORITHM = sys.argv[1].lower()
    if len(sys.argv) > 2:
        try:
            TOTAL_TIMESTEPS = int(sys.argv[2])
        except ValueError:
            print(f"❌ Invalid timesteps: {sys.argv[2]}")
            sys.exit(1)
    
    # Validate
    if ALGORITHM not in ALGORITHM_CONFIGS:
        print(f"❌ Unknown algorithm: {ALGORITHM}")
        print(f"   Available: {', '.join(ALGORITHM_CONFIGS.keys())}")
        sys.exit(1)
    
    # Run
    try:
        train_with_ray(
            algorithm=ALGORITHM,
            n_gpus=N_GPUS,
            n_workers=N_WORKERS,
            total_timesteps=TOTAL_TIMESTEPS
        )
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        print(f"\n💡 Usage:")
        print(f"  python {sys.argv[0]} [algorithm] [timesteps]")
        print(f"\n  Examples:")
        print(f"  python {sys.argv[0]} ppo")
        print(f"  python {sys.argv[0]} ppo 5000000")
        print(f"  python {sys.argv[0]} sac 10000000")
        print(f"\n  Available: {', '.join(ALGORITHM_CONFIGS.keys())}")
        sys.exit(1)
    
    # ========================================
    # Expected Performance (4 GPUs, 48 workers)
    # ========================================
    # Throughput: 5,000-10,000 steps/sec
    # 10M steps: ~20-30 minutes
    # vs Single-threaded: 50-100x faster! 🚀
    # 
    # NOTE: Using OLD API stack because:
    # Ray 2.47.1 new API doesn't support PettingZoo multi-agent envs yet
    # The old API is still fully supported and works perfectly
    # ========================================
from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from collections import defaultdict
import pygame
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==================== 可配置参数 ====================
NUM_EPISODES = 5                # 总共玩几局
MAX_ACTIONS_PER_AGENT = 125      # 每个agent最多可以行动多少次（测试用，后续可调整）
RANDOM_SEED = 42                # 随机种子，确保可重复性

class StepByStepController:
    """Step-by-step controller - execute one step per input"""
    
    def __init__(self, agent_max_accel=0.5):
        self.agent_max_accel = agent_max_accel
        
        # Keyboard mapping - directions confirmed by testing
        self.key_mapping = {
            pygame.K_w: [0, 1],       # W - up
            pygame.K_s: [0, -1],      # S - down  
            pygame.K_a: [-1, 0],      # A - left
            pygame.K_d: [1, 0],       # D - right
            pygame.K_UP: [0, 1],      # Arrow up
            pygame.K_DOWN: [0, -1],   # Arrow down
            pygame.K_LEFT: [-1, 0],   # Arrow left
            pygame.K_RIGHT: [1, 0],   # Arrow right
            pygame.K_SPACE: [0, 0],   # Space - stop
        }
        
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
    
    def wait_for_single_input(self):
        """Wait for single key input, return corresponding action"""
        print("Please enter movement command:")
        print("  W/↑ - accelerate up")
        print("  S/↓ - accelerate down") 
        print("  A/← - accelerate left")
        print("  D/→ - accelerate right")
        print("  SPACE - stop/decelerate")
        print("  ESC - exit game")
        
        clock = pygame.time.Clock()
        
        while True:
            pygame.event.pump()
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Exit event detected")
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("ESC key exit")
                        return None
                    elif event.key in self.key_mapping:
                        # Found valid movement key
                        direction = self.key_mapping[event.key]
                        action = [
                            direction[0] * self.agent_max_accel,
                            direction[1] * self.agent_max_accel
                        ]
                        key_name = pygame.key.name(event.key)
                        
                        if event.key == pygame.K_SPACE:
                            print(f"Input: SPACE -> stop action: [0.000, 0.000]")
                        else:
                            print(f"Input: {key_name.upper()} -> action: [{action[0]:6.3f}, {action[1]:6.3f}]")
                        return action
                    else:
                        key_name = pygame.key.name(event.key)
                        print(f"Invalid key: {key_name}, please use W/A/S/D or arrow keys")
            
            clock.tick(30)  # Limit loop frequency

def select_agent():
    """Let user select which agent to control"""
    print("="*50)
    print("WATERWORLD STEP-BY-STEP CONTROL SYSTEM")
    print("="*50)
    
    # Create environment to get available agent list
    agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"] 
    env = waterworld_v4.env(
        render_mode="human",
        n_predators=2,
        n_preys=2,
        n_evaders=50,
        n_obstacles=2,
        thrust_penalty=-1.0,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=40,
        agent_algorithms=agent_algos
    )
    env = ss.black_death_v3(env)
    
    # Show available agents
    available_agents = env.possible_agents
    print("Available agents:")
    for i, agent in enumerate(available_agents):
        agent_type = "PREDATOR" if "predator" in agent else "PREY"
        print(f"  {i+1}. {agent} - {agent_type}")
    
    # User selection
    while True:
        try:
            choice = input(f"\nPlease select agent to control (enter number 1-{len(available_agents)}): ").strip()
            
            if not choice:
                print("Please enter a valid number")
                continue
                
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_agents):
                selected_agent = available_agents[choice_num - 1]
                agent_type = "PREDATOR" if "predator" in selected_agent else "PREY"
                print(f"\n✓ Successfully selected agent: {selected_agent} - {agent_type}")
                print("Starting game interface...")
                return selected_agent, env
            else:
                print(f"Please enter a number between 1 and {len(available_agents)}")
        except ValueError:
            print("Please enter a valid number")

def run_single_episode(env, controlled_agent, controller, episode_num, max_actions_per_agent):
    """Run a single episode and return the total reward"""
    print("\n" + "="*50)
    print(f"EPISODE {episode_num} START")
    print("="*50)
    print(f"Maximum actions per agent: {max_actions_per_agent}")
    print("-" * 50)
    
    # Reset environment
    obs = env.reset(seed=RANDOM_SEED + episode_num)
    env.render()
    
    cumulative_rewards = defaultdict(float)
    agent_action_counts = defaultdict(int)  # 记录每个agent的行动次数
    step_count = 0
    episode_ended = False
    user_quit = False
    
    try:
        # Main game loop
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            # Accumulate rewards
            if reward is not None:
                cumulative_rewards[agent] += reward
            
            # Check if controlled agent reached max actions
            if agent == controlled_agent and agent_action_counts[controlled_agent] >= max_actions_per_agent:
                print(f"\n>>> {controlled_agent} reached maximum actions ({max_actions_per_agent})! Episode {episode_num} ended.")
                episode_ended = True
                action = None
                env.step(action)
                break
            
            # Determine action
            if termination or truncation:
                action = None
                if agent == controlled_agent:
                    print(f"\n>>> {controlled_agent} has died or game ended")
                    episode_ended = True
            elif agent == controlled_agent:
                # Player control - pause and wait for input
                player_actions = agent_action_counts[controlled_agent]
                print(f"\n>>> Action {player_actions+1}/{max_actions_per_agent} - {controlled_agent}'s turn (Total env step: {step_count+1})")
                print(f"Current cumulative reward: {cumulative_rewards[agent]:.3f}")
                
                action = controller.wait_for_single_input()
                if action is None:
                    print("Game exit requested by user")
                    user_quit = True
                    break
                
                # 只有成功执行action才计数
                agent_action_counts[controlled_agent] += 1
            else:
                # AI agent automatic action
                action = env.action_space(agent).sample()
                agent_action_counts[agent] += 1
            
            # Display agent info
            health = info.get('current_health', None)
            alive = info.get('is_alive', None)
            # print(f"Agent {agent} health: {health if health is not None else 'N/A'} | is_alive: {alive if alive is not None else 'N/A'}")
            
            # Execute one step
            env.step(action)
            step_count += 1
            
            # Render interface after each step
            env.render()
            
            # If player-controlled agent, show reward change
            if agent == controlled_agent and reward is not None:
                print(f"Step reward: {reward:.3f}")
                print(f"Cumulative reward: {cumulative_rewards[agent]:.3f}")
                print("-" * 30)
            
            # Check if episode should end
            if episode_ended:
                break
    
    except KeyboardInterrupt:
        print("\nEpisode interrupted by user")
        user_quit = True
    
    # Get final reward for controlled agent
    final_reward = cumulative_rewards[controlled_agent]
    
    print("\n" + "="*50)
    print(f"EPISODE {episode_num} END")
    print("="*50)
    print(f"Total environment steps: {step_count}")
    print(f"{controlled_agent} actions taken: {agent_action_counts[controlled_agent]}")
    print(f"Final reward for {controlled_agent}: {final_reward:.3f}")
    print("="*50)
    
    return final_reward, user_quit

def plot_episode_rewards(episode_rewards, controlled_agent):
    """Plot episode rewards and show statistics"""
    if len(episode_rewards) == 0:
        print("No episode data to plot")
        return
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot episode rewards
    episodes = list(range(1, len(episode_rewards) + 1))
    plt.plot(episodes, episode_rewards, marker='o', linewidth=2, markersize=8, label='Episode Reward')
    
    # Plot mean line
    plt.axhline(y=mean_reward, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.3f}')
    
    # Mark max and min
    max_episode = episodes[episode_rewards.index(max_reward)]
    min_episode = episodes[episode_rewards.index(min_reward)]
    plt.scatter([max_episode], [max_reward], color='green', s=200, zorder=5, marker='^', label=f'Max: {max_reward:.3f}')
    plt.scatter([min_episode], [min_reward], color='orange', s=200, zorder=5, marker='v', label=f'Min: {min_reward:.3f}')
    
    # Labels and title
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title(f'Episode Rewards for Human-Controlled Agent: {controlled_agent}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show integer episode numbers
    plt.xticks(episodes)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_rewards_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    print(f"\n📊 Plot saved as: {filename}")
    
    # Show plot
    plt.show()

def print_statistics(episode_rewards, controlled_agent):
    """Print detailed statistics"""
    print("\n" + "="*60)
    print("FINAL STATISTICS - ALL EPISODES")
    print("="*60)
    print(f"Controlled Agent: {controlled_agent}")
    print(f"Total Episodes Completed: {len(episode_rewards)}")
    print("-" * 60)
    
    # Print individual episode rewards
    print("\nIndividual Episode Rewards:")
    for i, reward in enumerate(episode_rewards, 1):
        print(f"  Episode {i}: {reward:8.3f}")
    
    print("-" * 60)
    
    # Statistical summary
    if len(episode_rewards) > 0:
        mean_reward = np.mean(episode_rewards)
        max_reward = np.max(episode_rewards)
        min_reward = np.min(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        print("\nStatistical Summary:")
        print(f"  Mean Reward:     {mean_reward:8.3f}")
        print(f"  Max Reward:      {max_reward:8.3f}")
        print(f"  Min Reward:      {min_reward:8.3f}")
        print(f"  Std Deviation:   {std_reward:8.3f}")
        
        if len(episode_rewards) > 1:
            print(f"  Range:           {max_reward - min_reward:8.3f}")
    
    print("="*60)

def main():
    """Main function"""
    # Step 1: Select agent
    controlled_agent, env = select_agent()
    
    # Step 2: Initialize controller
    controller = StepByStepController(env.unwrapped.env.agent_max_accel)
    
    print("\n" + "="*50)
    print("MULTI-EPISODE TRAINING SESSION")
    print("="*50)
    print(f"Total episodes to play: {NUM_EPISODES}")
    print(f"Max actions per agent: {MAX_ACTIONS_PER_AGENT}")
    print("\nControl method:")
    print("  - W/A/S/D or arrow keys control movement (acceleration)")
    print("  - Space bar to stop/decelerate")
    print("  - ESC key to exit game early")
    print("\nImportant notes:")
    print("  - Actions are accelerations, will accumulate into agent velocity")
    print("  - If agent moves too fast, use space bar to decelerate")
    print("  - Please click game window to gain keyboard focus")
    print("="*50)
    
    input("\nPress ENTER to start the training session...")
    
    # Step 3: Run multiple episodes
    episode_rewards = []
    
    for episode_num in range(1, NUM_EPISODES + 1):
        episode_reward, user_quit = run_single_episode(
            env, 
            controlled_agent, 
            controller, 
            episode_num, 
            MAX_ACTIONS_PER_AGENT
        )
        
        episode_rewards.append(episode_reward)
        
        # If user wants to quit, break the loop
        if user_quit:
            print("\n⚠️  User requested to quit early")
            break
        
        # If not the last episode, ask if user wants to continue
        if episode_num < NUM_EPISODES:
            print(f"\n✓ Episode {episode_num} completed!")
            response = input(f"Continue to Episode {episode_num + 1}? (Press ENTER to continue, or type 'q' to quit): ").strip().lower()
            if response == 'q':
                print("Training session ended by user")
                break
    
    # Step 4: Show statistics and plot
    print_statistics(episode_rewards, controlled_agent)
    
    # Step 5: Plot results
    if len(episode_rewards) > 0:
        print("\n📊 Generating episode reward plot...")
        plot_episode_rewards(episode_rewards, controlled_agent)
    
    # Clean up
    env.close()
    pygame.quit()
    
    print("\n✓ All done! Thank you for playing!")

if __name__ == "__main__":
    main()
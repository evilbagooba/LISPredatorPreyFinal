from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from collections import defaultdict
import pygame
import numpy as np

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
            # Add stop command
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
        render_mode="human",  # Ensure correct render mode
        n_predators=2,
        n_preys=2,
        n_evaders=50,
        sensor_range=0.3,
        n_obstacles=2,
        thrust_penalty =-0.5,
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

def main():
    """Main function"""
    # Step 1: Select agent
    controlled_agent, env = select_agent()
    
    # Step 2: Initialize controller
    controller = StepByStepController(env.unwrapped.env.agent_max_accel)
    
    print("\n" + "="*50)
    print("GAME START")
    print("="*50)
    print("Control method:")
    print("  - Environment starts in paused state by default")
    print("  - After each movement input, environment executes one step")
    print("  - W/A/S/D or arrow keys control movement (acceleration)")
    print("  - Space bar to stop/decelerate")
    print("  - ESC key to exit game")
    print()
    print("Important notes:")
    print("  - Actions are accelerations, will accumulate into agent velocity")
    print("  - If agent moves too fast, use space bar to decelerate")
    print("  - Please click game window to gain keyboard focus")
    print("-" * 50)
    
    # Step 3: Reset environment and show initial interface
    obs = env.reset(seed=42)
    env.render()  # Immediately render initial state
    print("Game interface started, environment reset")
    
    cumulative_rewards = defaultdict(float)
    step_count = 0
    
    try:
        # Main game loop
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            # Accumulate rewards
            if reward is not None:
                cumulative_rewards[agent] += reward
            
            # Determine action
            if termination or truncation:
                action = None
                if agent == controlled_agent:
                    print(f"\n>>> {controlled_agent} has died or game ended")
                    break
            elif agent == controlled_agent:
                # Player control - pause and wait for input
                print(f"\n>>> Step {step_count+1} - {controlled_agent}'s turn")
                print(f"Current cumulative reward: {cumulative_rewards[agent]:.3f}")
                
                action = controller.wait_for_single_input()
                if action is None:
                    print("Game exit")
                    break
            else:
                # AI agent automatic action
                action = env.action_space(agent).sample()
            health = info.get('current_health', None)
            alive = info.get('is_alive', None)
            print(f"Agent {agent} health: {health if health is not None else 'N/A'} | is_alive: {alive if alive is not None else 'N/A'}")


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
    
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    
    finally:
        print("\n" + "="*50)
        print("GAME END - Final Statistics")
        print("="*50)
        
        for agent_name, total_reward in cumulative_rewards.items():
            agent_type = "PREDATOR" if "predator" in agent_name else "PREY"
            if agent_name == controlled_agent:
                print(f">>> {agent_name} ({agent_type}) [Player Controlled]: {total_reward:.3f}")
            else:
                # print(f"    {agent_name} ({agent_type}) [AI Controlled]: {total_reward:.3f}")
                pass
        
        print(f"\nPlayer-controlled {controlled_agent} final score: {cumulative_rewards[controlled_agent]:.3f}")
        
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main()
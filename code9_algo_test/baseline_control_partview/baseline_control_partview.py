from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from collections import defaultdict
import pygame
import numpy as np

# 导入 Ego 渲染器（需要将上面的代码保存为 ego_renderer.py）
from ego_renderer import EgoRenderer

class StepByStepController:
    """Step-by-step controller - execute one step per input"""
    
    def __init__(self, agent_max_accel=0.5):
        self.agent_max_accel = agent_max_accel
        
        # Keyboard mapping
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
        print("\n" + "="*50)
        print("Waiting for your command...")
        print("  W/↑ - up | S/↓ - down | A/← - left | D/→ - right")
        print("  SPACE - stop | ESC - quit")
        print("="*50)
        
        clock = pygame.time.Clock()
        
        while True:
            # 🔥 关键：使用 get() 而不是 pump()，这样才能真正处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Window closed")
                    return None
                    
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("ESC pressed - exiting")
                        return None
                        
                    elif event.key in self.key_mapping:
                        direction = self.key_mapping[event.key]
                        action = [
                            direction[0] * self.agent_max_accel,
                            direction[1] * self.agent_max_accel
                        ]
                        key_name = pygame.key.name(event.key)
                        
                        if event.key == pygame.K_SPACE:
                            print(f">>> Action: STOP [0.000, 0.000]")
                        else:
                            print(f">>> Action: {key_name.upper()} [{action[0]:6.3f}, {action[1]:6.3f}]")
                        return action
                        
                    else:
                        key_name = pygame.key.name(event.key)
                        print(f"Invalid key: {key_name}, use W/A/S/D or arrows")
            
            # 保持窗口响应
            pygame.display.flip()
            clock.tick(30)  # 限制循环频率为 30 FPS

def select_agent_last_position():
    """自动选择最后位置的 agent（推荐用于最佳时序体验）"""
    print("="*60)
    print("WATERWORLD EGO VIEW CONTROL SYSTEM")
    print("="*60)
    
    # Create environment
    agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"] 
    env = waterworld_v4.env(
        render_mode=None,  # 🔥 关键：不使用环境渲染
        n_predators=2,
        n_preys=3,
        n_evaders=50,
        n_obstacles=2,
        thrust_penalty=-1.0,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=40,
        agent_algorithms=agent_algos
    )
    env = ss.black_death_v3(env)
    
    available_agents = env.possible_agents
    
    # 自动选择最后一个 agent（最佳时序体验）
    selected_agent = available_agents[-1]
    agent_type = "PREDATOR" if "predator" in selected_agent else "PREY"
    
    print(f"\n✓ Auto-selected LAST agent: {selected_agent} ({agent_type})")
    print("  → This ensures minimal observation delay")
    print("  → Your action triggers immediate environment update")
    
    return selected_agent, env

def select_agent_manual():
    """手动选择 agent（可能有观测延迟）"""
    print("="*60)
    print("WATERWORLD EGO VIEW CONTROL SYSTEM")
    print("="*60)
    
    agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"] 
    env = waterworld_v4.env(
        render_mode=None,  # 🔥 不使用环境渲染
        n_predators=2,
        n_preys=3,
        n_evaders=50,
        n_obstacles=2,
        thrust_penalty=-1.0,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=40,
        agent_algorithms=agent_algos
    )
    env = ss.black_death_v3(env)
    
    available_agents = env.possible_agents
    print("\nAvailable agents:")
    for i, agent in enumerate(available_agents):
        agent_type = "PREDATOR" if "predator" in agent else "PREY"
        position_note = " (LAST - best timing)" if i == len(available_agents) - 1 else ""
        print(f"  {i+1}. {agent} - {agent_type}{position_note}")
    
    while True:
        try:
            choice = input(f"\nSelect agent (1-{len(available_agents)}) or press Enter for auto: ").strip()
            
            if not choice:
                # 默认选择最后一个
                selected_agent = available_agents[-1]
                break
                
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_agents):
                selected_agent = available_agents[choice_num - 1]
                
                # 警告非最后位置的选择
                if choice_num != len(available_agents):
                    print("  ⚠️  Warning: Non-last agent may have observation delay")
                break
            else:
                print(f"Please enter 1-{len(available_agents)}")
        except ValueError:
            print("Invalid input")
    
    agent_type = "PREDATOR" if "predator" in selected_agent else "PREY"
    print(f"\n✓ Selected: {selected_agent} ({agent_type})")
    
    return selected_agent, env

def main():
    """Main function with Ego View rendering"""
    
    # Step 1: Select agent (推荐使用自动选择最后位置)
    controlled_agent, env = select_agent_last_position()
    # 或者使用手动选择：
    # controlled_agent, env = select_agent_manual()
    
    # Step 2: Initialize controller and Ego renderer
    controller = StepByStepController(env.unwrapped.env.agent_max_accel)
    
    # 🔥 创建 Ego 渲染器（独立窗口）
    ego_renderer = EgoRenderer(
        n_sensors=30,
        window_size=900,
        danger_zone=0.3
    )
    
    print("\n" + "="*60)
    print("GAME START - EGO VIEW MODE")
    print("="*60)
    print("Features:")
    print("  ✓ Independent Ego window (sensor-based view only)")
    print("  ✓ No global information leak")
    print("  ✓ Real-time health, metrics, and event display")
    print("  ✓ Strategy pointer (hunt/escape suggestion)")
    print("\nControls:")
    print("  - W/A/S/D or Arrow keys: Move (acceleration)")
    print("  - SPACE: Stop/decelerate")
    print("  - ESC: Quit game")
    print("\nRadar Layers (from inner to outer):")
    print("  - Innermost: Obstacles (gray)")
    print("  - Barriers (gray)")
    print("  - Food (green)")
    print("  - Poison (red)")
    print("  - Outermost: Other agents (red=predator, blue=prey)")
    print("\nVelocity Arrows:")
    print("  - Inward arrow: Object approaching")
    print("  - Outward arrow: Object moving away")
    print("-" * 60)
    
    # Step 3: Reset environment
    obs = env.reset(seed=42)
    # 不调用 env.render() - 使用 Ego 视图
    
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
                    print(f"\n>>> {controlled_agent} has terminated")
                    break
            elif agent == controlled_agent:
                # 🔥 Player control - render Ego view and wait for input
                print(f"\n{'='*60}")
                print(f"STEP {step_count+1} - YOUR TURN ({controlled_agent})")
                print(f"{'='*60}")
                print(f"Cumulative Reward: {cumulative_rewards[agent]:.2f}")
                
                # 🔥 渲染 Ego 视图（独立窗口）
                ego_renderer.render(observation, info)
                
                # 等待玩家输入
                action = controller.wait_for_single_input()
                if action is None:
                    print("Game quit by player")
                    break
                
                # 显示即时反馈
                if reward is not None:
                    reward_symbol = "+" if reward >= 0 else ""
                    print(f"Last Step Reward: {reward_symbol}{reward:.3f}")
            else:
                # AI agent automatic action
                action = env.action_space(agent).sample()
            
            # Execute step
            env.step(action)
            step_count += 1
            
            # 🔥 不调用 env.render() - 只在玩家回合显示 Ego 视图
            
            # Print status for controlled agent
            if agent == controlled_agent and reward is not None:
                health = info.get('current_health', 'N/A')
                print(f"Health: {health}")
                print("-" * 60)
    
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user (Ctrl+C)")
    
    finally:
        print("\n" + "="*60)
        print("GAME END - FINAL STATISTICS")
        print("="*60)
        
        agent_type = "PREDATOR" if "predator" in controlled_agent else "PREY"
        final_reward = cumulative_rewards[controlled_agent]
        
        print(f"\nControlled Agent: {controlled_agent} ({agent_type})")
        print(f"Final Score: {final_reward:.2f}")
        print(f"Total Steps: {step_count}")
        
        # 显示其他统计
        if controlled_agent in cumulative_rewards:
            avg_reward = final_reward / max(1, step_count)
            print(f"Average Reward per Step: {avg_reward:.3f}")
        
        print("\n" + "="*60)
        
        # Close resources
        ego_renderer.close()
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main()
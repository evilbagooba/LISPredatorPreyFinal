from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from collections import defaultdict
import pygame
import numpy as np

class StepByStepController:
    """Step-by-step controller - execute one step per input"""
    
    def __init__(self, agent_max_accel=0.5):
        self.agent_max_accel = agent_max_accel
        
        # Keyboard mapping
        self.key_mapping = {
            pygame.K_w: [0, 1],       
            pygame.K_s: [0, -1],      
            pygame.K_a: [-1, 0],      
            pygame.K_d: [1, 0],       
            pygame.K_UP: [0, 1],      
            pygame.K_DOWN: [0, -1],   
            pygame.K_LEFT: [-1, 0],   
            pygame.K_RIGHT: [1, 0],   
            pygame.K_SPACE: [0, 0],   
        }
        
        if not pygame.get_init():
            pygame.init()
    
    def wait_for_single_input(self):
        """Wait for single key input"""
        print("Commands: W/A/S/D or arrows, SPACE=stop, ESC=exit")
        
        clock = pygame.time.Clock()
        
        while True:
            pygame.event.pump()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    elif event.key in self.key_mapping:
                        direction = self.key_mapping[event.key]
                        action = [
                            direction[0] * self.agent_max_accel,
                            direction[1] * self.agent_max_accel
                        ]
                        key_name = pygame.key.name(event.key)
                        print(f"Input: {key_name.upper()} -> action: [{action[0]:6.3f}, {action[1]:6.3f}]")
                        return action
            
            clock.tick(30)

def select_agent():
    """Let user select which agent to control"""
    print("="*50)
    print("WATERWORLD STEP-BY-STEP CONTROL SYSTEM")
    print("="*50)
    
    agent_algos = ["PPO", "PPO", "DQN", "DQN"] * 40
    env = waterworld_v4.env(
        render_mode="human",
        n_predators=1,
        n_preys=1,
        n_evaders=200,
        n_obstacles=2,

        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=20,
        thrust_penalty=0,  # 🔥 改回负值，这样能看到基础奖励
        agent_algorithms=agent_algos,
    )

    # 🔥 关键修复：移除 black_death wrapper
    # env = ss.black_death_v3(env)  # ← 注释掉这行
    
    available_agents = env.possible_agents
    print("Available agents:")
    for i, agent in enumerate(available_agents):
        agent_type = "PREDATOR" if "predator" in agent else "PREY"
        print(f"  {i+1}. {agent} - {agent_type}")
    
    while True:
        try:
            choice = input(f"\nSelect agent (1-{len(available_agents)}): ").strip()
            
            if not choice:
                continue
                
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_agents):
                selected_agent = available_agents[choice_num - 1]
                agent_type = "PREDATOR" if "predator" in selected_agent else "PREY"
                print(f"\n✓ Selected: {selected_agent} - {agent_type}")
                return selected_agent, env
            else:
                print(f"Please enter 1-{len(available_agents)}")
        except ValueError:
            print("Please enter a valid number")

def main():
    controlled_agent, env = select_agent()
    controller = StepByStepController(env.unwrapped.env.agent_max_accel)
    
    print("\n" + "="*50)
    print("GAME START")
    print("="*50)
    
    obs = env.reset(seed=42)
    env.render()
    
    # 🔥 增强：追踪所有agent的奖励
    episode_rewards = defaultdict(list)  # 存储每步的奖励
    cumulative_rewards = defaultdict(float)
    step_count = 0
    last_round_events = {}  # 存储上一轮的事件信息
    
    try:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            print(f"\n reward: {reward:.3f}")
            # 🔥 记录奖励
            episode_rewards[agent].append(reward)
            cumulative_rewards[agent] += reward
            
            # 🔥 检查事件（从info中）
            events = []
            if info.get('food_caught', False):
                events.append("🥕 ATE FOOD")
            if info.get('predator_catch', False):
                events.append("🎯 CAUGHT PREY")
            if info.get('prey_caught', False):
                events.append("💀 GOT CAUGHT")
            
            # 🔥 详细显示controlled agent的信息
            if agent == controlled_agent:
                print(f"\n{'='*60}")
                print(f">>> Step {step_count+1} - {controlled_agent}'s turn")
                print(f"{'='*60}")
                
                # 显示上一步的结果
                if step_count > 0:
                    print(f"📊 LAST STEP RESULTS:")
                    print(f"   Reward received: {reward:.3f}")
                    if events:
                        print(f"   Events: {', '.join(events)}")
                    if agent in last_round_events and last_round_events[agent]:
                        print(f"   Previous events: {', '.join(last_round_events[agent])}")
                
                # 显示当前状态
                print(f"📈 CURRENT STATUS:")
                print(f"   Cumulative reward: {cumulative_rewards[agent]:.3f}")
                health = info.get('current_health', 'N/A')
                print(f"   Health: {health}")
                
                # 显示性能指标
                if 'performance_metrics' in info:
                    perf = info['performance_metrics']
                    print(f"   Performance: foraging={perf.get('foraging_rate', 0):.2f}, "
                          f"escape={perf.get('escape_rate', 0):.2f}, "
                          f"hunting={perf.get('hunting_rate', 0):.2f}")
                
                if termination or truncation:
                    print(f"\n💀 {controlled_agent} terminated!")
                    print(f"   Final reward: {cumulative_rewards[agent]:.3f}")
                    break
                
                action = controller.wait_for_single_input()
                if action is None:
                    break
            else:
                # AI agent
                if termination or truncation:
                    action = None
                else:
                    action = env.action_space(agent).sample()
                
                # 🔥 显示其他agent的重要事件
                if events:
                    print(f"🔔 {agent}: {', '.join(events)} (reward: {reward:.3f})")
            
            # 保存事件信息
            last_round_events[agent] = events
            
            env.step(action)
            step_count += 1
            env.render()
    
    except KeyboardInterrupt:
        print("\nGame interrupted")
    
    finally:
        print("\n" + "="*50)
        print("GAME END - Final Statistics")
        print("="*50)
        
        # 🔥 详细统计
        for agent_name in env.possible_agents:
            agent_type = "PREDATOR" if "predator" in agent_name else "PREY"
            total_reward = cumulative_rewards[agent_name]
            num_steps = len(episode_rewards[agent_name])
            avg_reward = total_reward / num_steps if num_steps > 0 else 0
            
            marker = ">>> " if agent_name == controlled_agent else "    "
            control_tag = "[PLAYER]" if agent_name == controlled_agent else "[AI]"
            
            print(f"{marker}{agent_name} ({agent_type}) {control_tag}:")
            print(f"       Total: {total_reward:.3f} | Avg/step: {avg_reward:.3f} | Steps: {num_steps}")
            
            # 显示奖励分布
            if num_steps > 0:
                positive_rewards = [r for r in episode_rewards[agent_name] if r > 0]
                negative_rewards = [r for r in episode_rewards[agent_name] if r < 0]
                print(f"       Positive events: {len(positive_rewards)} | Negative: {len(negative_rewards)}")
        
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main()
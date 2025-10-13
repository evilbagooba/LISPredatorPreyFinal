"""
终端输入版本 - 如果 Pygame 键盘不工作，用这个
Ego 窗口只负责显示，输入从终端读取
"""

from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from collections import defaultdict
import pygame
import numpy as np
from ego_renderer import EgoRenderer
import threading

class TerminalInputController:
    """从终端读取输入的控制器"""
    
    def __init__(self, agent_max_accel=0.5):
        self.agent_max_accel = agent_max_accel
        
        # 命令映射
        self.command_mapping = {
            'w': [0, 1],
            'W': [0, 1],
            'up': [0, 1],
            
            's': [0, -1],
            'S': [0, -1],
            'down': [0, -1],
            
            'a': [-1, 0],
            'A': [-1, 0],
            'left': [-1, 0],
            
            'd': [1, 0],
            'D': [1, 0],
            'right': [1, 0],
            
            'space': [0, 0],
            ' ': [0, 0],
            'stop': [0, 0],
            
            'q': None,
            'Q': None,
            'quit': None,
            'exit': None,
        }
    
    def wait_for_single_input(self):
        """从终端等待输入"""
        print("\n" + "="*60)
        print("🎮 YOUR TURN - Enter command:")
        print("="*60)
        print("Commands:")
        print("  w/up    - Move up")
        print("  s/down  - Move down")
        print("  a/left  - Move left")
        print("  d/right - Move right")
        print("  space   - Stop")
        print("  q/quit  - Quit game")
        print("-"*60)
        
        while True:
            try:
                command = input(">>> ").strip().lower()
                
                if not command:
                    print("⚠️  Empty input, please enter a command")
                    continue
                
                if command in self.command_mapping:
                    action_dir = self.command_mapping[command]
                    
                    if action_dir is None:
                        print("❌ Quitting game...")
                        return None
                    
                    action = [
                        action_dir[0] * self.agent_max_accel,
                        action_dir[1] * self.agent_max_accel
                    ]
                    
                    if action == [0, 0]:
                        print(f"✅ Action: STOP [0.000, 0.000]")
                    else:
                        print(f"✅ Action: {command.upper()} [{action[0]:6.3f}, {action[1]:6.3f}]")
                    
                    return action
                else:
                    print(f"⚠️  Unknown command: '{command}'")
                    print("    Use: w/s/a/d, arrows, space, or q")
            
            except EOFError:
                print("\n❌ EOF detected - exiting")
                return None
            
            except KeyboardInterrupt:
                print("\n❌ Interrupted - exiting")
                return None

def update_ego_display(ego_renderer, observation, info, stop_event):
    """在单独的线程中更新 Ego 显示"""
    while not stop_event.is_set():
        try:
            ego_renderer.render(observation, info)
            pygame.time.wait(33)  # ~30 FPS
            
            # 处理窗口关闭事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    stop_event.set()
                    return
        except:
            break

def terminal_main():
    """使用终端输入的主函数"""
    print("="*60)
    print("WATERWORLD - TERMINAL INPUT MODE")
    print("="*60)
    print("Note: Ego window shows visualization")
    print("      Commands are entered in terminal")
    print("="*60)
    
    # 创建环境
    agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"]
    env = waterworld_v4.env(
        render_mode=None,
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
    controlled_agent = available_agents[-1]
    agent_type = "PREDATOR" if "predator" in controlled_agent else "PREY"
    
    print(f"\n✓ Controlled agent: {controlled_agent} ({agent_type})")
    
    controller = TerminalInputController(env.unwrapped.env.agent_max_accel)
    ego_renderer = EgoRenderer(n_sensors=30, window_size=900, danger_zone=0.3)
    
    print("\n" + "="*60)
    print("STARTING GAME")
    print("="*60)
    print("📺 Ego window: Visual feedback")
    print("⌨️  Terminal: Command input")
    print("-"*60)
    
    obs = env.reset(seed=42)
    cumulative_rewards = defaultdict(float)
    step_count = 0
    
    # 检查初始状态
    print("\nInitial agent status:")
    for agent in env.possible_agents:
        if agent == controlled_agent:
            # 获取初始观测
            env.reset(seed=42)
            for a in env.agent_iter():
                if a == controlled_agent:
                    observation, reward, termination, truncation, info = env.last()
                    health = info.get('current_health', 100)
                    is_alive = info.get('is_alive', True)
                    print(f"  {controlled_agent}: Health={health:.1f}, Alive={is_alive}")
                    break
                env.step(None)
            break
    
    # 重新开始游戏
    obs = env.reset(seed=42)
    
    # 用于控制显示线程的事件
    stop_display = threading.Event()
    current_obs = None
    current_info = None
    
    try:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if reward is not None:
                cumulative_rewards[agent] += reward
            
            # 检查是否终止
            if termination or truncation:
                action = None
                if agent == controlled_agent:
                    # 显示终止原因
                    death_cause = info.get('death_cause', 'unknown')
                    is_alive = info.get('is_alive', False)
                    health = info.get('current_health', 0)
                    
                    print(f"\n{'='*60}")
                    print(f"AGENT TERMINATED: {controlled_agent}")
                    print(f"{'='*60}")
                    print(f"Status: {'ALIVE' if is_alive else 'DEAD'}")
                    print(f"Health: {health:.1f}")
                    print(f"Cause: {death_cause if death_cause else 'Game ended'}")
                    print(f"Total Steps: {step_count}")
                    print(f"Final Reward: {cumulative_rewards[agent]:.2f}")
                    break
            
            elif agent == controlled_agent:
                print(f"\n{'='*60}")
                print(f"STEP {step_count+1}")
                print(f"{'='*60}")
                print(f"Agent: {controlled_agent}")
                print(f"Cumulative Reward: {cumulative_rewards[agent]:.2f}")
                
                # 显示详细状态
                health = info.get('current_health', 100)
                is_alive = info.get('is_alive', True)
                print(f"Health: {health:.1f}")
                print(f"Status: {'ALIVE' if is_alive else 'DEAD'}")
                
                # 如果已经死亡但还没 terminate，给出警告
                if not is_alive:
                    print("WARNING: Agent health depleted but still in game!")
                
                # 更新 Ego 显示
                ego_renderer.render(observation, info)
                
                # 从终端获取输入
                action = controller.wait_for_single_input()
                
                if action is None:
                    print("Game quit by player")
                    break
                
                if reward is not None:
                    reward_symbol = "+" if reward >= 0 else ""
                    print(f"Last Reward: {reward_symbol}{reward:.3f}")
            
            else:
                # AI agent
                action = env.action_space(agent).sample()
            
            env.step(action)
            step_count += 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by Ctrl+C")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n" + "="*60)
        print("GAME END")
        print("="*60)
        
        final_reward = cumulative_rewards[controlled_agent]
        print(f"Final Score: {final_reward:.2f}")
        print(f"Total Steps: {step_count}")
        
        stop_display.set()
        ego_renderer.close()
        env.close()
        pygame.quit()
        print("✓ Cleanup complete")

if __name__ == "__main__":
    terminal_main()
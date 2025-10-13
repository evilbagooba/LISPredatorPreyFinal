"""
调试版本 - 用于排查键盘输入问题
"""

from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from collections import defaultdict
import pygame
import numpy as np
from ego_renderer import EgoRenderer

class DebugController:
    """带调试输出的控制器"""
    
    def __init__(self, agent_max_accel=0.5):
        self.agent_max_accel = agent_max_accel
        
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
            print("✓ Pygame initialized")
    
    def wait_for_single_input(self):
        """带详细调试输出的输入等待"""
        print("\n" + "="*60)
        print("🎮 WAITING FOR INPUT")
        print("="*60)
        print("Controls: W/A/S/D or Arrows | SPACE=stop | ESC=quit")
        print("Debugging: Press any key to see if it's detected...")
        print("-"*60)
        
        clock = pygame.time.Clock()
        frame_count = 0
        
        while True:
            frame_count += 1
            
            # 每 30 帧输出一次心跳
            if frame_count % 30 == 0:
                print(f"💓 Waiting... (frame {frame_count}) - Click the Ego window first!")
            
            # 获取所有事件
            events = pygame.event.get()
            
            if events:
                print(f"📥 Detected {len(events)} event(s)")
            
            for event in events:
                print(f"  Event type: {pygame.event.event_name(event.type)}")
                
                if event.type == pygame.QUIT:
                    print("❌ QUIT event - window closed")
                    return None
                
                elif event.type == pygame.KEYDOWN:
                    key_name = pygame.key.name(event.key)
                    print(f"  ⌨️  Key pressed: {key_name} (code: {event.key})")
                    
                    if event.key == pygame.K_ESCAPE:
                        print("❌ ESC pressed - exiting")
                        return None
                    
                    elif event.key in self.key_mapping:
                        direction = self.key_mapping[event.key]
                        action = [
                            direction[0] * self.agent_max_accel,
                            direction[1] * self.agent_max_accel
                        ]
                        
                        if event.key == pygame.K_SPACE:
                            print(f"✅ Action: STOP [0.000, 0.000]")
                        else:
                            print(f"✅ Action: {key_name.upper()} [{action[0]:6.3f}, {action[1]:6.3f}]")
                        return action
                    
                    else:
                        print(f"⚠️  Invalid key: {key_name}")
                
                elif event.type == pygame.KEYUP:
                    print(f"  Key released: {pygame.key.name(event.key)}")
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    print(f"  Mouse clicked at {event.pos}")
            
            # 保持窗口响应
            pygame.display.flip()
            clock.tick(30)

def debug_main():
    """调试版主函数"""
    print("="*60)
    print("DEBUG MODE - Keyboard Input Troubleshooting")
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
    controlled_agent = available_agents[-1]  # 选择最后一个
    
    print(f"\n✓ Environment created")
    print(f"✓ Controlled agent: {controlled_agent}")
    
    # 创建控制器和渲染器
    controller = DebugController(env.unwrapped.env.agent_max_accel)
    print(f"✓ Controller created")
    
    ego_renderer = EgoRenderer(n_sensors=30, window_size=900, danger_zone=0.3)
    print(f"✓ Ego renderer created")
    print(f"✓ Pygame windows: {len(pygame.display.list_modes())} mode(s) available")
    
    # 检查窗口焦点
    print("\n" + "="*60)
    print("⚠️  IMPORTANT: Make sure to CLICK on the Ego window")
    print("    to give it keyboard focus before pressing keys!")
    print("="*60)
    
    input("\nPress ENTER to start the game...")
    
    # 重置环境
    obs = env.reset(seed=42)
    print(f"\n✓ Environment reset")
    
    cumulative_rewards = defaultdict(float)
    step_count = 0
    
    try:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if reward is not None:
                cumulative_rewards[agent] += reward
            
            if termination or truncation:
                action = None
                if agent == controlled_agent:
                    print(f"\n>>> {controlled_agent} terminated")
                    break
            
            elif agent == controlled_agent:
                print(f"\n{'='*60}")
                print(f"STEP {step_count+1} - YOUR TURN")
                print(f"{'='*60}")
                print(f"Agent: {controlled_agent}")
                print(f"Reward: {cumulative_rewards[agent]:.2f}")
                print(f"Health: {info.get('current_health', 'N/A')}")
                print(f"Observation shape: {observation.shape}")
                
                # 渲染 Ego 视图
                print(f"\n🎨 Rendering Ego view...")
                ego_renderer.render(observation, info)
                print(f"✓ Ego view rendered")
                
                # 等待输入
                action = controller.wait_for_single_input()
                
                if action is None:
                    print("Game quit")
                    break
                
                print(f"✓ Action received: {action}")
            
            else:
                # AI agent
                action = env.action_space(agent).sample()
            
            env.step(action)
            step_count += 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by Ctrl+C")
    
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n" + "="*60)
        print("CLEANUP")
        print("="*60)
        
        final_reward = cumulative_rewards[controlled_agent]
        print(f"Final Score: {final_reward:.2f}")
        print(f"Total Steps: {step_count}")
        
        ego_renderer.close()
        env.close()
        pygame.quit()
        print("✓ All resources closed")

if __name__ == "__main__":
    debug_main()
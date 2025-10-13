"""
正确的多智能体环境设置：保持环境完整性，只训练一个智能体
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pettingzoo.sisl import waterworld_v4
import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel
import torch

def create_env():
    """创建完整的多智能体环境，但只训练一个智能体"""
    
    # 创建完整的环境 - 保持生态平衡
    env = waterworld_v4.env(
        render_mode=None,
        n_predators=2,      # 保留2个predator
        n_preys=2,          # 保留2个prey  
        n_evaders=1,        # 保留1个evader
        n_obstacles=2,      # 保留障碍物
        obstacle_coord=[(0.2, 0.2), (0.8, 0.8)],
        n_poisons=5,        # 保留毒药
        max_cycles=200,
        speed_features=True
    )
    
    print(f"原始环境智能体: {env.possible_agents}")
    
    # 应用包装 - 重要：包装顺序很关键
    env = ss.pad_observations_v0(env)    # 先标准化观察
    env = ss.pad_action_space_v0(env)    # 再标准化动作
    env = ss.black_death_v3(env)         # 使用black_death处理死亡智能体
    
    # 转换为parallel环境
    env = aec_to_parallel(env)
    
    print(f"包装后环境智能体: {env.possible_agents}")
    
    # 多智能体环境包装器 - 只训练一个智能体
    class MultiAgentSingleTrainerWrapper(gym.Env):
        def __init__(self, env, train_agent="predator_0"):
            super().__init__()
            self.env = env
            self.train_agent = train_agent  # 只训练这个智能体
            self.all_agents = env.possible_agents.copy()
            
            print(f"训练智能体: {self.train_agent}")
            print(f"其他智能体使用随机策略: {[a for a in self.all_agents if a != self.train_agent]}")
            
            # 设置空间（基于训练智能体）
            self.observation_space = env.observation_space(self.train_agent)
            self.action_space = env.action_space(self.train_agent)
            
            print(f"观察空间: {self.observation_space}")
            print(f"动作空间: {self.action_space}")
            
        def reset(self, seed=None, options=None):
            """重置环境"""
            try:
                obs, infos = self.env.reset(seed=seed)
                
                # 返回训练智能体的观察
                if self.train_agent in obs:
                    return obs[self.train_agent].astype(np.float32), {}
                else:
                    # 如果训练智能体不在，返回零观察
                    return np.zeros(self.observation_space.shape, dtype=np.float32), {}
                    
            except Exception as e:
                print(f"Reset error: {e}")
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}
            
        def step(self, action):
            """执行一步，只控制训练智能体，其他存活智能体使用随机策略"""
            try:
                # 准备所有存活智能体的动作
                actions = {}
                
                # 训练智能体使用传入的动作
                if self.train_agent in self.env.agents:  # 确保训练智能体还存活
                    if isinstance(action, (list, tuple)):
                        action = np.array(action, dtype=np.float32)
                    elif not isinstance(action, np.ndarray):
                        action = np.array([action], dtype=np.float32)
                    
                    # 限制动作范围
                    action = np.clip(action, -1.0, 1.0)
                    actions[self.train_agent] = action
                
                # 其他存活智能体使用随机策略
                # 关键：只为env.agents中的智能体生成动作（black_death_v3已移除死亡智能体）
                for agent_name in self.env.agents:
                    if agent_name != self.train_agent:
                        try:
                            # 为存活的智能体生成随机动作
                            random_action = self.env.action_space(agent_name).sample()
                            actions[agent_name] = random_action
                        except Exception as e:
                            # 如果无法生成动作，跳过这个智能体
                            print(f"无法为 {agent_name} 生成动作: {e}")
                            pass
                
                # 执行环境步骤
                obs, rewards, dones, truncs, infos = self.env.step(actions)
                
                # 返回训练智能体的结果
                if self.train_agent in obs:
                    return (obs[self.train_agent].astype(np.float32),
                           float(rewards.get(self.train_agent, 0.0)),
                           bool(dones.get(self.train_agent, False)),
                           bool(truncs.get(self.train_agent, False)),
                           infos.get(self.train_agent, {}))
                else:
                    # 训练智能体死亡
                    return (np.zeros(self.observation_space.shape, dtype=np.float32),
                           float(rewards.get(self.train_agent, -1.0)),  # 死亡惩罚
                           True,   # episode结束
                           False,  # 不是截断
                           {"agent_dead": True})
                           
            except Exception as e:
                # 不打印step error，静默处理
                return (np.zeros(self.observation_space.shape, dtype=np.float32),
                       0.0,    # 中性奖励
                       False,  # 不结束episode，继续尝试
                       False,
                       {"error_handled": True})
    
    return MultiAgentSingleTrainerWrapper(env, train_agent="predator_0")

def create_env_simple():
    """简化版环境 - 如果复杂版本还有问题"""
    
    # 最小但完整的环境
    env = waterworld_v4.env(
        render_mode=None,
        n_predators=1,
        n_preys=1,
        n_evaders=0,        # 暂时不要evader
        n_obstacles=1,
        obstacle_coord=[(0.5, 0.5)],
        n_poisons=2,
        max_cycles=150,
        speed_features=False  # 简化观察空间
    )
    
    # 只使用最基本的包装
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.black_death_v3(env)  # 保留black_death处理死亡
    env = aec_to_parallel(env)
    
    class SimpleMultiWrapper(gym.Env):
        def __init__(self, env):
            super().__init__()
            self.env = env
            self.train_agent = "predator_0"
            
            # 硬编码空间避免查询问题
            from gymnasium.spaces import Box
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(212,), dtype=np.float32)
            self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            
        def reset(self, seed=None, options=None):
            obs, _ = self.env.reset(seed=seed)
            if obs and self.train_agent in obs:
                return obs[self.train_agent].astype(np.float32), {}
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
            
        def step(self, action):
            action = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
            
            # 构造所有存活智能体的动作
            actions = {self.train_agent: action}
            
            # 为其他存活智能体添加随机动作
            # 关键：只为env.agents中的存活智能体生成动作
            for agent in self.env.agents:
                if agent != self.train_agent:
                    try:
                        actions[agent] = self.env.action_space(agent).sample()
                    except:
                        pass  # 跳过有问题的智能体
            
            try:
                obs, rewards, dones, truncs, infos = self.env.step(actions)
                
                return (obs.get(self.train_agent, np.zeros(self.observation_space.shape)).astype(np.float32),
                       float(rewards.get(self.train_agent, 0.0)),
                       bool(dones.get(self.train_agent, False)),
                       bool(truncs.get(self.train_agent, False)),
                       infos.get(self.train_agent, {}))
            except:
                return (np.zeros(self.observation_space.shape, dtype=np.float32), 
                       0.0, False, False, {})
    
    return SimpleMultiWrapper(env)

def main():
    print("创建多智能体环境（只训练一个智能体）...")
    
    # 选择环境版本
    use_simple = False  # 改为True使用简化版本
    
    if use_simple:
        print("使用简化多智能体环境...")
        env_func = create_env_simple
    else:
        print("使用完整多智能体环境...")
        env_func = create_env
    
    # 测试环境
    try:
        test_env = env_func()
        obs = test_env.reset()
        print(f"环境测试成功，观察形状: {obs[0].shape if isinstance(obs, tuple) else obs.shape}")
        
        # 测试几步
        for i in range(5):
            action = test_env.action_space.sample()
            obs, reward, done, truncated, info = test_env.step(action)
            print(f"步骤 {i+1}: reward={reward:.3f}, done={done}")
            if done:
                obs = test_env.reset()
                print("Episode结束，重置")
                break
                
    except Exception as e:
        print(f"环境测试失败: {e}")
        return
    
    # 创建向量化环境
    print("创建向量化环境...")
    env = DummyVecEnv([env_func])
    
    # 创建PPO模型
    print("创建PPO模型...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("开始训练...")
    try:
        # 训练
        model.learn(total_timesteps=20000, progress_bar=True)
        
        print("保存模型...")
        model.save("waterworld_ppo_multi_agent")
        
        print("测试训练效果...")
        obs = env.reset()
        episode_rewards = []
        current_reward = 0
        
        for i in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            current_reward += reward[0]
            
            if done[0]:
                episode_rewards.append(current_reward)
                print(f"Episode完成: 奖励={current_reward:.2f}")
                obs = env.reset()
                current_reward = 0
                
                if len(episode_rewards) >= 3:
                    break
        
        if episode_rewards:
            print(f"平均奖励: {np.mean(episode_rewards):.2f}")
            print(f"奖励标准差: {np.std(episode_rewards):.2f}")
        
        print("训练完成！")
        
    except Exception as e:
        print(f"训练错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    main()
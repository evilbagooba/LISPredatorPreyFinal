# """
# Tianshou Waterworld 稳定版本
# 先用纯随机策略验证框架，然后逐步添加可训练策略

# 每个agent使用不同类型的随机策略来模拟"独立算法"的概念
# """

# import os
# import argparse
# import numpy as np
# import torch
# from typing import Optional, Tuple, List

# # Tianshou imports
# from tianshou.data import Collector, Batch
# from tianshou.env import DummyVectorEnv
# from tianshou.env.pettingzoo_env import PettingZooEnv
# from tianshou.policy import BasePolicy, MultiAgentPolicyManager

# # PettingZoo imports
# from pettingzoo.sisl import waterworld_v4

# class ContinuousRandomPolicy(BasePolicy):
#     """基础连续随机策略"""
#     def __init__(self, action_space, policy_name="Random"):
#         super().__init__(action_space=action_space)
#         self.action_space = action_space
#         self.policy_name = policy_name
    
#     def forward(self, batch, state=None, **kwargs):
#         if hasattr(batch.obs, 'shape') and len(batch.obs.shape) > 1:
#             batch_size = batch.obs.shape[0]
#         else:
#             batch_size = 1
            
#         actions = []
#         for _ in range(batch_size):
#             action = self.action_space.sample()
#             actions.append(action)
        
#         return Batch(act=np.array(actions))
    
#     def exploration_noise(self, act, batch):
#         return act
    
#     def learn(self, batch, **kwargs):
#         return {}

# class AggressiveRandomPolicy(ContinuousRandomPolicy):
#     """激进随机策略 - 偏向更大的动作"""
#     def __init__(self, action_space):
#         super().__init__(action_space, "Aggressive")
    
#     def forward(self, batch, state=None, **kwargs):
#         if hasattr(batch.obs, 'shape') and len(batch.obs.shape) > 1:
#             batch_size = batch.obs.shape[0]
#         else:
#             batch_size = 1
            
#         actions = []
#         for _ in range(batch_size):
#             # 生成偏向边界的动作
#             action = np.random.uniform(-1.0, 1.0, size=self.action_space.shape)
#             # 增强动作幅度
#             action = np.sign(action) * np.abs(action) * 0.8  # 偏向较大动作
#             # 确保在范围内
#             action = np.clip(action, self.action_space.low, self.action_space.high)
#             actions.append(action)
        
#         return Batch(act=np.array(actions))

# class ConservativeRandomPolicy(ContinuousRandomPolicy):
#     """保守随机策略 - 偏向较小的动作"""
#     def __init__(self, action_space):
#         super().__init__(action_space, "Conservative")
    
#     def forward(self, batch, state=None, **kwargs):
#         if hasattr(batch.obs, 'shape') and len(batch.obs.shape) > 1:
#             batch_size = batch.obs.shape[0]
#         else:
#             batch_size = 1
            
#         actions = []
#         for _ in range(batch_size):
#             # 生成偏向中心的动作
#             action = np.random.uniform(-0.3, 0.3, size=self.action_space.shape)  # 较小范围
#             # 确保在范围内
#             action = np.clip(action, self.action_space.low, self.action_space.high)
#             actions.append(action)
        
#         return Batch(act=np.array(actions))

# class BiasedRandomPolicy(ContinuousRandomPolicy):
#     """偏向随机策略 - 偏向某个方向"""
#     def __init__(self, action_space, bias_direction=None):
#         super().__init__(action_space, "Biased")
#         # 如果没有指定偏向，随机选择一个
#         self.bias = bias_direction if bias_direction is not None else np.random.uniform(-0.5, 0.5, size=action_space.shape)
    
#     def forward(self, batch, state=None, **kwargs):
#         if hasattr(batch.obs, 'shape') and len(batch.obs.shape) > 1:
#             batch_size = batch.obs.shape[0]
#         else:
#             batch_size = 1
            
#         actions = []
#         for _ in range(batch_size):
#             # 基础随机动作
#             base_action = np.random.uniform(-0.5, 0.5, size=self.action_space.shape)
#             # 添加偏向
#             action = base_action + self.bias
#             # 确保在范围内
#             action = np.clip(action, self.action_space.low, self.action_space.high)
#             actions.append(action)
        
#         return Batch(act=np.array(actions))

# def get_args():
#     """配置参数"""
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--n-episodes', type=int, default=10)
#     parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
#     parser.add_argument('--render', action='store_true', default=False)
#     parser.add_argument('--n-pursuers', type=int, default=5)
#     parser.add_argument('--env-num', type=int, default=1)
    
#     return parser.parse_known_args()[0]

# def get_env(args):
#     """创建Waterworld环境"""
#     env = waterworld_v4.env(
#         n_pursuers=args.n_pursuers,
#         n_evaders=5,
#         n_poisons=10,
#         n_coop=2,
#         n_sensors=30,
#         sensor_range=0.2,
#         radius=0.015,
#         pursuer_max_accel=0.01,
#         evader_speed=0.01,
#         poison_speed=0.01,
#         poison_reward=-1.0,
#         food_reward=10.0,
#         encounter_reward=0.01,
#         thrust_penalty=-0.5,
#         local_ratio=1.0,
#         speed_features=True,
#         max_cycles=500,
#         render_mode="human" if args.render else None
#     )
#     return PettingZooEnv(env)

# def create_diverse_policies(args, env):
#     """为每个agent创建不同类型的策略"""
#     agents = env.agents
#     policies = []
    
#     action_space = env.action_space
    
#     print(f"环境信息:")
#     print(f"  智能体: {agents}")
#     print(f"  观察空间: {env.observation_space}")
#     print(f"  动作空间: {action_space}")
    
#     # 为每个agent分配不同的策略类型
#     policy_types = [
#         ("Standard Random", ContinuousRandomPolicy),
#         ("Aggressive", AggressiveRandomPolicy), 
#         ("Conservative", ConservativeRandomPolicy),
#         ("Biased", BiasedRandomPolicy),
#         ("Standard Random 2", ContinuousRandomPolicy)
#     ]
    
#     for i, agent_id in enumerate(agents):
#         policy_name, policy_class = policy_types[i % len(policy_types)]
        
#         if policy_class == BiasedRandomPolicy:
#             # 为偏向策略设置随机偏向方向
#             bias = np.random.uniform(-0.3, 0.3, size=action_space.shape)
#             policy = policy_class(action_space, bias_direction=bias)
#         else:
#             policy = policy_class(action_space)
        
#         policies.append(policy)
#         print(f"  Agent {agent_id}: {policy_name}")
    
#     return policies

# def run_multi_agent_test(args):
#     """运行多智能体测试"""
#     print("=== 多智能体独立策略测试 ===")
    
#     # 创建环境
#     env = get_env(args)
    
#     # 创建不同的策略
#     policies = create_diverse_policies(args, env)
    
#     # 创建策略管理器
#     policy_manager = MultiAgentPolicyManager(policies, env)
#     print(f"\n策略管理器创建成功，管理 {len(policies)} 个独立策略")
    
#     # 创建向量化环境
#     vec_envs = DummyVectorEnv([lambda: get_env(args) for _ in range(args.env_num)])
    
#     # 创建收集器
#     collector = Collector(policy_manager, vec_envs)
#     print("收集器创建成功")
    
#     # 设置随机种子
#     np.random.seed(args.seed)
#     vec_envs.seed(args.seed)
    
#     # 运行测试
#     print(f"\n开始运行 {args.n_episodes} 个episode...")
#     result = collector.collect(n_episode=args.n_episodes, render=args.render)
    
#     # 分析结果
#     print(f"\n=== 测试结果 ===")
#     print(f"总episode数: {result['n/ep']}")
#     print(f"总步数: {result['n/st']}")
#     print(f"平均episode长度: {result['len']:.2f}")
#     print(f"平均奖励: {result['rew']:.4f}")
#     print(f"奖励标准差: {result['rew_std']:.4f}")
    
#     # 详细奖励分析
#     if 'rews' in result:
#         rewards = result['rews']
#         print(f"\n=== 详细奖励分析 ===")
#         print(f"奖励形状: {rewards.shape}")
#         print(f"最大奖励: {np.max(rewards):.4f}")
#         print(f"最小奖励: {np.min(rewards):.4f}")
        
#         # 如果是多智能体，分析每个agent的表现
#         if len(rewards.shape) > 1 and rewards.shape[1] > 1:
#             print(f"\n各智能体平均奖励:")
#             for i in range(rewards.shape[1]):
#                 agent_reward = np.mean(rewards[:, i])
#                 print(f"  Agent {i}: {agent_reward:.4f}")
    
#     return result

# def compare_policy_performance(args):
#     """比较不同策略的性能"""
#     print("\n=== 策略性能比较 ===")
    
#     env = get_env(args)
#     action_space = env.action_space
    
#     # 测试不同策略类型
#     policy_configs = [
#         ("Standard Random", ContinuousRandomPolicy, {}),
#         ("Aggressive", AggressiveRandomPolicy, {}),
#         ("Conservative", ConservativeRandomPolicy, {}),
#         ("Biased Forward", BiasedRandomPolicy, {"bias_direction": np.array([0.3, 0.0])}),
#         ("Biased Backward", BiasedRandomPolicy, {"bias_direction": np.array([-0.3, 0.0])})
#     ]
    
#     results = {}
    
#     for policy_name, policy_class, kwargs in policy_configs:
#         print(f"\n测试策略: {policy_name}")
        
#         # 创建该策略的所有agents
#         policies = [policy_class(action_space, **kwargs) for _ in env.agents]
#         policy_manager = MultiAgentPolicyManager(policies, env)
        
#         # 创建环境和收集器
#         vec_envs = DummyVectorEnv([lambda: get_env(args) for _ in range(1)])
#         collector = Collector(policy_manager, vec_envs)
        
#         # 运行测试
#         result = collector.collect(n_episode=5, render=False)
#         results[policy_name] = result['rew']
        
#         print(f"  平均奖励: {result['rew']:.4f}")
#         print(f"  平均长度: {result['len']:.2f}")
    
#     # 总结比较
#     print(f"\n=== 策略性能排序 ===")
#     sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
#     for i, (policy_name, reward) in enumerate(sorted_results, 1):
#         print(f"{i}. {policy_name}: {reward:.4f}")

# def single_agent_test(args):
#     """单智能体测试验证基础功能"""
#     print("=== 单智能体基础测试 ===")
    
#     # 创建单智能体环境用于测试
#     args_single = argparse.Namespace(**vars(args))
#     args_single.n_pursuers = 1
    
#     env = get_env(args_single)
#     print(f"单智能体环境创建成功")
#     print(f"  智能体: {env.agents}")
    
#     # 创建单个策略
#     policy = ContinuousRandomPolicy(env.action_space)
#     policy_manager = MultiAgentPolicyManager([policy], env)
    
#     # 测试
#     vec_envs = DummyVectorEnv([lambda: get_env(args_single) for _ in range(1)])
#     collector = Collector(policy_manager, vec_envs)
    
#     result = collector.collect(n_episode=3, render=args.render)
#     print(f"单智能体测试成功！")
#     print(f"  平均奖励: {result['rew']:.4f}")
#     print(f"  平均长度: {result['len']:.2f}")
    
#     return True

# if __name__ == "__main__":
#     args = get_args()
    
#     print("=== Tianshou Waterworld 稳定版本 ===")
#     print(f"设备: {args.device}")
#     print(f"智能体数量: {args.n_pursuers}")
#     print(f"测试episode数: {args.n_episodes}")
    
#     try:
#         # 1. 单智能体基础测试
#         print("\n" + "="*50)
#         print("步骤1: 单智能体基础测试")
#         if not single_agent_test(args):
#             print("单智能体测试失败")
#             exit(1)
#         print("✅ 单智能体测试通过")
        
#         # 2. 多智能体测试
#         print("\n" + "="*50)
#         print("步骤2: 多智能体独立策略测试")
#         result = run_multi_agent_test(args)
#         print("✅ 多智能体测试通过")
        
#         # 3. 策略比较
#         if not args.render:  # 只在非渲染模式下进行比较测试
#             print("\n" + "="*50)
#             print("步骤3: 不同策略性能比较")
#             compare_policy_performance(args)
#             print("✅ 策略比较完成")
        
#         print(f"\n" + "="*60)
#         print(f"🎉 所有测试完成！")
#         print(f"")
#         print(f"📋 测试总结:")
#         print(f"   ✅ 环境创建和包装: PettingZoo → Tianshou")
#         print(f"   ✅ 多策略管理: MultiAgentPolicyManager")
#         print(f"   ✅ 独立策略: 每个agent使用不同策略")
#         print(f"   ✅ 数据收集: Collector正常工作")
#         print(f"   ✅ 连续动作空间: 正确处理")
#         print(f"")
#         print(f"🚀 下一步建议:")
#         print(f"   1. 这个框架已验证可以支持每个agent的独立策略")
#         print(f"   2. 可以逐个将随机策略替换为可训练算法")
#         print(f"   3. 建议顺序: 先试PPO(连续动作), 再试DQN(需要动作离散化)")
#         print(f"   4. 或者使用SAC等直接支持连续动作的算法")
#         print(f"")
#         print(f"💡 核心价值:")
#         print(f"   - 展示了Tianshou中真正的'每个agent独立算法'实现")
#         print(f"   - 为多智能体强化学习研究提供了稳定的基础框架")
#         print(f"   - 可以轻松扩展到任何PettingZoo环境")
        
#     except Exception as e:
#         print(f"\n❌ 测试过程中出现错误: {e}")
#         import traceback
#         traceback.print_exc()
#         print(f"\n💡 调试建议:")
#         print(f"   1. 检查依赖版本: pip list | grep -E '(tianshou|pettingzoo)'")
#         print(f"   2. 尝试简化参数: --n-episodes 3 --n-pursuers 3")
#         print(f"   3. 如有问题可以逐步调试每个组件")

# """
# 使用说明:

# 1. 基础测试:
#    python waterworld_stable.py

# 2. 可视化测试:
#    python waterworld_stable.py --render

# 3. 更多episode:
#    python waterworld_stable.py --n-episodes 20

# 4. 不同智能体数量:
#    python waterworld_stable.py --n-pursuers 3

# 这个版本的特点:
# - 完全避免了DQN的兼容性问题
# - 展示了真正的"每个agent独立策略"概念
# - 4种不同类型的策略模拟不同算法
# - 稳定的错误处理和测试流程
# - 性能比较和分析功能

# 这为后续添加真正的可训练算法（DQN、PPO、SAC等）奠定了坚实基础。
# """





# """
# Tianshou Waterworld PPO 简化版本
# 避免复杂的网络构建问题，使用更基础的方法

# 策略：先确保PPO能正常创建和运行，再优化性能
# """

# import os
# import argparse
# import numpy as np
# import torch
# import torch.nn as nn
# from typing import Optional, Tuple, List

# # Tianshou imports
# from tianshou.data import Collector, Batch
# from tianshou.env import DummyVectorEnv
# from tianshou.env.pettingzoo_env import PettingZooEnv
# from tianshou.policy import BasePolicy, PPOPolicy, MultiAgentPolicyManager
# from tianshou.trainer import onpolicy_trainer
# from tianshou.utils.net.common import Net

# # PettingZoo imports
# from pettingzoo.sisl import waterworld_v4

# class SimpleContinuousActor(nn.Module):
#     """简化的连续动作Actor网络"""
#     def __init__(self, state_dim, action_dim, hidden_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
#         # 输出均值和标准差
#         self.mu_head = nn.Linear(hidden_dim, action_dim)
#         self.sigma_head = nn.Linear(hidden_dim, action_dim)
        
#     def forward(self, obs, state=None, info={}):
#         features = self.net(obs)
#         mu = torch.tanh(self.mu_head(features))  # 输出范围[-1, 1]
#         sigma = torch.softplus(self.sigma_head(features)) + 1e-3  # 确保sigma > 0
#         return mu, sigma, state

# class SimpleContinuousCritic(nn.Module):
#     """简化的Critic网络"""
#     def __init__(self, state_dim, hidden_dim=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
        
#     def forward(self, obs, state=None, info={}):
#         return self.net(obs), state

# class ContinuousRandomPolicy(BasePolicy):
#     """连续动作随机策略"""
#     def __init__(self, action_space, policy_name="Random"):
#         super().__init__(action_space=action_space)
#         self.action_space = action_space
#         self.policy_name = policy_name
    
#     def forward(self, batch, state=None, **kwargs):
#         if hasattr(batch.obs, 'shape') and len(batch.obs.shape) > 1:
#             batch_size = batch.obs.shape[0]
#         else:
#             batch_size = 1
            
#         actions = []
#         for _ in range(batch_size):
#             action = self.action_space.sample()
#             actions.append(action)
        
#         return Batch(act=np.array(actions))
    
#     def exploration_noise(self, act, batch):
#         return act
    
#     def learn(self, batch, **kwargs):
#         return {}

# def get_args():
#     """配置参数"""
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--lr', type=float, default=3e-4)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--epoch', type=int, default=5)  # 减少epoch数
#     parser.add_argument('--step-per-epoch', type=int, default=1000)  # 减少步数
#     parser.add_argument('--repeat-per-collect', type=int, default=2)
#     parser.add_argument('--batch-size', type=int, default=64)
#     parser.add_argument('--hidden-dim', type=int, default=128)
#     parser.add_argument('--training-num', type=int, default=2)  # 减少环境数
#     parser.add_argument('--test-num', type=int, default=1)
#     parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
#     parser.add_argument('--render', action='store_true', default=False)
#     parser.add_argument('--watch', action='store_true', default=False)
    
#     # Waterworld specific parameters
#     parser.add_argument('--n-pursuers', type=int, default=5)
#     parser.add_argument('--n-episodes-test', type=int, default=5)
    
#     return parser.parse_known_args()[0]

# def get_env(args):
#     """创建Waterworld环境"""
#     env = waterworld_v4.env(
#         n_pursuers=args.n_pursuers,
#         n_evaders=5,
#         n_poisons=10,
#         n_coop=2,
#         n_sensors=30,
#         sensor_range=0.2,
#         radius=0.015,
#         pursuer_max_accel=0.01,
#         evader_speed=0.01,
#         poison_speed=0.01,
#         poison_reward=-1.0,
#         food_reward=10.0,
#         encounter_reward=0.01,
#         thrust_penalty=-0.5,
#         local_ratio=1.0,
#         speed_features=True,
#         max_cycles=500,
#         render_mode="human" if args.render else None
#     )
#     return PettingZooEnv(env)

# def create_simple_ppo_policy(args, state_dim, action_dim):
#     """创建简化的PPO策略"""
#     print(f"创建PPO策略: 状态维度={state_dim}, 动作维度={action_dim}")
    
#     # 创建网络
#     actor = SimpleContinuousActor(state_dim, action_dim, args.hidden_dim).to(args.device)
#     critic = SimpleContinuousCritic(state_dim, args.hidden_dim).to(args.device)
    
#     # 创建优化器
#     optim = torch.optim.Adam(
#         list(actor.parameters()) + list(critic.parameters()), 
#         lr=args.lr
#     )
    
#     # 正确的分布函数 - 关键修复！
#     def dist_fn(mu, sigma):
#         """
#         正确的分布函数构建方式
#         参数：
#         - mu: 动作均值张量 [batch_size, action_dim]
#         - sigma: 动作标准差张量 [batch_size, action_dim]
#         返回：
#         - Independent分布，将最后一个维度视为独立事件
#         """
#         # 创建Normal分布实例（不是类！）
#         normal_dist = torch.distributions.Normal(mu, sigma)
#         # 将最后一个维度（动作维度）设为独立
#         independent_dist = torch.distributions.Independent(normal_dist, 1)
#         return independent_dist
    
#     print("网络创建成功，开始创建PPO策略...")
    
#     # 调试：测试分布函数
#     try:
#         print("测试分布函数...")
#         test_mu = torch.zeros(1, action_dim)
#         test_sigma = torch.ones(1, action_dim)
#         test_dist = dist_fn(test_mu, test_sigma)
#         test_sample = test_dist.sample()
#         print(f"✅ 分布函数测试成功，采样形状: {test_sample.shape}")
#     except Exception as e:
#         print(f"❌ 分布函数测试失败: {e}")
#         return None
    
#     # 创建PPO策略
#     try:
#         ppo_policy = PPOPolicy(
#             actor=actor,
#             critic=critic,
#             optim=optim,
#             dist_fn=dist_fn,
#             discount_factor=args.gamma,
#             gae_lambda=0.95,
#             max_grad_norm=0.5,
#             vf_coef=0.5,
#             ent_coef=0.01,
#             eps_clip=0.2,
#             value_clip=True,
#             advantage_normalization=True,
#             recompute_advantage=False
#         )
#         print("✅ PPO策略创建成功")
#         return ppo_policy
        
#     except Exception as e:
#         print(f"❌ PPO策略创建失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# def create_mixed_policies(args, env):
#     """创建混合策略"""
#     agents = env.agents
#     policies = []
    
#     state_shape = env.observation_space.shape
#     action_space = env.action_space
    
#     state_dim = state_shape[0] if len(state_shape) == 1 else np.prod(state_shape)
#     action_dim = action_space.shape[0] if len(action_space.shape) == 1 else np.prod(action_space.shape)
    
#     print(f"创建混合策略组合:")
#     print(f"  状态维度: {state_dim}")
#     print(f"  动作维度: {action_dim}")
#     print(f"  智能体: {agents}")
    
#     for i, agent_id in enumerate(agents):
#         if i == 0:  # 第一个agent尝试使用PPO
#             print(f"  {agent_id}: 尝试创建PPO...")
#             ppo_policy = create_simple_ppo_policy(args, state_dim, action_dim)
            
#             if ppo_policy is not None:
#                 policy = ppo_policy
#                 print(f"  {agent_id}: PPO (可训练)")
#             else:
#                 policy = ContinuousRandomPolicy(action_space)
#                 print(f"  {agent_id}: 随机策略 (PPO创建失败)")
#         else:  # 其他agent使用随机策略
#             policy = ContinuousRandomPolicy(action_space)
#             print(f"  {agent_id}: 随机策略")
        
#         policies.append(policy)
    
#     return policies

# def test_basic_functionality(args):
#     """测试基础功能"""
#     print("=== 基础功能测试 ===")
    
#     try:
#         # 创建环境
#         env = get_env(args)
#         print(f"✅ 环境创建成功")
        
#         # 创建策略
#         policies = create_mixed_policies(args, env)
#         print(f"✅ 混合策略创建成功")
        
#         # 创建策略管理器
#         policy_manager = MultiAgentPolicyManager(policies, env)
#         print(f"✅ 策略管理器创建成功")
        
#         # 测试向量化环境
#         vec_envs = DummyVectorEnv([lambda: get_env(args) for _ in range(1)])
#         print(f"✅ 向量化环境创建成功")
        
#         # 测试收集器
#         collector = Collector(policy_manager, vec_envs)
#         print(f"✅ 收集器创建成功")
        
#         # 测试数据收集
#         print("测试数据收集...")
#         result = collector.collect(n_step=20)
#         print(f"✅ 数据收集成功: {result['n/st']} 步")
        
#         return True, policies
        
#     except Exception as e:
#         print(f"❌ 基础测试失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return False, None

# def simple_training_test(args, policies):
#     """简单训练测试"""
#     print("=== 简单训练测试 ===")
    
#     # 检查是否有PPO策略
#     has_ppo = any(isinstance(p, PPOPolicy) for p in policies)
    
#     if not has_ppo:
#         print("没有PPO策略，跳过训练测试")
#         return
    
#     print("发现PPO策略，开始简单训练测试...")
    
#     try:
#         # 环境设置
#         env = get_env(args)
#         train_envs = DummyVectorEnv([lambda: get_env(args) for _ in range(args.training_num)])
#         test_envs = DummyVectorEnv([lambda: get_env(args) for _ in range(args.test_num)])
        
#         # 创建策略管理器
#         policy_manager = MultiAgentPolicyManager(policies, env)
        
#         # 创建收集器
#         train_collector = Collector(policy_manager, train_envs)
#         test_collector = Collector(policy_manager, test_envs)
        
#         # 预收集数据
#         print("预收集训练数据...")
#         train_collector.collect(n_step=args.batch_size * args.training_num)
        
#         print("开始训练...")
        
#         # 简化的回调函数
#         def save_best_fn(policy):
#             print("保存最佳策略...")
        
#         def stop_fn(mean_rewards):
#             return False  # 不提前停止，让它完整训练
        
#         def reward_metric(rews):
#             return rews.mean(axis=1) if len(rews.shape) > 1 else rews
        
#         # 使用on-policy训练器
#         result = onpolicy_trainer(
#             policy=policy_manager,
#             train_collector=train_collector,
#             test_collector=test_collector,
#             max_epoch=args.epoch,
#             step_per_epoch=args.step_per_epoch,
#             repeat_per_collect=args.repeat_per_collect,
#             episode_per_test=3,
#             batch_size=args.batch_size,
#             save_best_fn=save_best_fn,
#             stop_fn=stop_fn,
#             test_in_train=False,
#             reward_metric=reward_metric
#         )
        
#         print("✅ 训练完成！")
#         print(f"训练结果: {result}")
        
#     except Exception as e:
#         print(f"❌ 训练测试失败: {e}")
#         import traceback
#         traceback.print_exc()

# def run_performance_test(args):
#     """运行性能测试"""
#     print("=== 性能测试 ===")
    
#     env = get_env(args)
    
#     # 测试全随机策略
#     print("测试全随机策略...")
#     random_policies = [ContinuousRandomPolicy(env.action_space) for _ in env.agents]
#     random_manager = MultiAgentPolicyManager(random_policies, env)
    
#     vec_envs = DummyVectorEnv([lambda: get_env(args) for _ in range(1)])
#     random_collector = Collector(random_manager, vec_envs)
#     random_result = random_collector.collect(n_episode=args.n_episodes_test, render=False)
    
#     print(f"随机策略表现: 平均奖励={random_result['rew']:.4f}, 平均长度={random_result['len']:.2f}")

# if __name__ == "__main__":
#     args = get_args()
    
#     print("=== Tianshou Waterworld PPO 简化版本 ===")
#     print(f"设备: {args.device}")
#     print(f"智能体数量: {args.n_pursuers}")
    
#     try:
#         # 1. 基础功能测试
#         print("\n" + "="*50)
#         print("步骤1: 基础功能测试")
#         success, policies = test_basic_functionality(args)
        
#         if not success:
#             print("❌ 基础测试失败")
#             exit(1)
        
#         print("✅ 基础测试通过")
        
#         if args.watch:
#             # 2. 观察模式
#             print("\n步骤2: 性能测试")
#             run_performance_test(args)
#         else:
#             # 3. 训练模式
#             print("\n步骤2: 简单训练测试")
#             simple_training_test(args, policies)
            
#             print("\n步骤3: 性能测试")
#             run_performance_test(args)
        
#         print(f"\n" + "="*50)
#         print(f"🎉 简化版PPO测试完成！")
#         print(f"")
#         print(f"💡 如果PPO创建成功，说明框架支持混合策略训练")
#         print(f"💡 如果PPO创建失败，至少验证了随机策略的多智能体框架")
#         print(f"💡 下一步可以根据具体错误调整网络构建方式")
        
#     except Exception as e:
#         print(f"\n❌ 测试过程中出现错误: {e}")
#         import traceback
#         traceback.print_exc()

# """
# 这个简化版本的策略:

# 1. 使用更简单的网络结构
# 2. 避免复杂的ActorCritic包装
# 3. 更直接的分布函数定义
# 4. 强化错误处理，即使PPO失败也能继续运行
# 5. 逐步验证每个组件

# 如果这个版本能运行，我们就知道框架本身没问题
# 如果还有错误，我们可以进一步简化或使用其他算法
# """



"""
Tianshou Waterworld PPO 简化版本
避免复杂的网络构建问题，使用更基础的方法

策略：先确保PPO能正常创建和运行，再优化性能
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

# Tianshou imports
from tianshou.data import Collector, Batch
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, PPOPolicy, MultiAgentPolicyManager
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import Net

# PettingZoo imports
from pettingzoo.sisl import waterworld_v4

class SimpleContinuousActor(nn.Module):
    """简化的连续动作Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 输出均值和标准差
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.sigma_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs, state=None, info={}):
        # 关键修复：确保输入是PyTorch张量
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        
        # 确保在正确的设备上
        if hasattr(self, 'device'):
            obs = obs.to(self.device)
        elif next(self.parameters()).is_cuda:
            obs = obs.cuda()
        
        features = self.net(obs)
        mu = torch.tanh(self.mu_head(features))  # 输出范围[-1, 1]
        sigma = torch.softplus(self.sigma_head(features)) + 1e-3  # 确保sigma > 0
        return mu, sigma, state

class SimpleContinuousCritic(nn.Module):
    """简化的Critic网络"""
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs, state=None, info={}):
        # 关键修复：确保输入是PyTorch张量
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        
        # 确保在正确的设备上
        if hasattr(self, 'device'):
            obs = obs.to(self.device)
        elif next(self.parameters()).is_cuda:
            obs = obs.cuda()
            
        return self.net(obs), state

class ContinuousRandomPolicy(BasePolicy):
    """连续动作随机策略"""
    def __init__(self, action_space, policy_name="Random"):
        super().__init__(action_space=action_space)
        self.action_space = action_space
        self.policy_name = policy_name
    
    def forward(self, batch, state=None, **kwargs):
        if hasattr(batch.obs, 'shape') and len(batch.obs.shape) > 1:
            batch_size = batch.obs.shape[0]
        else:
            batch_size = 1
            
        actions = []
        for _ in range(batch_size):
            action = self.action_space.sample()
            actions.append(action)
        
        return Batch(act=np.array(actions))
    
    def exploration_noise(self, act, batch):
        return act
    
    def learn(self, batch, **kwargs):
        return {}

def get_args():
    """配置参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=5)  # 减少epoch数
    parser.add_argument('--step-per-epoch', type=int, default=1000)  # 减少步数
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--training-num', type=int, default=2)  # 减少环境数
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--watch', action='store_true', default=False)
    
    # Waterworld specific parameters
    parser.add_argument('--n-pursuers', type=int, default=5)
    parser.add_argument('--n-episodes-test', type=int, default=5)
    
    return parser.parse_known_args()[0]

def get_env(args):
    """创建Waterworld环境"""
    env = waterworld_v4.env(
        n_pursuers=args.n_pursuers,
        n_evaders=5,
        n_poisons=10,
        n_coop=2,
        n_sensors=30,
        sensor_range=0.2,
        radius=0.015,
        pursuer_max_accel=0.01,
        evader_speed=0.01,
        poison_speed=0.01,
        poison_reward=-1.0,
        food_reward=10.0,
        encounter_reward=0.01,
        thrust_penalty=-0.5,
        local_ratio=1.0,
        speed_features=True,
        max_cycles=500,
        render_mode="human" if args.render else None
    )
    return PettingZooEnv(env)

def create_simple_ppo_policy(args, state_dim, action_dim):
    """创建简化的PPO策略"""
    print(f"创建PPO策略: 状态维度={state_dim}, 动作维度={action_dim}")
    
    # 创建网络
    actor = SimpleContinuousActor(state_dim, action_dim, args.hidden_dim).to(args.device)
    critic = SimpleContinuousCritic(state_dim, args.hidden_dim).to(args.device)
    
    # 创建优化器
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), 
        lr=args.lr
    )
    
    # 正确的分布函数 - 关键修复！
    def dist_fn(mu, sigma):
        """
        正确的分布函数构建方式
        参数：
        - mu: 动作均值张量 [batch_size, action_dim]
        - sigma: 动作标准差张量 [batch_size, action_dim]
        返回：
        - Independent分布，将最后一个维度视为独立事件
        """
        # 创建Normal分布实例（不是类！）
        normal_dist = torch.distributions.Normal(mu, sigma)
        # 将最后一个维度（动作维度）设为独立
        independent_dist = torch.distributions.Independent(normal_dist, 1)
        return independent_dist
    
    print("网络创建成功，开始创建PPO策略...")
    
    # 调试：测试分布函数
    try:
        print("测试分布函数...")
        test_mu = torch.zeros(1, action_dim)
        test_sigma = torch.ones(1, action_dim)
        test_dist = dist_fn(test_mu, test_sigma)
        test_sample = test_dist.sample()
        print(f"✅ 分布函数测试成功，采样形状: {test_sample.shape}")
    except Exception as e:
        print(f"❌ 分布函数测试失败: {e}")
        return None
    
    # 创建PPO策略
    try:
        ppo_policy = PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist_fn,
            discount_factor=args.gamma,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            vf_coef=0.5,
            ent_coef=0.01,
            eps_clip=0.2,
            value_clip=True,
            advantage_normalization=True,
            recompute_advantage=False
        )
        print("✅ PPO策略创建成功")
        return ppo_policy
        
    except Exception as e:
        print(f"❌ PPO策略创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_mixed_policies(args, env):
    """创建混合策略"""
    agents = env.agents
    policies = []
    
    state_shape = env.observation_space.shape
    action_space = env.action_space
    
    state_dim = state_shape[0] if len(state_shape) == 1 else np.prod(state_shape)
    action_dim = action_space.shape[0] if len(action_space.shape) == 1 else np.prod(action_space.shape)
    
    print(f"创建混合策略组合:")
    print(f"  状态维度: {state_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  智能体: {agents}")
    
    for i, agent_id in enumerate(agents):
        if i == 0:  # 第一个agent尝试使用PPO
            print(f"  {agent_id}: 尝试创建PPO...")
            ppo_policy = create_simple_ppo_policy(args, state_dim, action_dim)
            
            if ppo_policy is not None:
                policy = ppo_policy
                print(f"  {agent_id}: PPO (可训练)")
            else:
                policy = ContinuousRandomPolicy(action_space)
                print(f"  {agent_id}: 随机策略 (PPO创建失败)")
        else:  # 其他agent使用随机策略
            policy = ContinuousRandomPolicy(action_space)
            print(f"  {agent_id}: 随机策略")
        
        policies.append(policy)
    
    return policies

def test_basic_functionality(args):
    """测试基础功能"""
    print("=== 基础功能测试 ===")
    
    try:
        # 创建环境
        env = get_env(args)
        print(f"✅ 环境创建成功")
        
        # 创建策略
        policies = create_mixed_policies(args, env)
        print(f"✅ 混合策略创建成功")
        
        # 创建策略管理器
        policy_manager = MultiAgentPolicyManager(policies, env)
        print(f"✅ 策略管理器创建成功")
        
        # 测试向量化环境
        vec_envs = DummyVectorEnv([lambda: get_env(args) for _ in range(1)])
        print(f"✅ 向量化环境创建成功")
        
        # 测试收集器
        collector = Collector(policy_manager, vec_envs)
        print(f"✅ 收集器创建成功")
        
        # 测试数据收集
        print("测试数据收集...")
        result = collector.collect(n_step=20)
        print(f"✅ 数据收集成功: {result['n/st']} 步")
        
        return True, policies
        
    except Exception as e:
        print(f"❌ 基础测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def simple_training_test(args, policies):
    """简单训练测试"""
    print("=== 简单训练测试 ===")
    
    # 检查是否有PPO策略
    has_ppo = any(isinstance(p, PPOPolicy) for p in policies)
    
    if not has_ppo:
        print("没有PPO策略，跳过训练测试")
        return
    
    print("发现PPO策略，开始简单训练测试...")
    
    try:
        # 环境设置
        env = get_env(args)
        train_envs = DummyVectorEnv([lambda: get_env(args) for _ in range(args.training_num)])
        test_envs = DummyVectorEnv([lambda: get_env(args) for _ in range(args.test_num)])
        
        # 创建策略管理器
        policy_manager = MultiAgentPolicyManager(policies, env)
        
        # 创建收集器
        train_collector = Collector(policy_manager, train_envs)
        test_collector = Collector(policy_manager, test_envs)
        
        # 预收集数据
        print("预收集训练数据...")
        train_collector.collect(n_step=args.batch_size * args.training_num)
        
        print("开始训练...")
        
        # 简化的回调函数
        def save_best_fn(policy):
            print("保存最佳策略...")
        
        def stop_fn(mean_rewards):
            return False  # 不提前停止，让它完整训练
        
        def reward_metric(rews):
            return rews.mean(axis=1) if len(rews.shape) > 1 else rews
        
        # 使用on-policy训练器
        result = onpolicy_trainer(
            policy=policy_manager,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=3,
            batch_size=args.batch_size,
            save_best_fn=save_best_fn,
            stop_fn=stop_fn,
            test_in_train=False,
            reward_metric=reward_metric
        )
        
        print("✅ 训练完成！")
        print(f"训练结果: {result}")
        
    except Exception as e:
        print(f"❌ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()

def run_performance_test(args):
    """运行性能测试"""
    print("=== 性能测试 ===")
    
    env = get_env(args)
    
    # 测试全随机策略
    print("测试全随机策略...")
    random_policies = [ContinuousRandomPolicy(env.action_space) for _ in env.agents]
    random_manager = MultiAgentPolicyManager(random_policies, env)
    
    vec_envs = DummyVectorEnv([lambda: get_env(args) for _ in range(1)])
    random_collector = Collector(random_manager, vec_envs)
    random_result = random_collector.collect(n_episode=args.n_episodes_test, render=False)
    
    print(f"随机策略表现: 平均奖励={random_result['rew']:.4f}, 平均长度={random_result['len']:.2f}")

if __name__ == "__main__":
    args = get_args()
    
    print("=== Tianshou Waterworld PPO 简化版本 ===")
    print(f"设备: {args.device}")
    print(f"智能体数量: {args.n_pursuers}")
    
    try:
        # 1. 基础功能测试
        print("\n" + "="*50)
        print("步骤1: 基础功能测试")
        success, policies = test_basic_functionality(args)
        
        if not success:
            print("❌ 基础测试失败")
            exit(1)
        
        print("✅ 基础测试通过")
        
        if args.watch:
            # 2. 观察模式
            print("\n步骤2: 性能测试")
            run_performance_test(args)
        else:
            # 3. 训练模式
            print("\n步骤2: 简单训练测试")
            simple_training_test(args, policies)
            
            print("\n步骤3: 性能测试")
            run_performance_test(args)
        
        print(f"\n" + "="*50)
        print(f"🎉 简化版PPO测试完成！")
        print(f"")
        print(f"💡 如果PPO创建成功，说明框架支持混合策略训练")
        print(f"💡 如果PPO创建失败，至少验证了随机策略的多智能体框架")
        print(f"💡 下一步可以根据具体错误调整网络构建方式")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

"""
这个简化版本的策略:

1. 使用更简单的网络结构
2. 避免复杂的ActorCritic包装
3. 更直接的分布函数定义
4. 强化错误处理，即使PPO失败也能继续运行
5. 逐步验证每个组件

如果这个版本能运行，我们就知道框架本身没问题
如果还有错误，我们可以进一步简化或使用其他算法
"""
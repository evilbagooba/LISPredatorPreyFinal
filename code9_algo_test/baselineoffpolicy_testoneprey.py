from datetime import datetime

import os
import argparse
from copy import deepcopy
from typing import Optional, Tuple

import gymnasium
import numpy as np
import torch
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.data import VectorReplayBuffer, Collector  # 修改这行
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.policy import C51Policy
from tianshou.policy import PGPolicy
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.policy import PPOPolicy
from torch.distributions import Normal, Independent
from tianshou.policy import A2CPolicy
from tianshou.policy import TRPOPolicy
from tianshou.policy import NPGPolicy
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from torch.distributions import Categorical, Normal  # 添加Normal导入
from torch.utils.tensorboard import SummaryWriter
# 环境导入
import supersuit as ss
from pettingzoo.sisl import waterworld_v4
from tianshou.data import Batch

# 移除这行：from collector import Collector

import logging

# 设置日志级别为 ERROR，这样只会记录 ERROR 及以上级别的日志
logging.basicConfig(level=logging.ERROR)
# 在现有全局变量后添加
global_episode_counter = 0  # 添加这一行
writer_global = None
agents_global = []
# 在文件顶部的导入部分添加
_Independent = Independent
_Normal = Normal
agent_algorithms = {
    # "predator_0": "Random",
    # "predator_1": "Random", 
    "prey_0": "SAC",
    "prey_1": "SAC"
}

# class AccumulatedRewardTracker:
#     """追踪每个智能体的累积奖励 - 修复版本"""
#     def __init__(self):
#         self.global_accumulated_rewards = defaultdict(float)  # 全局累积奖励（不重置）
#         self.episode_rewards = defaultdict(float)            # 当前episode累积奖励
#         self.episode_history = defaultdict(list)             # 历史episode累积奖励
#         self.episode_count = defaultdict(int)                # episode计数
        
#     def update_step_reward(self, agent_rewards):
#         """每步更新奖励（不涉及episode结束）"""
#         for agent, reward in agent_rewards.items():
#             if reward is not None:
#                 self.global_accumulated_rewards[agent] += reward
#                 self.episode_rewards[agent] += reward
    
#     def complete_episode(self, agent_list):
#         """episode完成时的处理"""
#         for agent in agent_list:
#             if agent in self.episode_rewards:
#                 # 记录这个episode的累积奖励
#                 self.episode_history[agent].append(self.episode_rewards[agent])
#                 self.episode_count[agent] += 1
#                 # 重置当前episode奖励
#                 self.episode_rewards[agent] = 0.0
    
#     def get_global_accumulated_rewards(self):
#         """获取全局累积奖励（持续增长）"""
#         return dict(self.global_accumulated_rewards)
    
#     def get_current_episode_rewards(self):
#         """获取当前episode的累积奖励"""
#         return dict(self.episode_rewards)
    
#     def get_episode_statistics(self):
#         """获取历史episode的统计信息"""
#         stats = {}
#         for agent, rewards in self.episode_history.items():
#             if rewards:
#                 stats[agent] = {
#                     'mean': np.mean(rewards),
#                     'std': np.std(rewards),
#                     'max': np.max(rewards),
#                     'min': np.min(rewards),
#                     'last': rewards[-1] if rewards else 0.0,
#                     'count': len(rewards)
#                 }
#         return stats


# 添加支持连续动作空间的RandomPolicy
class ContinuousCompatibleRandomPolicy(BasePolicy):
    """兼容连续动作空间的随机策略"""
    
    def __init__(self, action_space=None):
        super().__init__()
        self.action_space = action_space
        print(f"RandomPolicy created for action_space: {action_space}")
        
    def forward(self, batch, state=None, **kwargs):
        """处理连续和离散动作空间"""
        # 确定batch大小
        batch_size = 1
        if hasattr(batch, 'obs'):
            if isinstance(batch.obs, np.ndarray):
                batch_size = batch.obs.shape[0]
            elif isinstance(batch.obs, torch.Tensor):
                batch_size = batch.obs.shape[0]
            elif hasattr(batch.obs, 'shape'):
                batch_size = batch.obs.shape[0]
        
        # 生成连续动作 (因为我们知道环境是连续的)
        if hasattr(self.action_space, 'shape') and self.action_space.shape:
            action_dim = self.action_space.shape[0]
            
            # 生成动作
            if batch_size == 1:
                actions = np.random.uniform(-1, 1, action_dim).astype(np.float32)
            else:
                actions = np.random.uniform(-1, 1, (batch_size, action_dim)).astype(np.float32)
                
            # print(f"Generated actions - batch_size: {batch_size}, action_dim: {action_dim}, actions_shape: {actions.shape}")
            
        else:
            # 备用离散动作
            actions = np.random.randint(0, 9, size=batch_size)
            print(f"Generated discrete actions - batch_size: {batch_size}, actions: {actions}")
        
        return Batch(act=actions, state=state)
    
    def learn(self, batch, **kwargs):
        return {}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--lr", type=float, default=3e-4)  # PPO通常使用更高的学习率
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor"
    )
    parser.add_argument("--n-step", type=int, default=1)  # PPO通常使用1步
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=100)  # PPO训练epoch数可以少一些
    parser.add_argument("--step-per-epoch", type=int, default=50000)  # PPO需要更多数据
    parser.add_argument("--step-per-collect", type=int, default=1)  # PPO典型值
    parser.add_argument("--repeat-per-collect", type=int, default=10)  # PPO特有参数
    parser.add_argument("--update-per-step", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=16)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.6,
        help="the expected winning rate",
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, watch the play of pre-trained models",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=2,
        help="the learned agent plays as the agent_id-th player",
    )

    # 连续动作空间特有参数
    parser.add_argument("--log-std-init", type=float, default=-1.0, 
                       help="initial log std for continuous actions")
    parser.add_argument("--log-std-min", type=float, default=-5.0, 
                       help="minimum log std for continuous actions")
    parser.add_argument("--log-std-max", type=float, default=1.5, 
                       help="maximum log std for continuous actions")
    parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="value function coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--value-clip", action="store_true", default=True, help="value clipping")
    parser.add_argument("--reward-normalization", action="store_true", default=False, 
                       help="reward normalization")
    parser.add_argument("--max-batchsize", type=int, default=128, help="max batch size for PPO")
    # PG特有参数调整建议
    parser.add_argument("--pg-lr", type=float, default=1e-4, 
                       help="learning rate for PG (usually higher than A2C/PPO)")
    parser.add_argument("--pg-baseline", action="store_true", default=False,
                       help="use baseline (moving average) for PG variance reduction")
    
        # TRPO特有参数
    parser.add_argument("--max-kl", type=float, default=0.01, 
                       help="max KL divergence for TRPO")
    parser.add_argument("--damping", type=float, default=0.1,
                       help="damping coefficient for TRPO")
    parser.add_argument("--max-backtracks", type=int, default=10,
                       help="max backtrack times for TRPO line search")
       # NPG特有参数（可选）
    parser.add_argument("--actor-step-size", type=float, default=0.5,
                       help="NPG actor step size for natural gradient")
    parser.add_argument("--advantage-normalization", action="store_true", default=True,
                       help="advantage normalization for NPG")
    parser.add_argument("--optim-critic-iters", type=int, default=1,
                       help="number of critic optimization iterations per update")
    # SAC特有参数
    parser.add_argument("--tau", type=float, default=0.005, 
                    help="soft update coefficient")
    parser.add_argument("--alpha", type=float, default=0.2, 
                    help="SAC temperature parameter")
    parser.add_argument("--auto-alpha", action="store_true", default=True,
                    help="automatic temperature adjustment")
    parser.add_argument("--target-entropy", type=float, default=-2.0,
                    help="target entropy for auto alpha, usually -action_dim")

    # off-policy通用参数调整
    parser.add_argument("--start-timesteps", type=int, default=10000,
                    help="random exploration steps before training")
    # 预训练模型加载参数
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="path to resume from pre-trained agent",
    )

    # 为每个智能体添加特定的resume路径参数
    parser.add_argument("--predator-0-resume-path", type=str, default="")
    parser.add_argument("--predator-1-resume-path", type=str, default="")
    parser.add_argument("--prey-0-resume-path", type=str, default="")
    parser.add_argument("--prey-1-resume-path", type=str, default="")

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_agents(
    args: argparse.Namespace = get_args(),
) -> Tuple[BasePolicy, dict, list]:
    """
    创建多智能体策略管理器 - PPO专用版本
    """
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or observation_space.n
    
    # 修复动作空间处理 - 支持连续动作空间
    if hasattr(env.action_space, 'shape') and env.action_space.shape:
        # 连续动作空间 Box(2,) -> action_shape = 2
        args.action_shape = env.action_space.shape[0]
    else:
        # 离散动作空间 Discrete(9) -> action_shape = 9
        args.action_shape = env.action_space.n
    
    print(f"Environment info:")
    print(f"  Action space: {env.action_space}")
    print(f"  Action shape: {args.action_shape}")
    print(f"  State shape: {args.state_shape}")

    # 获取环境中的所有智能体
    agents = env.agents

    # # 定义每个智能体使用的算法 - 全部使用PPO
    # agent_algorithms = {
    #     "predator_0": "Random",
    #     "predator_1": "Random", 
    #     "prey_0": "Random",
    #     "prey_1": "Random"
    # }

    # 处理预训练模型路径
    agent_resume_paths = {}
    for agent_name in agents:
        agent_resume_attr = f"{agent_name.replace('_', '_')}_resume_path"
        if hasattr(args, agent_resume_attr):
            path = getattr(args, agent_resume_attr)
            if path:
                agent_resume_paths[agent_name] = path

    # 多智能体配置逻辑
    policies = {}
    optimizers = {}

    for agent_name in agents:
        algo = agent_algorithms.get(agent_name, "Random")
        print(f"Creating {algo} policy for agent: {agent_name}")
        if algo == "PPO":
            # ===== 使用原始环境效果最好的 ActorProb/Critic PPO =====
            from tianshou.utils.net.continuous import ActorProb, Critic

            # 基础网络
            actor_base = Net(
                state_shape=args.state_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device
            )
            critic_base = Net(
                state_shape=args.state_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device
            )

            # ActorProb 网络（连续动作）
            actor = ActorProb(
                actor_base,
                action_shape=args.action_shape,
                unbounded=True,
                conditioned_sigma=False,
                device=args.device
            ).to(args.device)

            # Critic 网络
            critic = Critic(
                critic_base,
                device=args.device
            ).to(args.device)

            # 优化器（保持原始环境中的设置）
            optimizer = torch.optim.Adam(
                list(actor.parameters()) + list(critic.parameters()),
                lr=args.lr
            )
            optimizers[agent_name] = optimizer

            # 分布函数（使用局部变量）
            def _dist_fn(*logits):
                return _Independent(_Normal(*logits), 1)

            # PPO 策略
            policy = PPOPolicy(
                actor=actor,
                critic=critic,
                optim=optimizer,
                dist_fn=_dist_fn,
                discount_factor=args.gamma,
                max_grad_norm=0.5,
                eps_clip=args.eps_clip,
                vf_coef=args.vf_coef,      # 原始环境里设得很小，例如 0.005
                ent_coef=args.ent_coef,    # 通常 0.0
                gae_lambda=args.gae_lambda,
                reward_normalization=args.reward_normalization,
                action_scaling=True,
                action_bound_method="clip",
                lr_scheduler=None,
                action_space=env.action_space
            )

            # 如果需要加载预训练模型
            if agent_name in agent_resume_paths:
                resume_path = agent_resume_paths[agent_name]
                try:
                    print(f"Loading PPO pretrained model for {agent_name} from {resume_path}")
                    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
                    print(f"Successfully loaded PPO model for {agent_name}")
                except Exception as e:
                    print(f"Warning: Failed to load PPO model for {agent_name}: {e}")

            print(f"Original ActorProb/Critic PPO policy created for {agent_name}")
            policies[agent_name] = policy

        elif algo == "A2C":
            # ==================== 连续动作空间A2C算法实现 ====================
            
            # 创建连续动作空间的 A2C Actor 网络
            class ContinuousA2CActorNet(torch.nn.Module):
                def __init__(self, state_shape, action_shape, hidden_sizes, device="cpu", 
                            log_std_init=-0.5, log_std_min=-10.0, log_std_max=2.0):
                    super().__init__()
                    self.output_shape = action_shape
                    self.log_std_min = log_std_min
                    self.log_std_max = log_std_max
                    
                    # 计算输入维度
                    input_dim = int(np.prod(state_shape))
                    
                    # 构建共享特征网络
                    layers = []
                    prev_dim = input_dim
                    
                    for hidden_dim in hidden_sizes:
                        layers.extend([
                            torch.nn.Linear(prev_dim, hidden_dim),
                            torch.nn.ReLU()
                        ])
                        prev_dim = hidden_dim
                    
                    self.feature_net = torch.nn.Sequential(*layers)
                    
                    # 连续动作需要输出均值和标准差
                    self.mu_layer = torch.nn.Linear(prev_dim, action_shape)
                    self.log_std_layer = torch.nn.Linear(prev_dim, action_shape)
                    
                    # 初始化参数
                    # 均值层使用较小的权重（A2C通常比PPO更保守）
                    torch.nn.init.orthogonal_(self.mu_layer.weight, gain=0.01)
                    torch.nn.init.constant_(self.mu_layer.bias, 0)
                    
                    # log_std层初始化
                    torch.nn.init.constant_(self.log_std_layer.weight, 0)
                    torch.nn.init.constant_(self.log_std_layer.bias, log_std_init)
                    
                def forward(self, obs, state=None, info={}):
                    """连续动作A2C Actor网络前向传播 - 返回 ([mu, log_std], state)"""
                    obs_data = self._extract_obs(obs)
                    features = self.feature_net(obs_data)
                    
                    # 计算均值和log标准差
                    mu = torch.tanh(self.mu_layer(features))  # 限制在[-1, 1]，匹配action_space
                    log_std = self.log_std_layer(features)
                    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                    
                    # 拼接mu和log_std
                    output = torch.cat([mu, log_std], dim=-1)
                    return output, state
                
                def _extract_obs(self, obs):
                    """提取实际的观测数据"""
                    obs_data = obs
                    
                    # 处理Tianshou的Batch对象
                    if hasattr(obs, '__class__') and 'Batch' in str(obs.__class__):
                        if hasattr(obs, 'obs'):
                            obs_data = obs.obs
                        else:
                            raise ValueError(f"Cannot find obs field in Batch object")
                    elif isinstance(obs, dict):
                        if 'obs' in obs:
                            obs_data = obs['obs']
                        else:
                            obs_data = obs.get('observation', obs)
                    
                    # 转换为tensor
                    if isinstance(obs_data, np.ndarray):
                        obs_data = torch.from_numpy(obs_data).float()
                    elif not isinstance(obs_data, torch.Tensor):
                        obs_data = torch.tensor(obs_data, dtype=torch.float32)
                    
                    # 设备转移
                    if hasattr(self.feature_net[0], 'weight'):
                        target_device = self.feature_net[0].weight.device
                        if obs_data.device != target_device:
                            obs_data = obs_data.to(target_device)
                    
                    # 形状处理
                    if obs_data.dim() > 2:
                        obs_data = obs_data.flatten(1)
                    elif obs_data.dim() == 1:
                        obs_data = obs_data.unsqueeze(0)
                    
                    return obs_data

            # 创建连续动作空间的 A2C Critic 网络（与PPO相同）
            class ContinuousA2CCriticNet(torch.nn.Module):
                def __init__(self, state_shape, hidden_sizes, device="cpu"):
                    super().__init__()
                    
                    # 计算输入维度
                    input_dim = int(np.prod(state_shape))
                    
                    # 构建网络层
                    layers = []
                    prev_dim = input_dim
                    
                    for hidden_dim in hidden_sizes:
                        layers.extend([
                            torch.nn.Linear(prev_dim, hidden_dim),
                            torch.nn.ReLU()
                        ])
                        prev_dim = hidden_dim
                    
                    # 输出层：1个值函数值
                    layers.append(torch.nn.Linear(prev_dim, 1))
                    
                    self.net = torch.nn.Sequential(*layers)
                    
                    # 初始化
                    for layer in self.net:
                        if isinstance(layer, torch.nn.Linear):
                            torch.nn.init.orthogonal_(layer.weight)
                            torch.nn.init.constant_(layer.bias, 0)
                    
                def forward(self, obs, **kwargs):
                    """连续动作A2C Critic网络前向传播 - 只返回tensor"""
                    obs_data = self._extract_obs(obs)
                    value = self.net(obs_data)
                    return value
                
                def __call__(self, obs, **kwargs):
                    """确保所有调用方式都只返回tensor"""
                    return self.forward(obs, **kwargs)
                
                def _extract_obs(self, obs):
                    """提取实际的观测数据 - 与Actor相同的逻辑"""
                    obs_data = obs
                    
                    # 处理Tianshou的Batch对象
                    if hasattr(obs, '__class__') and 'Batch' in str(obs.__class__):
                        if hasattr(obs, 'obs'):
                            obs_data = obs.obs
                        else:
                            raise ValueError(f"Cannot find obs field in Batch object")
                    elif isinstance(obs, dict):
                        if 'obs' in obs:
                            obs_data = obs['obs']
                        else:
                            obs_data = obs.get('observation', obs)
                    
                    # 转换为tensor
                    if isinstance(obs_data, np.ndarray):
                        obs_data = torch.from_numpy(obs_data).float()
                    elif not isinstance(obs_data, torch.Tensor):
                        obs_data = torch.tensor(obs_data, dtype=torch.float32)
                    
                    # 设备转移
                    if hasattr(self.net[0], 'weight'):
                        target_device = self.net[0].weight.device
                        if obs_data.device != target_device:
                            obs_data = obs_data.to(target_device)
                    
                    # 形状处理
                    if obs_data.dim() > 2:
                        obs_data = obs_data.flatten(1)
                    elif obs_data.dim() == 1:
                        obs_data = obs_data.unsqueeze(0)
                    
                    return obs_data

            # 创建连续动作A2C网络
            actor_net = ContinuousA2CActorNet(
                args.state_shape,
                args.action_shape,  # 对于Box(2,)，这应该是2
                hidden_sizes=args.hidden_sizes,
                device=args.device,
                log_std_init=args.log_std_init,  # 使用参数值
                log_std_min=args.log_std_min,    # 使用参数值
                log_std_max=args.log_std_max,    # 使用参数值
            ).to(args.device)

            critic_net = ContinuousA2CCriticNet(
                args.state_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            # 优化器：同时更新 actor 和 critic
            optimizer = torch.optim.Adam(
                list(actor_net.parameters()) + list(critic_net.parameters()),
                lr=args.lr  # A2C通常使用较小的学习率
            )

            # 存储每个智能体的优化器
            optimizers[agent_name] = optimizer

            # 连续动作的分布函数 - 与PPO相同的实现
            def continuous_dist_fn(x):
                """连续动作分布函数 - 从网络输出创建Normal分布"""
                action_dim = args.action_shape  # 2
                mu = x[..., :action_dim]        # 前2维是均值
                log_std = x[..., action_dim:]   # 后2维是log标准差
                std = torch.exp(log_std)        # 转换为标准差
                return Normal(mu, std)

            # 创建连续动作 A2C 策略
            policy = A2CPolicy(
                actor=actor_net,
                critic=critic_net,
                optim=optimizer,
                dist_fn=continuous_dist_fn,          # 使用连续动作分布函数
                discount_factor=args.gamma,
                vf_coef=args.vf_coef,               # 值函数损失系数，A2C默认0.5
                ent_coef=args.ent_coef,             # 熵正则化系数，A2C默认0.01
                max_batchsize=args.max_batchsize,
                gae_lambda=args.gae_lambda,         # GAE参数（如果A2C支持）
                reward_normalization=args.reward_normalization,
                # 连续动作空间关键参数
                action_space=env.action_space,       # 传入Box空间
                action_scaling=True,                 # 启用动作缩放
                action_bound_method="clip",          # 动作边界处理方法
            )

            # 加载预训练模型（连续动作A2C专用）
            if agent_name in agent_resume_paths:
                resume_path = agent_resume_paths[agent_name]
                try:
                    print(f"Loading continuous A2C pretrained model for {agent_name} from {resume_path}")
                    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
                    print(f"Successfully loaded continuous A2C model for {agent_name}")
                except Exception as e:
                    print(f"Warning: Failed to load continuous A2C model for {agent_name}: {e}")

            print(f"Continuous A2C policy created for {agent_name}")
            policies[agent_name] = policy
            pass  # 后续扩展
        elif algo == "PG":
            # ==================== 连续动作空间PG算法实现 ====================
            
            # 创建连续动作空间的 PG Actor 网络
            class ContinuousPGActorNet(torch.nn.Module):
                def __init__(self, state_shape, action_shape, hidden_sizes, device="cpu", 
                            log_std_init=-0.5, log_std_min=-10.0, log_std_max=2.0):
                    super().__init__()
                    self.output_shape = action_shape
                    self.log_std_min = log_std_min
                    self.log_std_max = log_std_max
                    
                    # 计算输入维度
                    input_dim = int(np.prod(state_shape))
                    
                    # 构建共享特征网络
                    layers = []
                    prev_dim = input_dim
                    
                    for hidden_dim in hidden_sizes:
                        layers.extend([
                            torch.nn.Linear(prev_dim, hidden_dim),
                            torch.nn.ReLU()
                        ])
                        prev_dim = hidden_dim
                    
                    self.feature_net = torch.nn.Sequential(*layers)
                    
                    # 连续动作需要输出均值和标准差
                    self.mu_layer = torch.nn.Linear(prev_dim, action_shape)
                    self.log_std_layer = torch.nn.Linear(prev_dim, action_shape)
                    
                    # 初始化参数 - PG使用稍微保守的初始化
                    torch.nn.init.orthogonal_(self.mu_layer.weight, gain=0.01)
                    torch.nn.init.constant_(self.mu_layer.bias, 0)
                    
                    # log_std层初始化
                    torch.nn.init.constant_(self.log_std_layer.weight, 0)
                    torch.nn.init.constant_(self.log_std_layer.bias, log_std_init)
                    
                def forward(self, obs, state=None, info={}):
                    """连续动作PG Actor网络前向传播 - 返回 ([mu, log_std], state)"""
                    obs_data = self._extract_obs(obs)
                    features = self.feature_net(obs_data)
                    
                    # 计算均值和log标准差
                    mu = torch.tanh(self.mu_layer(features))  # 限制在[-1, 1]，匹配action_space
                    log_std = self.log_std_layer(features)
                    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                    
                    # 拼接mu和log_std
                    output = torch.cat([mu, log_std], dim=-1)
                    return output, state
                
                def _extract_obs(self, obs):
                    """提取实际的观测数据"""
                    obs_data = obs
                    
                    # 处理Tianshou的Batch对象
                    if hasattr(obs, '__class__') and 'Batch' in str(obs.__class__):
                        if hasattr(obs, 'obs'):
                            obs_data = obs.obs
                        else:
                            raise ValueError(f"Cannot find obs field in Batch object")
                    elif isinstance(obs, dict):
                        if 'obs' in obs:
                            obs_data = obs['obs']
                        else:
                            obs_data = obs.get('observation', obs)
                    
                    # 转换为tensor
                    if isinstance(obs_data, np.ndarray):
                        obs_data = torch.from_numpy(obs_data).float()
                    elif not isinstance(obs_data, torch.Tensor):
                        obs_data = torch.tensor(obs_data, dtype=torch.float32)
                    
                    # 设备转移
                    if hasattr(self.feature_net[0], 'weight'):
                        target_device = self.feature_net[0].weight.device
                        if obs_data.device != target_device:
                            obs_data = obs_data.to(target_device)
                    
                    # 形状处理
                    if obs_data.dim() > 2:
                        obs_data = obs_data.flatten(1)
                    elif obs_data.dim() == 1:
                        obs_data = obs_data.unsqueeze(0)
                    
                    return obs_data

            # 创建连续动作PG网络 - 只需要Actor，不需要Critic
            actor_net = ContinuousPGActorNet(
                args.state_shape,
                args.action_shape,  # 对于Box(2,)，这应该是2
                hidden_sizes=args.hidden_sizes,
                device=args.device,
                log_std_init=args.log_std_init,  # 使用参数值
                log_std_min=args.log_std_min,    # 使用参数值
                log_std_max=args.log_std_max,    # 使用参数值
            ).to(args.device)

            # 优化器：只优化 actor 网络（PG比A2C/PPO通常使用更高的学习率）
            optimizer = torch.optim.Adam(
                actor_net.parameters(),
                lr=args.pg_lr if hasattr(args, 'pg_lr') and args.pg_lr > 0 else args.lr * 3  # PG通常需要更高学习率
            )

            # 存储每个智能体的优化器
            optimizers[agent_name] = optimizer

            # 连续动作的分布函数 - 与PPO/A2C相同
            def continuous_dist_fn(x):
                """连续动作分布函数 - 从网络输出创建Normal分布"""
                action_dim = args.action_shape  # 2
                mu = x[..., :action_dim]        # 前2维是均值
                log_std = x[..., action_dim:]   # 后2维是log标准差
                std = torch.exp(log_std)        # 转换为标准差
                return Normal(mu, std)

            # 创建连续动作 PG 策略 - 简化版本，移除不支持的参数
            policy = PGPolicy(
                model=actor_net,
                optim=optimizer,
                dist_fn=continuous_dist_fn,          # 使用连续动作分布函数
                discount_factor=args.gamma,
                reward_normalization=args.reward_normalization,
                # 连续动作空间关键参数（根据Tianshou版本，这些参数可能需要调整）
                action_scaling=True,                 # 启用动作缩放
                action_bound_method="clip",          # 动作边界处理方法
                deterministic_eval=False,            # 测试时是否确定性
                # 注意：PG是最基础的策略梯度算法，没有复杂的参数
                # 如果需要基线减少方差，需要在算法实现层面添加，或使用A2C
            )

            # 加载预训练模型（连续动作PG专用）
            if agent_name in agent_resume_paths:
                resume_path = agent_resume_paths[agent_name]
                try:
                    print(f"Loading continuous PG pretrained model for {agent_name} from {resume_path}")
                    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
                    print(f"Successfully loaded continuous PG model for {agent_name}")
                except Exception as e:
                    print(f"Warning: Failed to load continuous PG model for {agent_name}: {e}")

            print(f"Continuous PG policy created for {agent_name}")
            policies[agent_name] = policy

        elif algo == "TRPO":
            # ==================== 使用天授官方连续动作空间TRPO实现 ====================
            
            from tianshou.utils.net.continuous import ActorProb, Critic
            from torch.distributions import Independent, Normal

            # 使用官方ActorProb网络（与PPO相同的结构，但用于TRPO）
            actor_base = Net(
                state_shape=args.state_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device
            )
            
            critic_base = Net(
                state_shape=args.state_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device
            )

            # ActorProb网络（连续动作）- 天授官方实现
            actor = ActorProb(
                actor_base,
                action_shape=args.action_shape,
                unbounded=True,                    # 连续动作使用unbounded=True
                conditioned_sigma=False,           # 可以尝试True来让标准差依赖状态
                device=args.device
            ).to(args.device)

            # Critic网络 - 天授官方实现
            critic = Critic(
                critic_base,
                device=args.device
            ).to(args.device)

            # 优化器 - TRPO特殊处理
            # Actor使用很小的学习率，因为TRPO使用自然梯度更新
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr * 0.1)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)
            
            # 合并优化器（天授接口需要）
            optimizer = torch.optim.Adam(
                list(actor.parameters()) + list(critic.parameters()),
                lr=args.lr
            )

            # 存储优化器
            optimizers[agent_name] = optimizer

            # 分布函数（与PPO相同）
            def _dist_fn(*logits):
                return Independent(Normal(*logits), 1)

            # 创建TRPO策略 - 使用天授官方实现
            policy = TRPOPolicy(
                actor=actor,
                critic=critic,
                optim=optimizer,
                dist_fn=_dist_fn,
                discount_factor=args.gamma,
                
                # GAE相关参数
                gae_lambda=args.gae_lambda,
                
                # 值函数相关
                vf_coef=args.vf_coef,
                reward_normalization=args.reward_normalization,
                max_batchsize=args.max_batchsize,
                
                # TRPO核心参数
                max_kl=args.max_kl,                      # KL散度约束，连续动作建议0.01-0.05
                backtrack_coeff=0.8,                     # 线搜索回溯系数
                max_backtracks=args.max_backtracks,      # 最大回溯次数
                
                # NPG基类参数
                advantage_normalization=args.advantage_normalization,
                optim_critic_iters=args.optim_critic_iters,
                
                # 连续动作关键参数
                action_space=env.action_space,
                action_scaling=True,
                action_bound_method="clip",
                deterministic_eval=False,
            )

            # 加载预训练模型
            if agent_name in agent_resume_paths:
                resume_path = agent_resume_paths[agent_name]
                try:
                    print(f"Loading TRPO pretrained model for {agent_name} from {resume_path}")
                    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
                    print(f"Successfully loaded TRPO model for {agent_name}")
                except Exception as e:
                    print(f"Warning: Failed to load TRPO model for {agent_name}: {e}")

            print(f"Official TRPO policy created for {agent_name}")
            policies[agent_name] = policy
        elif algo == "NPG":
            # ==================== 连续动作空间NPG算法实现 ====================
            
            # 创建连续动作空间的 NPG Actor 网络
            class ContinuousNPGActorNet(torch.nn.Module):
                def __init__(self, state_shape, action_shape, hidden_sizes, device="cpu", 
                            log_std_init=-0.5, log_std_min=-10.0, log_std_max=2.0):
                    super().__init__()
                    self.output_shape = action_shape
                    self.log_std_min = log_std_min
                    self.log_std_max = log_std_max
                    
                    # 计算输入维度
                    input_dim = int(np.prod(state_shape))
                    
                    # 构建共享特征网络
                    layers = []
                    prev_dim = input_dim
                    
                    for hidden_dim in hidden_sizes:
                        layers.extend([
                            torch.nn.Linear(prev_dim, hidden_dim),
                            torch.nn.ReLU()
                        ])
                        prev_dim = hidden_dim
                    
                    self.feature_net = torch.nn.Sequential(*layers)
                    
                    # 连续动作需要输出均值和标准差
                    self.mu_layer = torch.nn.Linear(prev_dim, action_shape)
                    self.log_std_layer = torch.nn.Linear(prev_dim, action_shape)
                    
                    # 初始化参数 - NPG使用保守的初始化，有利于自然梯度计算
                    torch.nn.init.orthogonal_(self.mu_layer.weight, gain=0.01)
                    torch.nn.init.constant_(self.mu_layer.bias, 0)
                    
                    # log_std层初始化
                    torch.nn.init.constant_(self.log_std_layer.weight, 0)
                    torch.nn.init.constant_(self.log_std_layer.bias, log_std_init)
                    
                def forward(self, obs, state=None, info={}):
                    """连续动作NPG Actor网络前向传播 - 返回 ([mu, log_std], state)"""
                    obs_data = self._extract_obs(obs)
                    features = self.feature_net(obs_data)
                    
                    # 计算均值和log标准差
                    mu = torch.tanh(self.mu_layer(features))  # 限制在[-1, 1]，匹配action_space
                    log_std = self.log_std_layer(features)
                    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                    
                    # 拼接mu和log_std
                    output = torch.cat([mu, log_std], dim=-1)
                    return output, state
                
                def _extract_obs(self, obs):
                    """提取实际的观测数据"""
                    obs_data = obs
                    
                    # 处理Tianshou的Batch对象
                    if hasattr(obs, '__class__') and 'Batch' in str(obs.__class__):
                        if hasattr(obs, 'obs'):
                            obs_data = obs.obs
                        else:
                            raise ValueError(f"Cannot find obs field in Batch object")
                    elif isinstance(obs, dict):
                        if 'obs' in obs:
                            obs_data = obs['obs']
                        else:
                            obs_data = obs.get('observation', obs)
                    
                    # 转换为tensor
                    if isinstance(obs_data, np.ndarray):
                        obs_data = torch.from_numpy(obs_data).float()
                    elif not isinstance(obs_data, torch.Tensor):
                        obs_data = torch.tensor(obs_data, dtype=torch.float32)
                    
                    # 设备转移
                    if hasattr(self.feature_net[0], 'weight'):
                        target_device = self.feature_net[0].weight.device
                        if obs_data.device != target_device:
                            obs_data = obs_data.to(target_device)
                    
                    # 形状处理
                    if obs_data.dim() > 2:
                        obs_data = obs_data.flatten(1)
                    elif obs_data.dim() == 1:
                        obs_data = obs_data.unsqueeze(0)
                    
                    return obs_data

            # 创建连续动作空间的 NPG Critic 网络
            class ContinuousNPGCriticNet(torch.nn.Module):
                def __init__(self, state_shape, hidden_sizes, device="cpu"):
                    super().__init__()
                    
                    # 计算输入维度
                    input_dim = int(np.prod(state_shape))
                    
                    # 构建网络层
                    layers = []
                    prev_dim = input_dim
                    
                    for hidden_dim in hidden_sizes:
                        layers.extend([
                            torch.nn.Linear(prev_dim, hidden_dim),
                            torch.nn.ReLU()
                        ])
                        prev_dim = hidden_dim
                    
                    # 输出层：1个值函数值
                    layers.append(torch.nn.Linear(prev_dim, 1))
                    
                    self.net = torch.nn.Sequential(*layers)
                    
                    # 初始化
                    for layer in self.net:
                        if isinstance(layer, torch.nn.Linear):
                            torch.nn.init.orthogonal_(layer.weight)
                            torch.nn.init.constant_(layer.bias, 0)
                    
                def forward(self, obs, **kwargs):
                    """连续动作NPG Critic网络前向传播 - 只返回tensor"""
                    obs_data = self._extract_obs(obs)
                    value = self.net(obs_data)
                    return value
                
                def __call__(self, obs, **kwargs):
                    """确保所有调用方式都只返回tensor"""
                    return self.forward(obs, **kwargs)
                
                def _extract_obs(self, obs):
                    """提取实际的观测数据"""
                    obs_data = obs
                    
                    # 处理Tianshou的Batch对象
                    if hasattr(obs, '__class__') and 'Batch' in str(obs.__class__):
                        if hasattr(obs, 'obs'):
                            obs_data = obs.obs
                        else:
                            raise ValueError(f"Cannot find obs field in Batch object")
                    elif isinstance(obs, dict):
                        if 'obs' in obs:
                            obs_data = obs['obs']
                        else:
                            obs_data = obs.get('observation', obs)
                    
                    # 转换为tensor
                    if isinstance(obs_data, np.ndarray):
                        obs_data = torch.from_numpy(obs_data).float()
                    elif not isinstance(obs_data, torch.Tensor):
                        obs_data = torch.tensor(obs_data, dtype=torch.float32)
                    
                    # 设备转移
                    if hasattr(self.net[0], 'weight'):
                        target_device = self.net[0].weight.device
                        if obs_data.device != target_device:
                            obs_data = obs_data.to(target_device)
                    
                    # 形状处理
                    if obs_data.dim() > 2:
                        obs_data = obs_data.flatten(1)
                    elif obs_data.dim() == 1:
                        obs_data = obs_data.unsqueeze(0)
                    
                    return obs_data

            # 创建连续动作NPG网络
            actor_net = ContinuousNPGActorNet(
                args.state_shape,
                args.action_shape,  # 对于Box(2,)，这应该是2
                hidden_sizes=args.hidden_sizes,
                device=args.device,
                log_std_init=args.log_std_init,  # 使用参数值
                log_std_min=args.log_std_min,    # 使用参数值
                log_std_max=args.log_std_max,    # 使用参数值
            ).to(args.device)

            critic_net = ContinuousNPGCriticNet(
                args.state_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            # 优化器：NPG的Actor使用特殊的自然梯度更新，Critic使用普通梯度下降
            # Actor优化器设置较小学习率，因为NPG会使用自然梯度
            actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=args.lr * 0.1)  # NPG Actor使用较小学习率
            critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=args.lr)
            
            # 合并优化器用于接口兼容性
            optimizer = torch.optim.Adam(
                list(actor_net.parameters()) + list(critic_net.parameters()),
                lr=args.lr
            )

            # 存储每个智能体的优化器
            optimizers[agent_name] = optimizer

            # 连续动作的分布函数
            def continuous_dist_fn(x):
                """连续动作分布函数 - 从网络输出创建Normal分布"""
                action_dim = args.action_shape  # 2
                mu = x[..., :action_dim]        # 前2维是均值
                log_std = x[..., action_dim:]   # 后2维是log标准差
                std = torch.exp(log_std)        # 转换为标准差
                return Normal(mu, std)

            # 创建连续动作 NPG 策略
            policy = NPGPolicy(
                actor=actor_net,
                critic=critic_net,
                optim=optimizer,
                dist_fn=continuous_dist_fn,          # 使用连续动作分布函数
                discount_factor=args.gamma,
                gae_lambda=args.gae_lambda,          # GAE参数
                vf_coef=args.vf_coef,               # 值函数损失系数
                reward_normalization=args.reward_normalization,
                max_batchsize=args.max_batchsize,
                
                # NPG特有参数
                advantage_normalization=args.advantage_normalization,  # 优势标准化，NPG中很重要
                optim_critic_iters=args.optim_critic_iters,           # Critic优化次数，通常3-5次
                actor_step_size=args.actor_step_size * 0.001,           # NPG Actor步长，连续动作需要更小步长

                # 连续动作空间关键参数
                action_space=env.action_space,       # 传入Box空间
                action_scaling=True,                 # 启用动作缩放
                action_bound_method="clip",          # 动作边界处理方法
                deterministic_eval=False,            # 测试时是否确定性
                
                # NPG特有的自然梯度计算参数（根据Tianshou版本调整）
                # NPG算法的核心是自然梯度，这些参数控制自然梯度的计算
            )

            # 加载预训练模型（连续动作NPG专用）
            if agent_name in agent_resume_paths:
                resume_path = agent_resume_paths[agent_name]
                try:
                    print(f"Loading continuous NPG pretrained model for {agent_name} from {resume_path}")
                    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
                    print(f"Successfully loaded continuous NPG model for {agent_name}")
                except Exception as e:
                    print(f"Warning: Failed to load continuous NPG model for {agent_name}: {e}")

            print(f"Continuous NPG policy created for {agent_name}")
            policies[agent_name] = policy
            pass  # 后续扩展
        elif algo == "Random":
            # 使用兼容连续动作空间的随机策略
            policy = ContinuousCompatibleRandomPolicy(env.action_space)
            print(f"Random policy created for {agent_name}")
            policies[agent_name] = policy
        # offpolicy
        # Replace the SAC section in get_agents() function with this complete implementation:

        # Replace your SAC section in get_agents() with this fixed version:
        elif algo == "SAC":
            # 使用tianshou官方的SAC网络 - 数据预处理修复版本
            from tianshou.utils.net.continuous import ActorProb, Critic
            from tianshou.policy import SACPolicy
            
            # 添加调试信息
            print(f"Creating SAC policy for {agent_name}")
            print(f"State shape: {args.state_shape}, Action shape: {args.action_shape}")
            
            # 自定义观测数据清理函数
            def clean_observation(obs):
                """清理观测数据，提取真正的数值观测"""
                # 如果是字典且包含'obs'键，只取数值部分
                if isinstance(obs, dict) and 'obs' in obs:
                    return obs['obs']
                # 如果是Batch对象
                elif hasattr(obs, '__class__') and 'Batch' in str(obs.__class__):
                    if hasattr(obs, 'obs'):
                        return obs.obs
                    else:
                        # 寻找数值型的观测数据，跳过字符串
                        for key, value in obs.items():
                            if isinstance(value, (np.ndarray, torch.Tensor)) and not isinstance(value, str):
                                return value
                        raise ValueError("No valid observation found in Batch")
                return obs
            
            # Actor网络基础类
            actor_base = Net(
                state_shape=args.state_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device
            )
            
            # 安全的ActorProb网络 - 添加数据清理
            class DataCleaningActorProb(ActorProb):
                def forward(self, obs, state=None, info={}):
                    # 清理观测数据
                    try:
                        cleaned_obs = clean_observation(obs)
                        # 确保是tensor格式
                        if isinstance(cleaned_obs, np.ndarray):
                            cleaned_obs = torch.from_numpy(cleaned_obs).float().to(self.device)
                        elif isinstance(cleaned_obs, str):
                            raise ValueError(f"Observation is still string after cleaning: {cleaned_obs}")
                        return super().forward(cleaned_obs, state, info)
                    except Exception as e:
                        print(f"Actor observation cleaning failed: {e}")
                        print(f"Original obs type: {type(obs)}")
                        if isinstance(obs, dict):
                            print(f"Dict keys: {list(obs.keys())}")
                        raise e
            
            actor = DataCleaningActorProb(
                actor_base,
                action_shape=args.action_shape,
                unbounded=True,  # SAC支持unbounded动作
                conditioned_sigma=True,  # 让std依赖于状态
                device=args.device
            ).to(args.device)
            
            # 安全的Critic网络 - 添加数据清理
            class DataCleaningCritic(Critic):
                def forward(self, obs, act=None, info={}):
                    # 清理观测数据
                    try:
                        cleaned_obs = clean_observation(obs)
                        # 确保是tensor格式
                        if isinstance(cleaned_obs, np.ndarray):
                            cleaned_obs = torch.from_numpy(cleaned_obs).float().to(self.device)
                        elif isinstance(cleaned_obs, str):
                            raise ValueError(f"Critic observation is still string after cleaning: {cleaned_obs}")
                        return super().forward(cleaned_obs, act, info)
                    except Exception as e:
                        print(f"Critic observation cleaning failed: {e}")
                        print(f"Original obs type: {type(obs)}")
                        if isinstance(obs, dict):
                            print(f"Dict keys: {list(obs.keys())}")
                        raise e
            
            # 双Critic网络
            critic1_base = Net(
                state_shape=args.state_shape,
                action_shape=args.action_shape,  # Critic需要state+action作为输入
                hidden_sizes=args.hidden_sizes,
                concat=True,  # 拼接state和action
                device=args.device
            )
            critic1 = DataCleaningCritic(critic1_base, device=args.device).to(args.device)
            
            critic2_base = Net(
                state_shape=args.state_shape,
                action_shape=args.action_shape,
                hidden_sizes=args.hidden_sizes,
                concat=True,
                device=args.device
            )
            critic2 = DataCleaningCritic(critic2_base, device=args.device).to(args.device)
            
            # 三个独立的优化器
            actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr)
            critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.lr)
            critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.lr)
            
            # 自定义SAC策略 - 添加数据预处理
            class DataCleaningSACPolicy(SACPolicy):
                def forward(self, batch, state=None, **kwargs):
                    # 在策略层面也进行数据清理
                    if hasattr(batch, 'obs'):
                        try:
                            # 清理观测数据
                            if isinstance(batch.obs, dict) and 'obs' in batch.obs:
                                # 创建新的batch，只包含数值观测
                                from tianshou.data import Batch
                                cleaned_batch = Batch()
                                cleaned_batch.obs = batch.obs['obs']  # 提取真正的观测数据
                                # 复制其他字段
                                for key, value in batch.items():
                                    if key != 'obs':
                                        cleaned_batch[key] = value
                                batch = cleaned_batch
                        except Exception as e:
                            print(f"Batch cleaning warning: {e}")
                    
                    return super().forward(batch, state, **kwargs)
                
                def learn(self, batch, **kwargs):
                    # 训练时也进行数据清理
                    if hasattr(batch, 'obs') and isinstance(batch.obs, dict) and 'obs' in batch.obs:
                        try:
                            from tianshou.data import Batch
                            cleaned_batch = Batch()
                            cleaned_batch.obs = batch.obs['obs']
                            # 处理obs_next
                            if hasattr(batch, 'obs_next') and isinstance(batch.obs_next, dict) and 'obs' in batch.obs_next:
                                cleaned_batch.obs_next = batch.obs_next['obs']
                            else:
                                cleaned_batch.obs_next = batch.obs_next
                            # 复制其他字段
                            for key, value in batch.items():
                                if key not in ['obs', 'obs_next']:
                                    cleaned_batch[key] = value
                            batch = cleaned_batch
                        except Exception as e:
                            print(f"Training batch cleaning warning: {e}")
                    
                    return super().learn(batch, **kwargs)
            
            # SAC策略
            policy = DataCleaningSACPolicy(
                actor=actor,
                critic1=critic1,
                critic2=critic2,
                actor_optim=actor_optim,
                critic1_optim=critic1_optim,
                critic2_optim=critic2_optim,
                tau=args.tau,
                gamma=args.gamma,
                alpha=args.alpha,
                estimation_step=args.n_step,
                action_space=env.action_space
            )
            
            # 如果启用自动温度调节
            if args.auto_alpha:
                target_entropy = args.target_entropy
                if target_entropy == -2.0:  # 使用默认值
                    target_entropy = -np.prod(env.action_space.shape)
                
                log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
                alpha_optim = torch.optim.Adam([log_alpha], lr=args.lr)
                
                # 更新SAC策略的alpha相关设置
                policy.alpha_optim = alpha_optim
                policy.log_alpha = log_alpha
                policy.target_entropy = target_entropy
            
            # 加载预训练模型（如果有）
            if agent_name in agent_resume_paths:
                resume_path = agent_resume_paths[agent_name]
                try:
                    print(f"Loading SAC pretrained model for {agent_name} from {resume_path}")
                    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
                    print(f"Successfully loaded SAC model for {agent_name}")
                except Exception as e:
                    print(f"Warning: Failed to load SAC model for {agent_name}: {e}")
            
            # 存储优化器（用于保存/加载）
            optimizers[agent_name] = {
                'actor': actor_optim,
                'critic1': critic1_optim, 
                'critic2': critic2_optim
            }
            if args.auto_alpha:
                optimizers[agent_name]['alpha'] = alpha_optim
            
            print(f"Data-cleaning SAC policy created for {agent_name}")
            policies[agent_name] = policy

        else:
            raise ValueError(f"This is PPO-only trainer. Unsupported algorithm: {algo}")

    # 创建多智能体策略管理器
    policy_list = []
    for agent in agents:
        policy = policies[agent]
        if hasattr(policy, 'to'):
            policy_list.append(policy.to(args.device))
        else:
            policy_list.append(policy)
    
    policy_manager = MultiAgentPolicyManager(policy_list, env)

    return policy_manager, optimizers, agents


def get_env(render_mode=None):
    """
    创建环境的函数
    """
    # waterworld环境 - 配置为PPO适合的设置
    agent_algos = ["SAC", "SAC", "SAC", "SAC"] * 50  # 给足够多的A2C算法
    env = waterworld_v4.env(
        render_mode=render_mode,
        n_predators=0,
        n_preys=2,   # 减少数量便于调试
        n_evaders=50, # 减少数量便于调试
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=40,  # 减少数量便于调试
        agent_algorithms=agent_algos
    )
    # 添加supersuit包装器
    env = ss.black_death_v3(env)
    return PettingZooEnv(env)

def _safe_iter_infos_from_buffer(buf):
    """把 (Vector)ReplayBuffer 中所有 env 的 info 扁平化迭代出来。
    只用索引，不做任何真值判断，避免触发 Batch.__len__ 的坑。
    """
    infos = []
    # VectorReplayBuffer：有 .buffers 列表
    if hasattr(buf, "buffers"):
        for b in buf.buffers:
            if b is None:
                continue
            size = getattr(b, "_size", 0)
            if size <= 0 or not hasattr(b, "info"):
                continue
            info_batch = b.info  # tianshou.data.Batch
            # 用显式索引，不做 `if item` 判断
            for i in range(size):
                infos.append(info_batch[i])
    else:
        # 单个 ReplayBuffer
        size = getattr(buf, "_size", 0)
        if size > 0 and hasattr(buf, "info"):
            info_batch = buf.info
            for i in range(size):
                infos.append(info_batch[i])
    return infos


# from collections import defaultdict

def _aggregate_perf_grouped(infos, agent_algo_map=None):
    """按 (role=predator/prey, algo) 分组聚合。
    返回：
      overall: (hunt_mean, escape_mean, forage_mean)
      grouped: dict[(role, algo)] -> {"hunt": mean, "escape": mean, "forage": mean}
    """
    # 收集
    overall = {"hunt": [], "escape": [], "forage": []}
    bucket = defaultdict(lambda: {"hunt": [], "escape": [], "forage": []})

    for info in infos:
        agent_type = getattr(info, "agent_type", None)  # "predator"/"prey"
        pm = getattr(info, "performance_metrics", None)

        if agent_type not in ("predator", "prey") or pm is None:
            continue

        # —— 算法名的稳健获取顺序 ——
        algo = None
        # 1) info 自带
        for key in ("agent_algorithm", "algorithm", "algo"):
            if hasattr(info, key):
                algo = getattr(info, key)
                break
        # 2) 用 agent_name / policy_id 去 agent_algo_map 查
        if not algo and agent_algo_map:
            # 常见字段尝试
            name_keys = ("agent_name", "policy_id", "agent_id", "policy_name")
            name = None
            for nk in name_keys:
                if hasattr(info, nk):
                    name = getattr(info, nk)
                    break
            if name in agent_algo_map:
                algo = agent_algo_map[name]
        if not algo:
            algo = "UNK"

        # 填值
        if agent_type == "predator" and hasattr(pm, "hunting_rate"):
            try:
                v = float(pm.hunting_rate)
                if not np.isnan(v):
                    overall["hunt"].append(v)
                    bucket[(agent_type, algo)]["hunt"].append(v)
            except Exception:
                pass

        if agent_type == "prey":
            if hasattr(pm, "escape_rate"):
                try:
                    v = float(pm.escape_rate); 
                    if not np.isnan(v):
                        overall["escape"].append(v)
                        bucket[(agent_type, algo)]["escape"].append(v)
                except Exception:
                    pass
            if hasattr(pm, "foraging_rate"):
                try:
                    v = float(pm.foraging_rate)
                    if not np.isnan(v):
                        overall["forage"].append(v)
                        bucket[(agent_type, algo)]["forage"].append(v)
                except Exception:
                    pass

    def _mean(xs): 
        return float(np.mean(xs)) if xs else float("nan")

    overall_mean = (_mean(overall["hunt"]), _mean(overall["escape"]), _mean(overall["forage"]))
    grouped_mean = {}
    for key, d in bucket.items():
        grouped_mean[key] = {
            "hunt": _mean(d["hunt"]),
            "escape": _mean(d["escape"]),
            "forage": _mean(d["forage"]),
        }
    return overall_mean, grouped_mean

class SimplifiedTensorboardLogger(TensorboardLogger):
    def __init__(self, writer: SummaryWriter, test_collector, ema_beta: float = 0.0,
                 agent_algo_map: Dict[str, str] = None):
        super().__init__(writer)
        self.test_collector = test_collector
        self.ema_beta = ema_beta
        self._ema_cache = {"hunt": None, "escape": None, "forage": None}
        self.agent_algo_map = agent_algo_map or {}

    def log_train_data(self, data: dict, step: int) -> None:
        """简化的训练数据日志记录"""
        super().log_train_data(data, step)

    def log_test_data(self, data: dict, step: int) -> None:
        """简化的测试数据日志记录"""
        super().log_test_data(data, step)

        # 保留原有的性能指标日志记录
        infos = _safe_iter_infos_from_buffer(self.test_collector.buffer)
        (h, e, f), grouped = _aggregate_perf_grouped(infos, agent_algo_map=self.agent_algo_map)

        # 总体曲线
        if not np.isnan(h):
            self.writer.add_scalar("Performance/Overall/Predator_Hunting_Rate", h, step)
        if not np.isnan(e):
            self.writer.add_scalar("Performance/Overall/Prey_Escape_Rate", e, step)
        if not np.isnan(f):
            self.writer.add_scalar("Performance/Overall/Prey_Foraging_Rate", f, step)

        # 分组曲线
        for (role, algo), m in grouped.items():
            if role == "predator" and not np.isnan(m["hunt"]):
                self.writer.add_scalar(f"Performance/{role.capitalize()}/{algo}/Hunting_Rate", m["hunt"], step)
            if role == "prey":
                if not np.isnan(m["escape"]):
                    self.writer.add_scalar(f"Performance/{role.capitalize()}/{algo}/Escape_Rate", m["escape"], step)
                if not np.isnan(m["forage"]):
                    self.writer.add_scalar(f"Performance/{role.capitalize()}/{algo}/Foraging_Rate", m["forage"], step)

        # EMA平滑
        if self.ema_beta:
            if not np.isnan(h):
                self._ema_cache["hunt"] = h if self._ema_cache["hunt"] is None else \
                    self.ema_beta * self._ema_cache["hunt"] + (1 - self.ema_beta) * h
                self.writer.add_scalar("Performance_Smoothed/Overall/Predator_Hunting_Rate_EMA",
                                       self._ema_cache["hunt"], step)
            if not np.isnan(e):
                self._ema_cache["escape"] = e if self._ema_cache["escape"] is None else \
                    self.ema_beta * self._ema_cache["escape"] + (1 - self.ema_beta) * e
                self.writer.add_scalar("Performance_Smoothed/Overall/Prey_Escape_Rate_EMA",
                                       self._ema_cache["escape"], step)
            if not np.isnan(f):
                self._ema_cache["forage"] = f if self._ema_cache["forage"] is None else \
                    self.ema_beta * self._ema_cache["forage"] + (1 - self.ema_beta) * f
                self.writer.add_scalar("Performance_Smoothed/Overall/Prey_Foraging_Rate_EMA",
                                       self._ema_cache["forage"], step)

        self.writer.flush()


def reward_metric(rews):
    """计算即时奖励并记录每个episode的奖励 - 简化版本"""
    global global_episode_counter, writer_global, agents_global
    
    # 原有的即时奖励计算（用于trainer）
    pred_idx = [i for i, n in enumerate(agents_global) if n.startswith("predator")]
    instant_reward = rews[:, pred_idx].mean(axis=1) if pred_idx else rews.mean(axis=1)
    
    # 记录每个episode的奖励
    if len(rews.shape) > 1 and len(agents_global) > 0 and writer_global:
        for episode_idx in range(rews.shape[0]):
            for agent_idx, agent_name in enumerate(agents_global):
                if agent_idx < rews.shape[1]:
                    episode_reward = float(rews[episode_idx, agent_idx])
                    
                    # 记录episode奖励到TensorBoard
                    role = "Predator" if agent_name.startswith("predator") else "Prey"
                    algo = agent_algorithms.get(agent_name, "Random")
                    writer_global.add_scalar(f"EpisodeReward/{role}/{algo}/{agent_name}", 
                                           episode_reward, global_episode_counter + episode_idx)
        
        # 更新全局episode计数器
        global_episode_counter += rews.shape[0]
        writer_global.flush()
    
    return instant_reward

def train_agent(args: argparse.Namespace = get_args()) -> Tuple[dict, BasePolicy]:
    """简化的训练函数"""
    global writer_global, agents_global
    
    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # ======== agent setup =========
    policy, optimizers, agents = get_agents(args)
    agents_global = agents  # 设置全局智能体列表

    print("Using off-policy trainer")  # 改变提示信息
    # collectors设置 - off-policy可以使用更小的step_per_collect
    train_collector = Collector(
        policy, train_envs,
        buffer=VectorReplayBuffer(args.buffer_size, args.training_num),
        exploration_noise=False  # SAC内置探索，不需要额外噪声
    )
    test_collector = Collector(
        policy, test_envs,
        buffer=VectorReplayBuffer(max(50000, args.buffer_size // 2), args.test_num),
        exploration_noise=False
    )
    # ======== 按智能体顺序构建算法名称字符串 =========
    # 获取每个智能体对应的算法，按agents顺序
    algo_sequence = []
    for agent_name in agents:
        algo = agent_algorithms.get(agent_name, "Unknown")
        algo_sequence.append(algo)
    
    # 用下划线连接
    algo_name_str = "_".join(algo_sequence)
    # ======== tensorboard logging setup =========
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("log", "waterworld", "NoPredator_preyTrain",algo_name_str, f"run_{time_str}")
    writer_global = SummaryWriter(log_path)
    writer_global.add_text("args", str(args))

    agent_algo_map = {name: agent_algorithms.get(name, "UNK") for name in agents}

    logger = SimplifiedTensorboardLogger(
        writer_global,
        test_collector=test_collector,
        ema_beta=0.9,
        agent_algo_map=agent_algo_map,
    )

    # 随机探索阶段
    print(f"Random exploration for {args.start_timesteps} steps...")
    train_collector.collect(n_step=args.start_timesteps, random=True)

    # 使用off-policy trainer
    from tianshou.trainer import offpolicy_trainer
    
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,  # 可以设为1
        update_per_step=args.update_per_step,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        train_fn=lambda epoch, env_step: None,
        test_fn=lambda epoch, env_step: None,
        save_best_fn=lambda pol: None,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
        show_progress=True,
    )

    main_policy = policy.policies[agents[0]]
    return result, main_policy


def watch(
    args: argparse.Namespace = get_args(),
) -> None:
    """
    测试预训练的PPO智能体
    """
    env = DummyVectorEnv([lambda: get_env(render_mode="human")])
    policy, optimizers, agents = get_agents(args)
    policy.eval()

    collector = Collector(policy, env, exploration_noise=False)  # 测试时不使用探索
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]

    # 打印所有PPO智能体的奖励
    print("Final rewards:")
    for i, agent_name in enumerate(agents):
        print(f"  {agent_name}: {rews[:, i].mean():.4f}")
    print(f"Episode length: {lens.mean()}")


if __name__ == "__main__":
    args = get_args()
    print("Training Configuration:")
    print(args)
    print("\nStarting training...")
    
    if args.watch:
        watch(args)
    else:
        result, agent = train_agent(args)
        print("Training completed!")
        print(f"Final result: {result}")
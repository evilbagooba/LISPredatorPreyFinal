from datetime import datetime
import os
import argparse
from copy import deepcopy
from typing import Optional, Tuple

import gymnasium
import numpy as np
import torch
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
from tianshou.policy import A2CPolicy
from tianshou.policy import TRPOPolicy
from tianshou.policy import NPGPolicy

from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
# 环境导入
import supersuit as ss
from pettingzoo.sisl import waterworld_v4

# 移除这行：from collector import Collector
agent_algorithms = {
    "predator_0": "Random",
    "predator_1": "Random", 
    "prey_0": "PPO",
    "prey_1": "PPO"
}

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)  # PPO通常使用更高的学习率
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="discount factor"
    )
    parser.add_argument("--n-step", type=int, default=1)  # PPO通常使用1步
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=4000)  # PPO训练epoch数可以少一些
    parser.add_argument("--step-per-epoch", type=int, default=50000)  # PPO需要更多数据
    parser.add_argument("--step-per-collect", type=int, default=2048)  # PPO典型值
    parser.add_argument("--repeat-per-collect", type=int, default=10)  # PPO特有参数
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128]
    )
    parser.add_argument("--training-num", type=int, default=10)
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

    # PPO特有参数
    parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="value function coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--value-clip", action="store_true", default=True, help="value clipping")
    parser.add_argument("--reward-normalization", action="store_true", default=False, 
                       help="reward normalization")
    parser.add_argument("--max-batchsize", type=int, default=256, help="max batch size for PPO")
    # PG特有参数调整建议
    parser.add_argument("--pg-lr", type=float, default=1e-3, 
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
    parser.add_argument("--optim-critic-iters", type=int, default=5,
                       help="number of critic optimization iterations per update")

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
    args.action_shape = env.action_space.shape or env.action_space.n

    # 获取环境中的所有智能体
    agents = env.agents

    # 定义每个智能体使用的算法 - 全部使用PPO
    # agent_algorithms = {
    #     "predator_0": "Random",
    #     "predator_1": "Random", 
    #     "prey_0": "PPO",
    #     "prey_1": "PPO"
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
        algo = agent_algorithms.get(agent_name, "PPO")

        if algo == "PPO":
            # 创建 Actor 网络 - 返回 (logits, state)
            class PPOActorNet(torch.nn.Module):
                def __init__(self, state_shape, action_shape, hidden_sizes, device="cpu"):
                    super().__init__()
                    self.output_shape = action_shape
                    
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
                    
                    # 输出层
                    layers.append(torch.nn.Linear(prev_dim, action_shape))
                    
                    self.net = torch.nn.Sequential(*layers)
                    
                def forward(self, obs, state=None, info={}):
                    """Actor网络前向传播 - 返回 (logits, state)"""
                    obs_data = self._extract_obs(obs)
                    output = self.net(obs_data)
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

            class PPOCriticNet(torch.nn.Module):
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
                    
                def forward(self, obs, **kwargs):
                    """PPO Critic网络前向传播 - 任何情况下都只返回tensor"""
                    obs_data = self._extract_obs(obs)
                    value = self.net(obs_data)
                    return value  # 永远只返回tensor，忽略其他参数
                
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

            # 创建PPO网络
            actor_net = PPOActorNet(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            critic_net = PPOCriticNet(
                args.state_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            # 优化器：同时更新 actor 和 critic
            optimizer = torch.optim.Adam(
                list(actor_net.parameters()) + list(critic_net.parameters()),
                lr=args.lr
            )

            # 存储每个智能体的优化器
            optimizers[agent_name] = optimizer

            # 创建 PPO 策略
            policy = PPOPolicy(
                actor=actor_net,
                critic=critic_net,
                optim=optimizer,
                dist_fn=lambda x: Categorical(logits=x),  # 重要：使用logits而不是probs
                discount_factor=args.gamma,
                eps_clip=args.eps_clip,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                gae_lambda=args.gae_lambda,
                value_clip=args.value_clip,
                reward_normalization=args.reward_normalization,
                max_batchsize=args.max_batchsize,
                action_space=None,    # 离散动作空间不需要
                action_scaling=False, # 离散动作空间不需要
                action_bound_method="", # 离散动作空间不需要
            )

            # 加载预训练模型（PPO专用）
            if agent_name in agent_resume_paths:
                resume_path = agent_resume_paths[agent_name]
                try:
                    print(f"Loading PPO pretrained model for {agent_name} from {resume_path}")
                    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
                    print(f"Successfully loaded PPO model for {agent_name}")
                except Exception as e:
                    print(f"Warning: Failed to load PPO model for {agent_name}: {e}")

            print(f"PPO policy created for {agent_name}")
            policies[agent_name] = policy
        elif algo == "A2C":
            # ==================== A2C算法实现 ====================
            
            # 创建 A2C Actor 网络 - 与PPO相同的架构
            class A2CActorNet(torch.nn.Module):
                def __init__(self, state_shape, action_shape, hidden_sizes, device="cpu"):
                    super().__init__()
                    self.output_shape = action_shape
                    
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
                    
                    # 输出层
                    layers.append(torch.nn.Linear(prev_dim, action_shape))
                    
                    self.net = torch.nn.Sequential(*layers)
                    
                def forward(self, obs, state=None, info={}):
                    """Actor网络前向传播 - 返回 (logits, state)"""
                    obs_data = self._extract_obs(obs)
                    output = self.net(obs_data)
                    return output, state
                
                def _extract_obs(self, obs):
                    """提取实际的观测数据 - 与PPO相同的逻辑"""
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

            # 创建 A2C Critic 网络 - 与PPO相同的架构
            class A2CCriticNet(torch.nn.Module):
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
                    
                def forward(self, obs, **kwargs):
                    """A2C Critic网络前向传播 - 只返回tensor"""
                    obs_data = self._extract_obs(obs)
                    value = self.net(obs_data)
                    return value
                
                def __call__(self, obs, **kwargs):
                    """确保所有调用方式都只返回tensor"""
                    return self.forward(obs, **kwargs)
                
                def _extract_obs(self, obs):
                    """提取实际的观测数据 - 与PPO相同的逻辑"""
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

            # 创建A2C网络
            actor_net = A2CActorNet(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            critic_net = A2CCriticNet(
                args.state_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            # 优化器：同时更新 actor 和 critic
            optimizer = torch.optim.Adam(
                list(actor_net.parameters()) + list(critic_net.parameters()),
                lr=args.lr
            )

            # 存储每个智能体的优化器
            optimizers[agent_name] = optimizer

            # 创建 A2C 策略
            policy = A2CPolicy(
                actor=actor_net,
                critic=critic_net,
                optim=optimizer,
                dist_fn=lambda x: Categorical(logits=x),  # 使用logits
                discount_factor=args.gamma,
                vf_coef=args.vf_coef,           # 值函数损失系数
                ent_coef=args.ent_coef,         # 熵正则化系数
                max_batchsize=args.max_batchsize,
                gae_lambda=args.gae_lambda,     # GAE参数（如果A2C支持）
                reward_normalization=args.reward_normalization,
                action_space=None,              # 离散动作空间不需要
                action_scaling=False,           # 离散动作空间不需要
                action_bound_method="",         # 离散动作空间不需要
            )

            # 加载预训练模型（A2C专用）
            if agent_name in agent_resume_paths:
                resume_path = agent_resume_paths[agent_name]
                try:
                    print(f"Loading A2C pretrained model for {agent_name} from {resume_path}")
                    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
                    print(f"Successfully loaded A2C model for {agent_name}")
                except Exception as e:
                    print(f"Warning: Failed to load A2C model for {agent_name}: {e}")

            print(f"A2C policy created for {agent_name}")
            policies[agent_name] = policy
        elif algo == "PG":

            # ==================== PG算法实现 ====================
            
            # 创建 PG Actor 网络 - 与PPO/A2C相同的架构，但不需要Critic
            class PGActorNet(torch.nn.Module):
                def __init__(self, state_shape, action_shape, hidden_sizes, device="cpu"):
                    super().__init__()
                    self.output_shape = action_shape
                    
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
                    
                    # 输出层
                    layers.append(torch.nn.Linear(prev_dim, action_shape))
                    
                    self.net = torch.nn.Sequential(*layers)
                    
                def forward(self, obs, state=None, info={}):
                    """Actor网络前向传播 - 返回 (logits, state)"""
                    obs_data = self._extract_obs(obs)
                    output = self.net(obs_data)
                    return output, state
                
                def _extract_obs(self, obs):
                    """提取实际的观测数据 - 与PPO/A2C相同的逻辑"""
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

            # 创建PG网络 - 只需要Actor，不需要Critic
            actor_net = PGActorNet(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            # 优化器：只优化 actor 网络
            optimizer = torch.optim.Adam(
                actor_net.parameters(),
                lr=args.lr
            )

            # 存储每个智能体的优化器
            optimizers[agent_name] = optimizer

            # 创建 PG 策略
            policy = PGPolicy(
                model=actor_net,
                optim=optimizer,
                dist_fn=lambda x: Categorical(logits=x),  # 使用logits
                discount_factor=args.gamma,
                reward_normalization=args.reward_normalization,
                action_space=None,              # 离散动作空间不需要
                action_scaling=False,           # 离散动作空间不需要
                action_bound_method="",         # 离散动作空间不需要
                # PG没有以下参数（与A2C/PPO的区别）：
                # - vf_coef (没有value function)
                # - ent_coef (基础PG通常不用熵正则化)
                # - gae_lambda (没有value function就没有GAE)
                # - eps_clip (没有clipping)
            )

            # 加载预训练模型（PG专用）
            if agent_name in agent_resume_paths:
                resume_path = agent_resume_paths[agent_name]
                try:
                    print(f"Loading PG pretrained model for {agent_name} from {resume_path}")
                    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
                    print(f"Successfully loaded PG model for {agent_name}")
                except Exception as e:
                    print(f"Warning: Failed to load PG model for {agent_name}: {e}")

            print(f"PG policy created for {agent_name}")
            policies[agent_name] = policy
        elif algo == "TRPO":
            # ==================== TRPO算法实现 ====================
            
            # 创建 TRPO Actor 网络 - 与PPO/A2C相同的架构
            class TRPOActorNet(torch.nn.Module):
                def __init__(self, state_shape, action_shape, hidden_sizes, device="cpu"):
                    super().__init__()
                    self.output_shape = action_shape
                    
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
                    
                    # 输出层
                    layers.append(torch.nn.Linear(prev_dim, action_shape))
                    
                    self.net = torch.nn.Sequential(*layers)
                    
                def forward(self, obs, state=None, info={}):
                    """Actor网络前向传播 - 返回 (logits, state)"""
                    obs_data = self._extract_obs(obs)
                    output = self.net(obs_data)
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

            # 创建 TRPO Critic 网络 - 与PPO/A2C相同的架构
            class TRPOCriticNet(torch.nn.Module):
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
                    
                def forward(self, obs, **kwargs):
                    """TRPO Critic网络前向传播 - 只返回tensor"""
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

            # 创建TRPO网络
            actor_net = TRPOActorNet(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            critic_net = TRPOCriticNet(
                args.state_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            # 优化器：同时更新 actor 和 critic
            optimizer = torch.optim.Adam(
                list(actor_net.parameters()) + list(critic_net.parameters()),
                lr=args.lr
            )

            # 存储每个智能体的优化器
            optimizers[agent_name] = optimizer

            # 创建 TRPO 策略 - 使用正确的参数
            policy = TRPOPolicy(
                actor=actor_net,
                critic=critic_net,
                optim=optimizer,
                dist_fn=lambda x: Categorical(logits=x),  # 使用logits
                # TRPO特有参数（根据源码）
                max_kl=0.0001,                           # KL散度约束
                backtrack_coeff=0.3,                   # 回溯系数
                max_backtracks=20,                     # 最大回溯次数
                # 从NPGPolicy继承的参数
                advantage_normalization=True,          # 优势标准化
                optim_critic_iters=5,                  # Critic优化次数
                # 从A2CPolicy继承的参数
                discount_factor=args.gamma,
                gae_lambda=args.gae_lambda,            # GAE参数
                vf_coef=args.vf_coef,                 # 值函数损失系数
                reward_normalization=args.reward_normalization,
                max_batchsize=args.max_batchsize,     # 最大批量大小
                # 动作空间参数
                action_space=None,                     # 离散动作空间不需要
                action_scaling=False,                  # 离散动作空间不需要
                action_bound_method="",                # 离散动作空间不需要
                deterministic_eval=False,              # 测试时是否确定性
            )

            # 加载预训练模型（TRPO专用）
            if agent_name in agent_resume_paths:
                resume_path = agent_resume_paths[agent_name]
                try:
                    print(f"Loading TRPO pretrained model for {agent_name} from {resume_path}")
                    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
                    print(f"Successfully loaded TRPO model for {agent_name}")
                except Exception as e:
                    print(f"Warning: Failed to load TRPO model for {agent_name}: {e}")

            print(f"TRPO policy created for {agent_name}")
            policies[agent_name] = policy
        elif algo == "NPG":
            # ==================== NPG算法实现 ====================
            
            # 创建 NPG Actor 网络 - 与A2C完全相同的架构
            class NPGActorNet(torch.nn.Module):
                def __init__(self, state_shape, action_shape, hidden_sizes, device="cpu"):
                    super().__init__()
                    self.output_shape = action_shape
                    
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
                    
                    # 输出层
                    layers.append(torch.nn.Linear(prev_dim, action_shape))
                    
                    self.net = torch.nn.Sequential(*layers)
                    
                def forward(self, obs, state=None, info={}):
                    """Actor网络前向传播 - 返回 (logits, state)"""
                    obs_data = self._extract_obs(obs)
                    output = self.net(obs_data)
                    return output, state
                
                def _extract_obs(self, obs):
                    """提取实际的观测数据 - 与其他算法相同的逻辑"""
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

            # 创建 NPG Critic 网络 - 与A2C完全相同的架构
            class NPGCriticNet(torch.nn.Module):
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
                    
                def forward(self, obs, **kwargs):
                    """NPG Critic网络前向传播 - 只返回tensor"""
                    obs_data = self._extract_obs(obs)
                    value = self.net(obs_data)
                    return value
                
                def __call__(self, obs, **kwargs):
                    """确保所有调用方式都只返回tensor"""
                    return self.forward(obs, **kwargs)
                
                def _extract_obs(self, obs):
                    """提取实际的观测数据 - 与其他算法相同的逻辑"""
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

            # 创建NPG网络
            actor_net = NPGActorNet(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            critic_net = NPGCriticNet(
                args.state_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)

            # 优化器：同时更新 actor 和 critic
            optimizer = torch.optim.Adam(
                list(actor_net.parameters()) + list(critic_net.parameters()),
                lr=args.lr
            )

            # 存储每个智能体的优化器
            optimizers[agent_name] = optimizer

            # 创建 NPG 策略 - 根据源码使用正确的参数
            policy = NPGPolicy(
                actor=actor_net,
                critic=critic_net,
                optim=optimizer,
                dist_fn=lambda x: Categorical(logits=x),  # 使用logits
                # NPG特有参数（从源码得出）
                advantage_normalization=True,          # 优势标准化，默认True
                optim_critic_iters=1,                  # Critic优化次数，默认5
                actor_step_size=0.003,                   # NPG特有：Actor步长，默认0.5
                # 从A2CPolicy继承的参数
                discount_factor=args.gamma,
                gae_lambda=args.gae_lambda,            # GAE参数
                vf_coef=args.vf_coef,                 # 值函数损失系数
                reward_normalization=args.reward_normalization,
                max_batchsize=args.max_batchsize,     # 最大批量大小
                # 动作空间参数
                action_space=None,                     # 离散动作空间不需要
                action_scaling=False,                  # 离散动作空间不需要
                action_bound_method="",                # 离散动作空间不需要
                deterministic_eval=False,              # 测试时是否确定性
            )

            # 加载预训练模型（NPG专用）
            if agent_name in agent_resume_paths:
                resume_path = agent_resume_paths[agent_name]
                try:
                    print(f"Loading NPG pretrained model for {agent_name} from {resume_path}")
                    policy.load_state_dict(torch.load(resume_path, map_location=args.device))
                    print(f"Successfully loaded NPG model for {agent_name}")
                except Exception as e:
                    print(f"Warning: Failed to load NPG model for {agent_name}: {e}")

            print(f"NPG policy created for {agent_name}")
            policies[agent_name] = policy
        elif algo == "Random":
            # 创建随机策略
            policies[agent_name] = RandomPolicy()
        else:
            raise ValueError(f"This is PPO-only trainer. Unsupported algorithm: {algo}")

    # 创建多智能体策略管理器
    policy_list = [policies[agent].to(args.device) for agent in agents]
    policy_manager = MultiAgentPolicyManager(policy_list, env)

    return policy_manager, optimizers, agents


def get_env(render_mode=None):
    """
    创建环境的函数
    """
    # waterworld环境 - 配置为PPO适合的设置
    agent_algos = ["PPO", "PPO", "PPO", "PPO"] *4  # 所有agent都使用PPO
    env = waterworld_v4.env(
        render_mode=None,
        n_predators=2,
        n_preys=14,
        n_evaders=50,
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=20,
        agent_algorithms=agent_algos
    )
    # 添加supersuit包装器
    env = ss.black_death_v3(env)
    return PettingZooEnv(env)


def train_agent(
    args: argparse.Namespace = get_args(),
) -> Tuple[dict, BasePolicy]:
    """
    PPO专用训练函数
    """
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

    print("Using on-policy trainer")

    # ======== collector setup =========
    # PPO不需要经验回放缓冲区
    train_collector = Collector(
        policy,
        train_envs,
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    
    # PPO需要更多的初始数据收集
    initial_collect_steps = max(args.step_per_collect, 200)
    train_collector.collect(n_step=initial_collect_steps)

    # ======== tensorboard logging setup =========
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("log", "waterworld", "ppo", f"run_{time_str}")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # ======== callback functions used during training =========
    # ======== callback functions used during training =========
    def save_best_fn(policy):
        # 1) 先确定基础的 model_save_path
        if hasattr(args, "model_save_path"):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, "waterworld", "ppo", "policy.pth"
            )

        # 2) 根据 model_save_path 的父目录，创建一个以时间命名的子文件夹
        base_dir = os.path.dirname(model_save_path)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, run_timestamp)
        os.makedirs(run_dir, exist_ok=True)

        # 3) 遍历所有 agent，保存模型
        for agent_name in agents:
            if agent_name not in agent_algorithms:
                raise ValueError(
                    f"Algorithm not found for agent {agent_name}. "
                    "Please specify an algorithm in 'agent_algorithms'."
                )
            algo = agent_algorithms[agent_name]

            # 只对有 state_dict 的 policy 进行保存
            if agent_name in policy.policies and hasattr(policy.policies[agent_name], "state_dict"):
                # 构造文件名：<agent>_<algo>.pth
                fname = f"{agent_name}_{algo}.pth"
                agent_model_path = os.path.join(run_dir, fname)
                torch.save(policy.policies[agent_name].state_dict(), agent_model_path)
                print(f"Saved {algo} model for {agent_name} to {agent_model_path}")
    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        # PPO不需要epsilon，但保留接口兼容性
        pass

    def test_fn(epoch, env_step):
        # PPO不需要epsilon，但保留接口兼容性
        pass
                
    def reward_metric(rews):
        # 返回所有智能体的平均奖励
        # print(rews)
        return rews[:, :2].mean(axis=1)

    # ======== PPO on-policy trainer =========
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
        show_progress=True,  # PPO训练显示进度
    )

    # 返回第一个智能体的策略
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
        print("training completed!")
        print(f"Final result: {result}")
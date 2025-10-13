from datetime import datetime

import os
import argparse
from copy import deepcopy
from typing import Optional, Tuple

import gymnasium
import numpy as np
import torch
from tianshou.data import VectorReplayBuffer, Collector
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
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
# 环境导入
import supersuit as ss
from pettingzoo.sisl import waterworld_v4
from tianshou.data import Batch

import logging

# 设置日志级别为 ERROR，这样只会记录 ERROR 及以上级别的日志
logging.basicConfig(level=logging.ERROR)

agent_algorithms = {
    "predator_0": "Random",
    "predator_1": "Random", 
    "prey_0": "Random",
    "prey_1": "Random"
}

# 全局变量
global_episode_counter = 0
writer_global = None
agents_global = []

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
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=50000)
    parser.add_argument("--step-per-collect", type=int, default=2000)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128, 128, 128])
    parser.add_argument("--training-num", type=int, default=16)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)
    parser.add_argument("--win-rate", type=float, default=0.6, help="the expected winning rate")
    parser.add_argument("--watch", default=False, action="store_true", help="no training, watch the play of pre-trained models")
    parser.add_argument("--agent-id", type=int, default=2, help="the learned agent plays as the agent_id-th player")
    
    # 连续动作空间特有参数
    parser.add_argument("--log-std-init", type=float, default=-1.0, help="initial log std for continuous actions")
    parser.add_argument("--log-std-min", type=float, default=-5.0, help="minimum log std for continuous actions")
    parser.add_argument("--log-std-max", type=float, default=1.5, help="maximum log std for continuous actions")
    parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="value function coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="entropy coefficient")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--value-clip", action="store_true", default=True, help="value clipping")
    parser.add_argument("--reward-normalization", action="store_true", default=False, help="reward normalization")
    parser.add_argument("--max-batchsize", type=int, default=128, help="max batch size for PPO")
    
    # 预训练模型加载参数
    parser.add_argument("--resume-path", type=str, default="", help="path to resume from pre-trained agent")
    parser.add_argument("--predator-0-resume-path", type=str, default="")
    parser.add_argument("--predator-1-resume-path", type=str, default="")
    parser.add_argument("--prey-0-resume-path", type=str, default="")
    parser.add_argument("--prey-1-resume-path", type=str, default="")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

def get_agents(args: argparse.Namespace = get_args()) -> Tuple[BasePolicy, dict, list]:
    """创建多智能体策略管理器"""
    env = get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    args.state_shape = observation_space.shape or observation_space.n
    
    # 修复动作空间处理 - 支持连续动作空间
    if hasattr(env.action_space, 'shape') and env.action_space.shape:
        args.action_shape = env.action_space.shape[0]
    else:
        args.action_shape = env.action_space.n
    
    print(f"Environment info:")
    print(f"  Action space: {env.action_space}")
    print(f"  Action shape: {args.action_shape}")
    print(f"  State shape: {args.state_shape}")

    # 获取环境中的所有智能体
    agents = env.agents

    # 多智能体配置逻辑
    policies = {}
    optimizers = {}

    for agent_name in agents:
        algo = agent_algorithms.get(agent_name, "Random")

        if algo == "Random":
            # 使用兼容连续动作空间的随机策略
            policy = ContinuousCompatibleRandomPolicy(env.action_space)
            print(f"Random policy created for {agent_name}")
            policies[agent_name] = policy
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

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
    """创建环境的函数"""
    agent_algos = ["Random", "Random", "Random", "Random"] * 50
    env = waterworld_v4.env(
        render_mode=render_mode,
        n_predators=2,
        n_preys=2,
        n_evaders=50,
        n_obstacles=2,
        obstacle_coord=[(0.2, 0.2), (0.8, 0.2)],
        n_poisons=20,
        agent_algorithms=agent_algos
    )
    # 添加supersuit包装器
    env = ss.black_death_v3(env)
    return PettingZooEnv(env)

def reward_metric(rews):
    """计算即时奖励并记录每个episode的奖励"""
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
                    writer_global.add_scalar(f"EpisodeReward/{role}/{agent_name}", 
                                           episode_reward, global_episode_counter + episode_idx)
        
        # 更新全局episode计数器
        global_episode_counter += rews.shape[0]
        writer_global.flush()
    
    return instant_reward

def train_agent(args: argparse.Namespace = get_args()) -> Tuple[dict, BasePolicy]:
    """训练函数"""
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

    # ======== collectors =========
    train_collector = Collector(
        policy, train_envs,
        buffer=VectorReplayBuffer(args.buffer_size, args.training_num),
        exploration_noise=False
    )
    test_collector = Collector(
        policy, test_envs,
        buffer=VectorReplayBuffer(max(50000, args.buffer_size // 2), args.test_num),
        exploration_noise=False
    )

    # ======== tensorboard logging setup =========
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("log", "waterworld", "random_random", f"run_{time_str}")
    writer_global = SummaryWriter(log_path)  # 设置全局writer
    
    # 简化的logger
    logger = TensorboardLogger(writer_global)

    # ======== 初始采样 =========
    initial_collect_steps = max(args.step_per_collect, 200)
    train_collector.collect(n_step=initial_collect_steps)

    # 简化的训练和测试函数
    def train_fn(epoch, env_step):
        pass  # 不需要特殊处理
        
    def test_fn(epoch, env_step):
        pass  # 不需要特殊处理

    # ======== on-policy trainer =========
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
        save_best_fn=lambda pol: None,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
        show_progress=True,
    )

    main_policy = policy.policies[agents[0]]
    return result, main_policy

def watch(args: argparse.Namespace = get_args()) -> None:
    """测试预训练的智能体"""
    env = DummyVectorEnv([lambda: get_env(render_mode="human")])
    policy, optimizers, agents = get_agents(args)
    policy.eval()

    collector = Collector(policy, env, exploration_noise=False)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]

    # 打印所有智能体的奖励
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
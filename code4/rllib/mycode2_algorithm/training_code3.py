#training_code2.py
from pathlib import Path
import torch
from PIL import Image
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian
from ray.tune.result import TRAINING_ITERATION
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core import (
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_RL_MODULE,
)
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentPendulum
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    check,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env

from pettingzoo.sisl import waterworld_v4

from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env
import os

import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy, softmax
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)

torch, _ = try_import_torch()

import wandb

class WandbLoggingCallback(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result, **kwargs):
        # 记录训练过程中的健康值
        print("[DEBUG] on_train_result 被调用")
        mean_reward = result.get("episode_reward_mean") or result.get("hist_stats", {}).get("episode_reward_mean")

        if mean_reward is not None:
            # Log 到 wandb
            wandb.log({"episode_reward_mean": mean_reward, "training_iteration": result.get("training_iteration", 0)})
        
        # 新增：记录健康值（如果可以从算法中获取）
        # 注意：这里可能需要根据实际情况调整获取健康值的方式
        try:
            # 尝试从环境中获取健康值数据
            env_runner_results = result.get("env_runners", {})
            if env_runner_results:
                # 这里需要根据RLLib的具体结构来获取健康值
                # 可能需要通过自定义metrics来实现
                pass
        except Exception as e:
            print(f"获取健康值时出错: {e}")


# 设置参数
import sys
sys.argv = [
    'notebook_script.py',
    '--enable-new-api-stack',
    '--num-agents=10',
    # 新增参数用于指定predator和prey的数量
    '--n-predators=5',
    '--wandb-key=fdd7656f474bba144dea1887bcdab534bc7df647',
    '--wandb-project=waterworld-v4',
    '--n-preys=5', 
    '--checkpoint-at-end',
    '--stop-reward=200.0',
    '--checkpoint-freq=1',
]

parser = add_rllib_example_script_args(
    default_iters=2,
    default_timesteps=10000,
    default_reward=0.0,
)

# 添加新的参数解析
parser.add_argument(
    "--n-predators",
    type=int,
    default=2,
    help="Number of predator agents"
)
parser.add_argument(
    "--n-preys", 
    type=int,
    default=2,
    help="Number of prey agents"
)
parser.add_argument(
    "--use-onnx-for-inference",
    action="store_true",
    help="Whether to convert the loaded module to ONNX format and then perform "
    "inference through this ONNX model.",
)
parser.add_argument(
    "--explore-during-inference",
    action="store_true",
    help="Whether the trained policy should use exploration during action "
    "inference.",
)
parser.add_argument(
    "--num-episodes-during-inference",
    type=int,
    default=10,
    help="Number of episodes to do inference over (after restoring from a checkpoint).",
)

agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C","PPO", "PPO", "DQN", "DQN", "A2C"]

args = parser.parse_args()

# 验证参数
assert args.n_predators > 0, "Must set --n-predators > 0 when running this script!"
assert args.n_preys > 0, "Must set --n-preys > 0 when running this script!"
assert (
    args.enable_new_api_stack
), "Must set --enable-new-api-stack when running this script!"

# 计算总智能体数量
total_agents = args.n_predators + args.n_preys
print(f"参数解析完成: n_predators={args.n_predators}, n_preys={args.n_preys}, total_agents={total_agents}, algo={args.algo}")

# 创建输出目录
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

wandb.init(project=args.wandb_project, config=vars(args))

# 修改环境注册，传递predator和prey的数量
register_env("env", lambda _: PettingZooEnv(
    waterworld_v4.env(
        n_predators=args.n_predators,
        n_preys=args.n_preys,
        agent_algorithms=agent_algos,
        initial_health=100.0
    )
))

# 创建新的policies字典，匹配环境中的agent命名
predator_policies = [f"predator_{i}" for i in range(args.n_predators)]
prey_policies = [f"prey_{i}" for i in range(args.n_preys)]
all_policies = predator_policies + prey_policies
print(all_policies)

# 创建RL module specs字典
rl_module_specs = {p: RLModuleSpec() for p in all_policies}
print(f"创建的policies: {list(all_policies)}")
print(f"创建的RL module specs: {list(rl_module_specs.keys())}")

base_config = (
    get_trainable_cls(args.algo)
    .get_default_config()
    .environment("env")
    .multi_agent(
        # 在新API中，只需要指定policy_mapping_fn
        policies=set(all_policies),
        policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
    )
    .training(
        vf_loss_coeff=0.005,
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs=rl_module_specs,
        ),
        model_config=DefaultModelConfig(vf_share_layers=True),
    )
    .callbacks(WandbLoggingCallback)
)

# 训练
print("开始训练...")
results = run_rllib_example_script_experiment(base_config, args, keep_ray_up=True)
print("训练完成")

###----------------------------------###
# 推理部分
###----------------------------------###        

# 获取最佳结果
print("获取最佳checkpoint...")
best_result = results.get_best_result(
    metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max"
)

print("加载所有智能体的RLModule...")
rl_modules = {}
# 修改：使用新的智能体命名方式
predator_agents = [f"predator_{i}" for i in range(args.n_predators)]
prey_agents = [f"prey_{i}" for i in range(args.n_preys)]
all_agent_names = predator_agents + prey_agents

for agent_name in all_agent_names:
    rl_module_path = os.path.join(
        best_result.checkpoint.path,
        "learner_group",
        "learner",
        "rl_module",
        agent_name,
    )
    
    if os.path.exists(rl_module_path):
        rl_modules[agent_name] = RLModule.from_checkpoint(rl_module_path)
        print(f"成功加载 {agent_name} 的模型")
    else:
        print(f"警告: 找不到 {agent_name} 的模型路径: {rl_module_path}")

print(f"总共加载了 {len(rl_modules)} 个智能体模型")

# 修复的保存GIF函数
def save_frames_as_gif(frames, filename, duration=100, output_dir="outputs"):
    """将帧序列保存为GIF
    
    Args:
        frames: 帧序列列表
        filename: 文件名（不包含扩展名）
        duration: 每帧持续时间（毫秒）
        output_dir: 输出目录
    """
    if not frames:
        print("没有帧可以保存")
        return None
    
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 生成完整的文件路径
    filepath = output_path / f"{filename}.gif"
    
    # 将numpy数组转换为PIL图像
    pil_frames = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            # 如果是numpy数组，转换为PIL图像
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame))
        else:
            pil_frames.append(frame)
    
    try:
        # 保存为GIF
        pil_frames[0].save(
            filepath,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF已保存到: {filepath.absolute()}")
        return str(filepath.absolute())
    except Exception as e:
        print(f"保存GIF时出错: {e}")
        return None

def get_action_from_rl_module(rl_module, observation, explore=False):
    """从RLModule获取动作"""
    # 将观察转换为torch tensor并添加batch维度
    input_dict = {Columns.OBS: torch.from_numpy(observation).unsqueeze(0)}
    
    if explore:
        # 使用探索性前向传播
        rl_module_out = rl_module.forward_exploration(input_dict)
        action_dist_inputs = rl_module_out["action_dist_inputs"][0]
        action = TorchDiagGaussian.from_logits(action_dist_inputs.unsqueeze(0)).sample().squeeze(0).numpy()
    else:
        # 使用推理前向传播
        rl_module_out = rl_module.forward_inference(input_dict)
        action_dist_inputs = rl_module_out["action_dist_inputs"][0]
        action = TorchDiagGaussian.from_logits(action_dist_inputs.unsqueeze(0)).sample().squeeze(0).numpy()
    
    return action

def get_agent_healths(env, all_agent_names):
    """获取所有智能体的健康值"""
    agent_healths = {}
    try:
        # 尝试从环境中获取健康值
        if hasattr(env, 'env') and hasattr(env.env, 'env') and hasattr(env.env.env, 'env'):
            inner_env = env.env.env.env
            if hasattr(inner_env, 'agents'):
                for i, agent_name in enumerate(all_agent_names):
                    if i < len(inner_env.agents):
                        agent_healths[agent_name] = float(inner_env.agents[i].shape.health)
        else:
            print("无法访问环境的内部结构获取健康值")
    except Exception as e:
        print(f"获取健康值时出错: {e}")
    
    return agent_healths

# 运行推理episodes
print("开始推理...")
for episode in range(args.num_episodes_during_inference):
    print(f"\n=== Episode {episode + 1} ===")
    
    # 创建环境
    env = waterworld_v4.env(
        n_predators=args.n_predators,
        n_preys=args.n_preys,
        agent_algorithms=agent_algos,
        initial_health=100.0,
        render_mode="rgb_array"
    )
    env.reset(seed=42 + episode)
    
    episode_rewards = {agent: 0 for agent in env.agents}
    step_count = 0
    frames = []
    gif_filename = f"waterworld_episode_{episode + 1}"

    try:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            # 累积奖励
            if agent in episode_rewards:
                episode_rewards[agent] += reward
            
            if termination or truncation:
                action = None
                print(f"{agent} 终止, 奖励: {reward}")
            else:
                # 使用对应的RLModule获取动作
                if agent in rl_modules:
                    try:
                        action = get_action_from_rl_module(
                            rl_modules[agent], 
                            observation, 
                            explore=args.explore_during_inference
                        )
                        # print(f"{agent} 执行动作: {action}")
                    except Exception as e:
                        # print(f"为 {agent} 获取动作时出错: {e}")
                        action = env.action_space(agent).sample()
                        # print(f"{agent} 使用随机动作: {action}")
                else:
                    action = env.action_space(agent).sample()
                    # print(f"{agent} 没有找到对应模型，使用随机动作: {action}")
            
            env.step(action)
            step_count += 1
            
            # 每隔几步捕获一帧
            if step_count % 10 == 0:
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    print(f"渲染第{step_count}步时出错: {e}")

            # 每100步输出一次进度
            if step_count % 100 == 0:
                print(f"已执行 {step_count} 步")
                
    except KeyboardInterrupt:
        print("用户中断")
        break
    except Exception as e:
        print(f"Episode {episode + 1} 出现错误: {e}")
    finally:
        # 获取最终的健康值
        final_healths = get_agent_healths(env, all_agent_names)
        env.close()
    
    # 保存GIF并记录到wandb
    gif_path = None
    if frames:
        gif_path = save_frames_as_gif(frames, gif_filename, duration=100, output_dir="outputs")
    
    # 准备日志数据
    log_data = {
        "episode_index": episode,
        "total_reward_mean": np.mean(list(episode_rewards.values())),
        "total_reward_std": np.std(list(episode_rewards.values())),
        "episode_steps": step_count,
    }
    
    # 添加每个智能体的奖励
    for agent_name, reward in episode_rewards.items():
        log_data[f"reward/{agent_name}"] = reward
    
    # 添加健康值信息
    if final_healths:
        for agent_name, health in final_healths.items():
            log_data[f"health/{agent_name}"] = health
        
        health_values = list(final_healths.values())
        log_data["health/mean"] = np.mean(health_values)
        log_data["health/std"] = np.std(health_values)
        log_data["health/min"] = np.min(health_values)
        log_data["health/max"] = np.max(health_values)
    
    # 添加GIF到wandb
    if gif_path and Path(gif_path).exists():
        try:
            log_data["episode_gif"] = wandb.Video(gif_path, fps=10, format="gif")
            print(f"成功添加GIF到wandb: {gif_path}")
        except Exception as e:
            print(f"添加GIF到wandb时出错: {e}")
    
    # 记录到wandb
    try:
        wandb.log(log_data)
        print(f"成功记录episode {episode + 1}的数据到wandb")
    except Exception as e:
        print(f"记录到wandb时出错: {e}")
    
    # 输出episode结果
    print(f"Episode {episode + 1} 完成")
    print("各智能体总奖励:")
    for agent, total_reward in episode_rewards.items():
        print(f"  {agent}: {total_reward:.2f}")
    print(f"总步数: {step_count}")
    
    if final_healths:
        print("各智能体最终健康值:")
        for agent, health in final_healths.items():
            print(f"  {agent}: {health:.2f}")

print("推理完成!")
wandb.finish()
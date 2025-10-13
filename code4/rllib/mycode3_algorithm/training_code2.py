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
    '--num-agents=4',
    # 新增参数用于指定predator和prey的数量
    '--n-predators=2',
    '--wandb-key=fdd7656f474bba144dea1887bcdab534bc7df647',
    '--wandb-project=waterworld-v4',
    '--n-preys=2', 
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


agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"]


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


import wandb

wandb.init(project=args.wandb_project, config=vars(args))  # 放在main训练逻辑之前


agent_algos = ["PPO", "PPO", "DQN", "DQN", "A2C"]


# 修改环境注册，传递predator和prey的数量
register_env("env", lambda _: PettingZooEnv(
    waterworld_v4.env(
        n_predators=args.n_predators,
        n_preys=args.n_preys,  # 注意：这里应该是n_preys而不是n_prey
        agent_algorithms=agent_algos,
        initial_health=100.0  # 新增：设置初始健康值

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
    .callbacks(WandbLoggingCallback)  # ✅ 加入回调
)


# 训练
print("开始训练...")
results =run_rllib_example_script_experiment(base_config, args, keep_ray_up=True)
# run_rllib_example_script_experiment(base_config, args, keep_ray_up=True)
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
# 在推理部分之前，添加这些函数
def save_frames_as_gif(frames, filename, duration=100, save_to_current_dir=True):
    """将帧序列保存为GIF
    
    Args:
        frames: 帧序列列表
        filename: 文件名（不包含扩展名）
        duration: 每帧持续时间（毫秒）
        save_to_current_dir: 是否保存到当前目录
    """
    if not frames:
        print("没有帧可以保存")
        return None
    
    # 根据参数选择保存路径
    if save_to_current_dir:
        # 保存到当前目录
        output_dir = Path(".")
        filepath = output_dir / f"{filename}.gif"
    else:
        # 保存到gifs子目录
        output_dir = Path("gifs")
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / f"{filename}.gif"
    
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
        return str(filepath)
    except Exception as e:
        print(f"保存GIF时出错: {e}")
        return None


# 推理阶段
print("开始推理...")

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





# 运行推理episodes
for episode in range(args.num_episodes_during_inference):
    print(f"\n=== Episode {episode + 1} ===")
    
    # 创建环境（使用新的参数方式）
    env = waterworld_v4.env(
        n_predators=args.n_predators,
        n_preys=args.n_preys,  # 注意：这里应该是n_preys而不是n_prey
        agent_algorithms=agent_algos,
        initial_health=100.0,  # 新增：设置初始健康值
        render_mode="rgb_array"  # 关键：启用渲染模式

    )
    env.reset(seed=42 + episode)
    
    episode_rewards = {agent: 0 for agent in env.agents}
    step_count = 0
    frames = []  # 存储渲染帧
    # 在episode开始时就定义gif_filename
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
                        print(f"{agent} 执行动作: {action}")
                    except Exception as e:
                        print(f"为 {agent} 获取动作时出错: {e}")
                        # 如果出错，使用随机动作作为备选
                        action = env.action_space(agent).sample()
                        print(f"{agent} 使用随机动作: {action}")
                else:
                    # 如果没有找到对应的模型，使用随机动作
                    action = env.action_space(agent).sample()
                    print(f"{agent} 没有找到对应模型，使用随机动作: {action}")
            
            env.step(action)
            step_count += 1
            # 每隔几步捕获一帧（可以调整频率）


            # 每隔几步捕获一帧（可以调整频率）
            if step_count % 10 == 0:  # 每10步捕获一帧
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    print(f"渲染第{step_count}步时出错: {e}")

            # 可选：每隔一定步数记录健康值
            if step_count % 50 == 0:  # 每50步记录一次
                # 获取每个agent的最终健康值
                agent_healths = {}
                try:
                    for i, agent_name in enumerate(all_agent_names):
                        # 通过索引访问对应的agent对象
                        if i < len(env.env.agents):
                            agent_healths[agent_name] = float(env.env.env.env.agents[i].shape.health)
                except Exception as e:
                    print(f"获取健康值时出错: {e}")
                    agent_healths = {}
                # 推理完每个 episode 后记录 reward、健康值和 GIF
                log_data = {
                    "episode_index": episode,
                    "total_reward_mean": np.mean(list(episode_rewards.values())),
                    "total_reward_std": np.std(list(episode_rewards.values())),
                    "gif": wandb.Video(f"gifs/{gif_filename}.gif", fps=10, format="gif")
                }

                # 添加每个agent的健康值
                for agent_name, health in agent_healths.items():
                    log_data[f"health/{agent_name}"] = health

                # 添加健康值的统计信息
                health_values = list(agent_healths.values())
                log_data["health/mean"] = np.mean(health_values)
                log_data["health/std"] = np.std(health_values)
                log_data["health/min"] = np.min(health_values)
                log_data["health/max"] = np.max(health_values)
                # # 处理GIF文件
                # if gif_filename:
                #     gif_path = Path(f"gifs/{gif_filename}.gif")
                    
                #     # 检查文件是否存在
                #     if gif_path.exists():
                #         try:
                #             # 方法1：使用wandb.Video（推荐用于视频）
                #             log_data["episode_gif"] = wandb.Video(str(gif_path), fps=10, format="gif")
                            
                #             # 方法2：或者使用wandb.Image（适用于GIF）
                #             # log_data["episode_gif"] = wandb.Image(str(gif_path))
                            
                #         except Exception as e:
                #             print(f"无法添加GIF到wandb日志: {e}")
                #             print(f"GIF文件路径: {gif_path}")
                #     else:
                #         print(f"GIF文件不存在: {gif_path}")
                # # 记录到wandb
                # try:
                #     wandb.log(log_data)
                #     print(f"成功记录episode {episode}的数据到wandb")
                # except Exception as e:
                #     print(f"记录到wandb时出错: {e}")
                # try:
                #     frame = env.render()
                #     if frame is not None:
                #         frames.append(frame)
                # except Exception as e:
                #     print(f"渲染第{step_count}步时出错: {e}")
                # 记录到wandb
                try:
                    wandb.log(log_data)
                    print(f"成功记录episode {episode} step {step_count}的数据到wandb")
                except Exception as e:
                    print(f"记录到wandb时出错: {e}")
            # 每100步输出一次进度
            if step_count % 100 == 0:
                print(f"已执行 {step_count} 步")
                
    except KeyboardInterrupt:
        print("用户中断")
        break
    except Exception as e:
        print(f"Episode {episode + 1} 出现错误: {e}")
    finally:
        env.close()
    # 保存GIF
    if frames:
        gif_filename = f"waterworld_episode_{episode + 1}"
        saved_gif_path= save_frames_as_gif(frames, gif_filename, duration=100)
        if saved_gif_path:
        # 推理完每个 episode 后记录 reward 和 GIF
        # 获取每个agent的最终健康值
            agent_healths = {}
            for agent_name in all_agent_names:
                # 通过agent名称找到对应的agent对象
                for i, agent_obj in enumerate(env.env.agents):
                    if agent_name == all_agent_names[i]:
                        agent_healths[agent_name] = float(env.env.env.env.agents[i].shape.health)
                        break

                # 推理完每个 episode 后记录 reward、健康值和 GIF
                log_data = {
                    "episode_index": episode,
                    "total_reward_mean": np.mean(list(episode_rewards.values())),
                    # "total_reward_std": np.std(list(episode_rewards.values())),
                    # "gif": wandb.Video(f"gifs/{gif_filename}.gif", fps=10, format="gif")
                }

                # 添加每个agent的健康值
                for agent_name, health in agent_healths.items():
                    log_data[f"health/{agent_name}"] = health

                # 添加健康值的统计信息
                if agent_healths:
                    health_values = list(agent_healths.values())
                    log_data["health/mean"] = np.mean(health_values)
                    log_data["health/std"] = np.std(health_values)
                    log_data["health/min"] = np.min(health_values)
                    log_data["health/max"] = np.max(health_values)

                wandb.log(log_data)
        else:
            print("没有捕获到帧，无法生成GIF")
        # 只记录 reward
        # 获取每个agent的最终健康值
        # agent_healths = {}
        # for agent_name in all_agent_names:
        #     # 通过agent名称找到对应的agent对象
        #     for i, agent_obj in enumerate(env.env.agents):
        #         if agent_name == all_agent_names[i]:
        #             agent_healths[agent_name] = float(env.env.env.env.agents[i].shape.health)
        #             break

        # # 记录 reward 和健康值
        # log_data = {
        #     "episode_index": episode,
        #     "total_reward_mean": np.mean(list(episode_rewards.values())),
        #     "total_reward_std": np.std(list(episode_rewards.values())),
        # }

        # # 添加每个agent的健康值
        # for agent_name, health in agent_healths.items():
        #     log_data[f"health/{agent_name}"] = health

        # # 添加健康值的统计信息
        # health_values = list(agent_healths.values())
        # log_data["health/mean"] = np.mean(health_values)
        # log_data["health/std"] = np.std(health_values)
        # log_data["health/min"] = np.min(health_values)
        # log_data["health/max"] = np.max(health_values)

        # wandb.log(log_data)
    # 输出episode结果
    print(f"Episode {episode + 1} 完成")
    print("各智能体总奖励:")
    for agent, total_reward in episode_rewards.items():
        print(f"  {agent}: {total_reward:.2f}")
    print(f"总步数: {step_count}")

print("推理完成!")



# 推理完每个 episode 后记录 reward
wandb.log({
    "episode_index": episode,
    "total_reward_mean": np.mean(list(episode_rewards.values())),
    "total_reward_std": np.std(list(episode_rewards.values())),
})

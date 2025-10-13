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


parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=1000000,
    default_reward=0.0,
)

args = parser.parse_args()

assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"

# Here, we use the "Agent Environment Cycle" (AEC) PettingZoo environment type.
# For a "Parallel" environment example, see the rock paper scissors examples
# in this same repository folder.
register_env("env", lambda _: PettingZooEnv(waterworld_v4.env()))

# Policies are called just like the agents (exact 1:1 mapping).
policies = {f"pursuer_{i}" for i in range(args.num_agents)}

base_config = (
    get_trainable_cls(args.algo)
    .get_default_config()
    .environment("env")
    .multi_agent(
        policies=policies,
        # Exact 1:1 mapping from AgentID to ModuleID.
        policy_mapping_fn=(lambda aid, *args, **kwargs: aid),
    )
    .training(
        vf_loss_coeff=0.005,
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={p: RLModuleSpec() for p in policies},
        ),
        model_config=DefaultModelConfig(vf_share_layers=True),
    )
)

run_rllib_example_script_experiment(base_config, args)
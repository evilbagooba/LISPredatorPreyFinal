# # from ray.rllib.algorithms.ppo import PPOConfig
# # from ray.rllib.connectors.env_to_module import FlattenObservations

# # # Configure the algorithm.
# # config = (
# #     PPOConfig()
# #     .environment("Taxi-v3")
# #     .env_runners(
# #         num_env_runners=2,
# #         # Observations are discrete (ints) -> We need to flatten (one-hot) them.
# #         env_to_module_connector=lambda env: FlattenObservations(),
# #     )
# #     .evaluation(evaluation_num_env_runners=1)
# # )

# # from pprint import pprint

# # # Build the algorithm.
# # algo = config.build_algo()

# # # Train it for 5 iterations ...
# # for _ in range(5):
# #     pprint(algo.train())
# # # ... and evaluate it.
# # pprint(algo.evaluate())

# # # Release the algo's resources (remote actors, like EnvRunners and Learners).
# # algo.stop()


# import gymnasium as gym

# from ray.rllib.env.multi_agent_env import MultiAgentEnv


# class RockPaperScissors(MultiAgentEnv):
#     """Two-player environment for the famous rock paper scissors game.

#     Both players always move simultaneously over a course of 10 timesteps in total.
#     The winner of each timestep receives reward of +1, the losing player -1.0.

#     The observation of each player is the last opponent action.
#     """

#     ROCK = 0
#     PAPER = 1
#     SCISSORS = 2
#     LIZARD = 3
#     SPOCK = 4

#     WIN_MATRIX = {
#         (ROCK, ROCK): (0, 0),
#         (ROCK, PAPER): (-1, 1),
#         (ROCK, SCISSORS): (1, -1),
#         (PAPER, ROCK): (1, -1),
#         (PAPER, PAPER): (0, 0),
#         (PAPER, SCISSORS): (-1, 1),
#         (SCISSORS, ROCK): (-1, 1),
#         (SCISSORS, PAPER): (1, -1),
#         (SCISSORS, SCISSORS): (0, 0),
#     }
#     def __init__(self, config=None):
#         super().__init__()

#         self.agents = self.possible_agents = ["player1", "player2"]

#         # The observations are always the last taken actions. Hence observation- and
#         # action spaces are identical.
#         self.observation_spaces = self.action_spaces = {
#             "player1": gym.spaces.Discrete(3),
#             "player2": gym.spaces.Discrete(3),
#         }
#         self.last_move = None
#         self.num_moves = 0
#     def reset(self, *, seed=None, options=None):
#         self.num_moves = 0

#         # The first observation should not matter (none of the agents has moved yet).
#         # Set them to 0.
#         return {
#             "player1": 0,
#             "player2": 0,
#         }, {}  # <- empty infos dict
#     def step(self, action_dict):
#         self.num_moves += 1

#         move1 = action_dict["player1"]
#         move2 = action_dict["player2"]

#         # Set the next observations (simply use the other player's action).
#         # Note that because we are publishing both players in the observations dict,
#         # we expect both players to act in the next `step()` (simultaneous stepping).
#         observations = {"player1": move2, "player2": move1}

#         # Compute rewards for each player based on the win-matrix.
#         r1, r2 = self.WIN_MATRIX[move1, move2]
#         rewards = {"player1": r1, "player2": r2}

#         # Terminate the entire episode (for all agents) once 10 moves have been made.
#         terminateds = {"__all__": self.num_moves >= 10}

#         # Leave truncateds and infos empty.
#         return observations, rewards, terminateds, {}, {}



"""Example of running a multi-agent experiment w/ agents always acting simultaneously.

This example:
    - demonstrates how to write your own (multi-agent) environment using RLlib's
    MultiAgentEnv API.
    - shows how to implement the `reset()` and `step()` methods of the env such that
    the agents act simultaneously.
    - shows how to configure and setup this environment class within an RLlib
    Algorithm config.
    - runs the experiment with the configured algo, trying to solve the environment.


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack --sheldon-cooper-mode`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
You should see results similar to the following in your console output:

+-----------------------------------+----------+--------+------------------+-------+
| Trial name                        | status   |   iter |   total time (s) |    ts |
|-----------------------------------+----------+--------+------------------+-------+
| PPO_RockPaperScissors_8cef7_00000 | RUNNING  |      3 |          16.5348 | 12000 |
+-----------------------------------+----------+--------+------------------+-------+
+-------------------+------------------+------------------+
|   combined return |   return player2 |   return player1 |
|-------------------+------------------+------------------|
|                 0 |            -0.15 |             0.15 |
+-------------------+------------------+------------------+

Note that b/c we are playing a zero-sum game, the overall return remains 0.0 at
all times.
"""



# from ray.rllib.examples.envs.classes.multi_agent.rock_paper_scissors import (
#     RockPaperScissors,
# )
# from ray.rllib.connectors.env_to_module.flatten_observations import FlattenObservations
# from ray.rllib.utils.test_utils import (
#     add_rllib_example_script_args,
#     run_rllib_example_script_experiment,
# )
# from ray.tune.registry import get_trainable_cls, register_env  # noqa


# parser = add_rllib_example_script_args(
#     default_reward=0.9, default_iters=50, default_timesteps=100000
# )
# parser.set_defaults(
#     enable_new_api_stack=True,
#     num_agents=2,
# )
# parser.add_argument(
#     "--sheldon-cooper-mode",
#     action="store_true",
#     help="Whether to add two more actions to the game: Lizard and Spock. "
#     "Watch here for more details :) https://www.youtube.com/watch?v=x5Q6-wMx-K8",
# )


# if __name__ == "__main__":
#     args = parser.parse_args()

#     assert args.num_agents == 2, "Must set --num-agents=2 when running this script!"

#     # You can also register the env creator function explicitly with:
#     # register_env("env", lambda cfg: RockPaperScissors({"sheldon_cooper_mode": False}))

#     # Or you can hard code certain settings into the Env's constructor (`config`).
#     # register_env(
#     #    "rock-paper-scissors-w-sheldon-mode-activated",
#     #    lambda config: RockPaperScissors({**config, **{"sheldon_cooper_mode": True}}),
#     # )

#     # Or allow the RLlib user to set more c'tor options via their algo config:
#     # config.environment(env_config={[c'tor arg name]: [value]})
#     # register_env("rock-paper-scissors", lambda cfg: RockPaperScissors(cfg))

#     base_config = (
#         get_trainable_cls(args.algo)
#         .get_default_config()
#         .environment(
#             RockPaperScissors,
#             env_config={"sheldon_cooper_mode": args.sheldon_cooper_mode},
#         )
#         .env_runners(
#             env_to_module_connector=(
#                 lambda env, spaces, device: FlattenObservations(multi_agent=True)
#             ),
#         )
#         .multi_agent(
#             # Define two policies.
#             policies={"player1", "player2"},
#             # Map agent "player1" to policy "player1" and agent "player2" to policy
#             # "player2".
#             policy_mapping_fn=lambda agent_id, episode, **kw: agent_id,
#         )
#     )

#     run_rllib_example_script_experiment(base_config, args)




"""Runs the PettingZoo Waterworld multi-agent env in RLlib using single policy learning.

Other than the `pettingzoo_independent_learning.py` example (in this same folder),
this example simply trains a single policy (shared by all agents).

See: https://pettingzoo.farama.org/environments/sisl/waterworld/
for more details on the environment.


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack --num-agents=2`

Control the number of agents and policies (RLModules) via --num-agents and
--num-policies.

This works with hundreds of agents and policies, but note that initializing
many policies might take some time.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
The above options can reach a combined reward of roughly ~0.0 after about 500k-1M env
timesteps. Keep in mind, though, that in this setup, the agents do not have the
opportunity to benefit from or even out other agents' mistakes (and behavior in general)
as everyone is using the same policy. Hence, this example learns a more generic policy,
which might be less specialized to certain "niche exploitation opportunities" inside
the env:

+---------------------+----------+-----------------+--------+-----------------+
| Trial name          | status   | loc             |   iter |  total time (s) |
|---------------------+----------+-----------------+--------+-----------------+
| PPO_env_91f49_00000 | RUNNING  | 127.0.0.1:63676 |    200 |         605.176 |
+---------------------+----------+-----------------+--------+-----------------+

+--------+-------------------+-------------+
|     ts |   combined reward |   reward p0 |
+--------+-------------------+-------------|
| 800000 |          0.323752 |    0.161876 |
+--------+-------------------+-------------+
"""
from pathlib import Path

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


parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=1000000,
    default_reward=0.0,
)


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"

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

    # run_rllib_example_script_experiment(base_config, args)
    results = run_rllib_example_script_experiment(base_config, args, keep_ray_up=True)

    # Now swap in the RLModule weights for policy 0.
    chkpt_path = results.get_best_result().checkpoint.path
    p_0_module_state_path = (
        Path(chkpt_path)  # <- algorithm's checkpoint dir
        / COMPONENT_LEARNER_GROUP  # <- learner group
        / COMPONENT_LEARNER  # <- learner
        / COMPONENT_RL_MODULE  # <- MultiRLModule
        / "pursuer_0"  # <- (single) RLModule
    )

    class LoadP0OnAlgoInitCallback(DefaultCallbacks):
        def on_algorithm_init(self, *, algorithm, **kwargs):
            module_p0 = algorithm.get_module("pursuer_0")
            weight_before = convert_to_numpy(next(iter(module_p0.parameters())))
            algorithm.restore_from_path(
                p_0_module_state_path,
                component=(
                    COMPONENT_LEARNER_GROUP
                    + "/"
                    + COMPONENT_LEARNER
                    + "/"
                    + COMPONENT_RL_MODULE
                    + "/pursuer_0"
                ),
            )
            # Make sure weights were updated.
            weight_after = convert_to_numpy(next(iter(module_p0.parameters())))
            check(weight_before, weight_after, false=True)

    base_config.callbacks(LoadP0OnAlgoInitCallback)

    # Define stopping criteria.
    stop = {
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": -800.0,
        f"{ENV_RUNNER_RESULTS}/{NUM_ENV_STEPS_SAMPLED_LIFETIME}": 100000,
        TRAINING_ITERATION: 100,
    }

    # Run the experiment again with the restored MultiRLModule.
    run_rllib_example_script_experiment(base_config, args, stop=stop)

# """
# Trains PPO hrl agent.
# """

# import argparse
# import logging
# import os
# from typing import Any

# import grid2op
# import ray
# from grid2op.Environment import BaseEnv
# from gymnasium.spaces import Discrete
# from ray import air, tune
# from ray.rllib.algorithms import ppo  # import the type of agents
# from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
# from ray.rllib.policy.policy import PolicySpec

# from mahrl.experiments.utils import (
#     calculate_action_space_asymmetry,
#     calculate_action_space_medha,
#     calculate_action_space_tennet,
# )
# from mahrl.experiments.yaml import load_config
# from mahrl.grid2op_env.custom_environment import (
#     HierarchicalCustomizedGrid2OpEnvironment,
# )
# from mahrl.multi_agent.policy import (
#     CapaPolicy,
#     DoNothingPolicy,
#     GreedyPolicy,
#     SelectAgentPolicy,
# )


# def run_training(config: dict[str, Any], setup: dict[str, Any]) -> None:
#     """
#     Function that runs the training script.
#     """
#     # init ray
#     ray.init()

#     # Create tuner
#     tuner = tune.Tuner(
#         ppo.PPO,
#         param_space=config,
#         run_config=air.RunConfig(
#             stop={"timesteps_total": setup["nb_timesteps"]},
#             storage_path=os.path.abspath(setup["storage_path"]),
#             checkpoint_config=air.CheckpointConfig(
#                 checkpoint_frequency=setup["checkpoint_freq"],
#                 checkpoint_at_end=True,
#                 checkpoint_score_attribute="evaluation/episode_reward_mean",
#             ),
#             verbose=setup["verbose"],
#         ),
#     )

#     # Launch tuning
#     try:
#         tuner.fit()
#     finally:
#         # Close ray instance
#         ray.shutdown()


# def find_list_of_agents(env: BaseEnv, action_space: str) -> list[int]:
#     """
#     Function that returns the number of controllable substations.
#     """
#     if action_space == "asymmetry":
#         _, _, list_of_agents = calculate_action_space_asymmetry(env)
#         return list_of_agents
#     if action_space == "medha":
#         _, _, list_of_agents = calculate_action_space_medha(env)
#         return list_of_agents
#     if action_space == "tennet":
#         _, _, list_of_agents = calculate_action_space_tennet(env)
#         return list_of_agents
#     raise ValueError("The action space is not supported.")


# def find_substation_per_lines(
#     env: BaseEnv, list_of_agents: list[int]
# ) -> dict[int, list[int]]:
#     """
#     Returns a dictionary connecting line ids to substations.
#     """
#     line_info = {agent: [] for agent in list_of_agents}
#     for sub_idx in list_of_agents:
#         for or_id in env.observation_space.get_obj_connect_to(substation_id=sub_idx)[
#             "lines_or_id"
#         ]:
#             line_info[sub_idx].append(or_id)
#         for ex_id in env.observation_space.get_obj_connect_to(substation_id=sub_idx)[
#             "lines_ex_id"
#         ]:
#             line_info[sub_idx].append(ex_id)

#     return line_info


# def setup_config(config_path: str) -> None:
#     """
#     Loads the json as config and sets it up for training.
#     """
#     # load base PPO config and load in hyperparameters
#     ppo_config = ppo.PPOConfig().to_dict()
#     custom_config = load_config(config_path)
#     ppo_config.update(custom_config["training"])
#     ppo_config.update(custom_config["debugging"])
#     ppo_config.update(custom_config["framework"])
#     ppo_config.update(custom_config["rl_module"])
#     ppo_config.update(custom_config["explore"])
#     ppo_config.update(custom_config["callbacks"])
#     ppo_config.update(custom_config["environment"])
#     ppo_config.update(custom_config["multi_agent"])

#     setup_env = grid2op.make(custom_config["environment"]["env_config"]["env_name"])
#     # Make as number additional policies as controllable substations
#     list_of_agents = find_list_of_agents(
#         setup_env,
#         custom_config["environment"]["env_config"]["action_space"],
#     )

#     print(f"list_of_agents={list_of_agents}")

#     line_info = find_substation_per_lines(setup_env, list_of_agents)
#     # TODO: Give these policies own parameters
#     # TODO: First use the rule-based policies
#     # TODO adjust policies to train config

#     policies = {
#         "high_level_policy": PolicySpec(  # chooses RL or do-nothing agent
#             policy_class=SelectAgentPolicy,
#             observation_space=None,  # infer automatically from env
#             action_space=Discrete(2),  # choose one of agents
#             config=(
#                 AlgorithmConfig()
#                 .training(
#                     _enable_learner_api=False,
#                     model={
#                         "custom_model_config": {
#                             "rho_threshold": custom_config["environment"]["env_config"][
#                                 "rho_threshold"
#                             ]
#                         }
#                     },
#                 )
#                 .rl_module(_enable_rl_module_api=False)
#                 .exploration(
#                     exploration_config={
#                         "type": "EpsilonGreedy",
#                     }
#                 )
#                 .rollouts(preprocessor_pref=None)
#             ),
#         ),
#         "choose_substation_policy": PolicySpec(  # rule based substation selection
#             policy_class=CapaPolicy,
#             observation_space=None,  # infer automatically from env
#             action_space=Discrete(len(list_of_agents)),  # infer automatically from env
#             config=(
#                 AlgorithmConfig()
#                 .training(
#                     _enable_learner_api=False,
#                     model={"custom_model_config": {"line_info": line_info}},
#                 )
#                 .rl_module(_enable_rl_module_api=False)
#                 .exploration(
#                     exploration_config={
#                         "type": "EpsilonGreedy",
#                     }
#                 )
#                 .rollouts(preprocessor_pref=None)
#             ),
#         ),
#         "do_nothing_policy": PolicySpec(  # performs do-nothing action
#             policy_class=DoNothingPolicy,
#             observation_space=None,  # infer automatically from env
#             action_space=Discrete(1),  # only perform do-nothing
#             config=(
#                 AlgorithmConfig()
#                 .training(_enable_learner_api=False)
#                 .rl_module(_enable_rl_module_api=False)
#                 .exploration(
#                     exploration_config={
#                         "type": "EpsilonGreedy",
#                     }
#                 )
#             ),
#         ),
#     }

#     # TODO: Change the rl policy to another reward function
#     # Add reinforcement learning policies to the dictionary
#     for sub_idx in list_of_agents:
#         policies[
#             f"reinforcement_learning_policy_{sub_idx}"
#         ] = PolicySpec(  # rule based substation selection
#             policy_class=GreedyPolicy,
#             observation_space=None,  # infer automatically from env
#             action_space=None,  # infer automatically from env
#             config=(
#                 AlgorithmConfig()
#                 .training(
#                     _enable_learner_api=False,
#                     model={
#                         "custom_model_config": {
#                             "env_config": custom_config["environment"]["env_config"]
#                         }
#                     },
#                 )
#                 .rl_module(_enable_rl_module_api=False)
#                 .exploration(
#                     exploration_config={
#                         "type": "EpsilonGreedy",
#                     }
#                 )
#                 .rollouts(preprocessor_pref=None)
#             ),
#         )

#     # load environment and agents manually
#     ppo_config.update({"policies": policies})
#     ppo_config.update({"env": HierarchicalCustomizedGrid2OpEnvironment})

#     run_training(ppo_config, custom_config["setup"])


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process possible variables.")

#     parser.add_argument(
#         "-f",
#         "--file_path",
#         type=str,
#         help="Path to the config file.",
#     )

#     # Parse the command-line arguments
#     args = parser.parse_args()

#     # Access the parsed arguments
#     input_file_path = args.file_path

#     if input_file_path:
#         setup_config(input_file_path)
#     else:
#         parser.print_help()
#         logging.error("\nError: --file_path is required to specify config location.")

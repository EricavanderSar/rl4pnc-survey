"""
Trains PPO hrl agent.
"""

import argparse
import logging
from typing import Any

import grid2op
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from ray.rllib.algorithms import ppo  # import the type of agents
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec

from mahrl.algorithms.custom_ppo import CustomPPO
from mahrl.experiments.utils import (
    find_list_of_agents,
    find_substation_per_lines,
    run_training,
)
from mahrl.experiments.yaml import load_config
from mahrl.grid2op_env.custom_environment import (
    GreedyHierarchicalCustomizedGrid2OpEnvironment,
    HierarchicalCustomizedGrid2OpEnvironment,
)
from mahrl.multi_agent.policy import CapaPolicy, DoNothingPolicy, SelectAgentPolicy


def select_mid_level_policy(
    middle_agent_type: str,
    list_of_agents: list[int],
    line_info: dict[int, list[int]],
    custom_config: dict[str, Any],
) -> PolicySpec:
    """
    Specifies the policy for the middle level agent.
    """
    # TODO Determine number of actions
    base_action = gymnasium.spaces.Discrete(100)

    capa_observation = gymnasium.spaces.Dict(
        {
            "previous_obs": gymnasium.spaces.Dict(
                {
                    "gen_p": gymnasium.spaces.Box(
                        -12.01,
                        np.array([22.01, 42.010002]),
                        (2,),
                        np.float32,
                    ),
                    "load_p": gymnasium.spaces.Box(-np.inf, np.inf, (3,), np.float32),
                    "p_ex": gymnasium.spaces.Box(-np.inf, np.inf, (8,), np.float32),
                    "p_or": gymnasium.spaces.Box(-np.inf, np.inf, (8,), np.float32),
                    "rho": gymnasium.spaces.Box(0.0, np.inf, (8,), np.float32),
                    "timestep_overflow": gymnasium.spaces.Box(
                        -2147483648, 2147483647, (8,), np.int32
                    ),
                    "topo_vect": gymnasium.spaces.Box(-1, 2, (21,), np.int32),
                }
            ),
            "proposed_actions": gymnasium.spaces.Dict(
                {str(i): base_action for i in list_of_agents}
            ),
            "reset_capa_idx": gymnasium.spaces.Discrete(2),
        }
    )

    if middle_agent_type == "capa":
        custom_config["environment"]["env_config"]["capa"] = True

        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=CapaPolicy,
            observation_space=capa_observation,  # information specifically for CAPA
            action_space=Discrete(len(list_of_agents)),  # choose one of agents
            config=(
                AlgorithmConfig()
                .training(
                    _enable_learner_api=False,
                    model={
                        "custom_model_config": {
                            "line_info": line_info,
                            "environment": custom_config["environment"],
                        },
                    },
                )
                .rl_module(_enable_rl_module_api=False)
                .rollouts(preprocessor_pref=None)
            ),
        )
    elif middle_agent_type == "rl":
        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=None,  # use default policy of PPO
            observation_space=capa_observation,
            action_space=Discrete(len(list_of_agents)),  # choose one of agents
            config=(
                AlgorithmConfig()
                # .rollouts(preprocessor_pref=None)
                .exploration(exploration_config={"type": "StochasticSampling"})
            ),
        )
    else:
        raise ValueError(
            f"Middle agent type {middle_agent_type} not recognized. Please use 'capa' or 'rl'."
        )
    return mid_level_policy


def setup_config(
    config_path: str, middle_agent_type: str, lower_agent_type: str
) -> None:
    """
    Loads the json as config and sets it up for training.
    """
    # load base PPO config and load in hyperparameters
    ppo_config = ppo.PPOConfig().to_dict()
    custom_config = load_config(config_path)

    ppo_config.update(custom_config["training"])
    ppo_config.update(custom_config["debugging"])
    ppo_config.update(custom_config["framework"])
    ppo_config.update(custom_config["rl_module"])
    ppo_config.update(custom_config["explore"])
    ppo_config.update(custom_config["callbacks"])
    ppo_config.update(custom_config["environment"])
    ppo_config.update(custom_config["multi_agent"])
    ppo_config.update(custom_config["evaluation"])

    setup_env = grid2op.make(custom_config["environment"]["env_config"]["env_name"])

    # Make as number additional policies as controllable substations
    agent_per_substation = find_list_of_agents(
        setup_env,
        custom_config["environment"]["env_config"]["action_space"],
    )

    list_of_substations = list(agent_per_substation.keys())

    line_info = find_substation_per_lines(setup_env, list_of_substations)
    # TODO: Give these policies own parameters
    # TODO: First use the rule-based policies
    # TODO adjust policies to train config

    mid_level_policy = select_mid_level_policy(
        middle_agent_type, list_of_substations, line_info, custom_config
    )

    policies = {
        "high_level_policy": PolicySpec(  # chooses RL or do-nothing agent
            policy_class=SelectAgentPolicy,
            observation_space=None,  # infer automatically from env
            action_space=Discrete(2),  # choose one of agents
            config=(
                AlgorithmConfig()
                .training(
                    _enable_learner_api=False,
                    model={
                        "custom_model_config": {
                            "rho_threshold": custom_config["environment"]["env_config"][
                                "rho_threshold"
                            ]
                        }
                    },
                )
                .rl_module(_enable_rl_module_api=False)
                .rollouts(preprocessor_pref=None)
            ),
        ),
        "choose_substation_policy": mid_level_policy,
        "do_nothing_policy": PolicySpec(  # performs do-nothing action
            policy_class=DoNothingPolicy,
            observation_space=None,  # infer automatically from env
            action_space=Discrete(1),  # only perform do-nothing
            config=(
                AlgorithmConfig()
                .training(_enable_learner_api=False)
                .rl_module(_enable_rl_module_api=False)
            ),
        ),
    }

    if lower_agent_type == "rl":  # add a rl agent for each substation
        # Add reinforcement learning policies to the dictionary
        for sub_idx, num_actions in agent_per_substation.items():
            policies[
                f"reinforcement_learning_policy_{sub_idx}"
            ] = PolicySpec(  # rule based substation selection
                policy_class=None,  # infer automatically from env (PPO)
                observation_space=None,  # infer automatically from env
                action_space=Discrete(int(num_actions)),
                config=None,
            )

    # if policy is rl, set an agent to train
    if middle_agent_type == "rl":
        ppo_config["policies_to_train"] = ["choose_substation_policy"]
    elif middle_agent_type == "capa":
        ppo_config["policies_to_train"] = []

    if lower_agent_type == "rl":
        ppo_config["policies_to_train"] += [
            f"reinforcement_learning_policy_{sub_idx}"
            for sub_idx in list_of_substations
        ]
        ppo_config.update({"env": HierarchicalCustomizedGrid2OpEnvironment})
    elif lower_agent_type == "greedy":
        ppo_config.update({"env": GreedyHierarchicalCustomizedGrid2OpEnvironment})

    # TODO: Get exploration from config explicitly?

    # load environment and agents manually
    ppo_config.update({"policies": policies})

    run_training(ppo_config, custom_config["setup"], CustomPPO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the config file.",
    )

    parser.add_argument(
        "-m",
        "--middle",
        type=str,
        default="capa",
        help="The type of middle level agent (capa or rl).",
    )

    parser.add_argument(
        "-l",
        "--lower",
        type=str,
        default="greedy",
        help="The type of middle level agent (greedy or rl).",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    input_config_path = args.config

    if input_config_path:
        setup_config(input_config_path, args.middle, args.lower)
    else:
        parser.print_help()
        logging.error("\nError: --file_path is required to specify config location.")

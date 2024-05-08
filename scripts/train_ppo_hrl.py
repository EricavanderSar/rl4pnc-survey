"""
Trains PPO hrl agent.
"""

import argparse
import logging
from typing import Any

import grid2op
import gymnasium as gym
import numpy as np
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
    HierarchicalCustomizedGrid2OpEnvironment,
)
from mahrl.grid2op_env.greedy_environment import (
    GreedyHierarchicalCustomizedGrid2OpEnvironment,
)
from mahrl.multi_agent.policy import (
    ArgMaxPolicy,
    CapaPolicy,
    DoNothingPolicy,
    RandomPolicy,
    SelectAgentPolicy,
)


def setup_gym_spaces(
    agent_per_substation: dict[int, int], env_info: dict[str, int]
) -> tuple[gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict]:
    """
    Set up the gym spaces for the RL environment.

    Args:
        agent_per_substation (dict[int, int]): A dictionary mapping substation IDs to the number of agents per substation.
        env_info (dict[str, int]): A dictionary containing information about the environment.

    Returns:
        tuple[gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict]: A tuple containing the gym spaces for
            previous observations, proposed actions, and proposed confidences.
    """
    gym_previous_obs = gym.spaces.Dict(
        {
            "gen_p": gym.spaces.Box(
                -np.inf,
                np.inf,
                (env_info["num_gen"],),
                np.float32,
            ),
            "load_p": gym.spaces.Box(
                -np.inf, np.inf, (env_info["num_load"],), np.float32
            ),
            "p_ex": gym.spaces.Box(
                -np.inf, np.inf, (env_info["num_line"],), np.float32
            ),
            "p_or": gym.spaces.Box(
                -np.inf, np.inf, (env_info["num_line"],), np.float32
            ),
            "rho": gym.spaces.Box(0.0, np.inf, (env_info["num_line"],), np.float32),
            "timestep_overflow": gym.spaces.Box(
                -2147483648, 2147483647, (env_info["num_line"],), np.int32
            ),
            "topo_vect": gym.spaces.Box(-1, 2, (env_info["dim_topo"],), np.int32),
        }
    )

    gym_proposed_actions = gym.spaces.Dict(
        {
            **{
                str(i): gym.spaces.Discrete(int(agent_per_substation[i]))
                for i in list(agent_per_substation.keys())
            },
            "-1": gym.spaces.Discrete(1),
        }
    )

    num_all_actions = (
        int(sum(agent_per_substation.values())) + 1
    )  # including do-nothing

    capa_gym_proposed_actions = gym.spaces.Dict(
        {
            **{
                str(i): gym.spaces.Discrete(num_all_actions)
                for i in list(agent_per_substation.keys())
            },
            "-1": gym.spaces.Discrete(1),
        }
    )

    gym_proposed_confidences = gym.spaces.Dict(
        {
            str(i): gym.spaces.Box(-np.inf, np.inf, tuple(), np.float32)
            for i in list(agent_per_substation.keys())
        }
    )

    return (
        gym_previous_obs,
        gym_proposed_actions,
        capa_gym_proposed_actions,
        gym_proposed_confidences,
    )


def select_mid_level_policy(
    middle_agent_type: str,
    agent_per_substation: dict[int, int],
    line_info: dict[int, list[int]],
    env_info: dict[str, int],
    custom_config: dict[str, Any],
) -> tuple[PolicySpec, dict[str, Any]]:
    """
    Specifies the policy for the middle level agent.

    Args:
        middle_agent_type (str): The type of middle level agent. Possible values are 'capa', 'random', 'argmax', 'rl_v',
            or 'rl'.
        agent_per_substation (dict[int, int]): A dictionary mapping substation IDs to the number of agents per substation.
        line_info (dict[int, list[int]]): A dictionary mapping line IDs to the list of connected substations.
        env_info (dict[str, int]): A dictionary containing environment information.
        custom_config (dict[str, Any]): A dictionary containing custom configuration parameters.

    Returns:
        PolicySpec: The policy specification for the middle level agent.

    Raises:
        ValueError: If the middle_agent_type is not recognized.

    """
    (
        gym_previous_obs,
        gym_proposed_actions,
        capa_gym_proposed_actions,
        gym_proposed_confidences,
    ) = setup_gym_spaces(agent_per_substation, env_info)

    if middle_agent_type in ("capa"):
        custom_config["environment"]["env_config"]["capa"] = True
        mid_level_observation = gym.spaces.Dict(
            {
                "previous_obs": gym_previous_obs,
                "reset_capa_idx": gym.spaces.Discrete(2),
                "proposed_actions": capa_gym_proposed_actions,
            }
        )
    elif middle_agent_type in ("random", "rl"):
        mid_level_observation = gym.spaces.Dict(
            {
                "previous_obs": gym_previous_obs,
                "proposed_actions": gym_proposed_actions,
            }
        )
    elif middle_agent_type in (
        "rl_v",
        "argmax",
    ):  # NOTE ORDER MIGHT BREAK IT WITH DICTIONARY
        mid_level_observation = gym.spaces.Dict(
            {
                "proposed_actions": gym_proposed_actions,
                "proposed_confidences": gym_proposed_confidences,
            }
        )

    act_space = gym.spaces.Discrete(len(list(agent_per_substation.keys())) + 1)

    if middle_agent_type == "capa":
        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=CapaPolicy,
            observation_space=mid_level_observation,  # information specifically for CAPA
            action_space=act_space,  # choose one of agents
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
    elif middle_agent_type == "random":
        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=RandomPolicy,
            observation_space=mid_level_observation,  # NOTE: Observation space is redundant but needed in custom_env
            action_space=act_space,  # choose one of agents
            config=(
                AlgorithmConfig()
                .training(_enable_learner_api=False)
                .rl_module(_enable_rl_module_api=False)
                .rollouts(preprocessor_pref=None)
            ),
        )

    elif middle_agent_type == "argmax":
        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=ArgMaxPolicy,
            observation_space=mid_level_observation,  # NOTE: Observation space is redundant but needed in custom_env
            action_space=act_space,  # choose one of agents
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
            observation_space=mid_level_observation,
            action_space=act_space,  # choose one of agents
            config=None,
        )
    elif middle_agent_type == "rl_v":
        env_config = custom_config["environment"]["env_config"]
        env_config["vf_rl"] = True
        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=None,  # use default policy of PPO
            observation_space=mid_level_observation,
            action_space=act_space,  # choose one of agents
            config={"env_config": env_config},
        )
    else:
        raise ValueError(
            f"Middle agent type {middle_agent_type} not recognized. Please use 'capa', 'random', 'argmax', 'rl_v' or 'rl'."
        )
    return mid_level_policy, custom_config


def select_low_level_policy(
    policies: dict[str, Any],
    lower_agent_type: str,
    agent_per_substation: dict[int, int],
) -> dict[str, Any]:
    """
    Selects and adds reinforcement learning policies to the given dictionary based on the lower agent type.

    Args:
        policies (dict[str, Any]): The dictionary to which the policies will be added.
        lower_agent_type (str): The type of the lower agent. Can be "rl" or "rl_v".
        agent_per_substation (dict[int, int]): A dictionary mapping substation indices to the number of actions.

    Returns:
        dict[str, Any]: The updated policies dictionary.
    """
    if lower_agent_type == "rl":  # add a rl agent for each substation
        # Add reinforcement learning policies to the dictionary
        for sub_idx, num_actions in agent_per_substation.items():
            policies[f"reinforcement_learning_policy_{sub_idx}"] = PolicySpec(
                policy_class=None,  # infer automatically from env (PPO)
                observation_space=None,  # infer automatically from env
                action_space=gym.spaces.Discrete(int(num_actions)),
                config=None,
            )
    elif (
        lower_agent_type == "rl_v"
    ):  # add a rl agent that outputs also the value function for each substation
        # Add reinforcement learning policies to the dictionary
        for sub_idx, num_actions in agent_per_substation.items():
            policies[f"value_reinforcement_learning_policy_{sub_idx}"] = PolicySpec(
                policy_class=None,
                observation_space=None,  # infer automatically from env
                action_space=gym.spaces.Dict(
                    {
                        "action": gym.spaces.Discrete(int(num_actions)),
                        "value": gym.spaces.Box(
                            float(-np.inf), float(np.inf), tuple(), np.float32
                        ),
                    }
                ),
                config=None,
            )
    return policies


def split_hub_into_agents(agent_per_substation: dict[int, int]) -> dict[int, int]:
    """
    Splits the hub into agents.

    Args:
        agent_per_substation (dict[int, int]): A dictionary mapping substation indices to the number of actions.

    Returns:
        dict[int, int]: The updated dictionary mapping substation indices to the number of actions.
    """
    extra_agents = 0
    # enumerate over dict to find the hub agent (over 1000 possible configurations)
    for sub_idx, num_actions in agent_per_substation.items():
        if num_actions > 1000:
            extra_agents += 1
        else:
            # replace sub_idx with sub_idx + extra_agents
            agent_per_substation[sub_idx + extra_agents] = num_actions
            # delete original sub_idx
            del agent_per_substation[sub_idx]
            agent_per_substation[sub_idx] = num_actions
    # TODO implement for 36-bus
    return agent_per_substation


def setup_config(
    config_path: str, middle_agent_type: str, lower_agent_type: str
) -> None:
    """
    Set up the configuration for training a hierarchical reinforcement learning (HRL) model.

    Args:
        config_path (str): The path to the configuration file.
        middle_agent_type (str): The type of middle-level agent to use.
        lower_agent_type (str): The type of lower-level agent to use.

    Returns:
        None
    """
    # load base PPO config and load in hyperparameters
    ppo_config = ppo.PPOConfig().to_dict()
    custom_config = load_config(config_path)

    # load in information about the environment
    setup_env = grid2op.make(custom_config["environment"]["env_config"]["env_name"])

    env_info = {
        "num_load": setup_env.n_load,
        "num_gen": setup_env.n_gen,
        "num_line": setup_env.n_line,
        "dim_topo": setup_env.dim_topo,
    }

    # Make as number additional policies as controllable substations
    agent_per_substation = find_list_of_agents(
        setup_env,
        custom_config["environment"]["env_config"]["action_space"],
    )

    agent_per_substation = split_hub_into_agents(agent_per_substation)

    list_of_substations = list(agent_per_substation.keys())

    line_info = find_substation_per_lines(setup_env, list_of_substations)

    # set-up the mid level policy
    mid_level_policy, custom_config = select_mid_level_policy(
        middle_agent_type, agent_per_substation, line_info, env_info, custom_config
    )

    # load whole config into ppoconfig
    for key in custom_config.keys():
        if key != "setup":
            ppo_config.update(custom_config[key])

    # define all level policies
    policies = {
        "high_level_policy": PolicySpec(  # chooses RL or do-nothing agent
            policy_class=SelectAgentPolicy,
            observation_space=gym.spaces.Box(-np.inf, np.inf),  # only the max rho
            action_space=gym.spaces.Discrete(2),  # choose one of agents
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
            observation_space=gym.spaces.Discrete(1),  # no observation space
            action_space=gym.spaces.Discrete(1),  # only perform do-nothing
            config=(
                AlgorithmConfig()
                .training(_enable_learner_api=False)
                .rl_module(_enable_rl_module_api=False)
            ),
        ),
    }

    policies = select_low_level_policy(policies, lower_agent_type, agent_per_substation)

    # if policy is rl, set an agent to train
    if middle_agent_type in ("rl", "rl_v"):
        ppo_config["policies_to_train"] = ["choose_substation_policy"]
    elif middle_agent_type in ("capa", "random", "argmax"):
        ppo_config["policies_to_train"] = []

    if lower_agent_type in ("rl", "rl_v"):
        if lower_agent_type == "rl":
            ppo_config["policies_to_train"] += [
                f"reinforcement_learning_policy_{sub_idx}"
                for sub_idx in list_of_substations
            ]
        elif lower_agent_type == "rl_v":
            ppo_config["policies_to_train"] += [
                f"value_reinforcement_learning_policy_{sub_idx}"
                for sub_idx in list_of_substations
            ]
        ppo_config.update({"env": HierarchicalCustomizedGrid2OpEnvironment})
    elif lower_agent_type == "greedy":
        ppo_config.update({"env": GreedyHierarchicalCustomizedGrid2OpEnvironment})

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
        default="rl",
        help="The type of middle level agent (capa or rl).",
    )

    parser.add_argument(
        "-l",
        "--lower",
        type=str,
        default="rl",
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

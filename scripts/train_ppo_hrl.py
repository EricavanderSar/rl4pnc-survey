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
    ActionFunctionTorchPolicy,
    ArgMaxPolicy,
    CapaPolicy,
    CustomFCN,
    DoNothingPolicy,
    OnlyValueFunctionTorchPolicy,
    RandomPolicy,
    SampleValuePolicy,
    SelectAgentPolicy,
)


def setup_gym_spaces(
    agent_per_substation: dict[str, int], env_info: dict[str, int]
) -> tuple[gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict]:
    """
    Set up the gym spaces for the RL environment.

    Args:
        agent_per_substation (dict[str, int]): A dictionary mapping substation IDs to the number of agents per substation.
        env_info (dict[str, int]): A dictionary containing information about the environment.

    Returns:
        tuple[gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict, gym.spaces.Dict]: A tuple containing the gym spaces for
            previous observations, proposed actions, proposed action capacities, and proposed confidences.
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
            # "-1": gym.spaces.Discrete(1),
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
            # "-1": gym.spaces.Discrete(1),
        }
    )

    gym_proposed_confidences = gym.spaces.Dict(
        {
            **{
                str(i): gym.spaces.Box(-np.inf, np.inf, tuple(), np.float32)
                for i in list(agent_per_substation.keys())
            },
            # "-1": gym.spaces.Box(-np.inf, np.inf, tuple(), np.float32),
        }
    )

    return (
        gym_previous_obs,
        gym_proposed_actions,
        capa_gym_proposed_actions,
        gym_proposed_confidences,
    )


def get_mid_level_observation(
    middle_agent_type: str,
    gym_previous_obs: gym.spaces.Dict,
    gym_proposed_actions: gym.spaces.Dict,
    capa_gym_proposed_actions: gym.spaces.Dict,
    gym_proposed_confidences: gym.spaces.Dict,
) -> gym.spaces.Dict:
    """
    Constructs the mid-level observation based on the middle agent type.

    Args:
        middle_agent_type (str): The type of middle agent.
        gym_previous_obs (gym.spaces.Dict): The previous observation from the environment.
        gym_proposed_actions (gym.spaces.Dict): The proposed actions from the environment.
        capa_gym_proposed_actions (gym.spaces.Dict): The proposed actions for CAPA agent.
        gym_proposed_confidences (gym.spaces.Dict): The proposed confidences from the environment.

    Returns:
        gym.spaces.Dict: The mid-level observation.

    Raises:
        None
    """
    if middle_agent_type in ("capa"):
        mid_level_observation = gym.spaces.Dict(
            {
                "previous_obs": gym_previous_obs,
                "reset_capa_idx": gym.spaces.Discrete(2),
                "proposed_actions": capa_gym_proposed_actions,
            }
        )
    elif middle_agent_type in ("rl"):
        mid_level_observation = gym.spaces.Dict(
            {
                "previous_obs": gym_previous_obs,
                "proposed_actions": gym_proposed_actions,
            }
        )
    elif middle_agent_type in ("random"):
        mid_level_observation = gym.spaces.Dict(
            {
                "proposed_actions": gym_proposed_actions,
            }
        )
    elif middle_agent_type in (
        "argmax",
        "sample",
    ):
        mid_level_observation = gym.spaces.Dict(
            {
                "proposed_actions": gym_proposed_actions,
                "proposed_confidences": gym_proposed_confidences,
            }
        )
    elif middle_agent_type in ("rlv",):
        mid_level_observation = gym.spaces.Dict(
            {
                "proposed_actions": gym_proposed_actions,
                "proposed_confidences": gym_proposed_confidences,
                "previous_obs": gym_previous_obs,
            }
        )
    return mid_level_observation


def select_mid_level_policy(
    middle_agent_type: str,
    agent_per_substation: dict[str, int],
    line_info: dict[str, list[int]],
    env_info: dict[str, int],
    custom_config: dict[str, Any],
) -> tuple[PolicySpec, dict[str, Any]]:  # noqa: C901
    """
    Specifies the policy for the middle level agent.

    Args:
        middle_agent_type (str): The type of middle level agent. Possible values are 'capa', 'random', 'argmax', 'rlv',
            'sample', or 'rl'.
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

    mid_level_observation = get_mid_level_observation(
        middle_agent_type,
        gym_previous_obs,
        gym_proposed_actions,
        capa_gym_proposed_actions,
        gym_proposed_confidences,
    )

    act_space_rl = gym.spaces.Discrete(len(list(agent_per_substation.keys())))

    # get the highest key as integer and add 1 to get the number of agents
    act_space_rulebased = gym.spaces.Discrete(
        max(int(k) for k in agent_per_substation.keys()) + 1
    )

    if middle_agent_type == "capa":
        custom_config["environment"]["env_config"]["capa"] = True
        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=CapaPolicy,
            observation_space=mid_level_observation,  # information specifically for CAPA
            action_space=act_space_rulebased,  # choose one of agents
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
        custom_config["environment"]["env_config"]["rulebased"] = True
        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=RandomPolicy,
            observation_space=mid_level_observation,  # NOTE: Observation space is redundant but needed in custom_env
            action_space=act_space_rulebased,  # choose one of agents
            config=(
                AlgorithmConfig()
                .training(_enable_learner_api=False)
                .rl_module(_enable_rl_module_api=False)
                .rollouts(preprocessor_pref=None)
            ),
        )

    elif middle_agent_type == "argmax":
        custom_config["environment"]["env_config"]["rulebased"] = True
        custom_config["environment"]["env_config"]["vf_rl"] = True
        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=ArgMaxPolicy,
            observation_space=mid_level_observation,
            action_space=act_space_rulebased,  # choose one of agents
            config=(
                AlgorithmConfig()
                .training(_enable_learner_api=False)
                .rl_module(_enable_rl_module_api=False)
                .rollouts(preprocessor_pref=None)
            ),
        )
    elif middle_agent_type == "sample":
        custom_config["environment"]["env_config"]["rulebased"] = True
        custom_config["environment"]["env_config"]["vf_rl"] = True
        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=SampleValuePolicy,
            observation_space=mid_level_observation,
            action_space=act_space_rulebased,  # choose one of agents
            config=(
                AlgorithmConfig()
                .training(_enable_learner_api=False)
                .rl_module(_enable_rl_module_api=False)
                .rollouts(preprocessor_pref=None)
            ),
        )
    elif middle_agent_type == "rl":
        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=None,  # use default policy of PPO
            observation_space=mid_level_observation,
            action_space=act_space_rl,  # choose one of agents
            config=None,
        )
    elif middle_agent_type == "rlv":
        custom_config["environment"]["env_config"]["vf_rl"] = True
        mid_level_policy = PolicySpec(  # rule based substation selection
            policy_class=None,  # use default policy of PPO
            observation_space=mid_level_observation,
            action_space=act_space_rl,  # choose one of agents
            config=None,
        )
    else:
        raise ValueError(
            f"Middle agent type {middle_agent_type} not recognized. \
            Please use 'capa', 'random', 'argmax', 'sample', 'rlv' or 'rl'."
        )
    return mid_level_policy, custom_config


def select_low_level_policy(
    policies: dict[str, Any],
    lower_agent_type: str,
    agent_per_substation: dict[str, int],
    ppo_config: dict[str, Any],
    env_info: dict[str, int],
) -> dict[str, Any]:
    """
    Selects and adds reinforcement learning policies to the given dictionary based on the lower agent type.

    Args:
        policies (dict[str, Any]): The dictionary to which the policies will be added.
        lower_agent_type (str): The type of the lower agent. Can be "rl" or "rlv".
        agent_per_substation (dict[int, int]): A dictionary mapping substation indices to the number of actions.
        ppo_config (dict[str, Any]): The PPO configuration.
        env_info (dict[str, int]): A dictionary containing information about the environment.

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
        lower_agent_type == "rlv"
    ):  # add a rl agent that outputs also the value function for each substation
        # Add reinforcement learning policies to the dictionary
        num_objects = (
            env_info["num_gen"]
            + env_info["num_load"]
            + 4 * env_info["num_line"]
            + env_info["dim_topo"]
        )
        for sub_idx, num_actions in agent_per_substation.items():
            shared_model = CustomFCN(
                obs_space=gym.spaces.Box(-1.0, 1.0, (num_objects,), np.float32),
                action_space=gym.spaces.Discrete(int(num_actions)),
                num_outputs=int(num_actions),
                model_config=ppo_config["model"],
                name=f"shared_model_{sub_idx}",
            )

            policies[f"value_reinforcement_learning_policy_{sub_idx}"] = PolicySpec(
                policy_class=ActionFunctionTorchPolicy,
                observation_space=None,  # infer automatically from env
                action_space=gym.spaces.Discrete(int(num_actions)),
                config={
                    "model": {
                        "custom_model_config": {"model": shared_model},
                    },
                },
            )

            policies[f"value_function_policy_{sub_idx}"] = PolicySpec(
                policy_class=OnlyValueFunctionTorchPolicy,
                observation_space=None,  # infer automatically from env
                action_space=gym.spaces.Box(-np.inf, np.inf, tuple(), np.float32),
                config={
                    "model": {
                        "custom_model_config": {"model": shared_model},
                    },
                },
            )

    return policies


def create_agents(agent_per_substation: dict[str, int]) -> dict[str, int]:
    """
    Create agents for all substations.

    Args:
        agent_per_substation (dict[int, int]): A dictionary mapping substation indices to the number of actions.

    Returns:
        dict[int, int]: The updated dictionary mapping substation indices to the number of actions.
    """
    new_agent_per_substation = {}

    for sub_idx, num_actions in agent_per_substation.items():
        new_agent_per_substation[str(sub_idx)] = int(num_actions)

    return new_agent_per_substation


def setup_policies(
    mid_level_policy: PolicySpec, custom_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Set up the policies for the hierarchical reinforcement learning (HRL) training.

    Args:
        mid_level_policy (PolicySpec): The policy specification for the mid-level policy.
        custom_config (dict[str, Any]): Custom configuration for the policies.

    Returns:
        dict[str, Any]: A dictionary containing the policy specifications for the high-level policy,
        choose_substation_policy, and do_nothing_policy.

    """
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
    return policies


def add_trainable_policies(
    middle_agent_type: str,
    lower_agent_type: str,
    list_of_substations: list[str],
    ppo_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Adds trainable policies to the PPO configuration based on the specified agent types.

    Args:
        middle_agent_type (str): The type of the middle agent.
        lower_agent_type (str): The type of the lower agent.
        list_of_substations (list[str]): A list of substations.
        ppo_config (dict[str, Any]): The PPO configuration.

    Returns:
        dict[str, Any]: The updated PPO configuration.
    """
    # if policy is rl, set an agent to train
    if middle_agent_type in ("rl", "rlv"):
        ppo_config["policies_to_train"] = ["choose_substation_policy"]
    elif middle_agent_type in ("capa", "random", "argmax", "sample"):
        ppo_config["policies_to_train"] = []

    if lower_agent_type in ("rl", "rlv"):
        if lower_agent_type == "rl":
            ppo_config["policies_to_train"] += [
                f"reinforcement_learning_policy_{sub_idx}"
                for sub_idx in list_of_substations
            ]
        elif lower_agent_type == "rlv":
            ppo_config["policies_to_train"] += [
                f"value_reinforcement_learning_policy_{sub_idx}"
                for sub_idx in list_of_substations
            ]
        ppo_config.update({"env": HierarchicalCustomizedGrid2OpEnvironment})
    elif lower_agent_type == "greedy":
        ppo_config.update({"env": GreedyHierarchicalCustomizedGrid2OpEnvironment})
    return ppo_config


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

    # NOTE: Manual for new action space, can still be automated
    if custom_config["environment"]["env_config"]["env_name"].startswith(
        "l2rpn_icaps_2021_large"
    ):
        agent_per_substation = {
            "29": 26,
            "9": 26,
            "4": 25,
            "28": 7,
            "26": 183,
            "33": 42,
            "21": 93,
            "23": 321,
            "16": 585,
            "7": 14,
        }
    else:
        # Make as number additional policies as controllable substations
        agent_per_substation = find_list_of_agents(
            setup_env,
            custom_config["environment"]["env_config"]["action_space"],
            add_dn_agents=False,
            add_dn_action_per_agent=True,
        )
        agent_per_substation = create_agents(agent_per_substation)

    original_list_of_substations = []
    for sub_id in agent_per_substation:
        if int(sub_id.split("_")[0]) not in original_list_of_substations:
            original_list_of_substations.append(sub_id.split("_")[0])

    line_info = find_substation_per_lines(setup_env, original_list_of_substations)

    # set-up the mid level policy
    mid_level_policy, custom_config = select_mid_level_policy(
        middle_agent_type, agent_per_substation, line_info, env_info, custom_config
    )

    # load whole config into ppoconfig
    for key in custom_config.keys():
        if key != "setup":
            ppo_config.update(custom_config[key])

    policies = setup_policies(mid_level_policy, custom_config)

    policies = select_low_level_policy(
        policies, lower_agent_type, agent_per_substation, ppo_config, env_info
    )

    ppo_config = add_trainable_policies(
        middle_agent_type,
        lower_agent_type,
        list(agent_per_substation.keys()),
        ppo_config,
    )

    # load environment and agents manually
    ppo_config.update({"policies": policies})

    # add tags of agent types to config
    ppo_config.update({"middle_agent_type": middle_agent_type})
    ppo_config.update({"lower_agent_type": lower_agent_type})

    specified_agent = f"{lower_agent_type}_{middle_agent_type}"

    if specified_agent.lower() not in custom_config["setup"]["experiment_name"].lower():
        raise ValueError("Probably using the wrong settings, check again.")

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
        help="Specify the type of coordinator you want to use (rl, rlv, capa, random, sample, argmax). Default: rl",
    )

    parser.add_argument(
        "-l",
        "--lower",
        type=str,
        default="rl",
        help="Specify the type of regional agents you want to use (rl, rlv, greedy). Default: rl",
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

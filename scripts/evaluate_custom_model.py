# TODO: Checki f reconnection goes well in evaluation because it's  only on the grid2op side.

"""
Class to evaluate custom RL models.
"""
import argparse
import importlib
import logging
import os
import re
import time
from statistics import mean
from typing import Any

import grid2op
from grid2op.Agent import BaseAgent, DoNothingAgent
from grid2op.Environment import BaseEnv
from grid2op.Reward import BaseReward
from grid2op.Runner import Runner
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms import Algorithm

from mahrl.evaluation.evaluation_agents import (
    CapaAndGreedyAgent,
    RllibAgent,
    TopologyGreedyAgent,
)
from mahrl.experiments.yaml import load_config
from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from mahrl.grid2op_env.utils import load_actions


def get_algorithm(alg_name) -> Algorithm:
    AGENT = {
        "ppo": ppo.PPO,
    }
    return AGENT[alg_name]

def setup_parser(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Set up the command-line argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser object.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser.add_argument("-a", "--agent_type", default="rl", choices=["nothing", "greedy", "rl"])
    parser.add_argument(
        "-n",
        "--nothing",
        action="store_true",
        help="Signals to evaluate a DoNothing agent.",
    )
    parser.add_argument(
        "-g",
        "--greedy",
        action="store_true",
        help="Signals to evaluate a Greedy agent.",
    )
    parser.add_argument(
        "-r",
        "--rule_based_hierarchical",
        action="store_true",
        help="Signals to evaluate a Capa and Greedy agent.",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Name of the config file, for Greedy and DoNothing agents.",
    )

    parser.add_argument(
        "-f",
        "--file_path",
        default="/Users/ericavandersar/Documents/Python_Projects/Research/mahrl_grid2op/scripts/runs/"
                "CustomPPO_2024-03-04_10-44-37/"
                "CustomPPO_CustomizedGrid2OpEnvironment_d0927_00000_0_2024-03-04_10-44-37",
        type=str,
        help="Path to the input file, only for the Rllib agent.",
    )
    parser.add_argument(
        "-p",
        "--checkpoint_name",
        type=str,
        help="Name of the checkpoint, only for the Rllib agent.",
    )

    return parser.parse_args()


def instantiate_reward_class(class_name: str) -> Any:
    """
    Instantiates the Reward class from json string.
    """
    # Split the class name into module and class
    class_name = class_name.replace("<", "")
    module_name, class_name = class_name.rsplit(".", 1)
    class_name = class_name.split(" ", 1)[0]
    # Import the module dynamically
    module = importlib.import_module(module_name)
    # Get the class from the module
    reward_class: BaseReward = getattr(module, class_name)
    # Instantiate the class
    if reward_class:
        return reward_class()
    raise ValueError("Problem instantiating reward class for evaluation.")


def instantiate_opponent_classes(class_name: str) -> Any:
    """
    Instantiates opponent classes from json string.
    """
    # extract the module and class names
    match = re.match(r"<class '(.*)\.(.*)'>", class_name)
    if match:
        module_name = match.group(1)
        class_name = match.group(2)

        # load the module
        module = importlib.import_module(module_name)

        # get the class from the module
        return getattr(module, class_name)
    raise ValueError("Problem instantiating opponent class for evaluation.")


def run_runner(env_config: dict[str, Any], agent_instance: BaseAgent, alg_name: str) -> None:
    """
    Perform runner on the implemented networks.
    """
    results_folder = f"{env_config['lib_dir']}/results/{env_config['env_name']}"
    # check if the folder exists
    if not os.path.exists(results_folder):
        # if not, create the folder
        os.makedirs(results_folder)

    env = grid2op.make(env_config["env_name"], **env_config["grid2op_kwargs"])

    params = env.get_params_for_runner()
    params["rewardClass"] = env_config["grid2op_kwargs"]["reward_class"]
    del env_config["grid2op_kwargs"]["reward_class"]

    if "kwargs_opponent" in env_config["grid2op_kwargs"]:
        env_config["grid2op_kwargs"]["opponent_kwargs"] = env_config["grid2op_kwargs"][
            "kwargs_opponent"
        ]
        del env_config["grid2op_kwargs"]["kwargs_opponent"]
        # only needed with opponent
        params.update(env_config["grid2op_kwargs"])

    store_trajectories_folder = os.path.join(
        env_config["lib_dir"], "runs/action_evaluation"
    )

    # check if the folder exists
    if not os.path.exists(store_trajectories_folder):
        # if not, create the folder
        os.makedirs(store_trajectories_folder)

    start_time = time.time()

    # run the environment 10 times if an opponent is active, with different seeds
    for i in range(10 if "opponent_kwargs" in env_config["grid2op_kwargs"] else 1):
        env_config["seed"] = env_config["seed"] + i

        # define the results folder path
        file_name = (
            f"{alg_name}"
            + f"_{'opponent' if 'opponent_kwargs' in env_config['grid2op_kwargs'] else 'no_opponent'}"
            + f"_{env_config['action_space']}_{env_config['rho_threshold']}_{i}_{time.strftime('%d%m%Y_%H%M%S')}.txt"
        )

        with open(
            f"{results_folder}/{file_name}",
            "w",
            encoding="utf-8",
        ) as file:
            file.write(f"Threshold={env_config['rho_threshold']}\n")

        print("params: ", params)
        print(f"storing results in: {store_trajectories_folder}/{env_config['env_name']}")
        print("nb_episodes: ", env.chronics_handler.subpaths)
        res = Runner(
            **params,
            agentClass=None,
            agentInstance=agent_instance,
        ).run(
            path_save=os.path.abspath(
                f"{store_trajectories_folder}/{env_config['env_name']}"
            ),
            nb_episode=len(env.chronics_handler.subpaths),
            max_iter=-1,
            nb_process=1,
        )

        individual_timesteps = []

        logging.info(f"The results for {agent_instance} agent are:")
        for _, chron_name, _, nb_time_step, max_ts in res:
            with open(
                f"{results_folder}/{file_name}",
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    f"\n\tFor chronics with id {chron_name}\n"
                    + f"\t\t - number of time steps completed: {nb_time_step:.0f} / {max_ts:.0f}"
                )

            individual_timesteps.append(nb_time_step)
            logging.info(
                f"\n\tFor chronics with id {chron_name}\n"
                + f"\t\t - number of time steps completed: {nb_time_step:.0f} / {max_ts:.0f}"
            )

        with open(
            f"{results_folder}/{file_name}",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                f"\nAverage timesteps survived: {mean(individual_timesteps)}\n{individual_timesteps}\n"
                + f"Total time = {time.time() - start_time}"
            )
        logging.info(f"Total time = {time.time() - start_time}")
        logging.info(f"Average timesteps survived: {mean(individual_timesteps)}")


def setup_greedy_evaluation(env_config: dict[str, Any], setup_env: BaseEnv) -> None:
    """
    Set up the evaluation of a greedy agent on a given environment configuration.

    Args:
        env_config (dict): The configuration of the environment.
        setup_env (object): The setup environment object.

    Returns:
        None
    """
    actions_path = os.path.abspath(
        f"{env_config['lib_dir']}/data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
    )

    possible_actions = load_actions(actions_path, setup_env)

    run_runner(
        env_config=env_config,
        agent_instance=TopologyGreedyAgent(
            action_space=setup_env.action_space,
            env_config=env_config,
            possible_actions=possible_actions,
        ),
        alg_name="greedy"
    )


def setup_do_nothing_evaluation(env_config: dict[str, Any], setup_env: BaseEnv) -> None:
    """
    Sets up and runs an evaluation using the DoNothingAgent.

    Args:
        env_config (dict): Configuration for the environment.
        setup_env (function): Function to set up the environment.

    Returns:
        None
    """
    run_runner(
        env_config=env_config,
        agent_instance=DoNothingAgent(setup_env.action_space),
        alg_name="do_nothing"
    )


def setup_rllib_evaluation(file_path: str, checkpoint_name: str) -> None:
    """
    Set up the evaluation of a custom RLlib model.

    Args:
        file_path (str): The file path of the model.
        checkpoint_name (str): The name of the checkpoint to load.

    Returns:
        None
    """
    # load config
    config_path = os.path.join(args.file_path, "params.json")
    config = load_config(config_path)
    env_config = config["env_config"]
    # change the env_name from _train to _test
    env_config["env_name"] = env_config["env_name"].replace("_train", "_test")

    env_config["grid2op_kwargs"]["reward_class"] = instantiate_reward_class(
        env_config["grid2op_kwargs"]["reward_class"]
    )

    # check if "opponent_action_class" is part of env_config["grid2op_kwargs"]
    if "opponent_action_class" in env_config["grid2op_kwargs"]:
        env_config["grid2op_kwargs"][
            "opponent_action_class"
        ] = instantiate_opponent_classes(
            env_config["grid2op_kwargs"]["opponent_action_class"]
        )
        env_config["grid2op_kwargs"][
            "opponent_budget_class"
        ] = instantiate_opponent_classes(
            env_config["grid2op_kwargs"]["opponent_budget_class"]
        )
        env_config["grid2op_kwargs"]["opponent_class"] = instantiate_opponent_classes(
            env_config["grid2op_kwargs"]["opponent_class"]
        )
    # TODO EVDS: make alg_type input param in future?
    alg_type='ppo'


    run_runner(
        env_config=env_config,
        agent_instance=RllibAgent(
            action_space=None,
            env_config=env_config,
            file_path=file_path,
            policy_name="reinforcement_learning_policy",
            algorithm=get_algorithm(alg_type),
            checkpoint_name=checkpoint_name,
            gym_wrapper=CustomizedGrid2OpEnvironment(env_config),
        ),
        alg_name=alg_type,
    )


def setup_capa_greedy_evaluation(
    env_config: dict[str, Any], setup_env: BaseEnv
) -> None:
    """
    Set up the evaluation of a greedy agent on a given environment configuration.

    Args:
        env_config (dict): The configuration of the environment.
        setup_env (object): The setup environment object.

    Returns:
        None
    """
    actions_path = os.path.abspath(
        f"{env_config['lib_dir']}/data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
    )

    possible_actions = load_actions(actions_path, setup_env)

    run_runner(
        env_config=env_config,
        agent_instance=CapaAndGreedyAgent(
            action_space=setup_env.action_space,
            env_config=env_config,
            possible_actions=possible_actions,
        ),
        alg_name="MidCapa_LowGreedy"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_parser = argparse.ArgumentParser(description="Process possible variables.")
    args = setup_parser(init_parser)

    # Check which arguments are provided
    if (
        args.greedy or args.nothing or args.rule_based_hierarchical
    ):  # run donothing or greedy evaluation
        if not args.config:
            init_parser.print_help()
            logging.error("\nError: --config is required for the agent.")
        else:
            # load config file
            environment_config = load_config(args.config)["environment"]["env_config"]
            # change the env_name from _train to _test
            environment_config["env_name"] = environment_config["env_name"].replace(
                "_train", "_test"
            )
            init_setup_env = grid2op.make(environment_config["env_name"])

            # start runners
            if args.greedy:
                setup_greedy_evaluation(environment_config, init_setup_env)
            elif args.rule_based_hierarchical:
                setup_capa_greedy_evaluation(environment_config, init_setup_env)
            else:
                setup_do_nothing_evaluation(environment_config, init_setup_env)
    else:
        if not args.file_path:
            init_parser.print_help()
            logging.error("\nError: --file_path is required.")
        else:
            if not args.checkpoint_name:
                logging.warning(
                    "\nWarning: --checkpoint_name not specified. Using checkpoint_000000."
                )
                CHECKPOINT_NAME = "checkpoint_000000"
                setup_rllib_evaluation(
                    file_path=args.file_path,
                    checkpoint_name=CHECKPOINT_NAME,
                )
            else:
                setup_rllib_evaluation(
                    file_path=args.file_path,
                    checkpoint_name=args.checkpoint_name,
                )

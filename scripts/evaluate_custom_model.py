"""
Class to evaluate custom RL models.
"""
import argparse
import importlib
import json
import logging
import os
import time
from statistics import mean
from typing import Any

import grid2op
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent, DoNothingAgent
from grid2op.Environment import BaseEnv
from grid2op.Reward import BaseReward
from grid2op.Runner import Runner
from ray.rllib.algorithms import ppo

from mahrl.evaluation.evaluation_agents import RllibAgent, TopologyGreedyAgent
from mahrl.experiments.yaml import load_config


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


def load_actions(path: str, env: BaseEnv) -> list[BaseAction]:
    """
    Loads the .json with specified topology actions.
    """
    with open(path, "rt", encoding="utf-8") as action_set_file:
        return list(
            (
                env.action_space(action_dict)
                for action_dict in json.load(action_set_file)
            )
        )


def run_runner(env_config: dict[str, Any], agent_instance: BaseAgent) -> list[int]:
    """
    Perform runner on the implemented networks.
    """
    if env_config["env_name"] == "rte_case5_example":
        env = grid2op.make(env_config["env_name"], test=True)
        nb_episode = 20
    elif env_config["env_name"] == "rte_case14_realistic":
        env = grid2op.make(env_config["env_name"])
        nb_episode = 1000
    else:
        raise NotImplementedError("This network was not implemented for evaluation.")

    params = env.get_params_for_runner()

    runner = Runner(
        **params,
        agentClass=None,
        agentInstance=agent_instance,
    )
    res = runner.run(nb_episode=nb_episode, max_iter=-1, nb_process=1)

    individual_timesteps = []

    logging.info(f"The results for {agent_instance} agent are:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = f"\n\tFor chronics with id {chron_name}\n"
        msg_tmp += f"\t\t - cumulative reward: {cum_reward:.6f}\n"
        msg_tmp += (
            f"\t\t - number of time steps completed: {nb_time_step:.0f} / {max_ts:.0f}"
        )
        with open(
            f"{env_config['env_name']}_{env_config['action_space']}_{env_config['rho_threshold']}.txt",
            "a",
            encoding="utf-8",
        ) as file:
            file.write(msg_tmp)

        individual_timesteps.append(nb_time_step)
        logging.info(msg_tmp)

    with open(
        f"{env_config['env_name']}_{env_config['action_space']}_{env_config['rho_threshold']}.txt",
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            f"Average timesteps survived: {mean(individual_timesteps)}\n{individual_timesteps}"
        )

    logging.info(f"Average timesteps survived: {mean(individual_timesteps)}")
    return individual_timesteps


def run_evaluation_rllib(file_path: str, checkpoint_name: str) -> None:
    """
    Loads config file and calls runner.
    """

    # load config
    config_path = os.path.join(file_path, "params.json")
    config = load_config(config_path)
    env_config = config["env_config"]

    # get reward class frmo object
    reward_object = instantiate_reward_class(
        env_config["grid2op_kwargs"]["reward_class"]
    )
    env_config["grid2op_kwargs"]["reward_class"] = reward_object

    # print and save results
    with open(
        f"{env_config['env_name']}_{env_config['action_space']}_{env_config['rho_threshold']}.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Threshold={env_config['rho_threshold']}\n")

    # start runner
    _ = run_runner(
        env_config,
        RllibAgent(
            action_space=None,
            env_config=env_config,
            file_path=file_path,
            policy_name="reinforcement_learning_policy",
            algorithm=ppo.PPO,
            checkpoint_name=checkpoint_name,
        ),
    )


def run_evaluation_greedy(actions_path: str, threshold: float) -> None:
    """
    Call runner for greedy agent.
    """
    # Get the environment and the action name from the path
    parts_of_action_path = actions_path.split("/")

    # setup env config
    env_config = {
        "env_name": parts_of_action_path[-2],
        "rho_threshold": threshold,
        "action_space": parts_of_action_path[-1].split(".json")[0],
    }

    setup_env = grid2op.make(env_config["env_name"], test=True)

    possible_actions = load_actions(actions_path, setup_env)

    # print and save results
    with open(
        f"{env_config['env_name']}_{env_config['action_space']}_{env_config['rho_threshold']}.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Threshold={env_config['rho_threshold']}\n")

    # start runner
    _ = run_runner(
        env_config,
        TopologyGreedyAgent(
            action_space=setup_env.action_space,
            env_config=env_config,
            possible_actions=possible_actions,
        ),
    )


def run_evaluation_nothing(environment_name: str) -> None:
    """
    Call runner for DoNothing agent.
    """
    # setup env config
    env_config = {
        "env_name": environment_name,
        "rho_threshold": None,
        "action_space": None,
    }
    setup_env = grid2op.make(env_config["env_name"], test=True)

    # print and save results
    with open(
        f"{env_config['env_name']}_{env_config['action_space']}_{env_config['rho_threshold']}.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("Agent=DoNothing\n")

    # start runner
    _ = run_runner(
        env_config,
        DoNothingAgent(setup_env.action_space),
    )


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Process possible variables.")

    # Define command-line arguments for two possibilities: greedy and rllib model
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        help="Path to the input file, only for the Rllib agent.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_name",
        type=str,
        help="Name of the checkpoint, only for the Rllib agent.",
    )
    parser.add_argument(
        "-g",
        "--greedy",
        action="store_true",
        help="Signals to evaluate a Greedy agent.",
    )
    parser.add_argument(
        "-n",
        "--nothing",
        action="store_true",
        help="Signals to evaluate a DoNothing agent.",
    )
    parser.add_argument(
        "-a",
        "--actions",
        type=str,
        help="Path to the action space file, only for the Greedy agent.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        help="Specify the threshold, only for the Greedy agent.",
    )
    parser.add_argument(
        "-e",
        "--environment",
        type=str,
        help="Specify the environment, only for the DoNothing agent.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    input_file_path = args.file_path
    input_checkpoint_name = args.checkpoint_name
    input_greedy = args.greedy
    input_nothing = args.nothing
    input_actions = args.actions
    input_threshold = args.threshold
    input_environment = args.environment

    # Check which arguments are provided
    if input_greedy:  # run greedy evaluation
        if not input_actions:
            parser.print_help()
            logging.error("\nError: --actions is required for the greedy agent.")
        else:
            if not input_threshold:
                logging.warning("\nWarning: --threshold not specified. Using 0.95.")
                INPUT_THRESHOLD = 0.95
                run_evaluation_greedy(input_actions, INPUT_THRESHOLD)
            else:
                run_evaluation_greedy(input_actions, input_threshold)
    elif input_nothing:
        if not input_environment:
            parser.print_help()
            logging.error("\nError: --environment is required for the greedy agent.")
        else:
            run_evaluation_nothing(input_environment)
    else:  # run Rllib evaluations
        if not input_file_path:
            parser.print_help()
            logging.error("\nError: --file_path is required.")
        else:
            if not input_checkpoint_name:
                logging.warning(
                    "\nWarning: --checkpoint_name not specified. Using checkpoint_000000."
                )
                CHECKPOINT_NAME = "checkpoint_000000"
                run_evaluation_rllib(input_file_path, CHECKPOINT_NAME)
            else:
                run_evaluation_rllib(input_file_path, input_checkpoint_name)

    logging.info(f"Total time = {time.time() - start_time}")

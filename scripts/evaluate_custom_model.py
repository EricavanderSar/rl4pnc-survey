"""
Class to evaluate custom RL models.
"""
import argparse
import importlib
import logging
import os
import time
from collections import OrderedDict
from statistics import mean
from typing import Any

import grid2op
import numpy as np
from grid2op.Action import ActionSpace, BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation
from grid2op.Reward import BaseReward
from grid2op.Runner import Runner
from ray.rllib.algorithms import Algorithm, ppo

from mahrl.experiments.yaml import load_config
from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment


class RLlib2Grid2Op(BaseAgent):
    """
    Class that runs a RLlib model in the Grid2Op environment.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        env_config: dict[str, Any],
        file_path: str,
        policy_name: str,
        algorithm: Algorithm,
        checkpoint_name: str,
    ):
        BaseAgent.__init__(self, action_space)

        # load PPO
        checkpoint_path = os.path.join(file_path, checkpoint_name)
        self._rllib_agent = algorithm.from_checkpoint(
            checkpoint_path, policy_ids=[policy_name]
        )

        # setup env
        self.gym_wrapper = CustomizedGrid2OpEnvironment(env_config)

        # setup threshold
        self.threshold = env_config["rho_threshold"]

    def act(
        self, observation: BaseObservation, reward: BaseReward, done: bool = False
    ) -> BaseAction:
        """
        Returns a grid2op action based on a RLlib observation.
        """
        # Grid2Op to RLlib observation
        gym_obs = self.gym_wrapper.env_gym.observation_space.to_gym(observation)
        gym_obs = OrderedDict(
            (k, gym_obs[k]) for k in self.gym_wrapper.observation_space.spaces
        )

        if np.max(gym_obs["rho"]) > self.threshold:
            # get action as int
            gym_act = self._rllib_agent.compute_single_action(
                gym_obs, policy_id="reinforcement_learning_policy"
            )
        else:
            gym_act = 0

        # convert Grid2Op action to RLlib
        grid2op_act = self.gym_wrapper.env_gym.action_space.from_gym(gym_act)
        return grid2op_act


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

    runner = Runner(
        **env.get_params_for_runner(), agentClass=None, agentInstance=agent_instance
    )
    res = runner.run(nb_episode=nb_episode, max_iter=-1)

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

    logging.info(f"Average timesteps survived: {mean(individual_timesteps)}")
    return individual_timesteps


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


def run_evaluation(file_path: str, checkpoint_name: str) -> None:
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
    start_time = time.time()
    _ = run_runner(
        env_config,
        RLlib2Grid2Op(
            action_space=None,
            env_config=env_config,
            file_path=file_path,
            policy_name="reinforcement_learning_policy",
            algorithm=ppo.PPO,
            checkpoint_name=checkpoint_name,
        ),
    )
    logging.info(f"done 5bus --- %s seconds --- {time.time() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process input file and checkpoint name."
    )

    # Define command-line arguments
    parser.add_argument("--file_path", type=str, help="Path to the input file")
    parser.add_argument("--checkpoint_name", type=str, help="Name of the checkpoint")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    input_file_path = args.file_path
    input_checkpoint_name = args.checkpoint_name

    # Check if both arguments are provided
    if not input_file_path:
        parser.print_help()
        logging.info("\nError: --file_path is required.")
    else:
        if not input_checkpoint_name:
            logging.info(
                "\nWarning: --checkpoint_name not specified. Using checkpoint_000000."
            )
            CHECKPOINT_NAME = "checkpoint_000000"
            run_evaluation(input_file_path, CHECKPOINT_NAME)
        else:
            run_evaluation(input_file_path, input_checkpoint_name)

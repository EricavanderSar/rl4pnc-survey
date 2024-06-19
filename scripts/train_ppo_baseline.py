"""
Trains PPO baseline agent.
"""

import argparse
import logging
import os
from typing import Any

import grid2op

import ray
import gymnasium as gym
from ray import air, tune, train
# from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms import ppo  # import the type of agents
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog

from mahrl.experiments.yaml import load_config
from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from mahrl.grid2op_env.custom_env2 import RlGrid2OpEnv
from mahrl.multi_agent.policy import (
    DoNothingPolicy,
    SelectAgentPolicy,
    DoNothingPolicy2,
    SelectAgentPolicy2
)
from mahrl.experiments.utils import run_training
from mahrl.models.linear_model import LinFCN

REPORT_END = False

ENV_TYPE = {
    "old_env": CustomizedGrid2OpEnvironment,
    "new_env": RlGrid2OpEnv,
}


def setup_config(workdir_path: str, input_path: str) -> (dict[str, Any], dict[str, Any]):
    """
    Loads the json as config and sets it up for training.
    """
    # load base PPO config and load in hyperparameters
    # Access the parsed arguments
    os.chdir(workdir_path)
    config_path = os.path.join(workdir_path, input_path)
    ppo_config = ppo.PPOConfig().to_dict()
    custom_config = load_config(config_path)
    for key in custom_config.keys():
        if key != "setup":
            ppo_config.update(custom_config[key])
    # Set eval duration equal to N available validation episodes
    ppo_config["evaluation_duration"] = len(
        os.listdir(os.path.join(
            f"{grid2op.get_current_local_dir()}",
            ppo_config["evaluation_config"]["env_config"]["env_name"],
            "chronics")
        ))
    change_workdir(workdir_path, ppo_config["env_config"]["env_name"])
    # ppo_config["env_config"]["lib_dir"] = os.path.join(workdir_path, ppo_config["env_config"]["lib_dir"])
    policies = {
        "high_level_policy": PolicySpec(  # chooses RL or do-nothing agent
            policy_class=SelectAgentPolicy2,
            config=(
                AlgorithmConfig()
                .training(
                    # _enable_learner_api=False,
                    model={
                        "custom_model_config": {
                            "rho_threshold": custom_config["environment"]["env_config"][
                                "rho_threshold"
                            ]
                        }
                    },
                )
                # .rl_module(_enable_rl_module_api=False)
                .rollouts(preprocessor_pref=None)
            ),
        ),
        "reinforcement_learning_policy": PolicySpec(  # performs RL topology
            policy_class=None,  # use default policy of PPO
            config=None,
        ),
        "do_nothing_policy": PolicySpec(  # performs do-nothing action
            policy_class=DoNothingPolicy2,
            config=(
                AlgorithmConfig()
                # .training(_enable_learner_api=False)
                # .rl_module(_enable_rl_module_api=False)
            ),
        ),
    }

    # load environment and agents manually
    ppo_config.update({"policies": policies})
    ppo_config.update({"env": ENV_TYPE[custom_config["environment"]["env_config"]["env_type"]]})
    ppo_config.update({"trial_info": "trial_id"})
    ppo_config.update({"my_log_level": custom_config["setup"]["my_log_level"]})

    return ppo_config, custom_config


def change_workdir(workdir: str, env_name: str) -> None:
    # Change grid2op path if this exists
    env_path = os.path.join(workdir, f"data_grid2op/{env_name}")
    if os.path.exists(env_path):
        grid2op_data_dir = os.path.join(workdir, "data_grid2op")
        grid2op.change_local_dir(grid2op_data_dir)
    else:
        grid2op.change_local_dir(os.path.expanduser("~/data_grid2op"))
    print(f"Environment data location used is: {grid2op.get_current_local_dir()}")
    # Change dir for RLlib ray_results output and disable the default output
    # os.environ["DEFAULT_STORAGE_PATH"] = os.path.join(workdir, f"runs/{env_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")
    ModelCatalog.register_custom_model(
        "linfcn",
        LinFCN
    )

    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        default="./configs/l2rpn_icaps_2021_small/ppo_baseline.yaml",  # "./configs/l2rpn_case14_sandbox/ppo_baseline.yaml", # "./configs/rte_case5_example/ppo_baseline.yaml", #
        help="Path to the config file.",
    )
    parser.add_argument(
        "-wd",
        "--workdir",
        type=str,
        default="/Users/ericavandersar/Documents/Python_Projects/Research/mahrl_grid2op/",
        help="path do store results.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.file_path:
        ppo_config, custom_config = setup_config(args.workdir, args.file_path)
        result_grid = run_training(ppo_config, custom_config["setup"], args.workdir)
    else:
        parser.print_help()
        logging.error("\nError: --file_path is required to specify config location.")

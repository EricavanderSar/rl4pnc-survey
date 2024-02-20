"""
Trains PPO baseline agent.
"""
import argparse
import logging
import os
from typing import Any

import grid2op

import ray
from gymnasium.spaces import Discrete
from ray import air, tune
from ray.rllib.algorithms import ppo  # import the type of agents
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec

from mahrl.experiments.yaml import load_config
from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from mahrl.multi_agent.policy import DoNothingPolicy, SelectAgentPolicy
from mahrl.algorithms.custom_ppo import CustomPPO

def run_training(config: dict[str, Any], setup: dict[str, Any]) -> None:
    """
    Function that runs the training script.
    """
    # runtime_env = {"env_vars": {"PYTHONWARNINGS": "ignore"}}
    # ray.init(runtime_env= runtime_env, local_mode=False)
    # init ray
    # Set the environment variable
    os.environ["RAY_DEDUP_LOGS"] = "0"
    ray.init()

    # Get the hostname and port
    address = ray.worker._real_worker._global_node.address
    host_name, port = address.split(":")

    print("Hostname:", host_name)
    print("Port:", port)

    # Create tuner
    tuner = tune.Tuner(
        trainable=CustomPPO,
        param_space=config,
        run_config=air.RunConfig(
            stop={"timesteps_total": setup["nb_timesteps"]}, #"training_iteration": 5}, #
            # storage_path=os.path.abspath(setup["storage_path"]),
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=setup["checkpoint_freq"],
                checkpoint_at_end=True,
                checkpoint_score_attribute="evaluation/episode_reward_mean",
            ),
            verbose=setup["verbose"],
        ),
    )

    # Launch tuning
    try:
        result_grid = tuner.fit()
    finally:
        # Close ray instance
        ray.shutdown()

    for i in range(len(result_grid)):
        result = result_grid[i]
        if not result.error:
            print(f"Trial finishes successfully with custom_metrics "
                  f"{result.metrics['custom_metrics']}.")
        else:
            print(f"Trial failed with error {result.error}.")


def setup_config(workdir_path: str, input_path: str) -> None:
    """
    Loads the json as config and sets it up for training.
    """
    # load base PPO config and load in hyperparameters
    # Access the parsed arguments
    config_path = os.path.join(workdir_path, input_path)
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
    ppo_config.update(custom_config["resources"])
    ppo_config.update(custom_config["rollouts"])
    # ppo_config.update(custom_config["evaluation"])

    change_workdir(workdir_path, ppo_config["env_config"]["env_name"])
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
        "reinforcement_learning_policy": PolicySpec(  # performs RL topology
            policy_class=None,  # use default policy of PPO
            observation_space=None,  # infer automatically from env
            action_space=None,  # infer automatically from env
            config=None,
        ),
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

    # load environment and agents manually
    ppo_config.update({"policies": policies})
    ppo_config.update({"env": CustomizedGrid2OpEnvironment})

    run_training(ppo_config, custom_config["setup"])


def change_workdir(workdir: str, env_name: str) -> None:
    # Change grid2op path if this exists
    env_path = os.path.join(workdir, f"data_grid2op/{env_name}")
    if os.path.exists(env_path):
        grid2op_data_dir = os.path.join(workdir, "data_grid2op")
        grid2op.change_local_dir(grid2op_data_dir)
    else:
        grid2op.change_local_dir(os.path.expanduser("~/data_grid2op"))
    print(f"Environment data location used is: {grid2op.get_current_local_dir()}")
    # Change dir for RLlib ray_results output tensorboard
    os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = os.path.join(workdir, "runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        default= "../configs/rte_case14_realistic/ppo_baseline_batchjob.yaml",  #"../configs/rte_case5_example/ppo_baseline.yaml", #
        help="Path to the config file.",
    )
    parser.add_argument(
        "-wd",
        "--workdir",
        type=str,
        default="/Users/ericavandersar/Documents/Python_Projects/Research/mahrl_grid2op/scripts/",
        help="path do store results.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.file_path:
        setup_config(args.workdir, args.file_path)
    else:
        parser.print_help()
        logging.error("\nError: --file_path is required to specify config location.")

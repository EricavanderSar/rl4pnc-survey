"""
Trains PPO baseline agent.
"""

import argparse
import logging
import os
from typing import Any
from tabulate import tabulate
import re

import grid2op

import ray
import gymnasium as gym
from ray import air, tune, train
# from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms import ppo  # import the type of agents
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.result_grid import ResultGrid
# import ray.rllib.models.torch.torch_modelv2
# from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import MedianStoppingRule

from mahrl.experiments.yaml import load_config
from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from mahrl.grid2op_env.custom_env2 import RlGrid2OpEnv
from mahrl.multi_agent.policy import (
    DoNothingPolicy,
    SelectAgentPolicy,
    DoNothingPolicy2,
    SelectAgentPolicy2
)
from mahrl.algorithms.custom_ppo import CustomPPO
from mahrl.algorithms.optuna_search import MyOptunaSearch
from mahrl.experiments.callback import Style, TuneCallback
from mahrl.experiments.utils import delete_nested_key

REPORT_END = False

ENV_TYPE = {
    "old_env": {
        'env': CustomizedGrid2OpEnvironment,
        'hl_policy': SelectAgentPolicy,
        'hl_obs_space': None, # infer automatically from env
        'dn_policy': DoNothingPolicy,
        'dn_obs_space': None, # infer automatically from env
    },
    "new_env": {
        'env': RlGrid2OpEnv,
        'hl_policy': SelectAgentPolicy2,
        'hl_obs_space': gym.spaces.Box(-1, 2), # Only give max rho as obs
        'dn_policy': DoNothingPolicy2,
        'dn_obs_space': gym.spaces.Discrete(1), # Do Nothing observation is irrelevant
    }
}


def run_training(config: dict[str, Any], setup: dict[str, Any], workdir: str) -> ResultGrid:
    """
    Function that runs the training script.
    """
    # runtime_env = {"env_vars": {"PYTHONWARNINGS": "ignore"}}
    # ray.init(runtime_env= runtime_env, local_mode=False)
    # init ray
    # Set the environment variable
    os.environ["RAY_DEDUP_LOGS"] = "0"
    # os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    # os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
    # Run wandb offline and to sync when finished use following command in result directory:
    # for d in $(ls -t -d */); do cd $d; wandb sync --sync-all; cd ..; done
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"
    ray.init()

    # Get the hostname and port
    address = ray.worker._real_worker._global_node.address
    host_name, port = address.split(":")
    print("Hostname:", host_name)
    print("Port:", port)

    # Use Optuna search algorithm to find good working parameters
    if setup['optimize']:
        algo = MyOptunaSearch(
            metric=setup["score_metric"],
            mode="max",
            points_to_evaluate=[setup['points_to_evaluate']] if 'points_to_evaluate' in setup else None,
        )
        if setup['result_dir']:
            print("Retrieving data old experiment from : ", setup['result_dir'])
            algo.restore_from_dir(setup['result_dir'])
            for key in algo._space.keys():
                if '/' in key:
                    delete_nested_key(config, key)
                else:
                    del config[key]
        # Scheduler determines if we should prematurely stop a certain experiment
        # scheduler = MedianStoppingRule(
        #     time_attr="timesteps_total", #Default = "time_total_s"
        #     metric=setup["score_metric"],
        #     mode="max",
        #     grace_period=setup["grace_period"], # First exploration before stopping
        #     min_samples_required=3, # Default = 3
        #     min_time_slice=3,
        # )

    # Create tuner
    tuner = tune.Tuner(
        trainable=CustomPPO,
        param_space=config,
        run_config=air.RunConfig(
            name=setup["folder_name"],
            # storage_path=os.path.join(workdir, os.path.join(setup["storage_path"], config["env_config"]["env_name"])),
            stop={"timesteps_total": setup["nb_timesteps"],
                  "custom_metrics/grid2op_end_mean": setup["max_ep_len"]},
            callbacks=[
                WandbLoggerCallback(
                    project=setup["experiment_name"],
                                    ),
                TuneCallback(
                    config["my_log_level"],
                    "evaluation/custom_metrics/grid2op_end_mean",
                    eval_freq=config["evaluation_interval"],
                    heartbeat_freq=60,
                ),
            ],
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=setup["checkpoint_freq"],
                checkpoint_at_end=True,
                checkpoint_score_attribute="custom_metrics/corrected_ep_len_mean",
                num_to_keep=5,
            ),
            verbose=setup["verbose"],
        ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=setup["num_samples"],
            # scheduler=scheduler,
        ) if setup["optimize"] else None
        ,
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
            print(Style.BOLD + f" *---- Trial {i} finished successfully with evaluation results ---*\n" + Style.END +
                  f"{tabulate([result.metrics['evaluation']['custom_metrics']], headers='keys', tablefmt='rounded_grid')}")

            # print("ALL RESULT METRICS: ", result.metrics)
            # print("ENV CONFIG: ", result.config['env_config'])
            # print("RESULT CONFIG: ", result.config['env_config'])
            # Print table with environment config.
            if REPORT_END:
                print(f"--- Environment Configuration  ---- \n"
                      f"{tabulate([result.config['env_config']], headers='keys', tablefmt='rounded_grid')}")
                # print other params:
                params_ppo = ['gamma', 'lr', 'exploration_config',  'vf_loss_coeff', 'entropy_coeff', 'clip_param',
                              'lambda', 'vf_clip_param', 'num_sgd_iter', 'sgd_minibatch_size', 'train_batch_size']
                values = [result.config[par] for par in params_ppo]
                print(f"--- PPO Configuration  ---- \n"
                      f"{tabulate([values], headers=params_ppo, tablefmt='rounded_grid')}")
                print(f"--- Model Configuration  ---- \n"
                      f"{tabulate([result.config['model']], headers='keys', tablefmt='rounded_grid')}")
        else:
            print(f"Trial failed with error {result.error}.")
    return result_grid


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
    ppo_config.update(custom_config["training"])
    ppo_config.update(custom_config["debugging"])
    ppo_config.update(custom_config["framework"])
    # ppo_config.update(custom_config["rl_module"])
    ppo_config.update(custom_config["explore"])
    ppo_config.update(custom_config["callbacks"])
    ppo_config.update(custom_config["environment"])
    ppo_config.update(custom_config["multi_agent"])
    if "resources" in custom_config.keys():
        ppo_config.update(custom_config["resources"])
    if "rollouts" in custom_config.keys():
        ppo_config.update(custom_config["rollouts"])
    if "scaling_config" in custom_config.keys():
        ppo_config.update(custom_config["scaling_config"])
    ppo_config.update(custom_config["evaluation"])
    ppo_config.update(custom_config["reporting"])
    env_type_config = ENV_TYPE[custom_config["environment"]["env_config"]["env_type"]]

    change_workdir(workdir_path, ppo_config["env_config"]["env_name"])
    # ppo_config["env_config"]["lib_dir"] = os.path.join(workdir_path, ppo_config["env_config"]["lib_dir"])
    policies = {
        "high_level_policy": PolicySpec(  # chooses RL or do-nothing agent
            policy_class=env_type_config["hl_policy"],
            # observation_space=env_type_config["hl_obs_space"],
            # action_space=gym.spaces.Discrete(2),  # choose one of agents
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
            # observation_space=None,  # infer automatically from env
            # action_space=None,  # infer automatically from env
            config=None,
        ),
        "do_nothing_policy": PolicySpec(  # performs do-nothing action
            policy_class=env_type_config["dn_policy"],
            # observation_space=env_type_config["dn_obs_space"],
            # action_space=gym.spaces.Discrete(1),  # only perform do-nothing
            config=(
                AlgorithmConfig()
                # .training(_enable_learner_api=False)
                # .rl_module(_enable_rl_module_api=False)
            ),
        ),
    }

    # load environment and agents manually
    ppo_config.update({"policies": policies})
    ppo_config.update({"env": env_type_config["env"]})
    ppo_config.update({"trial_info": "trial_id"})
    ppo_config.update({"my_log_level": custom_config["my_log_level"]})

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

    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        default="./configs/rte_case14_realistic/ppo_baseline.yaml",  #"./configs/rte_case5_example/ppo_baseline.yaml", #
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

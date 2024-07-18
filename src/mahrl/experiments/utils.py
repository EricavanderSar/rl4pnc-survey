"""
Utilities in the grid2op experiments.
"""

import logging
import os
from typing import Any, Dict, List, OrderedDict, Union
from tabulate import tabulate
import json
import numpy as np

from grid2op.Environment import BaseEnv
import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.result_grid import ResultGrid
# from ray.tune.schedulers import MedianStoppingRule
# from ray.air.integrations.mlflow import MLflowLoggerCallback
# from ray.rllib.algorithms import Algorithm
# from ray.rllib.models import ModelCatalog

from mahrl.algorithms.custom_ppo import CustomPPO
from mahrl.algorithms.optuna_search import MyOptunaSearch
from mahrl.experiments.callback import Style, TuneCallback
from ray.tune.stopper.stopper import Stopper
from ray.tune.experiment import Trial
from ray.util.annotations import PublicAPI

REPORT_END = True


def calculate_action_space_asymmetry(env: BaseEnv, add_dn:bool = False) -> tuple[int, int, dict[int, int]]:
    """
    Function prints and returns the number of legal actions and topologies without symmetries.
    """

    nr_substations = len(env.sub_info)

    logging.info("no symmetries")
    action_space = 0
    controllable_substations = {}
    possible_topologies = 1
    for sub in range(nr_substations):
        nr_elements = len(env.observation_space.get_obj_substations(substation_id=sub))
        nr_non_lines = sum(
            1
            for row in env.observation_space.get_obj_substations(substation_id=sub)
            if row[1] != -1 or row[2] != -1
        )

        alpha = 2 ** (nr_elements - 1) - (2**nr_non_lines - 1)
        action_space += alpha if alpha > 1 else 0
        # if alpha > 1:  # without do nothings for single substations
        if (add_dn and alpha > 0) or (alpha > 1):
            controllable_substations[sub] = alpha
        possible_topologies *= max(alpha, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def calculate_action_space_medha(env: BaseEnv, add_dn: bool = False) -> tuple[int, int, dict[int, int]]:
    """
    Function prints and returns the number of legal actions and topologies following Subrahamian (2021).
    """
    nr_substations = len(env.sub_info)

    logging.info("medha")
    action_space = 0
    controllable_substations = {}
    possible_topologies = 1
    for sub in range(nr_substations):
        nr_elements = len(env.observation_space.get_obj_substations(substation_id=sub))
        nr_non_lines = sum(
            1
            for row in env.observation_space.get_obj_substations(substation_id=sub)
            if row[1] != -1 or row[2] != -1
        )
        alpha = 2 ** (nr_elements - 1)
        beta = nr_elements - (1 if nr_elements == 2 else 0)
        gamma = 2**nr_non_lines - 1 - nr_non_lines
        combined = alpha - beta - gamma
        action_space += combined if combined > 1 else 0
        # if combined > 1:  # without do nothings for single substations
        if (add_dn and combined > 0) or (combined > 1):
            controllable_substations[sub] = combined
        possible_topologies *= max(combined, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    print(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def calculate_action_space_tennet(env: BaseEnv, add_dn=False) -> tuple[int, int, dict[int, int]]:
    """
    Function prints and returns the number of legal actions and topologies following the proposed action space.
    """
    nr_substations = len(env.sub_info)

    logging.info("TenneT")
    action_space = 0
    controllable_substations = {}
    possible_topologies = 1
    for sub in range(nr_substations):
        nr_elements = len(env.observation_space.get_obj_substations(substation_id=sub))
        nr_non_lines = sum(
            1
            for row in env.observation_space.get_obj_substations(substation_id=sub)
            if row[1] != -1 or row[2] != -1
        )
        nr_lines = nr_elements - nr_non_lines

        combined = (
            (
                2**nr_non_lines - 2
            )  # configuratations of non-lines except when all lines are same colour
            * (
                2**nr_lines  # configurations of lines
                - 2 * nr_lines  # minus lines that there is exactly one line at a busbar
                - 2  # minus case where all lines have the same colour
                + (2 if nr_lines == 1 else 0)  # due to doubles with 1 line
                + (2 if nr_lines == 2 else 0)  # due to doubles with 2 lines
            )
            + 2  # configurations where non-lines all have the same colour
            * (
                2**nr_lines  # configurations of lines
                - 2 * nr_lines  # minus lines that there is exactly one line at a busbar
                - 1  # if all non-lines have the same colour, then if all lines are also this colour, it's allowed
                + (2 if nr_lines == 2 else 0)  # due to doubles with 2 lines
                + (1 if nr_lines == 1 else 0)  # due to doubles with 1 line
            )
        ) / 2  # remove symmetries

        action_space += int(combined) if combined > 1 else 0
        if (add_dn and combined > 0) or (combined > 1):  # combined > 1: without do nothings for single substations
            controllable_substations[sub] = combined
        possible_topologies *= max(combined, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def get_capa_substation_id(
    line_info: dict[int, list[int]],
    obs_batch: Union[List[Dict[str, Any]], Dict[str, Any]],
    controllable_substations: dict[int, int],
) -> list[int]:
    """
    Returns the substation id of the substation to act on according to CAPA.
    """
    # calculate the mean rho per substation
    connected_rhos: dict[int, list[float]] = {agent: [] for agent in line_info}
    for sub_idx in line_info:
        for line_idx in line_info[sub_idx]:
            if isinstance(obs_batch, OrderedDict):
                connected_rhos[sub_idx].append(
                    obs_batch["previous_obs"]["rho"][0][line_idx]
                    # obs_batch["original_obs"]["rho"][0][line_idx]
                )
            elif isinstance(obs_batch, dict):
                connected_rhos[sub_idx].append(
                    obs_batch["previous_obs"]["rho"][line_idx]
                    # obs_batch["original_obs"]["rho"][line_idx]
                )
            else:
                raise ValueError("The observation batch is not supported.")
    for sub_idx in connected_rhos:
        connected_rhos[sub_idx] = [float(np.mean(connected_rhos[sub_idx]))]

    # set non-controllable substations to 0
    for sub_idx in connected_rhos:
        if sub_idx not in list(controllable_substations.keys()):
            connected_rhos[sub_idx] = [0.0]

    # order the substations by the mean rho, maximum first
    connected_rhos = dict(
        sorted(connected_rhos.items(), key=lambda item: item[1], reverse=True)
    )

    # # find substation with max average rho
    # max_value = max(connected_rhos.values())
    # return [key for key, value in connected_rhos.items() if value == max_value][0]

    # return the ordered entries
    # NOTE: When there are two equal max values, the first one is returned first
    return list(connected_rhos.keys())


def find_list_of_agents(env: BaseEnv, action_space: str) -> dict[int, int]:
    """
    Function that returns the number of controllable substations.
    """
    add_dn = "dn" in action_space
    if action_space.startswith("asymmetry"):
        _, _, list_of_agents = calculate_action_space_asymmetry(env, add_dn)
        return list_of_agents
    if action_space.startswith("medha"):
        _, _, list_of_agents = calculate_action_space_medha(env, add_dn)
        return list_of_agents
    if action_space.startswith("tennet"):
        _, _, list_of_agents = calculate_action_space_tennet(env, add_dn)
        return list_of_agents
    raise ValueError("The action space is not supported.")


def find_substation_per_lines(
    env: BaseEnv, list_of_agents: list[int]
) -> dict[int, list[int]]:
    """
    Returns a dictionary connecting line ids to substations.
    """
    line_info: dict[int, list[int]] = {agent: [] for agent in list_of_agents}
    for sub_idx in list_of_agents:
        for or_id in env.observation_space.get_obj_connect_to(substation_id=sub_idx)[
            "lines_or_id"
        ]:
            line_info[sub_idx].append(or_id)
        for ex_id in env.observation_space.get_obj_connect_to(substation_id=sub_idx)[
            "lines_ex_id"
        ]:
            line_info[sub_idx].append(ex_id)

    return line_info


def delete_nested_key(d, path):
    keys = path.split('/')
    current = d

    # Traverse through the dictionary using keys from the path
    for key in keys[:-1]:  # Iterate until the second last key
        if key in current:
            current = current[key]
        else:
            return  # If any key is missing, return without making changes

    # Now current points to the dictionary containing the key to be deleted
    last_key = keys[-1]
    if last_key in current:
        del current[last_key]


class MaxCustomMetricStopper(Stopper):
    """Stop trials after reaching a maximum value for the custom metric

    Args:
        metric: Metric to use.
        max_value: If custom metric reaches this value stop trials
    """

    def __init__(self, metric: str, max_value: int):
        self._max_value = max_value
        self._custom_Metric = metric

    def __call__(self, trial_id: str, result: Dict):
        print("current value custom metric: ", result["custom_metric"][self._custom_Metric])
        return result["custom_metric"][self._custom_Metric] >= self._max_value

    def stop_all(self):
        return False


def trial_str_creator(trial: Trial, job_id=""):
    trial.trial_id = trial.trial_id.split("_")[0]
    if job_id:
        return "{}_{}_{}_{}".format(trial.trainable_name, trial.config["env_config"]["env_type"], job_id, trial.trial_id)
    else:
        return "{}_{}_{}".format(trial.trainable_name, trial.config["env_config"]["env_type"], trial.trial_id)


def run_training(config: dict[str, Any], setup: dict[str, Any], job_id: str) -> ResultGrid:
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
        if 'result_dir' in setup.keys():
            print("Retrieving data old experiment from : ", setup['result_dir'])
            algo.restore_from_dir(setup['result_dir'])
            for key in algo._space.keys():
                if '/' in key:
                    delete_nested_key(config, key)
                else:
                    del config[key]
        # # Scheduler determines if we should prematurely stop a certain experiment - NOTE: DOES NOT WORK AS EXPECTED!!!
        # scheduler = MedianStoppingRule(
        #     time_attr="timesteps_total", #Default = "time_total_s"
        #     metric=setup["score_metric"],
        #     mode="max",
        #     grace_period=setup["grace_period"], # First exploration before stopping
        #     min_samples_required=5, # Default = 3
        #     min_time_slice=10_000,
        #     hard_stop=False,
        # )

    # Create tuner
    tuner = tune.Tuner(
        trainable=CustomPPO,
        param_space=config,
        run_config=air.RunConfig(
            name=setup["folder_name"],
            # storage_path=os.path.join(workdir, os.path.join(setup["storage_path"], config["env_config"]["env_name"])),
            stop={"timesteps_total": setup["nb_timesteps"]}, #MaxCustomMetricStopper("total_agent_interact", setup["nb_timesteps"]), #
            # "custom_metrics/grid2op_end_mean": setup["max_ep_len"]},
            callbacks=[
                WandbLoggerCallback(
                    project=setup["experiment_name"],
                                    ),
                TuneCallback(
                    setup["my_log_level"],
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
            trial_name_creator=lambda t: trial_str_creator(t, job_id),
            trial_dirname_creator=lambda t: trial_str_creator(t, job_id),
            search_alg=algo,
            num_samples=setup["num_samples"],
            # scheduler=scheduler,
        ) if setup["optimize"] else
        tune.TuneConfig(
            trial_name_creator=lambda t: trial_str_creator(t, job_id),
            trial_dirname_creator=lambda t: trial_str_creator(t, job_id),)
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
            # Print and save available checkpoints
            checkpoints_tojson = {
                os.path.basename(checkpoint.path): metrics['evaluation']['custom_metrics'] for
                checkpoint, metrics in result.best_checkpoints
            }
            with open(os.path.join(result.path, "checkpoint_results.json"), "w") as outfile:
                json.dump(checkpoints_tojson, outfile)

            print(Style.BOLD + f" *---- Trial {i} finished successfully with evaluation results ---*\n" + Style.END +
                  tabulate(
                      [[k] + list(v.values()) for k, v in checkpoints_tojson.items()],
                      headers=['checkpoint'] + list(result.metrics['evaluation']['custom_metrics'].keys()),
                      tablefmt='rounded_grid')
                  )

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
                params_model = ['fcnet_hiddens', 'fcnet_activation', 'post_fcnet_hiddens', 'post_fcnet_activation']
                values = [result.config['model'][par] for par in params_model]
                print(f"--- Model Configuration  ---- \n"
                      f"{tabulate([values], headers=params_model, tablefmt='rounded_grid')}")
        else:
            print(f"Trial failed with error {result.error}.")
    return result_grid

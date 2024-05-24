"""
Utilities in the grid2op experiments.
"""

import logging
import os
import random
from typing import Any, Dict, List, OrderedDict, Union

import numpy as np
import ray
import torch
from grid2op.Environment import BaseEnv
from ray import air, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.rllib.algorithms import Algorithm
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.bohb import TuneBOHB


def calculate_action_space_asymmetry(
    env: BaseEnv, add_dn: bool = False
) -> tuple[int, int, dict[str, int]]:
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
            controllable_substations[str(sub)] = alpha
        possible_topologies *= max(alpha, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def calculate_action_space_medha(
    env: BaseEnv, add_dn: bool = False
) -> tuple[int, int, dict[str, int]]:
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
        if (add_dn and combined > 0) or (combined > 1):
            controllable_substations[str(sub)] = combined
        possible_topologies *= max(combined, 1)

    return action_space, possible_topologies, controllable_substations


def calculate_action_space_tennet(
    env: BaseEnv, add_dn: bool = False
) -> tuple[int, int, dict[str, int]]:
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
        if (add_dn and combined > 0) or (combined > 1):
            controllable_substations[str(sub)] = combined
        possible_topologies *= max(combined, 1)

    return action_space, possible_topologies, controllable_substations


def get_capa_substation_id(
    line_info: dict[str, list[int]],
    obs_batch: Union[List[Dict[str, Any]], Dict[str, Any]],
    controllable_substations: dict[str, int],
) -> list[str]:
    """
    Returns the substation id of the substation to act on according to CAPA.
    """
    # calculate the mean rho per substation
    connected_rhos: dict[str, list[float]] = {agent: [] for agent in line_info}
    for sub_idx in line_info:
        for line_idx in line_info[sub_idx]:
            # extract rho
            if isinstance(obs_batch, OrderedDict):
                rho = obs_batch["previous_obs"]["rho"][0][line_idx]
            elif isinstance(obs_batch, dict):
                rho = obs_batch["previous_obs"]["rho"][line_idx]
            else:
                raise ValueError("The observation batch is not supported.")

            # add rho to sub
            if rho > 0:
                connected_rhos[sub_idx].append(rho)
            else:  # line is disconnected, set to 3
                connected_rhos[sub_idx].append(3.0)

    for sub_idx in connected_rhos:
        # connected_rhos[sub_idx] = [float(np.mean(connected_rhos[sub_idx]))]
        connected_rhos[sub_idx] = [
            float(np.max(connected_rhos[sub_idx])),
            float(np.min(connected_rhos[sub_idx])),
        ]

    # set non-controllable substations to 0
    for sub_idx in connected_rhos:
        if str(sub_idx) not in list(controllable_substations.keys()):
            connected_rhos[sub_idx] = [0.0]

    # order the substations by the max rho, using min rho as tiebreaker
    ordered_keys = sorted(
        connected_rhos.keys(),
        key=lambda x: (connected_rhos[x][0], connected_rhos[x][1]),
        reverse=True,
    )

    # return the ordered entries
    return list(ordered_keys)


def find_list_of_agents(env: BaseEnv, action_space: str) -> dict[str, int]:
    """
    Function that returns the number of controllable substations.
    """
    if action_space.startswith("asymmetry"):
        _, _, list_of_agents = calculate_action_space_asymmetry(env)
        return list_of_agents
    if action_space.startswith("medha"):
        _, _, list_of_agents = calculate_action_space_medha(env)
        return list_of_agents
    if action_space.startswith("tennet"):
        _, _, list_of_agents = calculate_action_space_tennet(env)
        return list_of_agents
    raise ValueError("The action space is not supported.")


def find_substation_per_lines(
    env: BaseEnv, list_of_agents: list[str]
) -> dict[str, list[int]]:
    """
    Returns a dictionary connecting line ids to substations.
    """
    line_info: dict[str, list[int]] = {agent: [] for agent in list_of_agents}
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


def run_training(
    config: dict[str, Any], setup: dict[str, Any], algorithm: Algorithm
) -> None:
    """
    Function that runs the training script.
    """
    # init ray
    ray.init(ignore_reinit_error=False)

    if "debugging" in config and "seed" in config["debugging"]:
        set_reproducibillity(config["debugging"]["seed"])

        # set seed in env
        config["env_config"]["seed"] = config["debugging"]["seed"]
        config["evaluation_config"]["env_config"]["seed"] = config["debugging"]["seed"]

    tuner = get_gridsearch_tuner(setup, config, algorithm)

    # Launch tuning
    try:
        tuner.fit()
    finally:
        # Close ray instance
        ray.shutdown()

    # save config to params.json in the runs file that is created
    with open(
        os.path.join(
            setup["storage_path"],
            # "mlruns",
            # "configs",
            "params.json",
        ),
        "w",
        encoding="utf-8",
    ) as config_file:
        config_file.write(str(config))


def get_gridsearch_tuner(
    setup: dict[str, Any], config: dict[str, Any], algorithm: Algorithm
) -> tune.Tuner:
    """
    Returns a GridSearch tuner for hyperparameter optimization.

    Args:
        setup (dict): A dictionary containing setup parameters.
        config (dict): A dictionary containing hyperparameter configurations.
        algorithm (Algorithm): The algorithm to be tuned.

    Returns:
        tune.Tuner: A GridSearch tuner object.

    """
    tuner = tune.Tuner(
        algorithm,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=setup["num_samples"],
        ),
        run_config=air.RunConfig(
            name="mlflow",
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=os.path.join(setup["storage_path"], "mlruns"),
                    experiment_name=setup["experiment_name"],
                    save_artifact=setup["save_artifact"],
                )
            ],
            stop={"timesteps_total": setup["nb_timesteps"]},
            storage_path=os.path.abspath(setup["storage_path"]),
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=setup["checkpoint_freq"],
                checkpoint_at_end=setup["checkpoint_at_end"],
                checkpoint_score_attribute=setup["checkpoint_score_attr"],
                # num_to_keep=setup["keep_checkpoints_num"],
            ),
            verbose=setup["verbose"],
        ),
    )
    return tuner


def get_bohb_tuner(
    setup: dict[str, Any], config: dict[str, Any], algorithm: Algorithm
) -> tune.Tuner:
    """
    Returns a BOHB tuner for hyperparameter optimization.

    Args:
        setup (dict): A dictionary containing setup parameters.
        config (dict): A dictionary containing hyperparameter configurations.
        algorithm (Algorithm): The algorithm to be tuned.

    Returns:
        tune.Tuner: The BOHB tuner.

    """
    # Create tuner
    algo = TuneBOHB()
    algo = ray.tune.search.ConcurrencyLimiter(algo, max_concurrent=4)
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=10,
        reduction_factor=4,
        stop_last_trials=False,
    )

    tuner = tune.Tuner(
        algorithm,
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="custom_metrics/corrected_ep_len_mean",
            mode="max",
            search_alg=algo,
            scheduler=scheduler,
            num_samples=10,
        ),
        run_config=air.RunConfig(
            name="mlflow",
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=os.path.join(setup["storage_path"], "mlruns"),
                    experiment_name=setup["experiment_name"],
                    save_artifact=setup["save_artifact"],
                )
            ],
            stop={"timesteps_total": setup["nb_timesteps"]},
            storage_path=os.path.abspath(setup["storage_path"]),
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=setup["checkpoint_freq"],
                checkpoint_at_end=setup["checkpoint_at_end"],
                checkpoint_score_attribute=setup["checkpoint_score_attr"],
                num_to_keep=setup["keep_checkpoints_num"],
            ),
            verbose=setup["verbose"],
        ),
    )
    return tuner


def get_bayesian_tuner(
    setup: dict[str, Any], config: dict[str, Any], algorithm: Algorithm
) -> tune.Tuner:
    """
    Returns a Bayesian tuner for hyperparameter optimization.

    Args:
        setup (dict): A dictionary containing setup parameters.
        config (dict): A dictionary containing hyperparameter configurations.
        algorithm (Algorithm): The algorithm to be used for tuning.

    Returns:
        tune.Tuner: A Bayesian tuner for hyperparameter optimization.
    """
    tuner = tune.Tuner(
        algorithm,
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="evaluation/episode_reward_mean",
            mode="max",
            search_alg=ConcurrencyLimiter(
                BayesOptSearch(utility_kwargs={"kind": "ucb", "kappa": 2.5, "xi": 0.0}),
                max_concurrent=4,
            ),
            num_samples=14,
        ),
        run_config=air.RunConfig(
            name="mlflow",
            callbacks=[
                MLflowLoggerCallback(
                    tracking_uri=os.path.join(setup["storage_path"], "mlruns"),
                    experiment_name=setup["experiment_name"],
                    save_artifact=setup["save_artifact"],
                )
            ],
            stop={"timesteps_total": setup["nb_timesteps"]},
            storage_path=os.path.abspath(setup["storage_path"]),
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=setup["checkpoint_freq"],
                checkpoint_at_end=setup["checkpoint_at_end"],
                checkpoint_score_attribute=setup["checkpoint_score_attr"],
                num_to_keep=setup["keep_checkpoints_num"],
            ),
            verbose=setup["verbose"],
        ),
    )
    return tuner


def set_reproducibillity(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Parameters:
        seed (int): The seed value to set for random number generators.

    Returns:
        None
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

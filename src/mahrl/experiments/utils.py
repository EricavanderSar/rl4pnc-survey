"""
Utilities in the grid2op experiments.
"""

import logging
import os
from typing import Any, Dict, List, OrderedDict, Union

import numpy as np
import ray
from grid2op.Environment import BaseEnv
from ray import air, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.rllib.algorithms import Algorithm
from ray.rllib.models import ModelCatalog

from mahrl.models.mlp import SimpleMlp


def calculate_action_space_asymmetry(env: BaseEnv) -> tuple[int, int, dict[int, int]]:
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
        if alpha > 1:
            controllable_substations[sub] = alpha
        possible_topologies *= max(alpha, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def calculate_action_space_medha(env: BaseEnv) -> tuple[int, int, dict[int, int]]:
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
        if combined > 1:
            controllable_substations[sub] = combined
        possible_topologies *= max(combined, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def calculate_action_space_tennet(env: BaseEnv) -> tuple[int, int, dict[int, int]]:
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
        if combined > 1:
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


def run_training(
    config: dict[str, Any], setup: dict[str, Any], algorithm: Algorithm
) -> None:
    """
    Function that runs the training script.
    """
    # init ray
    # ray.shutdown()
    ray.init(ignore_reinit_error=False)

    ModelCatalog.register_custom_model("fcn", SimpleMlp)

    # Create tuner
    tuner = tune.Tuner(
        algorithm,
        param_space=config,
        tune_config=tune.TuneConfig(num_samples=setup["num_samples"]),
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

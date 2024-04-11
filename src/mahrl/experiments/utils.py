"""
Utilities in the grid2op experiments.
"""

import logging
from typing import Any, Dict, List, OrderedDict, Union

import numpy as np
from grid2op.Environment import BaseEnv


def calculate_action_space_asymmetry(env: BaseEnv) -> tuple[int, int, list[int]]:
    """
    Function prints and returns the number of legal actions and topologies without symmetries.
    """

    nr_substations = len(env.sub_info)

    logging.info("no symmetries")
    action_space = 0
    controllable_substations = []
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
            controllable_substations.append(sub)
        possible_topologies *= max(alpha, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def calculate_action_space_medha(env: BaseEnv) -> tuple[int, int, list[int]]:
    """
    Function prints and returns the number of legal actions and topologies following Subrahamian (2021).
    """
    nr_substations = len(env.sub_info)

    logging.info("medha")
    action_space = 0
    controllable_substations = []
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
            controllable_substations.append(sub)
        possible_topologies *= max(combined, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def calculate_action_space_tennet(env: BaseEnv) -> tuple[int, int, list[int]]:
    """
    Function prints and returns the number of legal actions and topologies following the proposed action space.
    """
    nr_substations = len(env.sub_info)

    logging.info("TenneT")
    action_space = 0
    controllable_substations = []
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
            controllable_substations.append(sub)
        possible_topologies *= max(combined, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies, controllable_substations


def get_capa_substation_id(
    line_info: dict[int, list[int]],
    obs_batch: Union[List[Dict[str, Any]], Dict[str, Any]],
    controllable_substations: list[int],
) -> list[int]:
    """
    Returns the substation id of the substation to act on according to CAPA.
    """
    # calculate the mean rho per substation
    connected_rhos: dict[int, list[float]] = {agent: [] for agent in line_info}
    for sub_idx in line_info:
        for line_idx in line_info[sub_idx]:
            if isinstance(obs_batch, OrderedDict):
                connected_rhos[sub_idx].append(obs_batch["rho"][0][line_idx])
            elif isinstance(obs_batch, dict):
                connected_rhos[sub_idx].append(obs_batch["rho"][line_idx])
            else:
                raise ValueError("The observation batch is not supported.")
    for sub_idx in connected_rhos:
        connected_rhos[sub_idx] = np.mean(connected_rhos[sub_idx])

    # set non-controllable substations to 0
    for sub_idx in connected_rhos:
        if sub_idx not in controllable_substations:
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


def find_list_of_agents(env: BaseEnv, action_space: str) -> list[int]:
    """
    Function that returns the number of controllable substations.
    """
    if action_space == "asymmetry":
        _, _, list_of_agents = calculate_action_space_asymmetry(env)
        return list_of_agents
    if action_space == "medha":
        _, _, list_of_agents = calculate_action_space_medha(env)
        return list_of_agents
    if action_space == "tennet":
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
"""
Utilities in the grid2op experiments.
"""

import logging

from grid2op.Environment import BaseEnv


def calculate_action_space_asymmetry(env: BaseEnv) -> tuple[int, int]:
    """
    Function prints and returns the number of legal actions and topologies without symmetries.
    """

    nr_substations = len(env.sub_info)

    logging.info("no symmetries")
    action_space = 0
    controllable_substations = 0
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
        controllable_substations += 1 if alpha > 1 else 0
        possible_topologies *= max(alpha, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies


def calculate_action_space_medha(env: BaseEnv) -> tuple[int, int]:
    """
    Function prints and returns the number of legal actions and topologies following Subrahamian (2021).
    """
    nr_substations = len(env.sub_info)

    logging.info("medha")
    action_space = 0
    controllable_substations = 0
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
        controllable_substations += 1 if combined > 1 else 0
        possible_topologies *= max(combined, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies


def calculate_action_space_tennet(env: BaseEnv) -> tuple[int, int]:
    """
    Function prints and returns the number of legal actions and topologies following the proposed action space.
    """
    nr_substations = len(env.sub_info)

    logging.info("TenneT")
    action_space = 0
    controllable_substations = 0
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
        controllable_substations += 1 if combined > 1 else 0
        possible_topologies *= max(combined, 1)

    logging.info(f"actions {action_space}")
    logging.info(f"topologies {possible_topologies}")
    logging.info(f"controllable substations {controllable_substations}")
    return action_space, possible_topologies

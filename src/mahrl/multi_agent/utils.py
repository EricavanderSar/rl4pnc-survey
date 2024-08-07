"""
Defines agent policies.
"""

import random

import numpy as np
from grid2op.Action import BaseAction

from mahrl.experiments.utils import get_capa_substation_id


def argmax_logic(proposed_confidences: dict[str, float]) -> str:
    """
    Selects a sub_id based on the max proposed confidence.

    Args:
        proposed_confidences (dict[str, float]): A dictionary mapping sub_ids to their corresponding confidence values.

    Returns:
        str: The selected sub_id.

    """
    # return max(proposed_confidences, key=proposed_confidences.get)
    return max(proposed_confidences, key=lambda x: proposed_confidences[x])


def softmax(x):
    """
    Compute the softmax function for an input array.

    Parameters:
    x (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Softmax values of the input array.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sample_logic(proposed_confidences: dict[str, float]) -> str:
    """
    Samples a sub_id based on the proposed confidences, using them as weights.

    Args:
        proposed_confidences (dict[str, float]): A dictionary mapping sub_ids to their corresponding confidence values.

    Returns:
        str: The selected sub_id.

    """
    # make all weights positive
    weights = softmax(list(proposed_confidences.values()))

    # take the sub_id based on a uniform sample of proposed_confidences
    sub_id = random.choices(
        list(proposed_confidences.keys()),
        weights=weights,
        k=1,
    )[0]

    return sub_id


def capa_logic(
    proposed_actions: dict[str, BaseAction],
    gym_obs: dict[str, list[int]],
    controllable_substations: dict[str, int],
    line_info: dict[str, list[int]],
    substation_order: list[str] = [],
    idx: int = 0,
) -> tuple[int, str]:
    """
    Selects a sub_id based on the proposed actions and capa logic.

    Args:
        proposed_actions (dict[str, int]): A dictionary mapping sub_ids to their corresponding proposed actions.

    Returns:
        int: The current index.
        str: The selected sub_id.

    """
    # if no list is created yet, do so
    if idx == 0 or not substation_order:
        idx = 0
        substation_order = get_capa_substation_id(
            line_info, gym_obs, controllable_substations
        )

    # find an action that is not the do nothing action by looping over the substations
    chosen_action = 0
    while (not chosen_action) and idx < len(controllable_substations):
        single_substation = substation_order[idx % len(controllable_substations)]
        chosen_action = proposed_actions[str(single_substation)]
        idx += 1

        # if it's not the do nothing action, return action
        # if it's the do nothing action, continue the loop
        if chosen_action:
            return idx, single_substation

    # grid is safe or no action is found, reset list count and return DoNothing
    return 0, "-1"

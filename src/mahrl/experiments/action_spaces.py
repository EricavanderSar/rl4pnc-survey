"""
Implements the codes to configure the three kinds of action spaces.
"""

import json
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv


def get_possible_topologies(
    env: BaseEnv, substations_list: list[int]
) -> list[BaseAction]:
    """
    Function that returns all possible topologies when only keeping in mind a certain number of substations.

    Parameters:
        env (BaseEnv): The environment object.
        substations_list (list[int]): A list of indices representing the substations.

    Returns:
        list[BaseAction]: A list of possible topologies as BaseAction objects.
    """
    possible_substation_actions = []
    for idx in substations_list:
        possible_substation_actions += IdToAct.get_all_unitary_topologies_set(
            env.action_space, idx
        )
    return possible_substation_actions


def get_changeable_substations_tennet(env: BaseEnv) -> list[int]:
    """
    Find all substations that have more than four lines and can therefore be acted upon
    according to the proposed TenneT action space.

    Parameters:
        env (BaseEnv): The environment object.

    Returns:
        list[int]: A list of substation IDs that have more than four lines.
    """
    changeable_substations = []

    # for all substations
    nr_substations = len(env.sub_info)
    for sub in range(nr_substations):
        nr_elements = len(env.observation_space.get_obj_substations(substation_id=sub))
        nr_non_lines = sum(
            1
            for row in env.observation_space.get_obj_substations(substation_id=sub)
            if row[1] != -1 or row[2] != -1
        )
        nr_lines = nr_elements - nr_non_lines

        # append this substation if it has more than four lines
        if nr_lines >= 4:
            changeable_substations.append(sub)

    return changeable_substations


def get_asymmetrical_action_space(env: BaseEnv) -> list[BaseAction]:
    """
    This function returns a list of legal actions for the given environment.

    Parameters:
        env (BaseEnv): The environment for which to generate the action space.

    Returns:
        list[BaseAction]: A list of legal actions.
    """
    legal_actions = []
    # all substations
    possible_substation_actions = get_possible_topologies(
        env, list(range(len(env.sub_info)))
    )

    # Create the DataFrame with defined columns
    dataframe = pd.DataFrame(
        columns=list(
            range(len(possible_substation_actions[0].to_json()["_set_topo_vect"]))
        )
    )

    for action in possible_substation_actions:
        topo_vect = action.to_json()["_set_topo_vect"]
        dataframe.loc[len(dataframe.index)] = topo_vect

    for index, row in dataframe.iterrows():
        # Identify positive entries in the current row
        positive_entries = row[row > 0].index.tolist()

        # Check if the positive entries in the current row have corresponding empty columns in other rows
        meets_condition = all(
            dataframe.loc[dataframe.index != index, positive_entries].eq(0).all(axis=1)
        )

        if not meets_condition:
            legal_actions.append(possible_substation_actions[index])

    return legal_actions


def get_medha_action_space(env: BaseEnv) -> list[BaseAction]:
    """
    Generate allowed actions based on the proposed action space by Subramanian et al. (2021).

    Parameters:
        env (BaseEnv): The environment object.

    Returns:
        list[BaseAction]: A list of legal actions based on the proposed action space.
    """

    # look at all substations (since all substations have more than one element)
    possible_substation_actions = get_possible_topologies(
        env, list(range(len(env.sub_info)))
    )

    legal_actions = []

    # Create the DataFrame with defined columns
    dataframe = pd.DataFrame(
        columns=list(
            range(len(possible_substation_actions[0].to_json()["_set_topo_vect"]))
        )
    )

    for action in possible_substation_actions:
        topo_vect = action.to_json()["_set_topo_vect"]
        # Isolate this substation
        non_zero_elements = [num for num in topo_vect if num != 0]

        # Count occurrences of each non-zero element
        element_counts = Counter(non_zero_elements)

        # Check if there are at least two occurrences of each non-zero element
        at_least_two_occurrences = all(count >= 2 for count in element_counts.values())

        # get_possible_topologies already takes into account that a generator connected to no
        # substation-substation lines does not work, so the constraint to check if one
        # substation-substation line is connected to each busbar does not need to be tested
        if at_least_two_occurrences:
            dataframe.loc[len(dataframe.index)] = topo_vect
        else:  # add dummy to keep index intact
            dataframe.loc[len(dataframe.index)] = [0] * len(
                list(
                    range(
                        len(possible_substation_actions[0].to_json()["_set_topo_vect"])
                    )
                )
            )

    for index, row in dataframe.iterrows():
        # Identify positive entries in the current row
        positive_entries = row[row > 0].index.tolist()

        # Check if the positive entries in the current row have corresponding empty columns in other rows
        meets_condition = all(
            dataframe.loc[dataframe.index != index, positive_entries].eq(0).all(axis=1)
        )

        if not meets_condition:
            legal_actions.append(possible_substation_actions[index])

    return legal_actions


def get_tennet_action_space(env: BaseEnv) -> list[BaseAction]:
    """
    Generate allowed actions based on the proposed action space by Subramanian et al. (2021).

    Parameters:
        env (BaseEnv): The environment for which to generate the action space.

    Returns:
        list[BaseAction]: A list of legal actions based on the proposed action space.
    """
    changeable_substations = get_changeable_substations_tennet(env)

    possible_substation_actions = get_possible_topologies(env, changeable_substations)

    legal_actions = []

    non_lines = np.concatenate(
        (env.action_space.gen_pos_topo_vect, env.action_space.load_pos_topo_vect)
    ).tolist()

    # Create the DataFrame with defined columns
    dataframe = pd.DataFrame(
        columns=list(
            range(len(possible_substation_actions[0].to_json()["_set_topo_vect"]))
        )
    )

    for action in possible_substation_actions:
        topo_vect = action.to_json()["_set_topo_vect"]

        lines_topo_vect = np.delete(topo_vect, non_lines)

        # Filter out non-zero elements
        non_zero_lines = [num for num in lines_topo_vect if num != 0]

        # Count occurrences of each non-zero element
        element_counts = Counter(non_zero_lines)

        # Check if there are at least two occurrences of each non-zero element
        at_least_two_occurrences = all(count >= 2 for count in element_counts.values())
        if at_least_two_occurrences:
            dataframe.loc[len(dataframe.index)] = topo_vect
        else:  # add dummy to keep index intact
            dataframe.loc[len(dataframe.index)] = [0] * len(
                possible_substation_actions[0].to_json()["_set_topo_vect"]
            )

    for index, row in dataframe.iterrows():
        # Identify positive entries in the current row
        positive_entries = row[row > 0].index.tolist()

        # Check if the positive entries in the current row have corresponding empty columns in other rows
        if not all(
            dataframe.loc[dataframe.index != index, positive_entries].eq(0).all(axis=1)
        ):
            legal_actions.append(possible_substation_actions[index])

    return legal_actions


def save_to_json(
    possible_substation_actions: List[BaseAction], json_file_path: str
) -> None:
    """
    Saves list of actions to .json that can be used in training.

    Args:
        possible_substation_actions (List[BaseAction]): List of possible substation actions.
        json_file_path (str): Path to the .json file where the actions will be saved.

    Returns:
        None
    """
    actions_to_json = [
        {
            "set_bus": {
                "loads_id": [
                    [int(elem_id), int(bus_id)]
                    for elem_id, bus_id in enumerate(action.load_set_bus)
                    if bus_id > 0
                ],
                "generators_id": [
                    [int(elem_id), int(bus_id)]
                    for elem_id, bus_id in enumerate(action.gen_set_bus)
                    if bus_id > 0
                ],
                "lines_or_id": [
                    [int(elem_id), int(bus_id)]
                    for elem_id, bus_id in enumerate(action.line_or_set_bus)
                    if bus_id > 0
                ],
                "lines_ex_id": [
                    [int(elem_id), int(bus_id)]
                    for elem_id, bus_id in enumerate(action.line_ex_set_bus)
                    if bus_id > 0
                ],
            }
        }
        for action in possible_substation_actions
    ]
    # Save to json
    with open(json_file_path, "wt", encoding="utf-8") as file:
        json.dump(actions_to_json, file)

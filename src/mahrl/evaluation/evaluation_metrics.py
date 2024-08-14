"""
This describes possible metrics to be used for evaluation of the agent..
"""

import ast
import os
from collections import Counter
from typing import Any, List, Set

import matplotlib.pyplot as plt
import numpy as np
from grid2op.Episode import EpisodeData
from matplotlib import cm


def get_number_of_scenarios(all_episodes: list[EpisodeData]) -> int:
    """
    Returns the number of scenarios in the dataset.

    Parameters:
    - all_episodes (list[EpisodeData]): A list of EpisodeData objects representing the episodes in the dataset.

    Returns:
    - int: The number of scenarios in the dataset.
    """
    return len(all_episodes)


def get_number_of_incomplete_scenarios(all_episodes: list[EpisodeData]) -> int:
    """
    Returns the number of imperfect scenarios in the dataset.

    Parameters:
    - all_episodes (list[EpisodeData]): A list of EpisodeData objects representing the episodes in the dataset.

    Returns:
    - int: The number of imperfect scenarios in the dataset.
    """
    return sum(
        1 for episode in all_episodes if len(episode.actions) != len(episode.times)
    )


def get_max_scenario_length(all_episodes: list[EpisodeData]) -> int:
    """
    Returns the maximum scenario length in the dataset.

    Parameters:
    - all_episodes (list[EpisodeData]): A list of EpisodeData objects representing the dataset.

    Returns:
    - int: The maximum scenario length in the dataset.
    """
    return len(all_episodes[0].times)


def get_mean_scenario_length(all_episodes: list[EpisodeData]) -> float:
    """
    Returns the mean scenario length in the dataset.

    Parameters:
    - all_episodes (list[EpisodeData]): A list of EpisodeData objects representing the episodes in the dataset.

    Returns:
    - float: The mean scenario length.

    """
    return float(np.mean([len(episode.actions) for episode in all_episodes]))


def get_std_scenario_length(all_episodes: list[EpisodeData]) -> float:
    """
    Returns the standard deviation of the scenario length in the dataset.

    Parameters:
    - all_episodes (list[EpisodeData]): A list of EpisodeData objects representing the episodes in the dataset.

    Returns:
    - float: The standard deviation of the scenario length.
    """
    return float(np.std([len(episode.actions) for episode in all_episodes]))


def get_number_of_actions(action_list: list[dict[str, Any]]) -> int:
    """
    Count the number of actions in the dataset.

    Parameters:
    action_list (list[dict[str, Any]]): A list of dictionaries representing actions.

    Returns:
    int: The number of actions in the dataset.
    """
    return len(action_list)


def get_unique_actions(action_list: list[dict[str, Any]]) -> set[str]:
    """
    List the unique actions in the dataset.

    Args:
        action_list (list[dict[str, Any]]): A list of dictionaries representing actions.

    Returns:
        set[str]: A set of unique actions.

    """
    return set(str(action) for action in action_list)


def get_number_of_unique_actions(action_list: list[dict[str, Any]]) -> int:
    """
    Count the number of unique actions in the dataset.

    Parameters:
    action_list (list[dict[str, Any]]): A list of dictionaries representing actions.

    Returns:
    int: The number of unique actions in the dataset.
    """
    return len(get_unique_actions(action_list))


def get_controlled_substations(action_list: list[dict[str, Any]]) -> list[int]:
    """List the controlled substations in the dataset.

    Args:
        action_list (list[dict[str, Any]]): A list of actions.

    Returns:
        list[int]: A list of controlled substations.

    """
    substation_list = []
    for action in action_list:
        if "set_line_status" in action:
            pass
        else:
            substation_list.append(action["set_bus_vect"]["modif_subs_id"][-1])
    return substation_list


def get_unique_controlled_substations(action_list: list[dict[str, Any]]) -> set[int]:
    """
    Returns a set of unique controlled substations in the dataset.

    Parameters:
        action_list (list[dict[str, Any]]): A list of actions.

    Returns:
        set[int]: A set of unique controlled substations.
    """
    substation_list = get_controlled_substations(action_list)
    return set(substation_list)


def get_number_of_controlled_substations(action_list: list[dict[str, Any]]) -> int:
    """Count the number of controlled substations in the dataset.

    Args:
        action_list (list[dict[str, Any]]): A list of actions.

    Returns:
        int: The number of controlled substations in the dataset.
    """
    substation_list = get_controlled_substations(action_list)
    return len(set(substation_list))


def get_unique_topologies(topology_list: list[list[int]]) -> set[str]:
    """
    List the unique topologies in the dataset.

    Parameters:
    - topology_list (list[list[int]]): A list of topologies represented as lists of integers.

    Returns:
    - unique_topologies (set[str]): A set of unique topologies represented as strings.
    """
    return set(str(topology) for topology in topology_list)


def get_number_of_topologies(topology_list: list[list[int]]) -> int:
    """
    Count the number of topologies in the dataset.

    Parameters:
    - topology_list (list[list[int]]): A list of topologies represented as lists of integers.

    Returns:
    - int: The number of topologies in the dataset.
    """
    return len(topology_list)


def get_number_of_unique_topologies(topology_list: list[list[int]]) -> int:
    """
    Count the number of unique topologies in the dataset.

    Parameters:
    - topology_list (list[list[int]]): A list of topologies represented as lists of integers.

    Returns:
    - int: The number of unique topologies in the dataset.
    """
    return len(get_unique_topologies(topology_list))


def get_topological_depth(
    topology_list: list[list[int]], grid_objects_types: list[list[int]]
) -> list[int]:
    """List the topological depth of each action.

    Args:
        topology_list (list[list[int]]): A list of topology vectors representing the connections between elements.
        grid_objects_types (list[list[int]]): A list of grid object types for each element.

    Returns:
        list[int]: A list of topological depths for each action.

    """
    summed_depth = []
    for topo_vect in topology_list:
        # get the indices of the elements that are connected to busbar 2
        indices = [i for i, x in enumerate(topo_vect) if x == 2]

        # get the substations of these elements
        substations = [grid_objects_types[i][0] for i in indices]

        # count the number of unique substations
        num_substations = len(set(substations))

        summed_depth.append(num_substations)
    return summed_depth


def get_max_topological_depth(
    topology_list: list[list[int]], grid_objects_types: list[list[int]]
) -> int:
    """
    Return the maximum topological depth.

    Parameters:
    - topology_list (list[list[int]]): A list of lists representing the topology.
    - grid_objects_types (list[list[int]]): A list of lists representing the types of grid objects.

    Returns:
    - int: The maximum topological depth.

    """
    return int(np.max(get_topological_depth(topology_list, grid_objects_types)))


def get_number_of_action_sequences(action_sequences: list[list[dict[str, Any]]]) -> int:
    """
    Count the number of action sequences in the dataset.

    Parameters:
    - action_sequences (list[list[dict[str, Any]]]): A list of action sequences.

    Returns:
    - int: The number of action sequences in the dataset.
    """
    return len(action_sequences)


def get_number_of_unique_action_sequences(
    action_sequences: list[list[dict[str, Any]]]
) -> int:
    """
    Count the number of unique action sequences in the dataset.

    Parameters:
        action_sequences (list[list[dict[str, Any]]]): A list of action sequences.

    Returns:
        int: The number of unique action sequences.
    """
    unique_action_sequences = set(
        str(action_sequence) for action_sequence in action_sequences
    )
    return len(unique_action_sequences)


def get_max_action_sequence_length(action_sequences: list[list[dict[str, Any]]]) -> int:
    """
    Return the maximum length of action sequences.

    Parameters:
    action_sequences (list[list[dict[str, Any]]]): A list of action sequences.

    Returns:
    int: The maximum length of action sequences.
    """
    if len(action_sequences) != 0:
        return int(
            np.max([len(action_sequence) for action_sequence in action_sequences])
        )
    return 0


def get_substation_counts(
    all_substations: range, substation_list: List[int]
) -> List[int]:
    """
    Get the counts of each substation action in the dataset.

    Parameters:
        all_substations (range): A range object representing all possible substation actions.
        substation_list (List[int]): A list of substation actions.

    Returns:
        List[int]: A list containing the counts of each substation action in the dataset.
    """
    substation_list = [int(substation) for substation in substation_list]
    substation_counter = Counter(substation_list)
    substation_counts = [
        substation_counter[substation] for substation in all_substations
    ]
    return substation_counts


def get_action_counts(
    unique_actions: Set[str], action_list: List[dict[str, Any]]
) -> List[int]:
    """
    Get the counts of each action in the dataset.

    Parameters:
        unique_actions (set[str]): A set of unique actions.
        action_list (list[dict[str, Any]]): A list of dictionaries representing actions.

    Returns:
        list[int]: A list of counts for each action in the dataset.
    """
    dict_counter = Counter(map(str, action_list))
    dict_counts = {
        str(dict_value): dict_counter[str(dict_value)] for dict_value in unique_actions
    }
    return list(map(int, dict_counts.values()))


def plot_substation_distribution(
    path: str, all_substations: range, substation_list: list[int]
) -> None:
    """
    Plot the distribution of controlled substations.

    Parameters:
        path (str): The path where the plot will be saved.
        all_substations (range): The range of all substations.
        substation_list (list[int]): The list of controlled substations.

    Returns:
        None
    """
    substation_counts = get_substation_counts(all_substations, substation_list)
    plt.bar(all_substations, substation_counts)
    plt.xlabel("Substations")
    plt.xticks(all_substations)
    plt.ylabel("Count")
    plt.savefig(os.path.join(path, "substation_distribution.png"))
    plt.close()


def get_colours_for_substations(
    modif_subs_ids: list[list[str]],
) -> tuple[list[list[float]], dict[int, Any]]:
    """
    Colour coordinates the actions per substation.

    Args:
        modif_subs_ids (list[list[str]]): A list of lists containing the modified substation IDs.

    Returns:
        tuple[list[list[float]], dict[int, Any]]: A tuple containing the list of color coordinates for each substation
        action and a dictionary mapping values to colors.
    """
    # flatten the list and convert to integers
    values = list(map(int, [item for sublist in modif_subs_ids for item in sublist]))

    # create a list of unique values
    unique_values = list(set(values))

    # create a colormap
    # pylint: disable=no-member
    colors = cm.rainbow(np.linspace(0, 1, len(unique_values)))  # type: ignore[attr-defined]

    # create a dictionary that maps values to colors
    color_dict = dict(zip(unique_values, colors))

    # create a list of colors for your values
    color_list = [color_dict[value] for value in values]
    return color_list, color_dict


def plot_action_distribution(path: str, action_list: list[dict[str, Any]]) -> None:
    """
    Plot the distribution of actions.

    Args:
        path (str): The path to save the plot.
        action_list (list[dict[str, Any]]): A list of dictionaries representing actions.

    Returns:
        None
    """
    unique_actions_list = get_unique_actions(action_list)
    action_counts = get_action_counts(unique_actions_list, action_list)
    action_dicts = [
        ast.literal_eval(action)
        for action in unique_actions_list
        if "set_line_status" not in action
    ]
    modif_subs_ids = [
        action["set_bus_vect"]["modif_subs_id"]
        for action in action_dicts
        if "set_line_status" not in action
    ]

    # Sort action_counts and unique_actions_list based on action_counts
    action_counts, unique_actions_list, modif_subs_ids = zip(
        *sorted(zip(action_counts, unique_actions_list, modif_subs_ids), reverse=True)
    )

    color_list, color_dict = get_colours_for_substations(modif_subs_ids)
    plt.bar(range(len(unique_actions_list)), action_counts, color=color_list)
    plt.xlabel("Unique actions")
    plt.xticks([])  # Remove x-axis labels
    plt.ylabel("Count")

    # Create a custom legend
    sorted_keys = sorted(color_dict.keys(), key=int)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_dict[label]) for label in sorted_keys
    ]
    plt.legend(handles, sorted_keys)

    plt.savefig(os.path.join(path, "action_distribution.png"))
    plt.close()

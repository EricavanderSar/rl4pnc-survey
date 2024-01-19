"""
This describes possible metrics to be used for evaluation of the agent..
"""
import os
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from grid2op.Episode import EpisodeData


def get_number_of_scenarios(all_episodes: list[EpisodeData]) -> int:
    """
    Returns the number of scenarios in the dataset.
    """
    return len(all_episodes)


def get_number_of_incomplete_scenarios(all_episodes: list[EpisodeData]) -> int:
    """
    Returns the number of imperfect scenarios in the dataset.
    """
    return sum(
        1 for episode in all_episodes if len(episode.actions) != len(episode.times)
    )


def get_max_scenario_length(all_episodes: list[EpisodeData]) -> int:
    """
    Returns the maximum scenario length in the dataset.
    """
    return len(all_episodes[0].times)


def get_mean_scenario_length(all_episodes: list[EpisodeData]) -> float:
    """
    Returns the mean scenario length in the dataset.
    """
    return float(np.mean([len(episode.actions) for episode in all_episodes]))


def get_std_scenario_length(all_episodes: list[EpisodeData]) -> float:
    """
    Returns the standard deviation of the scenario length in the dataset.
    """
    return float(np.std([len(episode.actions) for episode in all_episodes]))


def get_number_of_actions(action_list: list[dict[str, Any]]) -> int:
    """Count the number of actions in the dataset"""
    return len(action_list)


def get_unique_actions(action_list: list[dict[str, Any]]) -> set[str]:
    """List the unique actions in the dataset"""
    return set(str(action) for action in action_list)


def get_number_of_unique_actions(action_list: list[dict[str, Any]]) -> int:
    """Count the number of unique actions in the dataset"""
    return len(get_unique_actions(action_list))


def get_controlled_substations(action_list: list[dict[str, Any]]) -> list[int]:
    """List the controlled substations in the dataset"""
    substation_list = []
    for action in action_list:
        substation_list.append(action["set_bus_vect"]["modif_subs_id"][0])
    return substation_list


def get_unique_controlled_substations(action_list: list[dict[str, Any]]) -> set[int]:
    """List the unique controlled substations in the dataset"""
    substation_list = get_controlled_substations(action_list)
    return set(substation_list)


def get_number_of_controlled_substations(action_list: list[dict[str, Any]]) -> int:
    """Count the number of controlled substations in the dataset"""
    substation_list = get_controlled_substations(action_list)
    return len(set(substation_list))


def get_unique_topologies(topology_list: list[list[int]]) -> set[str]:
    """List the unique topologies in the dataset"""
    return set(str(topology) for topology in topology_list)


def get_number_of_topologies(topology_list: list[list[int]]) -> int:
    """Count the number of topologies in the dataset"""
    return len(topology_list)


def get_number_of_unique_topologies(topology_list: list[list[int]]) -> int:
    """Count the number of unique topologies in the dataset"""
    return len(get_unique_topologies(topology_list))


def get_topological_depth(topology_list: list[list[int]]) -> list[int]:
    """List the topological depth of the each action"""
    # TODO revisit definition
    default = topology_list[0]
    summed_depth = []

    for topo in topology_list:
        topo_diff = np.array(topo) - np.array(default)
        topo_no_disconnection = np.where(topo_diff > 0, topo_diff, 0)

        # print(topo_no_disconnection)
        summed_depth.append(np.sum(topo_no_disconnection))

    return summed_depth


def get_max_topological_depth(topology_list: list[list[int]]) -> int:
    """Return the maximum topological."""
    return int(np.max(get_topological_depth(topology_list)))


def get_mean_topological_depth(topology_list: list[list[int]]) -> float:
    """Return the mean topological depth."""
    # TODO, not actually correct because it doesn't take into account how long the topology is held
    return float(np.mean(get_topological_depth(topology_list)))


def get_number_of_action_sequences(action_sequences: list[list[dict[str, Any]]]) -> int:
    """Count the number of action sequences in the dataset"""
    return len(action_sequences)


def get_number_of_unique_action_sequences(
    action_sequences: list[list[dict[str, Any]]]
) -> int:
    """Count the number of unique action sequences in the dataset"""
    unique_action_sequences = set(
        str(action_sequence) for action_sequence in action_sequences
    )
    return len(unique_action_sequences)


def get_max_action_sequence_length(action_sequences: list[list[dict[str, Any]]]) -> int:
    """Return the maximum action sequence length."""
    return int(np.max([len(action_sequence) for action_sequence in action_sequences]))


def get_substation_counts(
    all_substations: range, substation_list: list[int]
) -> list[int]:
    """
    Get the counts of each substation action in the dataset.
    """
    substation_list = [int(substation) for substation in substation_list]
    substation_counter = Counter(substation_list)
    substation_counts = [
        substation_counter[substation] for substation in all_substations
    ]
    return substation_counts


def get_action_counts(
    unique_actions: set[str], action_list: list[dict[str, Any]]
) -> list[int]:
    """ "
    Get the counts of each action in the dataset.
    """
    dict_counter = Counter(map(str, action_list))
    dict_counts = {
        str(dict_value): dict_counter[str(dict_value)] for dict_value in unique_actions
    }
    return list(map(int, dict_counts.values()))


def plot_substation_distribution(
    path: str, all_substations: range, substation_list: list[int]
) -> None:
    """Plot the distribution of controlled substations."""
    plt.bar(all_substations, get_substation_counts(all_substations, substation_list))
    plt.xlabel("Substations")
    plt.xticks(all_substations)
    plt.ylabel("Count")
    plt.savefig(os.path.join(path, "substation_distribution.png"))
    plt.close()


def plot_action_distribution(path: str, action_list: list[dict[str, Any]]) -> None:
    """Plot the distribution of actions."""
    unique_actions_list = get_unique_actions(action_list)
    action_counts = get_action_counts(unique_actions_list, action_list)
    plt.bar(list(unique_actions_list), action_counts)
    plt.xlabel("Unique actions")
    plt.xticks([])  # Remove x-axis labels
    plt.ylabel("Count")
    plt.savefig(os.path.join(path, "action_distribution.png"))
    plt.close()

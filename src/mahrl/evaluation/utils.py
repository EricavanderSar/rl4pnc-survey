"""
Utils for evaluation of agents.
"""
import os
from typing import Any

from grid2op.Episode import EpisodeData

from mahrl.evaluation import evaluation_metrics


def load_episodes(path: str) -> list[EpisodeData]:
    """
    Loads all evaluated episodes from a path.
    """
    li_episode = EpisodeData.list_episode(path)

    all_episodes = []
    for full_path, episode_studied in li_episode:
        this_episode = EpisodeData.from_disk(full_path, episode_studied)
        all_episodes.append(this_episode)

    return all_episodes


def save_global_statistics(
    path: str,
    all_episodes: list[EpisodeData],
    global_action_sequences: list[list[list[dict[str, Any]]]],
    global_topology_list: list[list[list[int]]],
) -> None:
    """
    Save global statistics to a file.
    """
    # flatten multi-dimensional lists
    all_action_sequences = [
        item for sublist in global_action_sequences for item in sublist
    ]
    all_topologies = [item for sublist in global_topology_list for item in sublist]
    all_actions = [item for sublist in all_action_sequences for item in sublist]

    with open(
        os.path.join(path, "global_statistics.txt"), "w", encoding="utf-8"
    ) as file:
        # - total number of chronics
        file.write(
            f"Number of scenarios: {evaluation_metrics.get_number_of_scenarios(all_episodes)}\n"
        )  # - number of imperfect chronics
        file.write(
            f"Number of incomplete scenarios: {evaluation_metrics.get_number_of_incomplete_scenarios(all_episodes)}\n"
        )  # - max scenario length,
        file.write(
            f"Max scenario length: {evaluation_metrics.get_max_scenario_length(all_episodes)}\n"
        )  # - mean scenario length,
        file.write(
            f"Mean scenario length: {evaluation_metrics.get_mean_scenario_length(all_episodes)}\n"
        )  # - std scenario length,
        file.write(
            f"Std scenario length: {evaluation_metrics.get_std_scenario_length(all_episodes)}\n"
        )
        # - topologies
        file.write(
            f"Number of topologies: {evaluation_metrics.get_number_of_topologies(all_topologies)}\n"
        )  # - unique topologies
        file.write(
            f"Number of unique topologies: {evaluation_metrics.get_number_of_unique_topologies(all_topologies)}\n"
        )  # - number of action sequences
        file.write(
            f"Number of action sequences: {evaluation_metrics.get_number_of_action_sequences(all_action_sequences)}\n"
        )  # - number of unique action sequences
        file.write(
            f"Number of unique action sequences: \
                {evaluation_metrics.get_number_of_unique_action_sequences(all_action_sequences)}\n"
        )  # - largest action sequence
        file.write(
            f"Max action sequence length: {evaluation_metrics.get_max_action_sequence_length(all_action_sequences)}\n"
        )
    all_substations = range(len(all_episodes[0].name_sub))
    evaluation_metrics.plot_substation_distribution(
        path,
        all_substations,
        evaluation_metrics.get_controlled_substations(all_actions),
    )

    evaluation_metrics.plot_action_distribution(path, all_actions)


def save_scenario_statistics(
    path: str,
    episode_name: str,
    action_sequences: list[list[dict[str, Any]]],
    topology_list: list[list[int]],
) -> None:
    """
    Save scenario-wise statistics to a file.
    """

    # flatten action sequences
    action_list = [item for sublist in action_sequences for item in sublist]

    with open(
        os.path.join(path, "scenario_statistics.txt"), "w", encoding="utf-8"
    ) as file:
        # - topologies
        file.write(f"Episode: {episode_name}\n")
        file.write(
            f"Number of topologies: {evaluation_metrics.get_number_of_topologies(topology_list)}\n"
        )
        # - unique topologies
        file.write(
            f"Number of unique topologies: {evaluation_metrics.get_number_of_unique_topologies(topology_list)}\n"
        )
        # - max topology depth
        file.write(
            f"Max topological depth: {evaluation_metrics.get_max_topological_depth(topology_list)}\n"
        )
        # - number of actions
        file.write(
            f"Number of actions: {evaluation_metrics.get_number_of_actions(action_list)}\n"
        )
        # - number of unique actions
        file.write(
            f"Number of unique actions: {evaluation_metrics.get_number_of_unique_actions(action_list)}\n"
        )
        # - number of substations with actions
        file.write(
            f"Number of substations: {evaluation_metrics.get_number_of_controlled_substations(action_list)}\n"
        )
        # - number of action sequences
        file.write(
            f"Number of action sequences: {evaluation_metrics.get_number_of_action_sequences(action_sequences)}\n"
        )
        # - number of unique action sequences
        file.write(
            f"Number of unique action sequences: \
                {evaluation_metrics.get_number_of_unique_action_sequences(action_sequences)}\n"
        )
        # - largest action sequence
        file.write(
            f"Max action sequence length: {evaluation_metrics.get_max_action_sequence_length(action_sequences)}\n"
        )

"""
Script that runs analysis on the behaviour of the agents.
Has to be run after the evaluation of agents is run.
"""
import argparse
import logging

import numpy as np
from grid2op.Episode import EpisodeData

from mahrl.evaluation.utils import (
    load_episodes,
    save_global_statistics,
    save_scenario_statistics,
)


def run_statistics(path: str, episode_list: list[EpisodeData]) -> None:
    """
    Calls the scenario and global statistics for the evaluation.
    """
    global_action_sequences = []
    global_topology_list = []
    grid_objects_types = episode_list[0].observations[0].grid_objects_types
    for episode in episode_list:
        action_sequences = []
        current_sequence = []
        topology_list = [episode.observations[0].topo_vect]

        for idx, _ in enumerate(episode.actions):
            # print(f"Observation before action: {episode.observations[idx].topo_vect}")
            # print(f"Action: {episode.actions[idx]}")
            # print(f"Observation after action: {episode.observations[idx+1].topo_vect}")

            implicit_do_nothing = np.array_equal(
                episode.observations[idx].topo_vect,
                episode.observations[idx + 1].topo_vect,
            )

            if implicit_do_nothing and episode.actions[idx].as_dict() != {}:
                print("IMPLICIT DO NOTHING CAUGHT")

            if episode.actions[idx].as_dict() != {} and not implicit_do_nothing:
                # print(f"Played Action: {episode.actions[idx].as_dict()}")
                topology_list.append(episode.observations[idx + 1].topo_vect)
                current_sequence.append(episode.actions[idx].as_dict())
            else:
                if current_sequence:
                    action_sequences.append(current_sequence)
                    current_sequence = []

        if (
            current_sequence
        ):  # if there is a current sequence after iterating over all actions
            action_sequences.append(current_sequence)

        # If scenario is not completed, exclude the final topology
        if np.all(topology_list[-1] == -1):
            topology_list = topology_list[:-1]

        global_action_sequences.append(action_sequences)
        global_topology_list.append(topology_list)
        save_scenario_statistics(
            path, episode.name, grid_objects_types, action_sequences, topology_list
        )

    save_global_statistics(
        path,
        episode_list,
        grid_objects_types,
        global_action_sequences,
        global_topology_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    # Define command-line arguments for two possibilities: greedy and rllib model
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        help="Path to the scenarios to be evaluated.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    input_file_path = args.file_path

    if not input_file_path:
        parser.print_help()
        logging.error("\nError: --file_path is required.")
    else:
        all_episodes = load_episodes(input_file_path)
        run_statistics(input_file_path, all_episodes)

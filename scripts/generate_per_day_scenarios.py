"""
This script splits up scenarios into individual or multiple days.
"""

import argparse
import logging
import math
import os
import shutil
from typing import Iterator

import grid2op
from grid2op.Environment import BaseEnv
from lightsim2grid import LightSimBackend

from mahrl.experiments.yaml import load_config

RHO_THRESHOLD = 1
LENGTH_DAY = 288
TIME = 3  # reset point (3AM)


def check_safe_starting_point(env: BaseEnv, scenario_id: int, tsteps: int) -> bool:
    """
    Checks if the starting point of the environment is safe.

    Args:
        env (BaseEnv): The environment object.
        scenario_id (int): The ID of the scenario.
        day (int): The day of the scenario.

    Returns:
        bool: True if the starting point is safe, False otherwise.
    """
    # setup scenario
    env.set_id(scenario_id)
    env.reset()

    # fast forward to current point
    env.fast_forward_chronics(tsteps)
    obs, _, _, _ = env.step(env.action_space())

    # check if the starting point is safe
    if max(obs.to_dict()["rho"]) > RHO_THRESHOLD:
        return False
    return True


def flatten_directory(root_dir: str) -> None:
    """
    Recursively flattens a directory structure by moving all files from subdirectories to the root directory.

    Args:
        root_dir (str): The root directory to flatten.

    Returns:
        None
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                os.remove(os.path.join(dirpath, filename))
        for dirname in dirnames:
            sub_dir = os.path.join(dirpath, dirname)
            for filename in os.listdir(sub_dir):
                shutil.move(os.path.join(sub_dir, filename), dirpath)
            os.rmdir(sub_dir)


def save_chronic(
    env: BaseEnv,
    save_path: str,
    dict_beg: dict[str, str],
    dict_end: dict[str, str],
    day: int,
    scenario_name: str,
) -> None:
    """
    Save chronic data for a specific day and scenario.

    Args:
        env (BaseEnv): The Grid2Op environment.
        tsteps (int): The number of time steps.
        save_path (str): The path to save the chronic data.
        dict_beg (dict): The dictionary containing the beginning of the chronic data.
        dict_end (dict): The dictionary containing the end of the chronic data.
        day (int): The day for which to save the chronic data.
        scenario_name (str): The name of the scenario.

    Raises:
        Exception: If an error occurs while saving the chronic data.

    Returns:
        None
    """
    # create folder if necessary
    file_path = os.path.join(
        save_path,
        f"{env.env_name}/generated_chronics/",
        scenario_name,
    )
    if not os.path.exists(os.path.abspath(file_path)):
        os.makedirs(os.path.abspath(file_path))

    # generate splits
    try:
        env.chronics_handler.real_data.split_and_save(
            dict_beg,
            dict_end,
            path_out=os.path.abspath(file_path),
        )
        flatten_directory(file_path)
    except Exception as exc:
        raise AssertionError(
            f"Not complete. Crashed with day {day} chronics {dict_beg} {dict_end}"
        ) from exc


def generate_split_points(
    env: BaseEnv, delta: int
) -> tuple[dict[str, list[int]], Iterator[str], dict[str, list[int]]]:
    """
    Generate splitting points for scenarios based on the given environment and delta.

    Args:
        env (BaseEnv): The environment object.
        delta (int): The time interval between splitting points.

    Returns:
        tuple[dict[str, list[int]], Iterable[str]]: A tuple containing the splitting points
        for each scenario and an iterable of formatted scenario IDs.
    """
    # determine environment specific parameters
    episode_duration = env.chronics_handler.real_data.max_timestep()
    total_number_of_days = math.ceil(
        episode_duration / (LENGTH_DAY * delta)
    )  # round up in case (LENGTH_DAY * delta) does not divide episode_duration evenly

    splitting_points: dict[str, list[int]] = {}
    reduction_in_days: dict[str, list[int]] = {}
    # loop over all scenarios to create a scenario for each day
    for scenario_id in env.chronics_handler.subpaths:
        reduction_in_days[scenario_id] = []
        splitting_points[scenario_id] = []
        if check_safe_starting_point(env, scenario_id, 0 + int(TIME * 60 / 5) - 1):
            # _, _, _, _ = env.step(env.action_space())
            splitting_points[scenario_id].append(int(TIME * 60 / 5) - 1)
        else:
            print(f"Day 0 not safe for {scenario_id}, appending anyways.")
            # _, _, _, _ = env.step(env.action_space())
            splitting_points[scenario_id].append(int(TIME * 60 / 5) - 1)

            # # no safe starting point found at day 0, skip day
            # not_safe = True
            # counter = 1
            # while not_safe:
            #     print(f"Day 0 not safe for {scenario_id}, trying day {counter}")
            #     reduction_in_days[scenario_id].append(True)

            #     if check_safe_starting_point(
            #         env, scenario_id, (LENGTH_DAY * counter) + int(TIME * 60 / 5) - 1
            #     ):
            #         _, _, _, _ = env.step(env.action_space())
            #         splitting_points[scenario_id].append(int(TIME * 60 / 5) - 1)
            #         not_safe = False

            # raise AssertionError(
            #     f"Safe starting point not found for {scenario_id} at day 0"
            # )

        for tsteps in range(LENGTH_DAY * delta, episode_duration, LENGTH_DAY * delta):
            number_of_days = int(tsteps / (LENGTH_DAY * delta))

            if number_of_days < total_number_of_days:
                total_tsteps = tsteps + int(TIME * 60 / 5) - 1
            else:  # for the last day, go from midnight to midnight
                total_tsteps = tsteps

            if check_safe_starting_point(env, scenario_id, total_tsteps):
                # _, _, _, _ = env.step(env.action_space())
                splitting_points[scenario_id].append(total_tsteps)
            else:
                reduction_in_days[scenario_id].append(True)
                continue

        # append final starting/ending point
        splitting_points[scenario_id].append(episode_duration)

    # flatten 2d list starting_points to find total number of scenarios
    total_number_of_generated_scenarios = 0
    for _, points in splitting_points.items():
        total_number_of_generated_scenarios += len(points) - 1

    formatted_scenario_ids_iter = iter(
        [
            f"{id:0{len(str(total_number_of_generated_scenarios))}d}"
            for id in range(total_number_of_generated_scenarios)
        ]
    )
    return splitting_points, formatted_scenario_ids_iter, reduction_in_days


def split_chronics_into_days(env: BaseEnv, save_path: str, delta: int) -> None:
    """
    Splits the chronics into individual days for a given number of input days.

    Args:
        env (BaseEnv): The Grid2Op environment.
        input_days (int): The number of days to split the chronics into.

    Returns:
        None
    """
    (
        splitting_points,
        formatted_scenario_ids_iter,
        reduction_in_days,
    ) = generate_split_points(env, delta)

    # print(reduction_in_days)
    for scenario_id, list_with_points in splitting_points.items():
        env.set_id(scenario_id)
        _ = env.reset()
        env.fast_forward_chronics(list_with_points[0])
        previous_obs, _, _, _ = env.step(env.action_space())

        for number_of_days, tsteps in enumerate(list_with_points[1:]):
            # go to the next day
            env.set_id(scenario_id)
            _ = env.reset()
            env.fast_forward_chronics(tsteps - 1)

            if (number_of_days + len(reduction_in_days[scenario_id])) < (
                math.ceil(  # round up in case (LENGTH_DAY * delta) does not divide episode_duration evenly
                    env.chronics_handler.real_data.max_timestep() / (LENGTH_DAY * delta)
                )
                - 1
            ):  # should be less than total number of days,
                time_str = f"0{TIME}:00"
                # # set at the specified starting night time
                # env.fast_forward_chronics(TIME * 60 / 5)
            else:  # for the last day, go from midnight to midnight
                time_str = "00:00"

            # get observation for next day (finishpoint)
            obs, _, _, _ = env.step(env.action_space())

            dict_beg = {
                os.path.basename(
                    scenario_id
                ): f"{previous_obs.year}-{previous_obs.month}-{previous_obs.day} {time_str}"
            }
            dict_end = {
                os.path.basename(
                    scenario_id
                ): f"{obs.year}-{obs.month}-{obs.day} {time_str}"
            }

            previous_obs = obs

            save_chronic(
                env,
                save_path,
                dict_beg,
                dict_end,
                number_of_days,
                next(formatted_scenario_ids_iter),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the config file.",
    )

    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default="/Users/barberademol/data_grid2op/",
        help="Path for scenarios to be saved",
    )

    # NOTE: Selecting a num_days that does not exactly splits up the day leads to the last day having another length
    parser.add_argument(
        "-d",
        "--num_days",
        type=int,
        default=1,
        help="The number of days that should be contained in a single scenario.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    input_config = args.config
    input_save_path = args.save_path
    input_days = args.num_days

    if input_config:
        custom_config = load_config(input_config)
        setup_env = grid2op.make(
            custom_config["environment"]["env_config"]["env_name"],
            **custom_config["environment"]["env_config"]["grid2op_kwargs"],
            backend=LightSimBackend(),
        )
        setup_env.seed(custom_config["environment"]["env_config"]["seed"])
        # generate_split_points(setup_env, input_days)

        split_chronics_into_days(setup_env, input_save_path, input_days)
    else:
        parser.print_help()
        logging.error("\nError: --config is required to specify config location.")

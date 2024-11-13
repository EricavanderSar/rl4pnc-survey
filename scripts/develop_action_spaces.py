"""
Script to define different action spaces.
"""

import argparse
import os

import grid2op
from lightsim2grid import LightSimBackend

from rl4pnc.experiments.action_spaces import (
    get_action_space,
    get_space_numpy,
    save_to_json,
)


def create_action_spaces(
        env_name: str,
        action_space_to_create: str,
        save_path: str,
        kwargs,
) -> None:
    """
    Creates action spaces for a specified grid2op environment.
    """
    env = grid2op.make(env_name, backend=LightSimBackend())
    env.chronics_handler.set_chunk_size(100)
    save_path = os.path.join(save_path, env_name.replace("_large", "").replace("_small", ""))
    os.makedirs(save_path, exist_ok=True)
    if action_space_to_create in ["binbinchen", "curriculumagent", "alphazero"]:
        possible_actions = get_space_numpy(env,
                                           action_space_to_create,
                                           path=save_path,
                                           **kwargs
                                           )
    else:
        possible_actions = get_action_space(env,
                                            action_space_to_create,
                                            **kwargs
                                            )
    name = action_space_to_create
    if kwargs.get("extra_donothing", False):
        name += f"_dn"
    adjust_shunt = kwargs.get("adjust_shunt", "")
    if adjust_shunt:
        name += f"_{adjust_shunt}shunt"
    rho_filter = kwargs.get("rho_filter", 2.0)
    if rho_filter < 2.0:
        name += f"_maxrho{rho_filter}"

    if kwargs.get("greedy_filter", False):
        pool_size = kwargs.get("act_pool_size", 0)
        nb_act = kwargs.get("nb_interact", 0)
        name += f"-{pool_size}-{nb_act}"
        # two action spaces will be returned.
        # One filtered based on the rho value and One filtered based on the greedy action selection for the pool
        greedy_actions, rho_actions = possible_actions
        print(f"\nActions will be saved at {save_path}/{name}.json")
        file_path = os.path.join(save_path, f"{name}.json")
        save_to_json(rho_actions, file_path)

        name += f"_gr"
        print(f"\nActions will be saved at {save_path}/{name}.json")
        file_path = os.path.join(save_path, f"{name}.json")
        save_to_json(greedy_actions, file_path)
    else:
        print(f"Action space of size {len(possible_actions)} created. "
              f"\nActions will be saved at {save_path}/{name}.json")
        file_path = os.path.join(save_path, f"{name}.json")
        save_to_json(possible_actions, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-w",
        "--nb_workers",
        default=8,
        type=int,
        help="Number of workers used to reduce the action space.",
    )
    parser.add_argument(
        "-e",
        "--environment",
        default="l2rpn_case14_sandbox",# "l2rpn_wcci_2022", # "l2rpn_neurips_2020_track1_small", #"l2rpn_icaps_2021_small", #
        type=str,
        help="Name of the environment to be used.",
    )
    parser.add_argument(
        "-a",
        "--action_space",
        type=str,
        help="Action space to be used.",
        default="assym",
        choices=["assym", "medha", "tennet", "binbinchen", "curriculumagent", "alphazero"]
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default="../data/action_spaces/",
        help="Path the action spaces must be saved.",
    )
    # Extra options to adjust the action_space
    parser.add_argument('-dn', '--extra_donothing', default=False, action='store_true',
                        help="adding extra do nothing actions for subs that dont have any other config to action space"
                        )
    parser.add_argument('-sh', "--adjust_shunt", type=str, default="", choices=["", "all", "opt"],
                        help="For subs with shunt the reversed action can be better."
                             "options: - all will add also reversed actions to action space"
                             "         - opt will pick the best action reversed or normal"
                        )
    parser.add_argument('-rf', "--rho_filter",  type=float, default=2.0,
                        help="Filter all actions with rho value larger than -rf. If >=2.0 no filtering is applied."
                        )
    parser.add_argument('-g', '--greedy_filter', default=False, action='store_true',
                        help="apply a greedy agent to select the best actions"
                        )
    parser.add_argument('-ps', "--act_pool_size", type=int, default=30,
                        help="mutiple greedy agents run in parallel. "
                             "Each greedy agent gets its own action pool of this size"
                        )
    parser.add_argument('-i', "--nb_interact", type=int, default=100,
                        help="number of interaction after which the greedy actions are filtered."
                        )
    args = parser.parse_args()

    input_environment = args.environment
    input_action_space = args.action_space
    input_save_path = args.save_path

    create_action_spaces(input_environment,
                         input_action_space,
                         input_save_path,
                         vars(args)
                         )

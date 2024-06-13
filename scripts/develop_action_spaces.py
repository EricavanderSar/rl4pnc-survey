"""
This script develops three variants of actions spaces (asymmetrical, based on medha and
based on TenneT) for a specified grid2op environment.
"""

import argparse
import os

import grid2op

from mahrl.experiments.action_spaces import (
    get_action_space,
    get_space_numpy,
    save_to_json,
)


def create_action_spaces(
        env_name: str,
        action_space_to_create: str,
        save_path: str,
        extra_dn: bool = False,
        adjust_shunt: str = "",
        rho_filter: float = 2.0,
        workers: int = 8,
) -> None:
    """
    Creates action spaces for a specified grid2op environment.
    """
    env = grid2op.make(env_name)
    save_path = os.path.join(save_path, env_name.replace("_large", "").replace("_small", ""))
    os.makedirs(save_path, exist_ok=True)
    if action_space_to_create in ["binbinchen", "curriculumagent", "alphazero"]:
        possible_actions = get_space_numpy(env,
                                           action_space_to_create,
                                           path=save_path,
                                           incl_dn=extra_dn,
                                           rho_filter=rho_filter,
                                           workers=workers,
                                           )
    else:
        possible_actions = get_action_space(env,
                                            action_space_to_create,
                                            incl_dn=extra_dn,
                                            adjust_shunt=adjust_shunt,
                                            rho_filter=rho_filter,
                                            workers=workers,
                                            )
    name = action_space_to_create
    if extra_dn:
        name += f"_dn"
    if adjust_shunt:
        name += f"_{adjust_shunt}shunt"
    if rho_filter < 2.0:
        name += f"_maxrho{rho_filter}"
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
        default="l2rpn_wcci_2022", # "l2rpn_neurips_2020_track1_small", #"l2rpn_icaps_2021_small", #"l2rpn_case14_sandbox",
        type=str,
        help="Name of the environment to be used.",
    )
    parser.add_argument(
        "-a",
        "--action_space",
        type=str,
        help="Action space to be used.",
        default="alphazero",
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

    args = parser.parse_args()

    input_environment = args.environment
    input_action_space = args.action_space
    input_save_path = args.save_path
    extra_dn = args.extra_donothing
    adj_shunt = args.adjust_shunt
    rho_filter = args.rho_filter
    nb_workers = args.nb_workers

    create_action_spaces(input_environment,
                         input_action_space,
                         input_save_path,
                         extra_dn,
                         adj_shunt,
                         rho_filter=rho_filter,
                         workers=nb_workers
                         )

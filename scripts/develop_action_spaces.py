"""
This script develops three variants of actions spaces (asymmetrical, based on medha and
based on TenneT) for a specified grid2op environment.
"""

import argparse
import os

import grid2op

from mahrl.experiments.action_spaces import (
    get_asymmetrical_action_space,
    get_medha_action_space,
    get_tennet_action_space,
    get_medha_dn_action_space,
    save_to_json,
)
from mahrl.experiments.utils import (
    calculate_action_space_asymmetry,
    calculate_action_space_medha,
    calculate_action_space_tennet,
)


def create_action_spaces(
    env_name: str, action_spaces_to_create: list[str], save_path: str
) -> None:
    """
    Creates action spaces for a specified grid2op environment.
    """
    env = grid2op.make(env_name)
    save_path = os.path.join(save_path, env_name)
    os.makedirs(save_path, exist_ok=True)
    if "asymmetry" in action_spaces_to_create:
        mathematically_possible_actions, _, _ = calculate_action_space_asymmetry(env)
        possible_actions = get_asymmetrical_action_space(env)
        if mathematically_possible_actions != len(possible_actions):
            raise ValueError(
                "The number of possible actions does not match the mathematically calculated number of actions."
            )

        file_path = os.path.join(save_path, "asymmetry.json")
        save_to_json(possible_actions, file_path)
    if "medha" in action_spaces_to_create:
        mathematically_possible_actions, _, _ = calculate_action_space_medha(env)
        possible_actions = get_medha_action_space(env)
        if mathematically_possible_actions != len(possible_actions):
            raise ValueError(
                "The number of possible actions does not match the mathematically calculated number of actions."
            )

        file_path = os.path.join(save_path, "medha.json")
        save_to_json(possible_actions, file_path)
    if "medha_dn" in action_spaces_to_create:
        possible_actions = get_medha_dn_action_space(env)
        file_path = os.path.join(save_path, "medha_DN.json")
        save_to_json(possible_actions, file_path)
    if "tennet" in action_spaces_to_create:
        mathematically_possible_actions, _, _ = calculate_action_space_tennet(env)
        possible_actions = get_tennet_action_space(env)
        if mathematically_possible_actions != len(possible_actions):
            raise ValueError(
                "The number of possible actions does not match the mathematically calculated number of actions."
            )

        file_path = os.path.join(save_path, "tennet.json")
        save_to_json(possible_actions, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-e",
        "--environment",
        default="l2rpn_case14_sandbox",
        type=str,
        help="Name of the environment to be used.",
    )
    parser.add_argument(
        "-a",
        "--action_space",
        type=str,
        help="Action space to be used.",
        default="medha_dn",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        default="../data/action_spaces/",
        help="Path the action spaces must be saved.",
    )

    args = parser.parse_args()

    input_environment = args.environment
    input_action_space = args.action_space
    input_save_path = args.save_path

    if input_action_space == "all":
        input_action_space = ["asymmetry", "medha", "tennet"]
    elif input_action_space == "asymmetry":
        input_action_space = ["asymmetry"]
    elif input_action_space == "medha":
        input_action_space = ["medha"]
    elif input_action_space == "medha_dn":
        input_action_space = ["medha_dn"]
    elif input_action_space == "tennet":
        input_action_space = ["tennet"]
    else:
        raise ValueError(
            "The action space must be either 'all', 'asymmetry', 'medha' or 'tennet'."
        )
    create_action_spaces(input_environment, input_action_space, input_save_path)

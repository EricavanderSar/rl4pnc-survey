"""
Utilities in the grid2op and gym convertion.
"""

import json
import os
from typing import Any
import numpy as np

import grid2op
import gymnasium
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Converter.Converters import Converter
from grid2op.Environment import BaseEnv
from lightsim2grid import LightSimBackend


class CustomDiscreteActions(gymnasium.spaces.Discrete):
    """
    Class that customizes the action space.

    Example usage:

    import grid2op
    from grid2op.Converter import IdToAct

    env = grid2op.make("rte_case14_realistic")

    all_actions = # a list of of desired actions
    converter = IdToAct(env.action_space)
    converter.init_converter(all_actions=all_actions)


    env.action_space = ChooseDiscreteActions(converter=converter)


    """

    def __init__(self, converter: Converter, do_nothing: BaseAction):
        self.converter = converter
        self.do_nothing = do_nothing
        super().__init__(n=converter.n)

    def from_gym(self, gym_action: dict[str, Any]) -> BaseAction:
        """
        Function that converts a gym action into a grid2op action.
        """
        return self.converter.convert_act(gym_action)

    def close(self) -> None:
        """Not implemented."""


def make_train_test_val_split(
    library_directory: str,
    env_name: str,
    pct_val: float,
    pct_test: float,
) -> None:
    """
    Function that splits an environment into a train, test and validation set.
    """
    if not os.path.exists(os.path.join(library_directory, env_name, "_train")):
        env = grid2op.make(os.path.join(library_directory, env_name))
        env.train_val_split_random(
            pct_val=pct_val, pct_test=pct_test, add_for_test="test"
        )


def get_possible_topologies(
    env: BaseEnv, substations_list: list[int]
) -> list[BaseAction]:
    """
    Function that returns all possible topologies when only keeping in mind a certain number of substations.
    """
    possible_substation_actions = []
    for idx in substations_list:
        possible_substation_actions += IdToAct.get_all_unitary_topologies_set(
            env.action_space, idx
        )
    return possible_substation_actions


def setup_converter(
    env: BaseEnv, possible_substation_actions: list[BaseAction]
) -> Converter:
    """
    Function that initializes and returns converter for gym to grid2op actions.
    """
    converter = IdToAct(env.action_space)
    converter.init_converter(all_actions=possible_substation_actions)
    return converter


def load_actions(path: str, env: BaseEnv) -> list[BaseAction]:
    """
    Loads the .json with specified topology actions.
    """
    with open(path, "rt", encoding="utf-8") as action_set_file:
        return list(
            (
                env.action_space(action_dict)
                for action_dict in json.load(action_set_file)
            )
        )


def rename_env(env: BaseEnv):
    # if the path contains _per_day or _train or _test or _val, then ignore this part of the string
    env_name = env.env_name
    if "_blazej" in env_name:
        path = env_name.replace("_blazej", "")
    if "_per_day" in env_name:
        env_name = env_name.replace("_per_day", "")
    if "_train" in env_name:
        env_name = env_name.replace("_train", "")
    if "_test" in env_name:
        env_name = env_name.replace("_test", "")
    if "_val" in env_name:
        env_name = env_name.replace("_val", "")
    env.set_env_name(env_name)


def make_g2op_env(env_config: dict[str, Any]) -> BaseEnv:
    """
    Function that makes a grid2op environment.
    """
    env = grid2op.make(
        env_config["env_name"],
        **env_config["grid2op_kwargs"],
        backend=LightSimBackend(),
    )
    env.seed(env_config["seed"])
    # *** RENAME THE ENVIRONMENT *** excl _train / _val etc
    # such that it can gather the action space and normalization/scaling parameters
    rename_env(env)

    if env.env_name == "rte_case14_realistic":
        env.set_thermal_limit(np.array(
            [
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                760,
                450,
                760,
                380,
                380,
                760,
                380,
                760,
                380,
                380,
                380,
                2000,
                2000,
            ])
        )
    return env

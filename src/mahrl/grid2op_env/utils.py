"""
Utilities in the grid2op and gym convertion.
"""

import json
import os
from typing import Any, List

import grid2op
import gymnasium
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import ScalerAttrConverter
from grid2op.gym_compat.gym_obs_space import GymnasiumObservationSpace
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend


class CustomIdToAct(IdToAct):
    """
    Defines also to_gym from actions.
    """

    def revert_act(self, action: BaseAction) -> int:
        """
        Do the opposite of convert_act. Given an action, return the id of this action in the list of all actions.
        """
        return int(np.where(self.all_actions == action)[0][0])


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

    def __init__(self, converter: CustomIdToAct):
        self.converter = converter
        super().__init__(n=converter.n)

    # # NOTE: Implementation before fixing single agent
    # def from_gym(self, gym_action: dict[str, Any]) -> BaseAction:
    #     """
    #     Function that converts a gym action into a grid2op action.
    #     """
    #     return self.converter.convert_act(gym_action)

    def from_gym(self, gym_action: int) -> BaseAction:
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
) -> CustomIdToAct:
    """
    Function that initializes and returns converter for gym to grid2op actions.
    """
    converter = CustomIdToAct(env.action_space)
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


def get_original_env_name(path: str) -> str:
    """
    If the path contains _per_day or _train or _test or _val, then ignore this part of the string
    """

    if "_blazej" in path:
        path = path.replace("_blazej", "")
    if "_per_day" in path:
        path = path.replace("_per_day", "")
    if "_train" in path:
        path = path.replace("_train", "")
    if "_test" in path:
        path = path.replace("_test", "")
    if "_val" in path:
        path = path.replace("_val", "")

    return path


def load_action_space(path: str, env: BaseEnv) -> List[BaseAction]:
    """
    Loads the action space from a specified folder.
    """
    path = get_original_env_name(path)

    return load_actions(path, env)


def rescale_observation_space(
    gym_observation_space: GymnasiumObservationSpace, g2op_env: BaseEnv
) -> GymnasiumObservationSpace:
    """
    Function that rescales the observation space.
    """
    # scale observations
    gym_observation_space = gym_observation_space.reencode_space(
        "gen_p",
        ScalerAttrConverter(substract=0.0, divide=g2op_env.gen_pmax),
    )
    gym_observation_space = gym_observation_space.reencode_space(
        "timestep_overflow",
        ScalerAttrConverter(
            substract=0.0,
            divide=Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED,  # assuming no custom params
        ),
    )

    grid_name = get_original_env_name(g2op_env.name)

    if grid_name in [
        "rte_case14_realistic",
        "rte_case5_example",
        "l2rpn_icaps_2021_large",
        "l2rpn_case14_sandbox",
    ]:
        for attr in ["p_ex", "p_or", "load_p"]:
            underestimation_constant = (
                1.2  # constant to account that our max/min are underestimated
            )
            max_arr, min_arr = np.load(
                os.path.join(
                    "/Users/barberademol/Documents/GitHub/mahrl_grid2op/",
                    "data/scaling_arrays",
                    grid_name,
                    f"{attr}.npy",
                )
            )

            gym_observation_space = gym_observation_space.reencode_space(
                attr,
                ScalerAttrConverter(
                    substract=underestimation_constant * min_arr,
                    divide=underestimation_constant * (max_arr - min_arr),
                ),
            )
    else:
        raise ValueError("This scaling is not yet implemented for this environment.")

    return gym_observation_space


def make_g2op_env(env_config: dict[str, Any]) -> BaseEnv:
    """
    Function that makes a grid2op environment.
    """
    env = grid2op.make(
        env_config["env_name"],
        **env_config["grid2op_kwargs"],
        backend=LightSimBackend(),
    )

    if "seed" in env_config:
        env.seed(env_config["seed"])

    if str(env_config["env_name"]).startswith("rte_case14_realistic"):
        env.set_thermal_limit(
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
            ]
        )
    return env

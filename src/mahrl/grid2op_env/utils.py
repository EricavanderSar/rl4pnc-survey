"""
Utilities in the grid2op and gym convertion.
"""

import json
import os
from typing import Any, List

import grid2op
import gymnasium
import numpy as np
import torch
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import ScalerAttrConverter
from grid2op.gym_compat.gym_obs_space import GymnasiumObservationSpace
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend


class CustomIdToAct(IdToAct):
    """
    This class extends the functionality of the base class `IdToAct` by providing a method `revert_act`
    to convert an action back to its corresponding ID in the list of all actions.

    Attributes:
        all_actions (list): A list of all possible actions.

    Methods:
        revert_act(action: BaseAction) -> int:
            Do the opposite of convert_act. Given an action, return the ID of this action in the list of all actions.
    """

    def revert_act(self, action: BaseAction) -> int:
        """
        Do the opposite of convert_act. Given an action, return the ID of this action in the list of all actions.

        Args:
            action (BaseAction): The action to be reverted.

        Returns:
            int: The ID of the given action in the list of all actions.
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
        """
        Initializes a CustomDiscreteActions object.

        Parameters:
        - converter (CustomIdToAct): The converter object used to convert actions.

        """
        self.converter = converter
        super().__init__(n=converter.n)

    def from_gym(self, gym_action: int) -> BaseAction:
        """
        Converts a gym action into a grid2op action.

        Parameters:
        - gym_action (int): The gym action to be converted.

        Returns:
        - BaseAction: The converted grid2op action.
        """
        return self.converter.convert_act(gym_action)

    def close(self) -> None:
        """
        Closes the CustomDiscreteActions object.

        """
        # Not implemented.


def make_train_test_val_split(
    library_directory: str,
    env_name: str,
    pct_val: float,
    pct_test: float,
) -> None:
    """
    Function that splits an environment into a train, test and validation set.

    Parameters:
    - library_directory (str): The directory where the environment library is located.
    - env_name (str): The name of the environment to split.
    - pct_val (float): The percentage of data to allocate for the validation set.
    - pct_test (float): The percentage of data to allocate for the test set.

    Returns:
    None
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

    Parameters:
        env (BaseEnv): The grid2op environment.
        substations_list (list[int]): A list of indices representing the substations to consider.

    Returns:
        list[BaseAction]: A list of possible topologies as BaseAction objects.
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

    Parameters:
        env (BaseEnv): The grid2op environment.
        possible_substation_actions (list[BaseAction]): List of possible substation actions.

    Returns:
        CustomIdToAct: The initialized converter for gym to grid2op actions.
    """
    converter = CustomIdToAct(env.action_space)
    converter.init_converter(all_actions=possible_substation_actions)
    return converter


def load_actions(path: str, env: BaseEnv) -> list[BaseAction]:
    """
    Loads the .json with specified topology actions.

    Args:
        path (str): The path to the .json file containing the topology actions.
        env (BaseEnv): The environment object.

    Returns:
        list[BaseAction]: A list of BaseAction objects representing the loaded actions.
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
    Returns the original environment name by removing specific parts from the given path.

    Args:
        path (str): The path containing the environment name.

    Returns:
        str: The original environment name without specific parts.
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

    Args:
        path (str): The path to the folder containing the action space.
        env (BaseEnv): The environment object.

    Returns:
        List[BaseAction]: The loaded action space.
    """
    path = get_original_env_name(path)

    return load_actions(path, env)


def rescale_observation_space(
    gym_observation_space: GymnasiumObservationSpace,
    g2op_env: BaseEnv,
    env_config: dict[str, Any],
) -> GymnasiumObservationSpace:
    """
    Rescales the observation space to better fit between 0 and 1.

    Args:
        gym_observation_space (GymnasiumObservationSpace): The original observation space.
        g2op_env (BaseEnv): The Grid2Op environment.
        env_config (dict[str, Any]): The environment configuration.

    Returns:
        GymnasiumObservationSpace: The rescaled observation space.
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
                    env_config["lib_dir"],
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
    Function that makes a grid2op environment and when needed sets thermal limits.

    Parameters:
        env_config (dict[str, Any]): A dictionary containing the configuration for the environment.

    Returns:
        BaseEnv: The created grid2op environment.

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


class ChronPrioMatrix:
    """
    A class representing the ChronPrioMatrix.

    This class is responsible for managing the priority scores of training chronics in the Grid2Op environment.

    Attributes:
        chronic_lengths (dict[str, int]): A dictionary mapping chronic names to their respective lengths.
        name_len (int): The length of the chronic name.
        chron_scores (torch.Tensor): The training chronic sampling weights.
        chronic_idx (int): The index of the sampled chronic.

    """

    def __init__(self, env: BaseEnv):
        """
        Initializes a new instance of the ChronPrioMatrix class.

        self.chron_scores is set to 2 when the chronic has not yet happend and to -1 when it was fully solved.
        The higher the score, the more likely it is to be sampled.

        Args:
            env (BaseEnv): The Grid2Op environment.

        """
        # Get the list of available chronics
        avail_chron = env.chronics_handler.real_data.available_chronics()

        # Get the length of each chronic, as it can vary per piece
        self.chronic_lengths: dict[str, int] = {}
        for chronic in avail_chron:
            env.set_id(chronic)
            env.reset()
            self.chronic_lengths[chronic.split("/")[-1]] = env.max_episode_duration()

        self.name_len = len(avail_chron[0].split("/")[-1])

        # initialize training chronic sampling weights
        self.chron_scores = torch.ones(len(avail_chron)) * 2.0  # NOTE: Why *2?
        self.chronic_idx = 0

    def sample_chron(self) -> int:
        """
        Samples a training chronic based on the chronic scores.

        Returns:
            int: The index of the sampled chronic.

        """
        dist = torch.distributions.categorical.Categorical(
            logits=torch.Tensor(self.chron_scores)
        )
        self.chronic_idx = dist.sample().item()
        return self.chronic_idx

    def update_prios(self, steps_surv: int) -> None:
        """
        Updates the priority scores based on the number of steps survived.

        Args:
            steps_surv (int): The number of steps survived.

        """
        scores = (
            1
            - np.sqrt(
                steps_surv
                / self.chronic_lengths[str(self.chronic_idx).zfill(self.name_len)]
            )
            * 2.0
        )

        chronic_idx_str = str(self.chronic_idx).lstrip("0")
        if chronic_idx_str == "":  # if chronic_idx_str is empty
            chronic_idx_str = "0"

        self.chron_scores[int(chronic_idx_str)] = scores

"""
Utilities in the grid2op and gym convertion.
"""

import json
import os
from typing import Any, List, Optional, Tuple

import grid2op
import gymnasium
import numpy as np
import torch
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import ScalerAttrConverter
from grid2op.gym_compat.gym_obs_space import GymnasiumObservationSpace
from grid2op.Observation import BaseObservation
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

    def to_gym(self, action: BaseAction) -> int:
        """
        Function that converts a grid2op action into a gym action.
        """
        return int(np.where(self.converter.all_actions == action)[0][0])

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
    if path.startswith("/home3/s3374610/"):
        # replace for runs on cluster
        path = path.replace("/home3/s3374610/", "/Users/barberademol/Documents/GitHub/")

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
    if "_original" in path:
        path = path.replace("_original", "")
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
            path = os.path.join(
                env_config["lib_dir"],
                "data/scaling_arrays",
                grid_name,
                f"{attr}.npy",
            )

            if path.startswith("/home3/s3374610/"):
                # replace for runs on cluster
                path = path.replace(
                    "/home3/s3374610/", "/Users/barberademol/Documents/GitHub/"
                )

            max_arr, min_arr = np.load(path)

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
    # enable acting on all subs to go back to base topology
    params = Parameters()
    if env_config["env_name"].startswith("rte_case5_example"):
        params.MAX_SUB_CHANGED = 5
        params.MAX_LINE_STATUS_CHANGED = 8
    elif env_config["env_name"].startswith("rte_case14_realistic") or env_config[
        "env_name"
    ].startswith("l2rpn_case14_sandbox"):
        params.MAX_SUB_CHANGED = 14
        params.MAX_LINE_STATUS_CHANGED = 20
    elif env_config["env_name"].startswith("l2rpn_icaps_2021_large"):
        params.MAX_SUB_CHANGED = 36
        params.MAX_LINE_STATUS_CHANGED = 59
    else:
        raise ValueError("Please specify the number of subs in this env.")

    env = grid2op.make(
        env_config["env_name"],
        **env_config["grid2op_kwargs"],
        backend=LightSimBackend(),
        param=params,
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


def go_to_abbc(
    env: BaseEnv,
    g2op_obs: BaseObservation,
    reconnect_line: list[BaseAction],
    on_cooldown: set[int],
    info: Optional[dict[str, Any]] = None,
) -> Tuple[list[BaseAction], dict[str, Any]]:
    """
    Performs the necessary actions to transition the system to ABBC state.

    Args:
        env (BaseEnv): The grid environment.
        g2op_obs (BaseObservation): The current observation of the grid.
        reconnect_line (list[BaseAction]): The list of actions to reconnect lines.
        on_cooldown (set[int]): The set of bus indices that are on cooldown.
        info (Optional[dict[str, Any]], optional): Additional information. Defaults to {}.

    Returns:
        Tuple[list[BaseAction], dict[str, Any]]: The updated list of reconnect actions and additional information.
    """
    if info is None:
        info = {}

    changed_busses = [
        i for i, x in enumerate(g2op_obs.topo_vect) if x != 1 and i not in on_cooldown
    ]
    # If we are no longer in base toplogy
    if changed_busses:
        new_status = [1] * len(changed_busses)

        # Get action that sets everything to ABBC
        g2op_act = env.action_space({"set_bus": list(zip(changed_busses, new_status))})

        try:
            # Simulate action to see that it doesn't cause the rho outside 0-1
            simul_obs, _, _, _ = g2op_obs.simulate(g2op_act)
            rho_values = simul_obs.to_dict()["rho"]
            # NOTE: Only do it if it's predicted to be in a 'very' safe state
            if all(0 <= value <= 0.9 for value in rho_values):
                # If succeeded, set to perform at next ts
                reconnect_line.append(g2op_act)

                # add info in reset to stop rewarding the last agent for this action
                info["abbc_action"] = True
        except:  # pylint: disable=bare-except # noqa: E722
            # no simulation available, skip step
            pass

    return reconnect_line, info


def reconnecting_and_abbc(
    env: BaseEnv,
    g2op_obs: BaseObservation,
    reconnect_line: list[BaseAction],
    info: Optional[dict[str, Any]] = None,
) -> Tuple[list[BaseAction], dict[str, Any]]:
    """
    Reconnects lines and performs ABBC (Automatic Busbar Closing) operation.

    Args:
        env (BaseEnv): The grid environment.
        g2op_obs (BaseObservation): The observation from the grid environment.
        reconnect_line (list[BaseAction]): List of actions to reconnect lines.
        info (dict[str, Any], optional): Additional information. Defaults to {}.

    Returns:
        Tuple[list[BaseAction], dict[str, Any]]: A Tuple containing the updated list of
        reconnect actions and additional information.
    """
    if info is None:
        info = {}

    # This should avoid converging power flows
    on_cooldown, cooldown_lines = ignore_cooldowns(env, g2op_obs)

    if g2op_obs.hour_of_day > 2 and g2op_obs.hour_of_day < 6:
        reconnect_line, info = go_to_abbc(
            env, g2op_obs, reconnect_line, on_cooldown, info
        )

    if any(g2op_obs.time_before_cooldown_sub):
        for sub_id, time in enumerate(g2op_obs.time_before_cooldown_sub):
            if time > 0:
                connected_elements = env.observation_space.get_obj_connect_to(
                    substation_id=sub_id
                )
                cooldown_lines = find_lines_on_cooldown(
                    connected_elements, cooldown_lines
                )

    to_reco = ~g2op_obs.line_status
    if np.any(to_reco):
        reco_id = np.where(to_reco)[0]
        for line_id in reco_id:
            if line_id not in cooldown_lines:
                g2op_act = env.action_space({"set_line_status": [(line_id, +1)]})
                reconnect_line.append(g2op_act)

    return reconnect_line, info


def find_lines_on_cooldown(
    connected_elements: dict[str, Any], cooldown_lines: set[int]
) -> set[int]:
    """
    Finds the lines on cooldown connected to a given substation.

    Parameters:
        connected_elements (list[int]): Connected elements.
        cooldown_lines (set[int]): A set to store the IDs of the lines on cooldown.

    Returns:
        set[int]: A set containing the IDs of the lines on cooldown connected to the substation.
    """
    for element in connected_elements:
        if element in ("lines_or_id", "lines_ex_id"):
            for l_id in connected_elements[element]:
                cooldown_lines.add(l_id)
    return cooldown_lines


def ignore_cooldowns(env: BaseEnv, obs: BaseObservation) -> Tuple[set[int], set[int]]:
    """
    Finds the elements and lines that are on cooldown based on the given observation.

    Args:
        env (BaseEnv): The environment object.
        obs (BaseObservation): The observation object.

    Returns:
        Tuple[set[int], set[int]]: A Tuple containing two sets:
            - elements_on_cooldown: A set of element IDs that are on cooldown.
            - lines_on_cooldown: A set of line IDs that are on cooldown.
    """
    elements_on_cooldown = set()
    substation_on_cooldown = set()
    lines_on_cooldown = set()
    if any(obs.time_before_cooldown_line):
        # print(obs.time_before_cooldown_line)
        for line_id, time in enumerate(obs.time_before_cooldown_line):
            if time > 0:
                lines_on_cooldown.add(line_id)
                elements_on_cooldown.add(obs.line_or_pos_topo_vect[line_id])
                elements_on_cooldown.add(obs.line_ex_pos_topo_vect[line_id])

                # manually add cooldown to other elements related to connected substations
                substation_on_cooldown.add(obs.line_or_to_subid[line_id])
                substation_on_cooldown.add(obs.line_ex_to_subid[line_id])
                for sub_id in substation_on_cooldown:
                    (
                        elements_on_cooldown,
                        lines_on_cooldown,
                    ) = find_substations_on_cooldown(
                        sub_id, elements_on_cooldown, lines_on_cooldown, obs
                    )
    # find related elements to substation that is nonzero
    if any(obs.time_before_cooldown_sub):
        # print(obs.time_before_cooldown_sub)
        for sub_id, time in enumerate(obs.time_before_cooldown_sub):
            if time > 0:
                elements_on_cooldown = find_elements_on_cooldown(
                    sub_id, elements_on_cooldown, obs
                )
        # print(on_cooldown)
    return elements_on_cooldown, lines_on_cooldown


def find_substations_on_cooldown(
    connected_elements: dict[str, Any],
    elements_on_cooldown: set[int],
    lines_on_cooldown: set[int],
    obs: BaseObservation,
) -> Tuple[set[int], set[int]]:
    """
    Finds the substations on cooldown based on the connected elements.

    Args:
        connected_elements (dict[str, Any]): A dictionary containing the connected elements.
        elements_on_cooldown (set[int]): A set to store the elements on cooldown.
        lines_on_cooldown (set[int]): A set to store the lines on cooldown.

    Returns:
        Tuple[set[int], set[int]]: A Tuple containing the elements on cooldown and lines on cooldown.
    """
    for element in connected_elements:
        if element == "loads_id":
            for l_id in connected_elements[element]:
                elements_on_cooldown.add(obs.load_pos_topo_vect[l_id])
        elif element == "lines_or_id":
            for l_id in connected_elements[element]:
                elements_on_cooldown.add(obs.line_or_pos_topo_vect[l_id])
                lines_on_cooldown.add(l_id)
        elif element == "lines_ex_id":
            for l_id in connected_elements[element]:
                elements_on_cooldown.add(obs.line_ex_pos_topo_vect[l_id])
                lines_on_cooldown.add(l_id)
        elif element == "generators_id":
            for l_id in connected_elements[element]:
                elements_on_cooldown.add(obs.gen_pos_topo_vect[l_id])
    return elements_on_cooldown, lines_on_cooldown


def find_elements_on_cooldown(
    connected_elements: dict[str, Any],
    elements_on_cooldown: set[int],
    obs: BaseObservation,
) -> set[int]:
    """
    Finds the elements on cooldown connected to a given substation.

    Args:
        sub_id (int): The ID of the substation.
        elements_on_cooldown (set[int]): The set of elements on cooldown.

    Returns:
        set[int]: The updated set of elements on cooldown.
    """
    for element in connected_elements:
        if element == "loads_id":
            for l_id in connected_elements[element]:
                elements_on_cooldown.add(obs.load_pos_topo_vect[l_id])
        elif element == "lines_or_id":
            for l_id in connected_elements[element]:
                elements_on_cooldown.add(obs.line_or_pos_topo_vect[l_id])
        elif element == "lines_ex_id":
            for l_id in connected_elements[element]:
                elements_on_cooldown.add(obs.line_ex_pos_topo_vect[l_id])
        elif element == "generators_id":
            for l_id in connected_elements[element]:
                elements_on_cooldown.add(obs.gen_pos_topo_vect[l_id])
    return elements_on_cooldown


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

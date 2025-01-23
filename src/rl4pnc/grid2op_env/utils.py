"""
Utilities in the grid2op and gym convertion.
"""

import json
import os
from typing import Any
import numpy as np
import torch

import grid2op
import gymnasium
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Converter.Converters import Converter
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

    def to_gym(self, action: BaseAction) -> int:
        return int(np.where(self.converter.all_actions == action)[0][0])

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
    if "_small" in env_name:
        env_name = env_name.replace("_small", "")
    if "_large" in env_name:
        env_name = env_name.replace("_large", "")
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
    print("Reward function used: ", env._rewardClass)
    env.chronics_handler.set_chunk_size(100)

    if "seed" in env_config:
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


class ChronPrioVect:
    """
        This class can be used when episodes are already split in smaller pieces for training.
        Each smaller episode gets a priority value: chronic_scores
    """
    def __init__(self, env: grid2op.Environment):
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
        # not used in this case since we have small episodes we dont ffw
        # initialized to generalize.
        self.cur_ffw = 0

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


class ChronPrioMatrix(ChronPrioVect):
    """
        The ChronPrioMatrix class defines a matrix that keeps track of the difficulty starting in some position (ffw)
        in the current chronic.
    """
    def __init__(self, env: grid2op.Environment):
        super().__init__(env)
        self.max_ep_dur = env.max_episode_duration()
        # initialize training chronic sampling weights
        self.ffw_size = 288
        self.max_ffw = int(np.ceil(self.max_ep_dur / self.ffw_size))
        avail_chron = env.chronics_handler.real_data.available_chronics()
        self.chron_scores = torch.ones(len(avail_chron), self.max_ffw) * 2.0

    def sample_chron(self):
        # sample training chronic
        dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(self.chron_scores.flatten()))
        record_idx = dist.sample().item()
        self.chronic_idx = record_idx // self.max_ffw
        self.cur_ffw = record_idx % self.max_ffw
        return self.chronic_idx

    def update_prios(self, steps_surv):
        pieces_played = int(np.ceil(steps_surv / self.ffw_size))
        max_steps = self.max_ep_dur - self.cur_ffw * self.ffw_size
        scores = torch.ones(pieces_played) * 2.0  # scale = 2.0
        if max_steps != self.ffw_size * pieces_played:
            for p in range(pieces_played):
                scores[p] *= 1 - np.sqrt((steps_surv - self.ffw_size * p) / (max_steps - self.ffw_size * p))
        # print("pieces played: ", pieces_played)
        # print("self.cur_ffw: ", self.cur_ffw)
        # print("steps surv: ", steps_surv)
        self.chron_scores[self.chronic_idx][self.cur_ffw: (self.cur_ffw + pieces_played)] = scores


def get_attr_list(attr_abbreviated: list):
    if "all" in attr_abbreviated:
        attr = ["topo_vect", "line_status", "load_p", "gen_p", "p_ex", "p_or", "load_q", "gen_q", "q_ex", "q_or", "load_v", "gen_v", "v_ex", "v_or", "load_theta", "gen_theta", "theta_ex", "theta_or", "a_ex", "a_or", "rho", "timestep_overflow", "time_next_maintenance"]
    else:
        attr = []
        if "t" in attr_abbreviated:
            attr = ["topo_vect"]
        if "l" in attr_abbreviated:
            # include line status
            attr.extend(["line_status"])
        if "p_i" in attr_abbreviated:
            # include active power input
            attr.extend(["load_p", "gen_p"])
        if "p_l" in attr_abbreviated:
            # include active power line flows
            attr.extend(["p_ex", "p_or"])
        if "q_i" in attr_abbreviated:
            # include reactive power input
            attr.extend(["load_q", "gen_q"])
        if "q_l" in attr_abbreviated:
            # include reactive power line flows
            attr.extend(["q_ex", "q_or"])
        if "v_i" in attr_abbreviated:
            # include voltage input
            attr.extend(["load_v", "gen_v"])
        if "v_l" in attr_abbreviated:
            # include voltage line flows
            attr.extend(["v_ex", "v_or"])
        if "theta_i" in attr_abbreviated:
            # include voltage angle input
            attr.extend(["load_theta", "gen_theta"])
        if "theta_l" in attr_abbreviated:
            # include voltage angle line flows
            attr.extend(["theta_ex", "theta_or"])
        if "a" in attr_abbreviated:
            # include current line flows
            attr.extend(["a_ex", "a_or"])
        if "r" in attr_abbreviated:
            # include rho (power flow / thermal limit lines)
            attr.append("rho")
        if "o" in attr_abbreviated:
            # include ts since overflow
            attr.append("timestep_overflow")
        if "m" in attr_abbreviated:
            attr.append("time_next_maintenance")
    return attr

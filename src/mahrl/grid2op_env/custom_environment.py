"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""

import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import gymnasium as gym
from grid2op.Action import BaseAction
from grid2op.Chronics import Multifolder
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import GymEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mahrl.evaluation.evaluation_agents import get_actions_per_substation
from mahrl.experiments.utils import find_list_of_agents
from mahrl.grid2op_env.utils import (
    ChronPrioMatrix,
    CustomDiscreteActions,
    load_action_space,
    make_g2op_env,
    reconnecting_and_abbc,
    rescale_observation_space,
    setup_converter,
)

OBSTYPE = TypeVar("OBSTYPE")
ACTTYPE = TypeVar("ACTTYPE")
RENDERFRAME = TypeVar("RENDERFRAME")


class ReconnectingGymEnv(GymEnv):
    """
    This class wrapps the Grid2Op GymEnv and automatically connects disconnected
    powerlines in the environment.
    """

    def __init__(self, env: BaseEnv, shuffle_chronics: bool = True):
        super().__init__(env, shuffle_chronics=shuffle_chronics)
        self.reconnect_line: list[BaseAction] = []

    def step(self, action: int) -> Tuple[OBSTYPE, float, bool, bool, dict[str, Any]]:
        """
        Perform a step in the environment.

        Parameters:
            action (ACTTYPE): The action to perform in the environment.

        Returns:
            Tuple[OBSTYPE, float, bool, dict[str, Any]]: The observation, reward, done, truncated flag and info dictionary.
        """

        g2op_act = self.action_space.from_gym(action)

        if self.reconnect_line:
            for line in self.reconnect_line:
                g2op_act = g2op_act + line
            self.reconnect_line = []

        g2op_obs, reward, terminated, info = self.init_env.step(g2op_act)

        self.reconnect_line, info = reconnecting_and_abbc(
            self.init_env, g2op_obs, self.reconnect_line, info
        )

        print(f"Inside: {info} and {self.reconnect_line}")
        gym_obs = self.observation_space.to_gym(g2op_obs)
        truncated = False  # see https://github.com/openai/gym/pull/2752
        return gym_obs, float(reward), terminated, truncated, info


# pylint: disable=too-many-instance-attributes
class CustomizedGrid2OpEnvironment(MultiAgentEnv):
    """Encapsulate Grid2Op environment and set action/observation space.

    Args:
        env_config (dict): Configuration parameters for the environment.

    Attributes:
        grid2op_env: The Grid2Op environment.
        env_gym: The gym environment.
        possible_substation_actions: List of possible substation actions.
        converter: Converter for action space.
        action_space: Action space for RLlib.
        observation_space: Observation space for RLlib.
        is_value_rl (bool): Flag indicating if it is a value RL environment.
        _agent_ids (list): List of agent IDs.
        previous_obs (OrderedDict): Previous observations.

    """

    def __init__(self, env_config: dict[str, Any]):
        super().__init__()

        # create the grid2op environment
        self.grid2op_env = make_g2op_env(env_config)

        self.stage = env_config["stage"]

        # create the gym environment
        if (
            self.stage == "val"
            or not env_config["shuffle_scenarios"]
            or env_config.get("prio", True)
        ):  # ensure the evaluation chronics are not shuffled
            self.env_gym = ReconnectingGymEnv(self.grid2op_env, shuffle_chronics=False)
        elif env_config["shuffle_scenarios"]:
            self.env_gym = ReconnectingGymEnv(self.grid2op_env, shuffle_chronics=True)
        else:
            raise ValueError("No valid shuffle scenario is defined.")

        # setting up custom action space
        path = os.path.join(
            env_config["lib_dir"],
            f"data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
        )
        self.possible_substation_actions = load_action_space(path, self.grid2op_env)

        # print(f"Actions loaded")
        # add the do-nothing action at index 0
        do_nothing_action = self.grid2op_env.action_space({})
        self.possible_substation_actions.insert(0, do_nothing_action)

        # create converter
        self.converter = setup_converter(
            self.grid2op_env, self.possible_substation_actions
        )

        # set gym action space to discrete
        self.env_gym.action_space = CustomDiscreteActions(self.converter)

        # specific to rllib
        self.action_space = gym.spaces.Discrete(len(self.possible_substation_actions))

        # customize observation space
        self.env_gym.observation_space = self.env_gym.observation_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect", "p_or", "p_ex", "timestep_overflow"]
        )

        # rescale observation space
        self.env_gym.observation_space = rescale_observation_space(
            self.env_gym.observation_space, self.grid2op_env, env_config
        )

        # specific to rllib
        self.observation_space = gym.spaces.Dict(
            dict(self.env_gym.observation_space.spaces.items())
        )

        # determine agent ids
        if "vf_rl" in env_config:
            self.is_value_rl = True

            # Group the value function and reinforcement policy
            self._agent_ids = [
                "high_level_agent",
                "vf_group",
                "do_nothing_agent",
            ]

            self.env = self.with_agent_groups(
                groups={
                    "vf_group": [
                        "value_reinforcement_learning_agent",
                        "value_function_agent",
                    ]
                },
            )
        else:
            self.is_value_rl = False
            self._agent_ids = [
                "high_level_agent",
                "reinforcement_learning_agent",
                "do_nothing_agent",
            ]

        # setup shared parameters
        self.previous_obs: OrderedDict[str, Any] = OrderedDict()

        # initialize training chronic sampling weights
        self.prio = env_config.get("prio", True)  # NOTE: Default is now set to true
        self.chron_prios = ChronPrioMatrix(self.grid2op_env)
        self.step_surv = 0
        self.pause_reward = False

    def no_shunt_reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[OrderedDict[str, Any], dict[str, Any]]:
        """
        This function resets the environment with prio sampling.
        """
        # adapted from the internal GymEnv _aux_reset method
        if self.env_gym._shuffle_chronics and isinstance(
            self.env_gym.init_env.chronics_handler.real_data, Multifolder
        ):
            self.env_gym.init_env.chronics_handler.sample_next_chronics()

        super().reset(seed=seed)
        if seed is not None:
            self.env_gym._aux_seed_spaces()
            seed, next_seed, underlying_env_seeds = self.env_gym._aux_seed_g2op(seed)

        g2op_obs = self.env_gym.init_env.reset()

        ##############################################
        # disable shunts
        g2op_obs, _, _, _ = self.env_gym.init_env.step(
            self.env_gym.init_env._helper_action_env({"shunt": {"shunt_q": [(0, 0.0)]}})
        )
        ##############################################

        gym_obs = self.env_gym.observation_space.to_gym(g2op_obs)

        chron_id = self.env_gym.init_env.chronics_handler.get_id()
        info = {"time serie id": chron_id}
        if seed is not None:
            info["seed"] = seed
            info["grid2op_env_seed"] = next_seed
            info["underlying_env_seeds"] = underlying_env_seeds

        return gym_obs, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        This function resets the environment. Observation is passed to HL agent.

        Args:
            seed (int, optional): Random seed for environment reset. Defaults to None.
            options (Dict[str, Any], optional): Additional options for environment reset. Defaults to None.

        Returns:
            Tuple[MultiAgentDict, MultiAgentDict]: Tuple containing the initial observations and info.

        """
        if self.prio and self.stage == "train":
            self.previous_obs, info = self.prio_reset()
        else:
            self.previous_obs, info = self.env_gym.reset()
            # self.previous_obs, info = self.no_shunt_reset() #TODO: Integrate shunt reset in others?

        return {"high_level_agent": max(self.previous_obs["rho"])}, {"__common__": info}

    def prio_reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[OrderedDict[str, Any], dict[str, Any]]:
        """
        This function resets the environment with prio sampling.
        """
        # TODO: CHeck if it doesn't sample the wrong one because of shuffling. Force shuffle off?
        # adapted from the internal GymEnv _aux_reset method
        if self.env_gym._shuffle_chronics and isinstance(
            self.env_gym.init_env.chronics_handler.real_data, Multifolder
        ):
            self.env_gym.init_env.chronics_handler.sample_next_chronics()

        ############################################################################
        # use chronic priority
        sampled_chron = self.chron_prios.sample_chron()
        self.env_gym.init_env.set_id(sampled_chron)
        self.step_surv = 0

        ############################################################################

        super().reset(seed=seed)
        if seed is not None:
            self.env_gym._aux_seed_spaces()
            seed, next_seed, underlying_env_seeds = self.env_gym._aux_seed_g2op(seed)

        g2op_obs = self.env_gym.init_env.reset()

        gym_obs = self.env_gym.observation_space.to_gym(g2op_obs)

        chron_id = self.env_gym.init_env.chronics_handler.get_id()
        info = {"time serie id": chron_id}
        if seed is not None:
            info["seed"] = seed
            info["grid2op_env_seed"] = next_seed
            info["underlying_env_seeds"] = underlying_env_seeds

        return gym_obs, info

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """
        This function performs a single step in the environment.

        Args:
            action_dict (MultiAgentDict): Dictionary containing the actions for each agent.

        Returns:
            Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
            Tuple containing the observations, rewards, termination flags, truncation flags, and info.

        """
        # build basic dicts, that are overwritten by acting agents
        observations: Dict[str, Any] = {}
        rewards: Dict[str, Any] = {}
        terminateds = {
            "__all__": False,
        }
        truncateds = {
            "__all__": False,
        }
        infos: Dict[str, Any] = {}

        # check which agent is acting
        if "high_level_agent" in action_dict.keys():
            observations = self.perform_high_level_action(action_dict)
        elif "do_nothing_agent" in action_dict.keys():
            # perform do nothing in the env
            (
                self.previous_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_gym.step(action_dict["do_nothing_agent"])

            self.perform_prio_update(terminated)

            # reward the RL agent for this step, go back to HL agent
            if not self.pause_reward:
                if self.is_value_rl:
                    rewards = {"value_reinforcement_learning_agent": reward}
                else:
                    rewards = {"reinforcement_learning_agent": reward}
            observations = {"high_level_agent": max(self.previous_obs["rho"])}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {"__common__": info}
        elif "reinforcement_learning_agent" in action_dict.keys():
            self.pause_reward = False
            # perform RL step in the env
            (
                self.previous_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_gym.step(action_dict["reinforcement_learning_agent"])

            self.perform_prio_update(terminated)

            # reward the RL agent for this step, go back to HL agent
            rewards = {"reinforcement_learning_agent": reward}
            observations = {"high_level_agent": max(self.previous_obs["rho"])}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {"__common__": info}
        elif "value_reinforcement_learning_agent" in action_dict.keys():
            self.pause_reward = False
            # perform RL step in the env
            (
                self.previous_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_gym.step(action_dict["value_reinforcement_learning_agent"])

            self.perform_prio_update(terminated)

            # reward the RL agent for this step, go back to HL agent
            rewards = {"value_reinforcement_learning_agent": reward}
            observations = {"high_level_agent": max(self.previous_obs["rho"])}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {"__common__": info}
        elif bool(action_dict) is False:
            # print("Caution: Empty action dictionary!")
            pass
        else:
            raise ValueError(
                f"No valid agent found in action dictionary in step(): {action_dict}."
            )

        if "__common__" in infos:
            if "abbc_action" in infos["__common__"]:
                self.pause_reward = True

        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError

    def perform_prio_update(self, terminated: bool) -> None:
        """
        Update the priority of the environment.

        Args:
            terminated (bool): Flag indicating whether the environment has terminated.

        Returns:
            None
        """
        if self.prio and self.stage == "train":
            self.step_surv += 1
            if terminated:
                self.chron_prios.update_prios(self.step_surv)

    def perform_high_level_action(self, action_dict: dict[str, int]) -> dict[str, Any]:
        """
        Perform high-level action based on the action dictionary.

        Args:
            action_dict (dict): Dictionary containing the high-level action.

        Returns:
            dict: Dictionary containing the observations.

        Raises:
            ValueError: If an invalid action is selected by the high_level_agent.

        """
        observations: dict[str, Union[int, OrderedDict[str, Any]]] = {}

        if action_dict["high_level_agent"] == 0:  # do something
            if self.is_value_rl:
                observations = {
                    "value_reinforcement_learning_agent": self.previous_obs,
                    "value_function_agent": self.previous_obs,
                }
            else:
                observations = {"reinforcement_learning_agent": self.previous_obs}
        elif action_dict["high_level_agent"] == 1:  # do nothing
            observations = {"do_nothing_agent": 0}
        else:
            raise ValueError(
                "An invalid action is selected by the high_level_agent in step()."
            )
        return observations


register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)


# pylint: disable=too-many-instance-attributes
class HierarchicalCustomizedGrid2OpEnvironment(CustomizedGrid2OpEnvironment):
    """
    Implement step function for hierarchical environment. This is made to work with greedy-agent lower
    level agents. Their action is built into the environment.
    """

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        # add all RL agents
        self.agent_per_substation = find_list_of_agents(
            env=self.grid2op_env,
            action_space=env_config["action_space"],
            add_dn_agents=False,
            add_dn_action_per_agent=True,
        )
        list_of_substations = list(self.agent_per_substation.keys())

        self.rl_agent_ids = []
        rl_agent_groups = {}

        # NOTE: All agents also have the explicit do-nothing action available
        actions_per_substations = get_actions_per_substation(
            possible_substation_actions=self.possible_substation_actions,
            agent_per_substation=self.agent_per_substation,
            add_dn_action_per_agent=True,
        )

        # map individual substation action to global action space
        # for each possible substation action, match the action in self.possible_substation_actions
        self.local_to_global_action_map = {}
        for sub_idx, actions in actions_per_substations.items():
            self.local_to_global_action_map[sub_idx] = {
                local_idx: self.possible_substation_actions.index(global_action)
                for local_idx, global_action in enumerate(actions)
            }

        # map the middle output substation to the substation id
        self.middle_to_substation_map = {
            str(i): v for i, v in enumerate(list_of_substations)
        }

        base_agents = [
            "high_level_agent",
            "choose_substation_agent",
            "do_nothing_agent",
        ]

        if self.is_value_rl:
            for sub_idx in list_of_substations:
                # add agent
                self.rl_agent_ids.append(
                    f"value_reinforcement_learning_agent_{sub_idx}"
                )
                self.rl_agent_ids.append(f"value_function_agent_{sub_idx}")

                rl_agent_groups[f"vf_group_{sub_idx}"] = [
                    f"value_reinforcement_learning_agent_{sub_idx}",
                    f"value_function_agent_{sub_idx}",
                ]

            self.env = self.with_agent_groups(
                groups=rl_agent_groups,
            )

            # determine the acting agents
            self._agent_ids = list(rl_agent_groups.keys()) + base_agents
        else:
            for sub_idx in list_of_substations:
                # add agent
                self.rl_agent_ids.append(f"reinforcement_learning_agent_{sub_idx}")

            # determine the acting agents
            self._agent_ids = self.rl_agent_ids + base_agents

        self.is_greedy = False
        self.is_capa = "capa" in env_config.keys()
        self.is_rulebased = "rulebased" in env_config.keys()
        self.reset_capa_idx = 1
        self.proposed_actions: dict[str, int] = {}
        self.proposed_confidences: dict[str, float] = {}
        self.last_action_agent: Union[str, None] = None

    # pylint: disable=too-many-branches
    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """
        This function performs a single step in the environment.
        """

        # build basic dicts, that are overwritten by acting agents
        observations: Dict[str, Union[OrderedDict[str, Any], int]] = {}
        rewards: Dict[str, Any] = {}
        terminateds = {
            "__all__": False,
        }
        truncateds = {
            "__all__": False,
        }
        infos: Dict[str, Any] = {}

        # print(action_dict)
        if "high_level_agent" in action_dict.keys():
            observations = self.perform_high_level_action(action_dict)
        elif (
            ("do_nothing_agent" in action_dict.keys())
            and (
                not any(
                    key.startswith("reinforcement_learning_agent")
                    for key in action_dict.keys()
                )
            )
            and (
                not any(
                    key.startswith("value_reinforcement_learning_agent")
                    for key in action_dict.keys()
                )
            )
        ):
            # step do nothing in environment
            self.previous_obs, reward, terminated, truncated, info = self.env_gym.step(
                0
            )

            self.perform_prio_update(terminated)

            if self.last_action_agent:
                rewards = self.assign_multi_agent_rewards(
                    self.last_action_agent, reward
                )

            observations = {"high_level_agent": max(self.previous_obs["rho"])}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {"__common__": info}

            self.reset_capa_idx = 1
        elif "choose_substation_agent" in action_dict.keys():
            output_substation_id = action_dict["choose_substation_agent"]
            substation_id, action = self.extract_substation_to_act(output_substation_id)

            self.previous_obs, reward, terminated, truncated, info = self.env_gym.step(
                action
            )
            self.perform_prio_update(terminated)

            # reward the RL agent for this step, go back to HL agent
            observations = {"high_level_agent": max(self.previous_obs["rho"])}

            # get the last action agent as ID, possibly convertable by middle_to_substation_map
            self.last_action_agent = substation_id
            rewards = self.assign_multi_agent_rewards(substation_id, reward)
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {"__common__": info}

        elif any(
            key.startswith("reinforcement_learning_agent") for key in action_dict.keys()
        ):
            self.proposed_actions = self.extract_proposed_actions(action_dict)
            observations = {
                "choose_substation_agent": OrderedDict(
                    {
                        "proposed_actions": {
                            str(sub_id): action
                            for sub_id, action in self.proposed_actions.items()
                        },
                    }
                )
            }

            if not self.is_rulebased:  # also provide additional informatoin
                if isinstance(observations["choose_substation_agent"], OrderedDict):
                    if self.is_capa:
                        # add reset_capa_idx to observations
                        observations["choose_substation_agent"][
                            "reset_capa_idx"
                        ] = self.reset_capa_idx
                    observations["choose_substation_agent"][
                        "previous_obs"
                    ] = self.previous_obs
                else:
                    raise ValueError("Capa observations is not an OrderedDict.")

            self.reset_capa_idx = 0
        elif any(
            key.startswith("value_reinforcement_learning_agent")
            for key in action_dict.keys()
        ):
            (
                self.proposed_actions,
                self.proposed_confidences,
            ) = self.extract_proposed_actions_values(action_dict)

            if (
                not self.is_rulebased
            ):  # also provide additional information, obs need to be first, otherwise ray breaks
                observations = {
                    "choose_substation_agent": OrderedDict(
                        {
                            "previous_obs": self.previous_obs,
                        }
                    )
                }

            # obs already exists
            if isinstance(observations["choose_substation_agent"], OrderedDict):
                observations["choose_substation_agent"]["proposed_actions"] = {
                    str(sub_id): action
                    for sub_id, action in self.proposed_actions.items()
                }
                observations["choose_substation_agent"]["proposed_confidences"] = {
                    str(sub_id): confidence
                    for sub_id, confidence in self.proposed_confidences.items()
                }
            else:
                observations = {
                    "choose_substation_agent": OrderedDict(
                        {
                            "proposed_actions": {
                                str(sub_id): action
                                for sub_id, action in self.proposed_actions.items()
                            },
                            "proposed_confidences": {
                                str(sub_id): confidence
                                for sub_id, confidence in self.proposed_confidences.items()
                            },
                        }
                    )
                }

        elif bool(action_dict) is False:
            pass
        else:
            raise ValueError("No agent found in action dictionary in step().")

        if "__common__" in infos:
            if "abbc_action" in infos["__common__"]:
                # print(f"Info in rllib: {infos['__common__']}")
                self.last_action_agent = None

        return observations, rewards, terminateds, truncateds, infos

    def assign_multi_agent_rewards(
        self, substation_id: str, reward: float
    ) -> dict[str, float]:
        """
        Assigns rewards to multiple agents in the environment.

        Parameters:
            substation_id (int): The ID of the substation agent to assign the reward to.
                Use -1 to assign the reward to the "choose_substation_agent".
            reward (float): The reward value to assign.

        Returns:
            dict[str, float]: A dictionary containing the assigned rewards for each agent.
        """
        # Do not reward do-nothing
        # if substation_id == "-1":
        #     rewards = {}
        # else:
        if self.is_value_rl:
            rewards = {
                f"value_reinforcement_learning_agent_{substation_id}": reward,
            }
        else:
            rewards = {
                f"reinforcement_learning_agent_{substation_id}": reward,
            }

        # print(f"Rewarded subid: {rewards}")

        # if the middle agent is not learned, award it
        if not self.is_capa and not self.is_rulebased:
            rewards["choose_substation_agent"] = reward

        return rewards

    def extract_substation_to_act(self, substation_id: str) -> tuple[str, int]:
        """
        Extracts the action corresponding to the given substation ID.

        Parameters:
            substation_id (int): The ID of the substation.

        Returns:
            int: The action corresponding to the given substation ID.

        Raises:
            ValueError: If the substation ID is not an integer.
        """
        if self.is_capa or self.is_rulebased:
            action = self.proposed_actions[str(substation_id)]
        else:
            substation_id = self.middle_to_substation_map[str(substation_id)]
            local_action = self.proposed_actions[str(substation_id)]
            action = self.local_to_global_action_map[str(substation_id)][local_action]
        return substation_id, action

    def extract_proposed_actions_values(
        self, action_dict: MultiAgentDict
    ) -> tuple[dict[str, int], dict[str, float]]:
        """
        Extract all proposed actions and vluesfrom the action_dict.
        """
        # assert that there are as many keys that start with "value_reinforcement_learning_agent"
        # as there are keys that start with "value_function_agent"
        assert len(
            [
                key
                for key in action_dict.keys()
                if key.startswith("value_reinforcement_learning_agent")
            ]
        ) == len(
            [
                key
                for key in action_dict.keys()
                if key.startswith("value_function_agent")
            ]
        )

        proposed_actions: dict[str, int] = {}
        proposed_confidences: dict[str, float] = {}

        for key, action in action_dict.items():
            if not key == "do_nothing_agent":
                # extract integer at end of key
                agent_id = str(key.split("_")[-1])
                if key.startswith("value_reinforcement_learning_agent"):
                    proposed_actions[agent_id] = int(action)
                if key.startswith("value_function_agent"):
                    proposed_confidences[agent_id] = float(action)
            # else:
            #     # NOTE: The confidence for the do-nothing action is set to 0
            #     proposed_actions["-1"] = 0
            #     proposed_confidences["-1"] = 0.0

        return proposed_actions, proposed_confidences

    def extract_proposed_actions(self, action_dict: MultiAgentDict) -> dict[str, int]:
        """
        Extract all proposed actions from the action_dict.
        """
        proposed_actions: dict[str, int] = {}
        for key, local_action in action_dict.items():
            if not key == "do_nothing_agent":
                # extract integer at end of key
                agent_id = str(key.split("_")[-1])

                if self.is_rulebased or self.is_capa:
                    # convert action back to global
                    global_action = self.local_to_global_action_map[agent_id][
                        local_action
                    ]
                    proposed_actions[agent_id] = int(global_action)
                else:
                    proposed_actions[agent_id] = int(local_action)
            # else:
            #     proposed_actions["-1"] = 0

        return proposed_actions

    def perform_high_level_action(self, action_dict: MultiAgentDict) -> MultiAgentDict:
        """
        Performs HL action for HRL-env.
        """
        # observations: Union[dict[str, int], dict[int, OrderedDict[str, Any]]] = {}
        observations: dict[str, Union[int, OrderedDict[str, Any]]] = {}
        action = action_dict["high_level_agent"]
        if action == 0:  # do something
            # add an observation key for all agents in self.rl_agent_ids
            for agent_id in self.rl_agent_ids:
                # observations[agent_id] = gym.spaces.Dict(self.previous_obs) # NOTE For the vf only?
                observations[agent_id] = self.previous_obs

            # # also add do nothing agent
            # observations["do_nothing_agent"] = 0
        elif action == 1:  # do nothing
            observations = {"do_nothing_agent": 0}
        else:
            raise ValueError(
                "An invalid action is selected by the high_level_agent in step()."
            )

        return observations

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError


register_env(
    "HierarchicalCustomizedGrid2OpEnvironment",
    HierarchicalCustomizedGrid2OpEnvironment,
)

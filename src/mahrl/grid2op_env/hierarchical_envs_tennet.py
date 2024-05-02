"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
import numpy as np

import gymnasium as gym
from grid2op.Action import BaseAction
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mahrl.evaluation.evaluation_agents import (
    create_greedy_agent_per_substation,
    get_actions_per_substation,
)
from mahrl.experiments.utils import (
    find_list_of_agents,
)
from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment

OBSTYPE = TypeVar("OBSTYPE")
ACTTYPE = TypeVar("ACTTYPE")
RENDERFRAME = TypeVar("RENDERFRAME")


class GreedyHierarchicalCustomizedGrid2OpEnvironment(CustomizedGrid2OpEnvironment):
    """
    Implement step function for hierarchical environment. This is made to work with greedy-agent lower
    level agents. Their action is built into the environment.
    """

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        # create greedy agents for each substation
        controllable_substations = find_list_of_agents(self.env_g2op, env_config["action_space"])

        self.agents = create_greedy_agent_per_substation(
            self.env_g2op,
            env_config,
            controllable_substations,
            self.possible_substation_actions,
        )

        # determine the acting agents
        self._agent_ids = [
            "high_level_agent",
            "choose_substation_agent",
            "do_nothing_agent",
        ]

        self.reset_capa_idx = 1
        self.g2op_obs = None
        self.proposed_g2op_actions: dict[int, BaseAction] = {}
        print('Environment is GREEDY!')

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        This function resets the environment.
        """
        # Adjusted reset to also get g2op_obs
        self.reset_capa_idx = 1
        self.g2op_obs = self.env_g2op.reset()
        self.cur_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)
        observations = {"high_level_agent": max(self.cur_obs["rho"])}
        return observations, {}

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """
        This function performs a single step in the environment.
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

        if "high_level_agent" in action_dict.keys():
            action = action_dict["high_level_agent"]
            if action == 0:  # do something
                self.proposed_g2op_actions = {
                    sub_id: agent.act(self.g2op_obs, reward=None)
                    for sub_id, agent in self.agents.items()
                }

                observation_for_middle_agent = OrderedDict(
                    {
                        # "cur_obs": self.cur_obs,  # NOTE Pass entire obs
                        "proposed_actions": {
                            str(sub_id): self.env_gym.action_space.to_gym(action)
                            for sub_id, action in self.proposed_g2op_actions.items()
                        },
                        # "reset_capa_idx": self.reset_capa_idx,
                    }
                )
                observations = {"choose_substation_agent": observation_for_middle_agent}
                self.reset_capa_idx = 0
            elif action == 1:  # do nothing
                self.reset_capa_idx = 1
                observations = {"do_nothing_agent": 0}
            else:
                raise ValueError(
                    "An invalid action is selected by the high_level_agent in step()."
                )
        elif "do_nothing_agent" in action_dict.keys():
            # step do nothing in environment
            self.g2op_obs, reward, terminated, info = self.env_g2op.step(
                self.env_g2op.action_space({})
            )
            self.cur_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

            # reward the RL agent for this step, go back to HL agent
            rewards = {"choose_substation_agent": reward}
            observations = {"high_level_agent": max(self.cur_obs["rho"])}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": False}
            infos = {"__common__": info}
        elif "choose_substation_agent" in action_dict.keys():
            substation_id = action_dict["choose_substation_agent"]
            if substation_id == -1:
                g2op_action = self.env_g2op.action_space({})
            else:
                g2op_action = self.proposed_g2op_actions[substation_id]

            self.g2op_obs, reward, terminated, info = self.env_g2op.step(g2op_action)
            self.cur_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

            # reward the RL agent for this step, go back to HL agent
            observations = {"high_level_agent": max(self.cur_obs["rho"])}
            rewards = {"choose_substation_agent": reward}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": False}
            infos = {"__common__": info}
        elif bool(action_dict) is False:
            pass
            # print("Caution: Empty action dictionary!")
        else:
            raise ValueError("No agent found in action dictionary in step().")

        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError


register_env(
    "GreedyHierarchicalCustomizedGrid2OpEnvironment",
    GreedyHierarchicalCustomizedGrid2OpEnvironment,
)


class HierarchicalCustomizedGrid2OpEnvironment(CustomizedGrid2OpEnvironment):
    """
    Implement step function for hierarchical environment. This is made to work with greedy-agent lower
    level agents. Their action is built into the environment.
    """

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        controllable_substations = find_list_of_agents(self.env_g2op, env_config["action_space"])
        # add all RL agents
        list_of_substations = list(controllable_substations.keys())

        print(" ENVIRONMENT : HierarchicalCustomizedGrid2OpEnvironment")
        actions_per_substations = get_actions_per_substation(
            controllable_substations=controllable_substations,
            possible_substation_actions=self.possible_substation_actions,
        )
        # map individual substation action to global action space
        # for each possible substation action, match the action in self.possible_substation_actions
        self.local_to_global_action_map = {}
        for sub_idx, actions in actions_per_substations.items():
            # print(f"sub id {sub_idx} has {len(actions)} actions")
            self.local_to_global_action_map[sub_idx] = {
                local_idx: self.possible_substation_actions.index(global_action)
                for local_idx, global_action in enumerate(actions)
            }

        # map the middle output substation to the substation id
        self.middle_to_substation_map = dict(enumerate(list_of_substations))

        self.is_capa = "capa" in env_config.keys()
        self.reset_capa_idx = 1
        self.proposed_actions: dict[int, int] = {}
        self.proposed_confidences: dict[int, float] = {}

    def define_agents(self, env_config: dict) -> list:
        controllable_substations = find_list_of_agents(self.env_g2op, env_config["action_space"])
        self.rl_agent_ids = []
        for sub_idx in controllable_substations.keys():
            # add agent
            self.rl_agent_ids.append(f"reinforcement_learning_agent_{sub_idx}")

        # determine the acting agents
        return self.rl_agent_ids + [
            "high_level_agent",
            "choose_substation_agent",
            "do_nothing_agent",
        ]

    def define_action_space(self, env_config: dict) -> gym.Space:
        # General action space only used if action_space for policy is set to None
        return gym.spaces.Discrete(
            len(self.possible_substation_actions)
        )

    def define_obs_space(self, env_config: dict) -> gym.Space:
        # General observation space only used if observation_space for policy is set to None
        return gym.spaces.Dict(
            dict(self.env_gym.observation_space.spaces.items())
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        This function resets the environment.
        """
        self.reset_capa_idx = 1
        return super().reset()

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

        if "high_level_agent" in action_dict.keys():
            observations = self.perform_high_level_action(action_dict)
        elif "do_nothing_agent" in action_dict.keys():
            # step do nothing in environment
            # Execute action DN agent:
            g2op_obs, reward, terminated, info = self.gym_act_in_g2op(0)

            # reward the RL agent for this step, go back to HL agent
            rewards = {"choose_substation_agent": reward}
            observations = {"high_level_agent": max(self.cur_obs["rho"]).flatten()}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": g2op_obs.current_step == g2op_obs.max_step}
            infos = {}

            self.reset_capa_idx = 1
        elif "choose_substation_agent" in action_dict.keys():
            substation_id = action_dict["choose_substation_agent"]
            action = self.extract_substation_to_act(
                action_dict["choose_substation_agent"]
            )

            # Execute action given by DN or RL agent:
            g2op_obs, reward, terminated, infos = self.gym_act_in_g2op(action)

            # reward the RL agent for this step, go back to HL agent
            observations = {"high_level_agent": max(self.cur_obs["rho"]).flatten()}
            rewards = self.assign_multi_agent_rewards(substation_id, reward)
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": g2op_obs.current_step == g2op_obs.max_step}
            infos = {}
        elif any(
            key.startswith("reinforcement_learning_agent") for key in action_dict.keys()
        ):
            self.proposed_actions = self.extract_proposed_actions(action_dict)
            # print('proposed actions: ', self.proposed_actions)
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

            if self.is_capa:
                if isinstance(observations["choose_substation_agent"], OrderedDict):
                    # add reset_capa_idx to observations
                    observations["choose_substation_agent"][
                        "reset_capa_idx"
                    ] = self.reset_capa_idx
                    observations["choose_substation_agent"][
                        "previous_obs"
                    ] = self.cur_obs
                else:
                    raise ValueError("Capa observations is not an OrderedDict.")
            # print('observation: ', observations)
            self.reset_capa_idx = 0
        elif any(
            key.startswith("value_reinforcement_learning_agent")
            for key in action_dict.keys()
        ):
            (
                self.proposed_actions,
                self.proposed_confidences,
            ) = self.extract_proposed_actions_values(action_dict)
            observations = {
                "choose_substation_agent": OrderedDict(
                    {
                        # "cur_obs": self.cur_obs,
                        "proposed_actions": {
                            str(sub_id): action
                            for sub_id, action in self.proposed_actions.items()
                        },
                        "proposed_confidences": {
                            str(sub_id): confidence
                            for sub_id, confidence in self.proposed_confidences.items()
                        },
                        # "reset_capa_idx": self.reset_capa_idx,
                    }
                )
            }
            self.reset_capa_idx = 0
        elif bool(action_dict) is False:
            # print("Caution: Empty action dictionary!")
            pass
        else:
            raise ValueError("No agent found in action dictionary in step().")

        return observations, rewards, terminateds, truncateds, infos

    def assign_multi_agent_rewards(
        self, substation_id: int, reward: float
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
        if substation_id == -1:
            rewards = {"choose_substation_agent": reward}
        else:
            rewards = {
                "choose_substation_agent": reward,
                f"reinforcement_learning_agent_{substation_id}": reward,
            }
        return rewards

    def extract_substation_to_act(self, substation_id: int) -> int:
        """
        Extracts the action corresponding to the given substation ID.

        Parameters:
            substation_id (int): The ID of the substation.

        Returns:
            int: The action corresponding to the given substation ID.

        Raises:
            ValueError: If the substation ID is not an integer.
        """
        if substation_id == -1:
            action = 0
        else:
            if not self.is_capa:
                substation_id = self.middle_to_substation_map[substation_id]
            local_action = self.proposed_actions[substation_id]
            action = self.local_to_global_action_map[substation_id][local_action]
        return action

    def extract_proposed_actions_values(
        self, action_dict: MultiAgentDict
    ) -> tuple[dict[int, int], dict[int, float]]:
        """
        Extract all proposed actions and vluesfrom the action_dict.
        """
        proposed_actions: dict[int, int] = {}
        proposed_confidences: dict[int, float] = {}
        for key, action in action_dict.items():
            # extract integer at end of key
            agent_id = int(key.split("_")[-1])
            proposed_actions[agent_id] = int(action["action"])
            proposed_confidences[agent_id] = float(action["value"])

        return proposed_actions, proposed_confidences

    def extract_proposed_actions(self, action_dict: MultiAgentDict) -> dict[int, int]:
        """
        Extract all proposed actions from the action_dict.
        """
        proposed_actions: dict[int, int] = {}
        for key, action in action_dict.items():
            # extract integer at end of key
            agent_id = int(key.split("_")[-1])
            proposed_actions[agent_id] = int(action)

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
                observations[agent_id] = self.cur_obs
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
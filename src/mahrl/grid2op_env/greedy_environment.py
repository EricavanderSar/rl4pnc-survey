"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, TypeVar

from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mahrl.evaluation.evaluation_agents import create_greedy_agent_per_substation
from mahrl.grid2op_env.custom_environment import (
    HierarchicalCustomizedGrid2OpEnvironment,
)
from mahrl.grid2op_env.utils import reconnecting_and_abbc

RENDERFRAME = TypeVar("RENDERFRAME")


class GreedyHierarchicalCustomizedGrid2OpEnvironment(
    HierarchicalCustomizedGrid2OpEnvironment
):
    """
    Implement step function for hierarchical environment. This is made to work with greedy-agent lower
    level agents. Their action is built into the environment.
    """

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        self.agents = create_greedy_agent_per_substation(
            self.grid2op_env,
            env_config,
            self.agent_per_substation,
            self.possible_substation_actions,
        )

        # determine the acting agents
        self._agent_ids = [
            "high_level_agent",
            "choose_substation_agent",
            "do_nothing_agent",
        ]

        self.is_greedy = True
        self.g2op_obs = None
        self.reconnect_lines: list[int] = []

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
        self.g2op_obs = self.grid2op_env.reset()
        self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)
        observations = {"high_level_agent": max(self.previous_obs["rho"])}
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
            observations = self.select_high_level_action(observations, action_dict)
        elif "do_nothing_agent" in action_dict.keys():
            (
                observations,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.execute_do_nothing()
        elif "choose_substation_agent" in action_dict.keys():
            (
                observations,
                rewards,
                terminateds,
                truncateds,
                infos,
            ) = self.select_substation(action_dict)
        elif bool(action_dict) is False:
            pass
        else:
            raise ValueError("No agent found in action dictionary in step().")

        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError

    def select_high_level_action(
        self, observations: MultiAgentDict, action_dict: MultiAgentDict
    ) -> MultiAgentDict:
        """
        Selects a high-level action based on the given action dictionary.

        Args:
            action_dict (MultiAgentDict): A dictionary containing the high-level agent's action.

        Returns:
            MultiAgentDict: A dictionary containing the selected high-level action.

        Raises:
            ValueError: If an invalid action is selected by the high_level_agent in step().
        """
        if action_dict["high_level_agent"] == 0:  # do something
            self.proposed_actions = {
                str(sub_id): self.get_local_from_global(
                    self.env_gym.action_space.to_gym(
                        agent.act(self.g2op_obs, reward=None)
                    )
                )
                for sub_id, agent in self.agents.items()
            }

            observation_for_middle_agent = OrderedDict(
                {
                    "previous_obs": self.previous_obs,
                    "proposed_actions": {
                        str(sub_id): action
                        for sub_id, action in self.proposed_actions.items()
                    },
                }
            )

            if self.is_capa:
                if isinstance(observations["choose_substation_agent"], OrderedDict):
                    # add reset_capa_idx to observations
                    observations["choose_substation_agent"][
                        "reset_capa_idx"
                    ] = self.reset_capa_idx
                else:
                    raise ValueError("Capa observations is not an OrderedDict.")

            observations = {"choose_substation_agent": observation_for_middle_agent}
            self.reset_capa_idx = 0
        elif action_dict["high_level_agent"] == 1:  # do nothing
            self.reset_capa_idx = 1
            observations = {"do_nothing_agent": 0}
        else:
            raise ValueError(
                "An invalid action is selected by the high_level_agent in step()."
            )
        return observations

    def execute_do_nothing(
        self,
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """
        Executes the "do nothing" action in the environment.

        Returns:
            A tuple containing the following:
            - observations: A dictionary of observations for each agent.
            - rewards: A dictionary of rewards for each agent.
            - terminateds: A dictionary indicating whether each agent is terminated.
            - truncateds: A dictionary indicating whether each agent's episode is truncated.
            - infos: A dictionary of additional information for each agent.
        """
        # step do nothing in environment
        g2op_action = self.grid2op_env.action_space({})

        if self.reconnect_lines:
            for line in self.reconnect_lines:
                g2op_action = g2op_action + line
            self.reconnect_lines = []

        self.g2op_obs, reward, terminated, info = self.grid2op_env.step(g2op_action)
        self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

        self.reconnect_lines, info = reconnecting_and_abbc(
            self.grid2op_env, self.g2op_obs, self.reconnect_lines, info
        )

        self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

        # reward the RL agent for this step, go back to HL agent
        if not self.pause_reward:
            rewards = {"choose_substation_agent": reward}

        observations = {"high_level_agent": max(self.previous_obs["rho"])}
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": False}
        infos = {"__common__": info}
        return observations, rewards, terminateds, truncateds, infos

    def select_substation(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """
        Selects a substation based on the given action dictionary.

        Args:
            action_dict (MultiAgentDict): The action dictionary containing the chosen substation agent.

        Returns:
            Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
            A tuple containing the observations, rewards, termination flags, truncation flags, and information dictionary.

        """
        self.pause_reward = False
        _, gym_action = self.extract_substation_to_act(
            action_dict["choose_substation_agent"]
        )

        g2op_action = self.converter.convert_act(gym_action)

        if self.reconnect_lines:
            for line in self.reconnect_lines:
                g2op_action = g2op_action + line
            self.reconnect_lines = []

        self.g2op_obs, reward, terminated, info = self.grid2op_env.step(g2op_action)
        self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

        self.reconnect_lines, info = reconnecting_and_abbc(
            self.grid2op_env, self.g2op_obs, self.reconnect_lines, info
        )

        # reward the RL agent for this step, go back to HL agent
        observations = {"high_level_agent": max(self.previous_obs["rho"])}
        rewards = {"choose_substation_agent": reward}
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": False}
        infos = {"__common__": info}
        return observations, rewards, terminateds, truncateds, infos

    def get_local_from_global(self, global_action: int) -> int:
        """
        Converts a global action to its corresponding local action.

        Parameters:
            global_action (int): The global action to be converted.

        Returns:
            int: The corresponding local action.

        Raises:
            ValueError: If the global action is not found in the inner dictionary.
        """
        # return the do-nothing action if asked
        if global_action == 0:
            return 0
        # otherwise get the local action
        for _, sub_dict in self.local_to_global_action_map.items():
            for local_action, value in sub_dict.items():
                if value == global_action:
                    return local_action
        raise ValueError("Value not found in inner dict.")


register_env(
    "GreedyHierarchicalCustomizedGrid2OpEnvironment",
    GreedyHierarchicalCustomizedGrid2OpEnvironment,
)

"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""

import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import gymnasium
from grid2op.gym_compat import GymEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mahrl.evaluation.evaluation_agents import (
    create_greedy_agent_per_substation,
    get_actions_per_substation,
)
from mahrl.experiments.utils import (
    calculate_action_space_asymmetry,
    calculate_action_space_medha,
    calculate_action_space_tennet,
    find_list_of_agents,
)
from mahrl.grid2op_env.utils import (
    CustomDiscreteActions,
    load_action_space,
    make_g2op_env,
    rescale_observation_space,
    setup_converter,
)

CHANGEABLE_SUBSTATIONS = [0, 2, 3]

OBSTYPE = TypeVar("OBSTYPE")
ACTTYPE = TypeVar("ACTTYPE")
RENDERFRAME = TypeVar("RENDERFRAME")


class CustomizedGrid2OpEnvironment(MultiAgentEnv):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__()

        # create the grid2op environment
        self.grid2op_env = make_g2op_env(env_config)

        # create the gym environment
        if env_config["shuffle_scenarios"]:
            self.env_gym = GymEnv(self.grid2op_env, shuffle_chronics=True)
        else:  # ensure the evaluation chronics are not shuffled
            self.env_gym = GymEnv(self.grid2op_env, shuffle_chronics=False)

        # setting up custom action space
        path = os.path.join(
            env_config["lib_dir"],
            f"data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
        )
        self.possible_substation_actions = load_action_space(path, self.grid2op_env)

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
        self.action_space = gymnasium.spaces.Discrete(
            len(self.possible_substation_actions)
        )

        # customize observation space
        self.env_gym.observation_space = self.env_gym.observation_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect", "p_or", "p_ex", "timestep_overflow"]
        )

        # rescale observation space
        self.env_gym.observation_space = rescale_observation_space(
            self.env_gym.observation_space, self.grid2op_env
        )

        # specific to rllib
        self.observation_space = gymnasium.spaces.Dict(
            dict(self.env_gym.observation_space.spaces.items())
        )

        # determine agent ids
        self._agent_ids = [
            "high_level_agent",
            "reinforcement_learning_agent",
            "do_nothing_agent",
        ]

        # setup shared parameters
        self.previous_obs: OrderedDict[str, Any] = OrderedDict()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        This function resets the environment. Observation is passed to HL agent.
        """
        self.previous_obs, info = self.env_gym.reset()
        return {"high_level_agent": max(self.previous_obs["rho"])}, {"__common__": info}

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

        # check which agent is acting
        if "high_level_agent" in action_dict.keys():
            if action_dict["high_level_agent"] == 0:  # do something
                observations = {"reinforcement_learning_agent": self.previous_obs}
            elif action_dict["high_level_agent"] == 1:  # do nothing
                observations = {"do_nothing_agent": 0}
            else:
                raise ValueError(
                    "An invalid action is selected by the high_level_agent in step()."
                )
        elif "do_nothing_agent" in action_dict.keys():
            # perform do nothing in the env
            (
                self.previous_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_gym.step(action_dict["do_nothing_agent"])

            # reward the RL agent for this step, go back to HL agent
            rewards = {"reinforcement_learning_agent": reward}
            observations = {"high_level_agent": max(self.previous_obs["rho"])}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {"__common__": info}
        elif "reinforcement_learning_agent" in action_dict.keys():
            # perform RL step in the env
            (
                self.previous_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_gym.step(action_dict["reinforcement_learning_agent"])
            # reward the RL agent for this step, go back to HL agent
            rewards = {"reinforcement_learning_agent": reward}
            observations = {"high_level_agent": max(self.previous_obs["rho"])}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {"__common__": info}
        elif bool(action_dict) is False:
            # print("Caution: Empty action dictionary!")
            pass
        else:
            raise ValueError("No valid agent found in action dictionary in step().")

        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError


register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)


class GreedyHierarchicalCustomizedGrid2OpEnvironment(CustomizedGrid2OpEnvironment):
    """
    Implement step function for hierarchical environment. This is made to work with greedy-agent lower
    level agents. Their action is built into the environment.
    """

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        # create greedy agents for each substation
        if env_config["action_space"] == "asymmetry":
            _, _, controllable_substations = calculate_action_space_asymmetry(
                self.grid2op_env
            )
        elif env_config["action_space"] == "medha":
            _, _, controllable_substations = calculate_action_space_medha(
                self.grid2op_env
            )
        elif env_config["action_space"] == "tennet":
            _, _, controllable_substations = calculate_action_space_tennet(
                self.grid2op_env
            )
        else:
            raise ValueError("No action valid space is defined.")

        self.agents = create_greedy_agent_per_substation(
            self.grid2op_env,
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
        self.proposed_actions: dict[int, int] = {}

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
            action = action_dict["high_level_agent"]
            if action == 0:  # do something
                proposed_g2op_actions = {
                    sub_id: agent.act(self.g2op_obs, reward=None)
                    for sub_id, agent in self.agents.items()
                }

                observation_for_middle_agent = OrderedDict(
                    {
                        "previous_obs": self.previous_obs,  # NOTE Pass entire obs
                        "proposed_actions": {
                            str(sub_id): self.converter.revert_act(action)
                            for sub_id, action in proposed_g2op_actions.items()
                        },
                        "reset_capa_idx": self.reset_capa_idx,
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
            self.g2op_obs, reward, terminated, info = self.grid2op_env.step(
                self.grid2op_env.action_space({})
            )
            self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

            # reward the RL agent for this step, go back to HL agent
            rewards = {"choose_substation_agent": reward}
            observations = {"high_level_agent": max(self.previous_obs["rho"])}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": False}
            infos = {"__common__": info}
        elif "choose_substation_agent" in action_dict.keys():
            substation_id = action_dict["choose_substation_agent"]
            if substation_id == -1:
                g2op_action = self.grid2op_env.action_space({})
            else:
                g2op_action = self.proposed_actions[substation_id]

            self.g2op_obs, reward, terminated, info = self.grid2op_env.step(g2op_action)
            self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

            # reward the RL agent for this step, go back to HL agent
            observations = {"high_level_agent": max(self.previous_obs["rho"])}
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

        # add all RL agents
        list_of_substations = list(
            find_list_of_agents(
                self.grid2op_env,
                env_config["action_space"],
            ).keys()
        )

        self.rl_agent_ids = []

        # get changeable substations
        if env_config["action_space"] == "asymmetry":
            _, _, controllable_substations = calculate_action_space_asymmetry(
                self.grid2op_env
            )
        elif env_config["action_space"] == "medha":
            _, _, controllable_substations = calculate_action_space_medha(
                self.grid2op_env
            )
        elif env_config["action_space"] == "tennet":
            _, _, controllable_substations = calculate_action_space_tennet(
                self.grid2op_env
            )
        else:
            raise ValueError("No action valid space is defined.")

        actions_per_substations = get_actions_per_substation(
            controllable_substations=controllable_substations,
            possible_substation_actions=self.possible_substation_actions,
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
        self.middle_to_substation_map = dict(enumerate(list_of_substations))

        for sub_idx in list_of_substations:
            # add agent
            self.rl_agent_ids.append(f"reinforcement_learning_agent_{sub_idx}")

        # determine the acting agents
        self._agent_ids = self.rl_agent_ids + [
            "high_level_agent",
            "choose_substation_agent",
            "do_nothing_agent",
        ]

        self.is_capa = env_config["capa"]
        self.reset_capa_idx = 1
        self.g2op_obs = None
        self.proposed_actions: dict[int, int] = {}

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
        self.previous_obs, _ = self.env_gym.reset()
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
            self.previous_obs, reward, terminated, truncated, info = self.env_gym.step(
                0
            )

            # reward the RL agent for this step, go back to HL agent
            rewards = {"choose_substation_agent": reward}
            observations = {"high_level_agent": max(self.previous_obs["rho"])}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {"__common__": info}

            self.reset_capa_idx = 1
        elif "choose_substation_agent" in action_dict.keys():
            substation_id = action_dict["choose_substation_agent"]
            action = self.extract_substation_to_act(
                action_dict["choose_substation_agent"]
            )

            self.previous_obs, reward, terminated, truncated, info = self.env_gym.step(
                action
            )

            # reward the RL agent for this step, go back to HL agent
            observations = {"high_level_agent": max(self.previous_obs["rho"])}
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
                        "previous_obs": self.previous_obs,
                        "proposed_actions": {
                            str(sub_id): action
                            for sub_id, action in self.proposed_actions.items()
                        },
                        "reset_capa_idx": self.reset_capa_idx,
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

    def extract_proposed_actions(self, action_dict: MultiAgentDict) -> dict[int, int]:
        """
        Extract all proposed actions from the action_dict.
        """
        proposed_actions: dict[int, int] = {}
        for key, action in action_dict.items():
            # extract integer at end of key
            agent_id = int(key.split("_")[-1])
            proposed_actions[agent_id] = action

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
                observations[agent_id] = self.previous_obs
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


class SingleAgentGrid2OpEnvironment(gymnasium.Env):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        # create the grid2op environment
        self.grid2op_env = make_g2op_env(env_config)

        # create the gym environment
        self.env_gym = GymEnv(self.grid2op_env)
        # NOTE: Difference is not using the custom made GymEnv

        # setting up custom action space
        path = os.path.join(
            env_config["lib_dir"],
            f"data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
        )
        self.possible_substation_actions = load_action_space(path, self.grid2op_env)

        # TODO: Implement a check that the 1st action is in fact do-nothing

        # create converter
        converter = setup_converter(self.grid2op_env, self.possible_substation_actions)

        # set gym action space to discrete
        self.env_gym.action_space = CustomDiscreteActions(converter)

        # specific to rllib
        self.action_space = gymnasium.spaces.Discrete(
            len(self.possible_substation_actions)
        )

        # customize observation space
        self.env_gym.observation_space = self.env_gym.observation_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect", "p_or", "p_ex", "timestep_overflow"]
        )

        # rescale observation space
        self.env_gym.observation_space = rescale_observation_space(
            self.env_gym.observation_space, self.grid2op_env
        )

        # specific to rllib
        self.observation_space = gymnasium.spaces.Dict(
            dict(self.env_gym.observation_space.spaces.items())
        )

        # setup shared parameters
        self.rho_threshold = env_config["rho_threshold"]
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[OBSTYPE, dict[str, Any]]:  # type: ignore
        """
        This function resets the environment.
        """
        done = True
        while done:
            obs = self.env_gym.reset()

            if obs is not None:
                obs = obs[0]  # remove timeseries ID
            else:
                raise ValueError("Observation is None.")

            # find first step that surpasses threshold
            done = False
            self.steps = 0
            while (max(obs["rho"]) < self.rho_threshold) and (not done):
                obs, _, done, _, _ = self.env_gym.step(0)
                self.steps += 1

        return obs, {}

    def step(
        self,
        action: int,
    ) -> tuple[OBSTYPE | None, float, bool, bool, dict[str, Any]]:
        """
        This function performs a single step in the environment.
        """
        cum_reward: float = 0.0
        # obs, reward, done, truncated, info = self.env_gym.step(action)
        obs, reward, done, truncated, info = self.env_gym.step(action)
        self.steps += 1
        cum_reward += reward
        while (max(obs["rho"]) < self.rho_threshold) and (not done):
            # obs, reward, done, truncated, _ = self.env_gym.step(0)
            obs, reward, done, truncated, _ = self.env_gym.step(0)
            # obs, reward, done, _ = self.env_gym.step(self.do_nothing_actions[0])
            self.steps += 1
            cum_reward += reward

        if done:
            info["steps"] = self.steps
        return obs, cum_reward, done, truncated, info

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError


register_env("SingleAgentGrid2OpEnvironment", SingleAgentGrid2OpEnvironment)

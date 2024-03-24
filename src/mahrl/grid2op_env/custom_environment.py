"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""

import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import grid2op
from grid2op.Observation import BaseObservation
import gymnasium
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import GymEnv, BoxGymObsSpace
from l2rpn_baselines.utils import GymEnvWithHeuristics
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mahrl.evaluation.evaluation_agents import create_greedy_agent_per_substation
from mahrl.experiments.utils import (
    calculate_action_space_asymmetry,
    calculate_action_space_medha,
    calculate_action_space_tennet,
)
from mahrl.grid2op_env.utils import (
    CustomDiscreteActions,
    make_g2op_env,
    setup_converter,
    load_action_space,
    rescale_observation_space,
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
        self.env_gym = GymEnv(self.grid2op_env)

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
        self.env_gym.action_space = CustomDiscreteActions(
            self.converter, self.grid2op_env.action_space()
        )

        # specific to rllib
        self.action_space = gymnasium.spaces.Discrete(
            len(self.possible_substation_actions)
        )

        # customize observation space
        self.env_gym.observation_space = self.env_gym.observation_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect", "p_or", "p_ex", "timestep_overflow"]
        )

        print(f"obs_space: {self.env_gym.observation_space}")

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
        return {"high_level_agent": self.previous_obs}, {"__common__": info}

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
                observations = {"do_nothing_agent": self.previous_obs}
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
            observations = {"high_level_agent": self.previous_obs}
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
            observations = {"high_level_agent": self.previous_obs}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {"__common__": info}
        elif bool(action_dict) is False:
            print("Caution: Empty action dictionary!")
        else:
            raise ValueError("No valid agent found in action dictionary in step().")

        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError


register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)


# class ProvideGreedyInfo(GymEnvWithHeuristics):
#     """Provides information about greedy actions per substation."""

#     def __init__(self, env: BaseEnv):
#         super().__init__(env)

#     def heuristic_actions(
#         self, g2op_obs: BaseObservation, reward: float, done: bool, info: Dict
#     ) -> List[BaseAction]:
#         """Returns a dictionary with greedy actions per substation."""
#         return []


# class CustomizedGrid2OpEnvironment(MultiAgentEnv):
#     """Encapsulate Grid2Op environment and set action/observation space."""

#     def __init__(self, env_config: dict[str, Any]):
#         super().__init__()

#         # create the grid2op environment
#         self.grid2op_env = make_g2op_env(env_config)

#         # create the gym environment
#         self.env_gym = ProvideGreedyInfo(self.grid2op_env)

#         # setting up custom action space
#         path = os.path.join(
#             env_config["lib_dir"],
#             f"data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
#         )
#         self.possible_substation_actions = load_action_space(path, self.grid2op_env)

#         # add the do-nothing action at index 0
#         do_nothing_action = self.grid2op_env.action_space({})
#         self.possible_substation_actions.insert(0, do_nothing_action)

#         # create converter
#         converter = setup_converter(self.grid2op_env, self.possible_substation_actions)

#         # set gym action space to discrete
#         self.env_gym.action_space = CustomDiscreteActions(
#             converter, self.grid2op_env.action_space()
#         )

#         # specific to rllib
#         self.action_space = gymnasium.spaces.Discrete(
#             len(self.possible_substation_actions)
#         )

#         # customize observation space
#         self.env_gym.observation_space = self.env_gym.observation_space.keep_only_attr(
#             ["rho", "gen_p", "load_p", "topo_vect", "p_or", "p_ex", "timestep_overflow"]
#         )

#         # rescale observation space
#         self.env_gym.observation_space = rescale_observation_space(
#             self.env_gym.observation_space, self.grid2op_env
#         )

#         # specific to rllib
#         self.observation_space = gymnasium.spaces.Dict(
#             dict(self.env_gym.observation_space.spaces.items())
#         )

#         # determine agent ids
#         self._agent_ids = [
#             "high_level_agent",
#             "reinforcement_learning_agent",
#             "do_nothing_agent",
#         ]

#         # setup shared parameters
#         self.previous_obs: OrderedDict[str, Any] = OrderedDict()

#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         options: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[MultiAgentDict, MultiAgentDict]:
#         """
#         This function resets the environment. Observation is passed to HL agent.
#         """
#         self.previous_obs, info = self.env_gym.reset()
#         return {"high_level_agent": self.previous_obs}, {"__common__": info}

#     def step(
#         self, action_dict: MultiAgentDict
#     ) -> Tuple[
#         MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
#     ]:
#         """
#         This function performs a single step in the environment.
#         """
#         # build basic dicts, that are overwritten by acting agents
#         observations: Dict[str, Any] = {}
#         rewards: Dict[str, Any] = {}
#         terminateds = {
#             "__all__": False,
#         }
#         truncateds = {
#             "__all__": False,
#         }
#         infos: Dict[str, Any] = {}

#         # check which agent is acting
#         if "high_level_agent" in action_dict.keys():
#             if action_dict["high_level_agent"] == 0:  # do something
#                 observations = {"reinforcement_learning_agent": self.previous_obs}
#             elif action_dict["high_level_agent"] == 1:  # do nothing
#                 observations = {"do_nothing_agent": self.previous_obs}
#             else:
#                 raise ValueError(
#                     "An invalid action is selected by the high_level_agent in step()."
#                 )
#         elif "do_nothing_agent" in action_dict.keys():
#             # perform do nothing in the env
#             (
#                 self.previous_obs,
#                 reward,
#                 terminated,
#                 truncated,
#                 info,
#             ) = self.env_gym.step(action_dict["do_nothing_agent"])

#             # reward the RL agent for this step, go back to HL agent
#             rewards = {"reinforcement_learning_agent": reward}
#             observations = {"high_level_agent": self.previous_obs}
#             terminateds = {"__all__": terminated}
#             truncateds = {"__all__": truncated}
#             infos = {"__common__": info}
#         elif "reinforcement_learning_agent" in action_dict.keys():
#             # perform RL step in the env
#             (
#                 self.previous_obs,
#                 reward,
#                 terminated,
#                 truncated,
#                 info,
#             ) = self.env_gym.step(action_dict["reinforcement_learning_agent"])
#             # reward the RL agent for this step, go back to HL agent
#             rewards = {"reinforcement_learning_agent": reward}
#             observations = {"high_level_agent": self.previous_obs}
#             terminateds = {"__all__": terminated}
#             truncateds = {"__all__": truncated}
#             infos = {"__common__": info}
#         elif bool(action_dict) is False:
#             print("Caution: Empty action dictionary!")
#         else:
#             raise ValueError("No valid agent found in action dictionary in step().")

#         return observations, rewards, terminateds, truncateds, infos

#     def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
#         """
#         Not implemented.
#         """
#         raise NotImplementedError


# register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)


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

        # the middle agent can either be capa, or RL based
        if "capa" in env_config:
            self.capa_middle = True
            self.reset_capa_idx = 1
        else:
            self.capa_middle = False

        # determine the acting agents
        self._agent_ids = [
            "high_level_agent",
            "choose_substation_agent",
            "do_nothing_agent",
        ]

        # TODO: Set observation space similar to CAPA, for middle agent.
        self.g2op_obs = None
        self.proposed_actions: dict[int, dict[str, Any]] = {}

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
        observations = {"high_level_agent": self.previous_obs}
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
                self.proposed_actions = {
                    sub_id: agent.act(self.g2op_obs, reward=None)
                    for sub_id, agent in self.agents.items()
                }

                observation_for_middle_agent = OrderedDict(
                    {
                        "previous_obs": self.previous_obs,  # NOTE Pass entire obs
                        "proposed_actions": {
                            str(sub_id): self.converter.revert_act(action)
                            for sub_id, action in self.proposed_actions.items()
                        },
                        "reset_capa_idx": self.reset_capa_idx,
                    }
                )
                observations = {"choose_substation_agent": observation_for_middle_agent}
                self.reset_capa_idx = 0
            elif action == 1:  # do nothing
                self.reset_capa_idx = 1
                observations = {"do_nothing_agent": self.previous_obs}
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
            observations = {"high_level_agent": self.previous_obs}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": False}
            infos = {"__common__": info}
        elif "choose_substation_agent" in action_dict.keys():
            # if self.capa_middle:  # capa middle agent returns g2op action
            #     g2op_action = action_dict["choose_substation_agent"]
            # else:  # RL agent returns rllib action, convert to g2op
            substation_id = action_dict["choose_substation_agent"]
            if substation_id == -1:
                g2op_action = self.grid2op_env.action_space({})
            else:
                g2op_action = self.proposed_actions[substation_id]

            print(g2op_action)

            self.g2op_obs, reward, terminated, info = self.grid2op_env.step(g2op_action)
            self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

            # reward the RL agent for this step, go back to HL agent
            observations = {"high_level_agent": self.previous_obs}
            rewards = {"choose_substation_agent": reward}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": False}
            infos = {"__common__": info}
        elif bool(action_dict) is False:
            print("Caution: Empty action dictionary!")
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

        # the middle agent can either be capa, or RL based
        if "capa" in env_config:
            self.capa_middle = True
            self.reset_capa_idx = 1
        else:
            self.capa_middle = False

        # determine the acting agents
        self._agent_ids = [
            "high_level_agent",
            "choose_substation_agent",
            "do_nothing_agent",
        ]

        # TODO: Set observation space similar to CAPA, for middle agent.
        self.g2op_obs = None
        self.proposed_actions: dict[int, dict[str, Any]] = {}

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
        observations = {"high_level_agent": self.previous_obs}
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
                self.proposed_actions = {
                    sub_id: agent.act(self.g2op_obs, reward=None)
                    for sub_id, agent in self.agents.items()
                }

                observation_for_middle_agent = OrderedDict(
                    {
                        "previous_obs": self.previous_obs,  # NOTE Pass entire obs
                        "proposed_actions": {
                            str(sub_id): self.converter.revert_act(action)
                            for sub_id, action in self.proposed_actions.items()
                        },
                        "reset_capa_idx": self.reset_capa_idx,
                    }
                )
                observations = {"choose_substation_agent": observation_for_middle_agent}
                self.reset_capa_idx = 0
            elif action == 1:  # do nothing
                self.reset_capa_idx = 1
                observations = {"do_nothing_agent": self.previous_obs}
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
            observations = {"high_level_agent": self.previous_obs}
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
            observations = {"high_level_agent": self.previous_obs}
            rewards = {"choose_substation_agent": reward}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": False}
            infos = {"__common__": info}
        elif any(
            key.startswith("reinforcement_learning_agent") for key in action_dict.keys()
        ):

            # extract key
            if len(action_dict) == 1:
                key = next(iter(action_dict))
            else:
                raise ValueError(
                    "Only one reinforcement_learning_agent should be in action_dict."
                )

        elif bool(action_dict) is False:
            print("Caution: Empty action dictionary!")
        else:
            raise ValueError("No agent found in action dictionary in step().")

        return observations, rewards, terminateds, truncateds, infos

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
        converter = setup_converter(self.grid2op_env, self.possible_substation_actions)

        # set gym action space to discrete
        self.env_gym.action_space = CustomDiscreteActions(
            converter, self.grid2op_env.action_space()
        )

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
        self.obs = None
        self.game_over = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[OBSTYPE, dict[str, Any]]:  # type: ignore
        """
        This function resets the environment.
        """
        done = False
        self.steps = 0
        self.obs = self.env_gym.reset()
        self.obs = self.obs[0]  # remove timeseries ID

        # do nothing until rho is above threshold and action is required
        while (max(self.obs["rho"]) < self.rho_threshold) and (not done):
            self.obs, _, done, _, _ = self.env_gym.step(0)
            self.steps += 1

        if done:
            self.game_over = True

        return self.obs, {}

    def step(
        self,
        action: int,
    ) -> tuple[OBSTYPE, float, bool, bool, dict[str, Any]]:
        """
        This function performs a single step in the environment.
        """
        if not self.game_over:
            self.obs, reward, done, truncated, info = self.env_gym.step(action)
            self.steps += 1
            cum_reward = reward

            # do nothing until rho is above threshold and action is required
            while (max(self.obs["rho"]) < self.rho_threshold) and (not done):
                self.obs, reward, done, truncated, _ = self.env_gym.step(0)
                self.steps += 1
                cum_reward += reward

        else:  # if done right after reset function, skip step function
            done = True
            truncated = False
            cum_reward = 0
            info = {}

        if done:
            info["steps"] = self.steps

        return self.obs, cum_reward, done, truncated, info

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError


register_env("SingleAgentGrid2OpEnvironment", SingleAgentGrid2OpEnvironment)

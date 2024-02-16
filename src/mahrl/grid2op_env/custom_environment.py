"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""

import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import grid2op
import gymnasium
from grid2op.Action import BaseAction
from grid2op.gym_compat import GymEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mahrl.grid2op_env.utils import (
    CustomDiscreteActions,
    get_possible_topologies,
    setup_converter,
)

CHANGEABLE_SUBSTATIONS = [0, 2, 3]

OBSTYPE = TypeVar("OBSTYPE")
ACTTYPE = TypeVar("ACTTYPE")
RENDERFRAME = TypeVar("RENDERFRAME")

# Configure the logging module
# logging.basicConfig(filename="example.log", level=logging.INFO)


class CustomizedGrid2OpEnvironment(MultiAgentEnv):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__()

        self._agent_ids = [
            "high_level_agent",
            "reinforcement_learning_agent",
            "do_nothing_agent",
        ]

        # 1. create the grid2op environment
        if "env_name" not in env_config:
            raise RuntimeError(
                "The configuration for RLLIB should provide the env name"
            )
        self.max_tsteps = env_config["max_tsteps"]
        lib_dir = env_config["lib_dir"]

        self.env_glop = grid2op.make(
            env_config["env_name"], **env_config["grid2op_kwargs"]
        )
        self.env_glop.seed(env_config["seed"])
        # TODO: CHeck if wnat to use
        self.env_glop.chronics_handler.set_chunk_size(288)

        # 1.a. Setting up custom action space
        if (
            env_config["action_space"] == "asymmetry"
            or env_config["action_space"] == "medha"
            or env_config["action_space"] == "tennet"
        ):
            path = os.path.join(
                lib_dir,
                f"data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
            )
            possible_substation_actions = self.load_action_space(path)
        elif env_config["action_space"] == "erica":
            possible_substation_actions = get_possible_topologies(
                self.env_glop, CHANGEABLE_SUBSTATIONS
            )
        else:
            path = os.path.join(
                lib_dir,
                f"data/action_spaces/{env_config['env_name']}/asymmetry.json",
            )
            possible_substation_actions = self.load_action_space(path)
            logging.warning(
                "No valid space is defined, using asymmetrical action space."
            )

        logging.info(f"LEN ACTIONS={len(possible_substation_actions)}")
        # Add the do-nothing action at index 0
        do_nothing_action = self.env_glop.action_space({})
        possible_substation_actions.insert(0, do_nothing_action)

        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_glop)
        self.env_gym.reset()

        # 3. customize action and observation space space to only change bus
        # create converter
        converter = setup_converter(self.env_glop, possible_substation_actions)

        # set gym action space to discrete
        self.env_gym.action_space = CustomDiscreteActions(
            converter, self.env_glop.action_space()
        )
        # customize observation space
        ob_space = self.env_gym.observation_space
        ob_space = ob_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect", "p_or", "p_ex", "timestep_overflow"]
        )

        self.env_gym.observation_space = ob_space

        # 4. specific to rllib
        self.action_space = gymnasium.spaces.Discrete(len(possible_substation_actions))
        self.observation_space = gymnasium.spaces.Dict(
            dict(self.env_gym.observation_space.spaces.items())
        )

        self.previous_obs: OrderedDict[str, Any] = OrderedDict()
        self.step_nb = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        This function resets the environment.
        """
        self.previous_obs, infos = self.env_gym.reset()
        observations = {"high_level_agent": self.previous_obs}
        return observations, infos

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """
        This function performs a single step in the environment.
        """

        # Increase step
        self.step_nb = self.step_nb + 1

        # Build termination dict
        terminateds = {
            "__all__": self.step_nb >= self.max_tsteps,
        }

        truncateds = {
            "__all__": False,
        }

        rewards: Dict[str, Any] = {}
        infos: Dict[str, Any] = {}

        logging.info(f"ACTION_DICT = {action_dict}")

        if "high_level_agent" in action_dict.keys():
            action = action_dict["high_level_agent"]
            if action == 0:
                # do something
                observations = {"reinforcement_learning_agent": self.previous_obs}
            elif action == 1:
                # do nothing
                observations = {"do_nothing_agent": self.previous_obs}
            else:
                raise ValueError(
                    "An invalid action is selected by the high_level_agent in step()."
                )
        elif "do_nothing_agent" in action_dict.keys():
            logging.info("do_nothing_agent IS CALLED: DO NOTHING")

            # overwrite action in action_dict to nothing
            action = action_dict["do_nothing_agent"]
            (
                self.previous_obs,
                reward,
                terminated,
                truncated,
                infos,
            ) = self.env_gym.step(action)

            # still give reward to RL agent
            rewards = {"reinforcement_learning_agent": reward}
            observations = {"high_level_agent": self.previous_obs}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {}
        elif "reinforcement_learning_agent" in action_dict.keys():
            logging.info("reinforcement_learning_agent IS CALLED: DO SOMETHING")
            action = action_dict["reinforcement_learning_agent"]

            (
                self.previous_obs,
                reward,
                terminated,
                truncated,
                infos,
            ) = self.env_gym.step(action)

            # give reward to RL agent
            rewards = {"reinforcement_learning_agent": reward}
            observations = {"high_level_agent": self.previous_obs}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {}
        elif bool(action_dict) is False:
            logging.info("Caution: Empty action dictionary!")
            rewards = {}
            observations = {}
            infos = {}
        else:
            logging.info(f"ACTION_DICT={action_dict}")
            raise ValueError("No agent found in action dictionary in step().")

        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError

    def load_action_space(self, path: str) -> List[BaseAction]:
        """
        Loads the action space from a specified folder.
        """
        with open(path, "rt", encoding="utf-8") as action_set_file:
            return list(
                (
                    self.env_glop.action_space(action_dict)
                    for action_dict in json.load(action_set_file)
                )
            )


register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)


# class HierarchicalCustomizedGrid2OpEnvironment(CustomizedGrid2OpEnvironment):
#     """
#     Implement step function for hierarchical environment.
#     """

#     def __init__(self, env_config: dict[str, Any]):
#         super().__init__(env_config)

#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         options: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[MultiAgentDict, MultiAgentDict]:
#         """
#         This function resets the environment.
#         """
#         return super().reset()

#     def step(
#         self, action_dict: MultiAgentDict
#     ) -> Tuple[
#         MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
#     ]:
#         """
#         This function performs a single step in the environment.
#         """

#         # TODO: Adjust step function to handle middle level agent

#         # Increase step
#         self.step_nb = self.step_nb + 1

#         # Build termination dict
#         terminateds = {
#             "__all__": self.step_nb >= self.max_tsteps,
#         }

#         truncateds = {
#             "__all__": False,
#         }

#         rewards: Dict[str, Any] = {}
#         infos: Dict[str, Any] = {}

#         logging.info(f"ACTION_DICT = {action_dict}")

#         if "high_level_agent" in action_dict.keys():
#             action = action_dict["high_level_agent"]
#             if action == 0:
#                 # do something
#                 observations = {"choose_substation_agent": self.previous_obs}
#             elif action == 1:
#                 # do nothing
#                 observations = {"do_nothing_agent": self.previous_obs}
#             else:
#                 raise ValueError(
#                     "An invalid action is selected by the high_level_agent in step()."
#                 )
#         elif "do_nothing_agent" in action_dict.keys():
#             logging.info("do_nothing_agent IS CALLED: DO NOTHING")

#             # overwrite action in action_dict to nothing
#             action = action_dict["do_nothing_agent"]
#             (
#                 self.previous_obs,
#                 reward,
#                 terminated,
#                 truncated,
#                 infos,
#             ) = self.env_gym.step(action)

#             # still give reward to RL agent
#             rewards = {"reinforcement_learning_agent": reward}
#             observations = {"high_level_agent": self.previous_obs}
#             terminateds = {"__all__": terminated}
#             truncateds = {"__all__": truncated}
#             infos = {}
#         elif any(
#             key.startswith("reinforcement_learning_agent") for key in action_dict.keys()
#         ):
#             logging.info("reinforcement_learning_agent IS CALLED: DO SOMETHING")

#             # extract key
#             if len(action_dict) == 1:
#                 key = next(iter(action_dict))
#             else:
#                 raise ValueError(
#                     "Only one reinforcement_learning_agent should be in action_dict."
#                 )

#             action = action_dict[key]

#             (
#                 self.previous_obs,
#                 reward,
#                 terminated,
#                 truncated,
#                 infos,
#             ) = self.env_gym.step(action)

#             # TODO: How to setup reward system?
#             rewards = {key: reward}
#             observations = {"high_level_agent": self.previous_obs}
#             terminateds = {"__all__": terminated}
#             truncateds = {"__all__": truncated}
#             infos = {}
#         elif "choose_substation_agent" in action_dict.keys():
#             logging.info("choose_substation_agent IS CALLED: DO SOMETHING")

#             action = action_dict["choose_substation_agent"]
#             observations = {f"reinforcement_learning_agent_{action}": self.previous_obs}
#         elif bool(action_dict) is False:
#             logging.info("Caution: Empty action dictionary!")
#             rewards = {}
#             observations = {}
#             infos = {}
#         else:
#             logging.info(f"ACTION_DICT={action_dict}")
#             raise ValueError("No agent found in action dictionary in step().")

#         return observations, rewards, terminateds, truncateds, infos

#     def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
#         """
#         Not implemented.
#         """
#         super().render()

#     def load_action_space(self, path: str) -> List[BaseAction]:
#         """
#         Loads the action space from a specified folder.
#         """
#         return super().load_action_space(path)

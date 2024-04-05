"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""

import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import grid2op
from lightsim2grid import LightSimBackend
import gymnasium as gym
from grid2op.Action import BaseAction
from grid2op.gym_compat import GymEnv
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
    get_possible_topologies,
    setup_converter,
    rename_env,
)

OBSTYPE = TypeVar("OBSTYPE")
ACTTYPE = TypeVar("ACTTYPE")
RENDERFRAME = TypeVar("RENDERFRAME")

# Configure the logging module
# logging.basicConfig(filename="example.log", level=logging.INFO)


class CustomizedGrid2OpEnvironment(MultiAgentEnv):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__()
        self._skip_env_checking = True

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

        self.env_g2op = grid2op.make(
            env_config["env_name"], **env_config["grid2op_kwargs"], backend=LightSimBackend()
        )
        self.env_g2op.seed(env_config["seed"])
        rename_env(self.env_g2op)
        # 1.a. Setting up custom action space
        if (
            env_config["action_space"] == "asymmetry"
            or env_config["action_space"] == "medha"
            or env_config["action_space"] == "tennet"
        ):
            path = os.path.join(
                lib_dir,
                f"data/action_spaces/{self.env_g2op.env_name}/{env_config['action_space']}.json",
            )
            self.possible_substation_actions = self.load_action_space(path)
        elif env_config["action_space"] == "masked":
            mask = env_config.get("mask", 3)
            subs = [i for i, big_enough in enumerate(self.env_g2op.action_space.sub_info > mask) if big_enough]
            self.possible_substation_actions = get_possible_topologies(
                self.env_g2op, subs
            )
            print('subs to act: ', subs)
        else:
            path = os.path.join(
                lib_dir,
                f"data/action_spaces/{self.env_g2op.env_name}/asymmetry.json",
            )
            self.possible_substation_actions = self.load_action_space(path)
            logging.warning(
                "No valid space is defined, using asymmetrical action space."
            )
        print('action_space is ', env_config.get("action_space"))
        print('number possible sub actions: ', len(self.possible_substation_actions))

        logging.info(f"LEN ACTIONS={len(self.possible_substation_actions)}")
        # Add the do-nothing action at index 0
        do_nothing_action = self.env_g2op.action_space({})
        self.possible_substation_actions.insert(0, do_nothing_action)

        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_g2op, shuffle_chronics=env_config["shuffle_scenarios"])
        self.env_gym.reset()

        # 3. customize action and observation space space to only change bus
        # create converter
        converter = setup_converter(self.env_g2op, self.possible_substation_actions)

        # set gym action space to discrete
        self.env_gym.action_space = CustomDiscreteActions(
            converter, self.env_g2op.action_space()
        )
        # customize observation space
        ob_space = self.env_gym.observation_space
        ob_space = ob_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "p_or", "p_ex", "timestep_overflow", "topo_vect"]
        )

        self.env_gym.observation_space = ob_space

        # 4. specific to rllib
        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict(
            {
                "high_level_agent": gym.spaces.Discrete(2),
                "reinforcement_learning_agent": gym.spaces.Discrete(len(self.possible_substation_actions)),
                "do_nothing_agent": gym.spaces.Discrete(1),
            }
        )
        self.observation_space = gym.spaces.Dict(
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

        # # Increase step
        # self.step_nb = self.step_nb + 1

        # Build termination dict
        terminateds = {
            "__all__": self.step_nb >= self.max_tsteps,
        }
        if self.step_nb >= self.max_tsteps:
            # terminate when train_batch_size is collected and reset step count.
            self.step_nb = 0

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
            # Increase step
            self.step_nb = self.step_nb + 1
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
                    self.env_g2op.action_space(action_dict)
                    for action_dict in json.load(action_set_file)
                )
            )

    # def observation_space_sample(self, agent_ids: list = None):
    #     return {}

    # def action_space_sample(self, agent_ids: list = None):
    #     return {}

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
            return False
        return all(self.observation_space.contains(val) for val in x.values())

register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)


# class GreedyHierarchicalCustomizedGrid2OpEnvironment(CustomizedGrid2OpEnvironment):
#     """
#     Implement step function for hierarchical environment. This is made to work with greedy-agent lower
#     level agents. Their action is build-in the environment.
#     """

#     def __init__(self, env_config: dict[str, Any]):
#         super().__init__(env_config)

#         self.g2op_obs = None

#         # get changeable substations
#         if env_config["action_space"] == "asymmetry":
#             _, _, controllable_substations = calculate_action_space_asymmetry(
#                 self.env_glop
#             )
#         elif env_config["action_space"] == "medha":
#             _, _, controllable_substations = calculate_action_space_medha(self.env_glop)
#         elif env_config["action_space"] == "tennet":
#             _, _, controllable_substations = calculate_action_space_tennet(
#                 self.env_glop
#             )
#         else:
#             raise ValueError("No action valid space is defined.")

#         self.agents = create_greedy_agent_per_substation(
#             self.env_glop,
#             env_config,
#             controllable_substations,
#             self.possible_substation_actions,
#         )

#     def reset(
#         self,
#         *,
#         seed: Optional[int] = None,
#         options: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[MultiAgentDict, MultiAgentDict]:
#         """
#         This function resets the environment.
#         """
#         # Adjusted reset to also get g2op_obs
#         self.g2op_obs = self.env_glop.reset()
#         self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)
#         observations = {"high_level_agent": self.previous_obs}
#         return observations, {}

#     def step(
#         self, action_dict: MultiAgentDict
#     ) -> Tuple[
#         MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
#     ]:
#         """
#         This function performs a single step in the environment.
#         """

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
#             g2op_action = self.env_glop.action_space({})

#             self.g2op_obs, reward, terminated, _ = self.env_glop.step(g2op_action)

#             self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

#             # still give reward to RL agent
#             rewards = {"choose_substation_agent": reward}
#             observations = {"high_level_agent": self.previous_obs}
#             terminateds = {"__all__": terminated}
#             truncateds = {"__all__": terminated}
#             infos = {}
#         elif "choose_substation_agent" in action_dict.keys():
#             logging.info("choose_substation_agent IS CALLED: DO SOMETHING")

#             action = action_dict["choose_substation_agent"]

#             # Implement correct action on environment side, e.g. don't step
#             # on gym side, but step on grid2op side and return gym observation
#             # determine action with greedy agent
#             g2op_action = self.agents[action].act(self.g2op_obs, reward=None)
#             self.g2op_obs, reward, terminated, _ = self.env_glop.step(g2op_action)
#             self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

#             # TODO: Manually setup terminated (episode complete) or truncated (episode fail prematurely)
#             rewards = {"choose_substation_agent": reward}
#             observations = {"high_level_agent": self.previous_obs}
#             terminateds = {"__all__": terminated}
#             truncateds = {"__all__": terminated}
#             infos = {}
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
#         raise NotImplementedError


class GreedyHierarchicalCustomizedGrid2OpEnvironment(CustomizedGrid2OpEnvironment):
    """
    Implement step function for hierarchical environment. This is made to work with greedy-agent lower
    level agents. Their action is build-in the environment.
    """

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        self.g2op_obs = None

        # get changeable substations
        if env_config["action_space"] == "asymmetry":
            _, _, controllable_substations = calculate_action_space_asymmetry(
                self.env_g2op
            )
        elif env_config["action_space"] == "medha":
            _, _, controllable_substations = calculate_action_space_medha(self.env_g2op)
        elif env_config["action_space"] == "tennet":
            _, _, controllable_substations = calculate_action_space_tennet(
                self.env_g2op
            )
        else:
            raise ValueError("No action valid space is defined.")

        self.agents = create_greedy_agent_per_substation(
            self.env_g2op,
            env_config,
            controllable_substations,
            self.possible_substation_actions,
        )

        self.reset_capa_idx = True
        self.proposed_actions = None

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
        self.reset_capa_idx = True
        self.g2op_obs = self.env_g2op.reset()
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
                # TODO: Change observation so that it is all greedy agents and their proposed action, given the observation.
                self.proposed_actions = {
                    sub_id: agent.act(self.g2op_obs, reward=None)
                    for sub_id, agent in self.agents.items()
                }

                observation_for_middle_agent = OrderedDict(
                    {
                        **self.previous_obs,
                        "proposed_actions": self.proposed_actions,
                        "do_nothing_action": self.env_g2op.action_space({}),
                        "reset_capa_idx": self.reset_capa_idx,
                    }
                )

                observations = {"choose_substation_agent": observation_for_middle_agent}
                self.reset_capa_idx = False
            elif action == 1:
                # do nothing
                self.reset_capa_idx = True
                observations = {"do_nothing_agent": self.previous_obs}
            else:
                raise ValueError(
                    "An invalid action is selected by the high_level_agent in step()."
                )
        elif "do_nothing_agent" in action_dict.keys():
            logging.info("do_nothing_agent IS CALLED: DO NOTHING")

            # overwrite action in action_dict to nothing
            g2op_action = self.env_g2op.action_space({})

            self.g2op_obs, reward, terminated, _ = self.env_g2op.step(g2op_action)

            self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

            # still give reward to RL agent
            rewards = {"choose_substation_agent": reward}
            observations = {"high_level_agent": self.previous_obs}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": terminated}
            infos = {}
        elif "choose_substation_agent" in action_dict.keys():
            logging.info("choose_substation_agent IS CALLED: DO SOMETHING")

            g2op_action = action_dict["choose_substation_agent"]

            # TODO: Step proposed action of substation returned by CAPA
            # Implement correct action on environment side, e.g. don't step
            # on gym side, but step on grid2op side and return gym observation
            # determine action with greedy agent
            self.g2op_obs, reward, terminated, _ = self.env_g2op.step(g2op_action)
            self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

            # TODO: Manually setup terminated (episode complete) or truncated (episode fail prematurely)
            rewards = {"choose_substation_agent": reward}
            observations = {"high_level_agent": self.previous_obs}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": terminated}
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


register_env(
    "GreedyHierarchicalCustomizedGrid2OpEnvironment",
    GreedyHierarchicalCustomizedGrid2OpEnvironment,
)

# elif any(
#     key.startswith("reinforcement_learning_agent") for key in action_dict.keys()
# ):
#     raise NotImplementedError
# logging.info("reinforcement_learning_agent IS CALLED: DO SOMETHING")

# # extract key
# if len(action_dict) == 1:
#     key = next(iter(action_dict))
# else:
#     raise ValueError(
#         "Only one reinforcement_learning_agent should be in action_dict."
#     )

# # Implement correct action on environment side, e.g. don't step
# # on gym side, but step on grid2op side and return gym observation
# g2op_action = self.agents[]  # TODO determine action with greedy agent
# self.g2op_obs, reward, terminated, info = self.env_glop.step(g2op_action)

# self.previous_obs = self.observation_space.to_gym(self.g2op_obs)

# # TODO: How to setup reward system?
# # TODO: Manually setup terminated (episode complete) or truncated (episode fail prematurely)
# rewards = {key: reward}
# observations = {"high_level_agent": self.previous_obs}
# terminateds = {"__all__": terminated}
# truncateds = {"__all__": terminated}
# infos = {}

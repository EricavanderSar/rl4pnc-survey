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
import numpy as np
from grid2op.Action import BaseAction
from grid2op.gym_compat import GymEnv
from lightsim2grid import LightSimBackend  # pylint: disable=wrong-import-order
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mahrl.evaluation.evaluation_agents import create_greedy_agent_per_substation
from mahrl.experiments.utils import (
    calculate_action_space_asymmetry,
    calculate_action_space_medha,
    calculate_action_space_tennet,
)
from mahrl.grid2op_env.utils import CustomDiscreteActions, setup_converter

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
            env_config["env_name"],
            **env_config["grid2op_kwargs"],
            backend=LightSimBackend(),
        )
        # self.env_glop.seed(env_config["seed"])

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
        else:
            path = os.path.join(
                lib_dir,
                f"data/action_spaces/{env_config['env_name']}/asymmetry.json",
            )
            logging.warning(
                "No valid space is defined, using asymmetrical action space."
            )
        self.possible_substation_actions = self.load_action_space(path)

        logging.info(f"LEN ACTIONS={len(self.possible_substation_actions)}")
        # Add the do-nothing action at index 0
        do_nothing_action = self.env_glop.action_space({})
        self.possible_substation_actions.insert(0, do_nothing_action)

        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_glop)
        self.env_gym.reset()

        # 3. customize action and observation space space to only change bus
        # create converter
        converter = setup_converter(self.env_glop, self.possible_substation_actions)

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
        self.action_space = gymnasium.spaces.Discrete(
            len(self.possible_substation_actions)
        )
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
        # if the path contains _per_day or _train or _test or _val, then ignore this part of the string
        if "_per_day" in path:
            path = path.replace("_per_day", "")
        if "_train" in path:
            path = path.replace("_train", "")
        if "_test" in path:
            path = path.replace("_test", "")
        if "_val" in path:
            path = path.replace("_val", "")

        with open(path, "rt", encoding="utf-8") as action_set_file:
            return list(
                (
                    self.env_glop.action_space(action_dict)
                    for action_dict in json.load(action_set_file)
                )
            )


register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)


class GreedyHierarchicalCustomizedGrid2OpEnvironment(CustomizedGrid2OpEnvironment):
    """
    Implement step function for hierarchical environment. This is made to work with greedy-agent lower
    level agents. Their action is build-in the environment.
    """

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        self._agent_ids = [
            "high_level_agent",
            "choose_substation_agent",
            "do_nothing_agent",
        ]

        self.g2op_obs = None

        # get changeable substations
        if env_config["action_space"] == "asymmetry":
            _, _, controllable_substations = calculate_action_space_asymmetry(
                self.env_glop
            )
        elif env_config["action_space"] == "medha":
            _, _, controllable_substations = calculate_action_space_medha(self.env_glop)
        elif env_config["action_space"] == "tennet":
            _, _, controllable_substations = calculate_action_space_tennet(
                self.env_glop
            )
        else:
            raise ValueError("No action valid space is defined.")

        self.agents = create_greedy_agent_per_substation(
            self.env_glop,
            env_config,
            controllable_substations,
            self.possible_substation_actions,
        )

        if "capa" in env_config:
            self.capa_middle = True
            self.reset_capa_idx = 1
        else:
            self.capa_middle = False

        # TODO: Set observation space similar to CAPA, for middle agent.

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
        self.g2op_obs = self.env_glop.reset()
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
                    sub_id: agent.act(self.g2op_obs, reward=None).as_dict()
                    for sub_id, agent in self.agents.items()
                }

                print(f"actions={self.proposed_actions}")

                observation_for_middle_agent = OrderedDict(
                    {
                        "rho": self.previous_obs["rho"],
                        "proposed_actions": self.proposed_actions,
                        "do_nothing_action": self.env_glop.action_space({}).as_dict(),
                        "reset_capa_idx": self.reset_capa_idx,
                    }
                )

                # if self.capa_middle:  # capa
                #     observation_for_middle_agent = OrderedDict(
                #         {
                #             "rho": self.previous_obs["rho"],
                #             "proposed_actions": self.proposed_actions,
                #             "do_nothing_action": self.env_glop.action_space({}),
                #             "reset_capa_idx": self.reset_capa_idx,
                #         }
                #     )
                # else:  # RL AGENT
                #     observation_for_middle_agent = self.previous_obs

                observations = {"choose_substation_agent": observation_for_middle_agent}
                self.reset_capa_idx = 0
            elif action == 1:
                # do nothing
                self.reset_capa_idx = 1
                observations = {"do_nothing_agent": self.previous_obs}
            else:
                raise ValueError(
                    "An invalid action is selected by the high_level_agent in step()."
                )
        elif "do_nothing_agent" in action_dict.keys():
            logging.info("do_nothing_agent IS CALLED: DO NOTHING")

            # overwrite action in action_dict to nothing
            g2op_action = self.env_glop.action_space({})

            self.g2op_obs, reward, terminated, _ = self.env_glop.step(g2op_action)

            self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

            # still give reward to RL agent
            rewards = {"choose_substation_agent": reward}
            observations = {"high_level_agent": self.previous_obs}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": terminated}
            infos = {}
        elif "choose_substation_agent" in action_dict.keys():
            logging.info("choose_substation_agent IS CALLED: DO SOMETHING")

            if self.capa_middle:  # capa
                g2op_action = action_dict["choose_substation_agent"]
            else:  # FOR RL AGENT
                action_id = action_dict["choose_substation_agent"]
                g2op_action = self.proposed_actions[action_id]

            # TODO: Step proposed action of substation returned by CAPA
            # Implement correct action on environment side, e.g. don't step
            # on gym side, but step on grid2op side and return gym observation
            # determine action with greedy agent
            self.g2op_obs, reward, terminated, _ = self.env_glop.step(g2op_action)
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


class LowerRLHierarchicalCustomizedGrid2OpEnvironment(CustomizedGrid2OpEnvironment):
    """
    Implement step function for hierarchical environment. This is made to work with greedy-agent lower
    level agents. Their action is build-in the environment.
    """

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        # TODO: Adjust for nr lower level RL agents
        self._agent_ids = [
            "high_level_agent",
            "choose_substation_agent",
            "do_nothing_agent",
        ]

        self.g2op_obs = None

        # get changeable substations
        if env_config["action_space"] == "asymmetry":
            _, _, controllable_substations = calculate_action_space_asymmetry(
                self.env_glop
            )
        elif env_config["action_space"] == "medha":
            _, _, controllable_substations = calculate_action_space_medha(self.env_glop)
        elif env_config["action_space"] == "tennet":
            _, _, controllable_substations = calculate_action_space_tennet(
                self.env_glop
            )
        else:
            raise ValueError("No action valid space is defined.")

        self.agents = create_greedy_agent_per_substation(
            self.env_glop,
            env_config,
            controllable_substations,
            self.possible_substation_actions,
        )

        if "capa" in env_config:
            self.capa_middle = True
            self.reset_capa_idx = 1
        else:
            self.capa_middle = False

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
        self.g2op_obs = self.env_glop.reset()
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
                    sub_id: agent.act(self.g2op_obs, reward=None).as_dict()
                    for sub_id, agent in self.agents.items()
                }

                observation_for_middle_agent = OrderedDict(
                    {
                        "rho": self.previous_obs["rho"],
                        "proposed_actions": self.proposed_actions,
                        "do_nothing_action": self.env_glop.action_space({}).as_dict(),
                        "reset_capa_idx": self.reset_capa_idx,
                    }
                )

                # if self.capa_middle:  # capa
                #     observation_for_middle_agent = OrderedDict(
                #         {
                #             "rho": self.previous_obs["rho"],
                #             "proposed_actions": self.proposed_actions,
                #             "do_nothing_action": self.env_glop.action_space(
                #                 {}
                #             ).as_dict(),
                #             "reset_capa_idx": self.reset_capa_idx,
                #         }
                #     )
                # else:  # RL AGENT
                #     observation_for_middle_agent = self.previous_obs

                observations = {"choose_substation_agent": observation_for_middle_agent}
                self.reset_capa_idx = 0
            elif action == 1:
                # do nothing
                self.reset_capa_idx = 1
                observations = {"do_nothing_agent": self.previous_obs}
            else:
                raise ValueError(
                    "An invalid action is selected by the high_level_agent in step()."
                )
        elif "do_nothing_agent" in action_dict.keys():
            logging.info("do_nothing_agent IS CALLED: DO NOTHING")

            # overwrite action in action_dict to nothing
            g2op_action = self.env_glop.action_space({})

            self.g2op_obs, reward, terminated, _ = self.env_glop.step(g2op_action)

            self.previous_obs = self.env_gym.observation_space.to_gym(self.g2op_obs)

            # still give reward to RL agent
            rewards = {"choose_substation_agent": reward}
            observations = {"high_level_agent": self.previous_obs}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": terminated}
            infos = {}
        elif "choose_substation_agent" in action_dict.keys():
            logging.info("choose_substation_agent IS CALLED: DO SOMETHING")

            if self.capa_middle:  # capa
                g2op_action = action_dict["choose_substation_agent"]
            else:  # FOR RL AGENT
                action_id = action_dict["choose_substation_agent"]
                g2op_action = self.proposed_actions[action_id]

            # TODO: Step proposed action of substation returned by CAPA
            # Implement correct action on environment side, e.g. don't step
            # on gym side, but step on grid2op side and return gym observation
            # determine action with greedy agent
            self.g2op_obs, reward, terminated, _ = self.env_glop.step(g2op_action)
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
    "LowerRLHierarchicalCustomizedGrid2OpEnvironment",
    LowerRLHierarchicalCustomizedGrid2OpEnvironment,
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


class SingleAgentGrid2OpEnvironment(gymnasium.Env):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        # 1. create the grid2op environment
        if "env_name" not in env_config:
            raise RuntimeError(
                "The configuration for RLLIB should provide the env name"
            )
        self.max_tsteps = env_config["max_tsteps"]
        lib_dir = env_config["lib_dir"]

        self.env_glop = grid2op.make(
            env_config["env_name"],
            **env_config["grid2op_kwargs"],
            backend=LightSimBackend(),
        )
        # self.env_glop.seed(env_config["seed"])

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
            self.possible_substation_actions = self.load_action_space(path)
        else:
            path = os.path.join(
                lib_dir,
                f"data/action_spaces/{env_config['env_name']}/asymmetry.json",
            )
            self.possible_substation_actions = self.load_action_space(path)
            logging.warning(
                "No valid space is defined, using asymmetrical action space."
            )

        logging.info(f"LEN ACTIONS={len(self.possible_substation_actions)}")
        # Add the do-nothing action at index 0
        do_nothing_action = self.env_glop.action_space({})
        self.possible_substation_actions.insert(0, do_nothing_action)

        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_glop)
        self.env_gym.reset()

        # 3. customize action and observation space space to only change bus
        # create converter
        converter = setup_converter(self.env_glop, self.possible_substation_actions)

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
        self.action_space = gymnasium.spaces.Discrete(
            len(self.possible_substation_actions)
        )
        self.observation_space = gymnasium.spaces.Dict(
            dict(self.env_gym.observation_space.spaces.items())
        )

        # setup rho-env side
        self.rho_threshold = env_config["rho_threshold"]
        self.last_rho = 0
        self.steps = 0
        self.cum_reward = 0
        self.g2op_obs = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[OBSTYPE, dict[str, Any]]:  # type: ignore
        """
        This function resets the environment.
        """
        self.g2op_obs = self.env_glop.reset()
        self.last_rho = np.max(self.g2op_obs.to_dict()["rho"])
        self.cum_reward = 0
        self.steps = 0
        done = False

        while (self.last_rho < self.rho_threshold) and (not done):
            do_nothing_action = self.env_glop.action_space({})
            self.g2op_obs, reward, done, info = self.env_glop.step(do_nothing_action)
            self.steps += 1
            self.cum_reward += reward

        obs = self.env_gym.observation_space.to_gym(self.g2op_obs)
        if done:
            info["steps"] = self.steps
            # self.step(action=0, done=True, reward=reward, info=info, obs=obs)
            self.reset()

        return obs, {}

    def step(
        self,
        action: int,
    ) -> tuple[OBSTYPE, float, bool, bool, dict[str, Any]]:
        """
        This function performs a single step in the environment.
        """
        # if done:
        #     # collect the reward and store the number of steps in info
        #     return obs, reward, True, False, info
        # else:
        # perform the proposed action
        g2op_action = self.env_gym.action_space.from_gym(action)
        self.g2op_obs, reward, done, info = self.env_glop.step(g2op_action)
        self.last_rho = np.max(self.g2op_obs.to_dict()["rho"])

        self.cum_reward += reward

        # whenever the lines are not near overloading, do nothing
        while self.last_rho < self.rho_threshold and not done:
            do_nothing_action = self.env_glop.action_space({})
            self.g2op_obs, reward, done, info = self.env_glop.step(do_nothing_action)
            self.last_rho = np.max(self.g2op_obs.to_dict()["rho"])
            self.cum_reward += reward
            self.steps += 1

        if done:
            info["steps"] = self.steps

        truncated = False
        obs = self.env_gym.observation_space.to_gym(self.g2op_obs)
        return obs, self.cum_reward, done, truncated, info

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError

    def load_action_space(self, path: str) -> List[BaseAction]:
        """
        Loads the action space from a specified folder.
        """
        # if the path contains _per_day or _train or _test or _val, then ignore this part of the string
        if "_per_day" in path:
            path = path.replace("_per_day", "")
        if "_train" in path:
            path = path.replace("_train", "")
        if "_test" in path:
            path = path.replace("_test", "")
        if "_val" in path:
            path = path.replace("_val", "")

        with open(path, "rt", encoding="utf-8") as action_set_file:
            return list(
                (
                    self.env_glop.action_space(action_dict)
                    for action_dict in json.load(action_set_file)
                )
            )


register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)

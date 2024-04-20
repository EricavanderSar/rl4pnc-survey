"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""

import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TypeVar
import numpy as np

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
    make_g2op_env,
    ChronPrioMatrix,
)


from grid2op.gym_compat import ScalerAttrConverter
from grid2op.Parameters import Parameters
from grid2op.gym_compat.gym_obs_space import GymnasiumObservationSpace

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
        lib_dir = env_config["lib_dir"]

        self.env_g2op = make_g2op_env(env_config)
        # 1.a. Setting up custom action space
        if env_config["action_space"] == "masked":
            mask = env_config.get("mask", 3)
            subs = [i for i, big_enough in enumerate(self.env_g2op.action_space.sub_info > mask) if big_enough]
            self.possible_substation_actions = get_possible_topologies(
                self.env_g2op, subs
            )
            # print('subs to act: ', subs)
        else:
            path = os.path.join(
                lib_dir,
                f"data/action_spaces/{self.env_g2op.env_name}/{env_config['action_space']}.json",
            )
            self.possible_substation_actions = self.load_action_space(path)
        # print('action_space is ', env_config.get("action_space"))
        # print('number possible sub actions: ', len(self.possible_substation_actions))

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
        self.env_gym.observation_space = self.rescale_observation_space(lib_dir)

        # 4. specific to rllib
        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict(
            {
                "high_level_agent": gym.spaces.Discrete(2),
                "reinforcement_learning_agent": gym.spaces.Discrete(len(self.possible_substation_actions)),
                "do_nothing_agent": gym.spaces.Discrete(1),
            }
        )

        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict(
            {
                "high_level_agent": gym.spaces.Discrete(2),
                "reinforcement_learning_agent":
                    gym.spaces.Dict(
                        dict(self.env_gym.observation_space.spaces.items())
                    ),
                "do_nothing_agent": gym.spaces.Discrete(1),
            }
        )

        self.previous_obs: OrderedDict[str, Any] = OrderedDict()

        # initialize training chronic sampling weights
        self.prio = env_config.get("prio", True)
        self.chron_prios = ChronPrioMatrix(self.env_g2op)
        self.step_surv = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        This function resets the environment.
        """
        if self.prio:
            # use chronic priority
            self.env_g2op.set_id(
                self.chron_prios.sample_chron()
            )  # NOTE: this will take the previous chronic since with env_glop.reset() you will get the next
        g2op_obs = self.env_g2op.reset()
        terminated = False
        if self.prio:
            if self.chron_prios.cur_ffw > 0:
                self.env_g2op.fast_forward_chronics(self.chron_prios.cur_ffw * self.chron_prios.ffw_size)
                (
                    g2op_obs,
                    reward,
                    terminated,
                    infos,
                ) = self.env_g2op.step(self.env_g2op.action_space())
            self.step_surv = 0

        # reconnect lines if needed.
        if not terminated:
           g2op_obs, _, _ = self.reconnect_lines(g2op_obs)

        observations = {"high_level_agent": g2op_obs.rho.max().flatten()}

        chron_id = self.env_g2op.chronics_handler.get_name()
        infos = {"time serie id": chron_id}

        self.previous_obs = self.env_gym.observation_space.to_gym(g2op_obs)
        # self.previous_obs, infos = self.env_gym.reset()
        # observations = {"high_level_agent": self.previous_obs['rho'].max().flatten()}

        return observations, infos

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """
        This function performs a single step in the environment.
        """

        # Build termination dict
        terminateds = {
            "__all__": False,
        }

        truncateds = {
            "__all__": False,
        }

        rewards: Dict[str, Any] = {}
        infos: Dict[str, Any] = {}
        observations = {}

        logging.info(f"ACTION_DICT = {action_dict}")

        if "high_level_agent" in action_dict.keys():
            action = action_dict["high_level_agent"]
            if action == 0:
                # do something
                observations = {"reinforcement_learning_agent": self.previous_obs}
            elif action == 1:
                # do nothing
                observations = {"do_nothing_agent": 0}
            else:
                raise ValueError(
                    "An invalid action is selected by the high_level_agent in step()."
                )
            return observations, rewards, terminateds, truncateds, infos
        elif "do_nothing_agent" in action_dict.keys():
            logging.info("do_nothing_agent IS CALLED: DO NOTHING")

            # overwrite action in action_dict to nothing
            action = action_dict["do_nothing_agent"]
        elif "reinforcement_learning_agent" in action_dict.keys():
            logging.info("reinforcement_learning_agent IS CALLED: DO SOMETHING")
            action = action_dict["reinforcement_learning_agent"]
        elif bool(action_dict) is False:
            logging.info("Caution: Empty action dictionary!")
            return observations, rewards, terminateds, truncateds, infos
        else:
            logging.info(f"ACTION_DICT={action_dict}")
            raise ValueError("No agent found in action dictionary in step().")

        # Execute action given by DN or RL agent:
        g2op_act = self.env_gym.action_space.from_gym(action)
        (
            g2op_obs,
            reward,
            terminated,
            infos,
        ) = self.env_g2op.step(g2op_act)
        # reconnect lines if needed.
        if not terminated:
            g2op_obs, rw, terminated = self.reconnect_lines(g2op_obs)
            reward += rw
        if self.prio:
            self.step_surv += 1
            if terminated:
                self.chron_prios.update_prios(self.step_surv)
        # Give reward to RL agent
        rewards = {"reinforcement_learning_agent": reward}
        # Let high-level agent decide to act or not
        observations = {"high_level_agent": g2op_obs.rho.max().flatten()}
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": g2op_obs.current_step == g2op_obs.max_step}
        infos = {}
        self.previous_obs = self.env_gym.observation_space.to_gym(g2op_obs)
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

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
            return False
        return all(self.observation_space.contains(val) for val in x.values())

    def rescale_observation_space(self, lib_dir: str) -> GymnasiumObservationSpace:
        """
        Function that rescales the observation space.
        """
        # scale observations
        gym_obs = self.env_gym.observation_space
        gym_obs = gym_obs.keep_only_attr(
            ["rho", "gen_p", "load_p", "p_or", "p_ex", "timestep_overflow", "topo_vect"]
        )

        gym_obs = gym_obs.reencode_space(
            "gen_p",
            ScalerAttrConverter(substract=0.0, divide=self.env_g2op.gen_pmax),
        )
        gym_obs = gym_obs.reencode_space(
            "timestep_overflow",
            ScalerAttrConverter(
                substract=0.0,
                divide=Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED,  # assuming no custom params
            ),
        )

        if self.env_g2op.env_name in [
            "rte_case14_realistic",
            "rte_case5_example",
            "l2rpn_icaps_2021_large",
        ]:
            underestimation_constant = 1.2  # constant to account that our max/min are underestimated
            for attr in ["p_ex", "p_or", "load_p"]:
                path = os.path.join(
                    lib_dir,
                    f"data/scaling_arrays/{self.env_g2op.env_name}/{attr}.npy",
                )
                max_arr, min_arr = np.load(path)

                gym_obs = gym_obs.reencode_space(
                    attr,
                    ScalerAttrConverter(
                        substract=underestimation_constant * min_arr,
                        divide=underestimation_constant * (max_arr - min_arr),
                    ),
                )
        else:
            raise ValueError("This scaling is not yet implemented for this environment.")

        return gym_obs

    def reconnect_lines(self, g2op_obs: grid2op.Observation):
        if False in g2op_obs.line_status:
            disc_lines = np.where(g2op_obs.line_status == False)[0]
            for i in disc_lines:
                act = None
                # Reconnecting the line when cooldown and maintenance is over:
                if (g2op_obs.time_next_maintenance[i] != 0) & (g2op_obs.time_before_cooldown_line[i] == 0):
                    status = self.env_g2op.action_space.get_change_line_status_vect()
                    status[i] = True
                    act = self.env_g2op.action_space({"change_line_status": status})
                    if act is not None:
                        if self.prio:
                            self.step_surv += 1
                        # Execute reconnection action
                        (
                            g2op_obs,
                            rw,
                            terminated,
                            infos,
                        ) = self.env_g2op.step(act)
                        return g2op_obs, rw, terminated
        return g2op_obs, 0, False


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

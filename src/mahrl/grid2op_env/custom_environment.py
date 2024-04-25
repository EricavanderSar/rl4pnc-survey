"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""
import os
import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
import numpy as np

import grid2op
from lightsim2grid import LightSimBackend
import gymnasium as gym
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
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
    get_possible_topologies,
    setup_converter,
    make_g2op_env,
    ChronPrioMatrix,
    get_attr_list,
    load_actions
)

from grid2op.gym_compat import ScalerAttrConverter
from grid2op.Parameters import Parameters
from grid2op.gym_compat.gym_obs_space import GymnasiumObservationSpace

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

        g2op_obs, reward, terminated, info = self.init_env.step(g2op_act)

        to_reco = ~g2op_obs.line_status
        self.reconnect_line = []
        if np.any(to_reco):
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space(
                    {"set_line_status": [(line_id, +1)]}
                )
                self.reconnect_line.append(g2op_act)

        gym_obs = self.observation_space.to_gym(g2op_obs)
        truncated = False  # see https://github.com/openai/gym/pull/2752
        return gym_obs, float(reward), terminated, truncated, info


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
        self.env_g2op = make_g2op_env(env_config)
        if "env_name" not in env_config:
            raise RuntimeError(
                "The configuration for RLLIB should provide the env name"
            )
        lib_dir = env_config["lib_dir"]
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
            self.possible_substation_actions = load_actions(path, self.env_g2op)
        # print('action_space is ', env_config.get("action_space"))
        # print('number possible sub actions: ', len(self.possible_substation_actions))

        # add the do-nothing action at index 0
        do_nothing_action = self.env_g2op.action_space({})
        self.possible_substation_actions.insert(0, do_nothing_action)

        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_g2op, shuffle_chronics=env_config["shuffle_scenarios"])
        self.env_gym.reset()

        # 3. customize action and observation space space to only change bus
        # create converter
        converter = setup_converter(self.env_g2op, self.possible_substation_actions)

        # set gym action space to discrete
        self.env_gym.action_space = CustomDiscreteActions(converter)
        # customize observation space
        self.env_gym.observation_space = self.rescale_observation_space(
            lib_dir,
            env_config.get("input", ["p_i", "p_l", "r", "o"])
        )

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
        # if self.prio:
        #     # use chronic priority
        #     g2op_obs, terminated = self.prio_reset()
        # else:
        g2op_obs = self.env_g2op.reset()
        terminated = False

        # # reconnect lines if needed.
        # if not terminated:
        #    g2op_obs, _, _ = self.reconnect_lines(g2op_obs)

        observations = {"high_level_agent": g2op_obs.rho.max().flatten()}
        chron_id = self.env_g2op.chronics_handler.get_name()
        infos = {"time serie id": chron_id}

        self.previous_obs = self.env_gym.observation_space.to_gym(g2op_obs)
        # self.previous_obs, infos = self.env_gym.reset()
        # observations = {"high_level_agent": self.previous_obs['rho'].max().flatten()}

        return observations, infos

    def prio_reset(self):
        # use chronic priority
        self.env_g2op.set_id(
            self.chron_prios.sample_chron()
        )  # NOTE: this will take the previous chronic since with env_glop.reset() you will get the next
        g2op_obs = self.env_g2op.reset()
        terminated = False
        if self.chron_prios.cur_ffw > 0:
            self.env_g2op.fast_forward_chronics(self.chron_prios.cur_ffw * self.chron_prios.ffw_size)
            (
                g2op_obs,
                reward,
                terminated,
                infos,
            ) = self.env_g2op.step(self.env_g2op.action_space())
        self.step_surv = 0
        return g2op_obs, terminated

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

        # check which agent is acting
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
            # overwrite action in action_dict to nothing
            action = action_dict["do_nothing_agent"]
        elif "reinforcement_learning_agent" in action_dict.keys():
            action = action_dict["reinforcement_learning_agent"]
        elif bool(action_dict) is False:
            return observations, rewards, terminateds, truncateds, infos
        else:
            raise ValueError("No agent found in action dictionary in step().")

        # Execute action given by DN or RL agent:
        g2op_act = self.env_gym.action_space.from_gym(action)
        (
            g2op_obs,
            reward,
            terminated,
            infos,
        ) = self.env_g2op.step(g2op_act)
        # # reconnect lines if needed.
        # if not terminated:
        #     g2op_obs, rw, terminated = self.reconnect_lines(g2op_obs)
        #     reward += rw
        # if self.prio:
        #     self.step_surv += 1
        #     if terminated:
        #         self.chron_prios.update_prios(self.step_surv)
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

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
            return False
        return all(self.observation_space.contains(val) for val in x.values())

    def rescale_observation_space(self, lib_dir: str, input_attr: list = ["p_i", "p_l", "r", "o"]) -> GymnasiumObservationSpace:
        """
        Function that rescales the observation space.
        """
        # scale observations
        attr_list = get_attr_list(input_attr)
        print("Observation attributes used are: ", attr_list)
        gym_obs = self.env_gym.observation_space
        gym_obs = gym_obs.keep_only_attr(attr_list)

        if "gen_p" in attr_list:
            gym_obs = gym_obs.reencode_space(
                "gen_p",
                ScalerAttrConverter(substract=0.0, divide=self.env_g2op.gen_pmax),
            )
        if "timestep_overflow" in attr_list:
            gym_obs = gym_obs.reencode_space(
                "timestep_overflow",
                ScalerAttrConverter(
                    substract=0.0,
                    divide=Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED,  # assuming no custom params
                ),
            )
        path = os.path.join(lib_dir, f"data/scaling_arrays")
        if self.env_g2op.env_name in os.listdir(path):
            # underestimation_constant = 1.2  # constant to account that our max/min are underestimated
            for attr in ["p_ex", "p_or", "load_p"]:
                if attr in attr_list:
                    max_arr, min_arr = np.load(os.path.join(path, f"{self.env_g2op.env_name}/{attr}.npy"))
                    # values are multiplied with a constant to account that our max/min are underestimated
                    gym_obs = gym_obs.reencode_space(
                        attr,
                        ScalerAttrConverter(
                            substract=0.8 * min_arr,
                            divide=(1.2 * max_arr - 0.8 * min_arr),
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


class GreedyHierarchicalCustomizedGrid2OpEnvironment(CustomizedGrid2OpEnvironment):
    """
    Implement step function for hierarchical environment. This is made to work with greedy-agent lower
    level agents. Their action is built into the environment.
    """

    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        # create greedy agents for each substation
        if env_config["action_space"].startswith("asymmetry"):
            _, _, controllable_substations = calculate_action_space_asymmetry(
                self.env_g2op
            )
        elif env_config["action_space"].startswith("medha"):
            _, _, controllable_substations = calculate_action_space_medha(
                self.env_g2op
            )
        elif env_config["action_space"].startswith("tennet"):
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

        # determine the acting agents
        self._agent_ids = [
            "high_level_agent",
            "choose_substation_agent",
            "do_nothing_agent",
        ]

        self.reset_capa_idx = 1
        self.g2op_obs = None
        self.proposed_g2op_actions: dict[int, BaseAction] = {}

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
                self.proposed_g2op_actions = {
                    sub_id: agent.act(self.g2op_obs, reward=None)
                    for sub_id, agent in self.agents.items()
                }

                observation_for_middle_agent = OrderedDict(
                    {
                        "previous_obs": self.previous_obs,  # NOTE Pass entire obs
                        "proposed_actions": {
                            str(sub_id): self.converter.revert_act(action)
                            for sub_id, action in self.proposed_g2op_actions.items()
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
            self.g2op_obs, reward, terminated, info = self.env_g2op.step(
                self.env_g2op.action_space({})
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
                g2op_action = self.env_g2op.action_space({})
            else:
                g2op_action = self.proposed_g2op_actions[substation_id]

            self.g2op_obs, reward, terminated, info = self.env_g2op.step(g2op_action)
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
                self.env_g2op,
                env_config["action_space"],
            ).keys()
        )

        self.rl_agent_ids = []

        # get changeable substations
        if env_config["action_space"].startswith("asymmetry"):
            _, _, controllable_substations = calculate_action_space_asymmetry(
                self.env_g2op
            )
        elif env_config["action_space"].startswith("medha"):
            _, _, controllable_substations = calculate_action_space_medha(
                self.env_g2op
            )
        elif env_config["action_space"].startswith("tennet"):
            _, _, controllable_substations = calculate_action_space_tennet(
                self.env_g2op
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

        self.is_capa = "capa" in env_config["action_space"]
        self.reset_capa_idx = 1
        self.proposed_actions: dict[int, int] = {}
        self.proposed_confidences: dict[int, float] = {}

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
                        # "previous_obs": self.previous_obs,
                        "proposed_actions": {
                            str(sub_id): action
                            for sub_id, action in self.proposed_actions.items()
                        },
                        # "reset_capa_idx": self.reset_capa_idx,
                    }
                )
            }

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
                        # "previous_obs": self.previous_obs,
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


# class SingleAgentGrid2OpEnvironment(gym.Env):
#     """
#         In this version highlevel and DN actions are included within the environment. (as in implementation Blazej)
#         Encapsulate Grid2Op environment and set action/observation space.
#     """
#
#     def __init__(self, env_config: dict[str, Any]):
#         # create the grid2op environment
#         self.grid2op_env = make_g2op_env(env_config)
#
#         # create the gym environment
#         if env_config["shuffle_scenarios"]:
#             self.env_gym = ReconnectingGymEnv(self.grid2op_env, shuffle_chronics=True)
#         else:  # ensure the evaluation chronics are not shuffled
#             self.env_gym = ReconnectingGymEnv(self.grid2op_env, shuffle_chronics=False)
#
#         # setting up custom action space
#         path = os.path.join(
#             env_config["lib_dir"],
#             f"data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
#         )
#         self.possible_substation_actions = load_actions(path, self.grid2op_env)
#
#         # insert do-nothing action at index 0
#         do_nothing_action = self.grid2op_env.action_space({})
#         self.possible_substation_actions.insert(0, do_nothing_action)
#
#         # create converter
#         converter = setup_converter(self.grid2op_env, self.possible_substation_actions)
#
#         # set gym action space to discrete
#         self.env_gym.action_space = CustomDiscreteActions(converter)
#
#         # specific to rllib
#         self.action_space = gym.spaces.Discrete(
#             len(self.possible_substation_actions)
#         )
#
#         # customize observation space
#         self.env_gym.observation_space = self.env_gym.observation_space.keep_only_attr(
#             ["rho", "gen_p", "load_p", "topo_vect", "p_or", "p_ex", "timestep_overflow"]
#         )
#
#         # rescale observation space
#         self.env_gym.observation_space = self.rescale_observation_space(
#             self.env_gym.observation_space, self.grid2op_env, env_config
#         )
#
#         # specific to rllib
#         self.observation_space = gym.spaces.Dict(
#             dict(self.env_gym.observation_space.spaces.items())
#         )
#
#         # setup shared parameters
#         self.rho_threshold = env_config["rho_threshold"]
#         self.steps = 0
#
#     def reset(
#         self,
#         *,
#         seed: int | None = None,
#         options: dict[str, Any] | None = None,
#     ) -> tuple[dict[str, Any], dict[str, Any]]:  # type: ignore
#         """
#         This function resets the environment.
#         """
#         done = True
#         while done:
#             obs = self.env_gym.reset()
#
#             if obs is not None:
#                 obs = obs[0]  # remove timeseries ID
#             else:
#                 raise ValueError("Observation is None.")
#
#             # find first step that surpasses threshold
#             done = False
#             self.steps = 0
#             while (max(obs["rho"]) < self.rho_threshold) and (not done):
#                 obs, _, done, _, _ = self.env_gym.step(0)
#                 self.steps += 1
#
#         return obs, {}
#
#     def step(
#         self,
#         action: int,
#     ) -> tuple[dict[str, Any] | None, float, bool, bool, dict[str, Any]]:
#         """
#         This function performs a single step in the environment.
#         """
#         cum_reward: float = 0.0
#         obs: dict[str, Any]
#         # obs, reward, done, truncated, info = self.env_gym.step(action)
#         obs, reward, done, truncated, info = self.env_gym.step(action)
#         self.steps += 1
#         cum_reward += reward
#         while (max(obs["rho"]) < self.rho_threshold) and (not done):
#             # obs, reward, done, truncated, _ = self.env_gym.step(0)
#             obs, reward, done, truncated, _ = self.env_gym.step(0)
#             # obs, reward, done, _ = self.env_gym.step(self.do_nothing_actions[0])
#             self.steps += 1
#             cum_reward += reward
#
#         if done:
#             info["steps"] = self.steps
#         return obs, cum_reward, done, truncated, info
#
#     def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
#         """
#         Not implemented.
#         """
#         raise NotImplementedError
#
#
# register_env("SingleAgentGrid2OpEnvironment", SingleAgentGrid2OpEnvironment)

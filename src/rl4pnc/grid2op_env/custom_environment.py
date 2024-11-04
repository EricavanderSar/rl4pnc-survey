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
from grid2op.Observation import BaseObservation
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import GymEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from rl4pnc.evaluation.evaluation_agents import (
    create_greedy_agent_per_substation,
    get_actions_per_substation,
)
from rl4pnc.experiments.utils import (
    calculate_action_space_asymmetry,
    calculate_action_space_medha,
    calculate_action_space_tennet,
    find_list_of_agents,
)
from rl4pnc.grid2op_env.utils import (
    CustomDiscreteActions,
    get_possible_topologies,
    setup_converter,
    make_g2op_env,
    ChronPrioMatrix,
    ChronPrioVect,
    get_attr_list,
    load_actions
)

from grid2op.gym_compat import ScalerAttrConverter
from grid2op.Parameters import Parameters
from grid2op.gym_compat.gym_obs_space import GymnasiumObservationSpace

OBSTYPE = TypeVar("OBSTYPE")
ACTTYPE = TypeVar("ACTTYPE")
RENDERFRAME = TypeVar("RENDERFRAME")


class CustomizedGrid2OpEnvironment(MultiAgentEnv):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__()
        self._skip_env_checking = True

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
        print('action_space is ', env_config.get("action_space"))
        print('number possible sub actions: ', len(self.possible_substation_actions))

        # add the do-nothing action at index 0
        do_nothing_action = self.env_g2op.action_space({})
        self.possible_substation_actions.insert(0, do_nothing_action)

        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_g2op) #, shuffle_chronics=env_config["shuffle_scenarios"])
        self.env_gym.reset()

        # 3. Define agents:
        self._agent_ids = self.define_agents(env_config)

        # 4. customize action space to only change bus
        # create converter
        converter = setup_converter(self.env_g2op, self.possible_substation_actions)

        # set gym action space to discrete
        self.env_gym.action_space = CustomDiscreteActions(converter)
        # specific to rllib
        self.action_space = self.define_action_space(env_config)

        # 5. customize observation space
        self.env_gym.observation_space = self.rescale_observation_space(
            lib_dir,
            env_config.get("input", ["p_i", "p_l", "r", "o"])
        )
        # specific to rllib
        self.observation_space = self.define_obs_space(env_config)

        self.cur_obs = None

        # initialize training chronic sampling weights
        self.prio = env_config.get("prio", True)
        if self.prio:
            self.chron_prios = ChronPrioMatrix(self.env_g2op) if env_config.get("use_ffw", True) \
                else ChronPrioVect(self.env_g2op)
        self.step_surv = 0

        # reset topo option
        self.reset_topo = env_config.get("reset_topo", 0)

    def reset_metrics(self):
        # different metrics to keep track of episode performance
        self.interact_count = 0
        self.activated = False
        self.active_dn_count = 0
        self.reconnect_count = 0
        self.reset_count = 0

    def define_agents(self, env_config: dict) -> list:
        return [
            "high_level_agent",
            "reinforcement_learning_agent",
            "do_nothing_agent",
        ]

    def define_action_space(self, env_config: dict) -> gym.Space:
        # Defines Single Agent action space
        self._action_space_in_preferred_format = True
        return gym.spaces.Dict(
            {
                "high_level_agent": gym.spaces.Discrete(2),
                "reinforcement_learning_agent": gym.spaces.Discrete(len(self.possible_substation_actions)),
                "do_nothing_agent": gym.spaces.Discrete(1),
            }
        )

    def define_obs_space(self, env_config: dict) -> gym.Space:
        self._obs_space_in_preferred_format = True
        return gym.spaces.Dict(
            {
                "high_level_agent": gym.spaces.Discrete(2),
                "reinforcement_learning_agent":
                    gym.spaces.Dict(
                        dict(self.env_gym.observation_space.spaces.items())
                    ),
                "do_nothing_agent": gym.spaces.Discrete(1),
            }
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
        if self.prio:
            terminated = True
            while terminated:
                # use chronic priority
                g2op_obs, terminated = self.prio_reset()
        else:
            g2op_obs = self.env_g2op.reset()
        self.update_obs(g2op_obs)
        # reset episode metrics
        self.reset_metrics()
        # reconnect lines if needed.
        g2op_obs, _, _ = self.reconnect_lines(g2op_obs)

        observations = {"high_level_agent": g2op_obs.rho.max().flatten()}
        chron_id = self.env_g2op.chronics_handler.get_name()
        infos = {"time serie id": chron_id}

        return observations, infos

    def update_obs(self, g2op_obs):
        self.cur_obs = dict(self.env_gym.observation_space.to_gym(g2op_obs))

    def prio_reset(self):
        # use chronic priority
        self.env_g2op.set_id(
            self.chron_prios.sample_chron()
        )  # NOTE: this will take the previous chronic since with env_glop.reset() you will get the next
        g2op_obs = self.env_g2op.reset()
        terminated = False
        if self.chron_prios.cur_ffw > 0:
            self.env_g2op.fast_forward_chronics(self.chron_prios.cur_ffw * self.chron_prios.ffw_size)
            # Get new observation of this time step
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
                observations = {"reinforcement_learning_agent": self.cur_obs}
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
            self.activated = False
        elif "reinforcement_learning_agent" in action_dict.keys():
            action = action_dict["reinforcement_learning_agent"]
            self.activated = True
            self.interact_count += 1
        elif bool(action_dict) is False:
            return observations, rewards, terminateds, truncateds, infos
        else:
            raise ValueError("No agent found in action dictionary in step().")

        # Execute action given by DN or RL agent:
        g2op_obs, reward, terminated, infos = self.gym_act_in_g2op(action)
        # Give reward to RL agent
        rewards = {"reinforcement_learning_agent": reward}
        # Let high-level agent decide to act or not
        observations = {"high_level_agent": g2op_obs.rho.max().flatten()}
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": g2op_obs.current_step == g2op_obs.max_step}
        infos = {}
        return observations, rewards, terminateds, truncateds, infos

    def gym_act_in_g2op(self, action) -> Tuple[BaseObservation, float, bool, dict]:
        g2op_act = self.env_gym.action_space.from_gym(action)
        if self.activated:
            act_config = g2op_act.set_bus
            if np.all(self.env_g2op.current_obs.topo_vect[act_config!=0] == act_config[act_config!=0]):
                self.active_dn_count += 1
        (
            g2op_obs,
            reward,
            terminated,
            infos,
        ) = self.env_g2op.step(g2op_act)
        self.update_obs(g2op_obs)
        # reconnect lines if needed.
        if not terminated:
            g2op_obs, rw, terminated = self.reconnect_lines(g2op_obs)
            reward += rw
            if self.reset_topo and not terminated:
                g2op_obs, rw, terminated = self.reset_ref_topo(g2op_obs)
                reward += rw
        if self.prio:
            self.step_surv += 1
            if terminated:
                self.chron_prios.update_prios(self.step_surv)
        return g2op_obs, reward, terminated, infos

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
            return False
        return all(self.observation_space.contains(val) for val in x.values())

    def rescale_observation_space(self, lib_dir: str,
                                  input_attr: list = ["p_i", "p_l", "r", "o"]) -> GymnasiumObservationSpace:
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
                        self.reconnect_count += 1
                        self.update_obs(g2op_obs)
                        return g2op_obs, rw, terminated
        return g2op_obs, 0, False

    def reset_ref_topo(self, g2op_obs: grid2op.Observation):
        # The environment goes back to the reference topology when safe
        if (g2op_obs.rho.max() < self.reset_topo) and (g2op_obs.current_step < g2op_obs.max_step-1):
            # Get all subs that are not in default topology
            subs_changed = np.unique(g2op_obs._topo_vect_to_sub[g2op_obs.topo_vect != 1])
            if len(subs_changed):
                action_options = []
                max_rhos = np.zeros(len(subs_changed))
                rewards = np.zeros(len(subs_changed))
                for i, sub in enumerate(subs_changed):
                    action = self.env_g2op.action_space(
                        {"set_bus": {
                            "substations_id":
                                [(sub, np.ones(g2op_obs.sub_info[sub], dtype=int))]
                        }
                        })
                    action_options.append(action)
                    sim_obs, rw, done, info = g2op_obs.simulate(action)
                    max_rhos[i] = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
                    rewards[i] = rw
                if max_rhos[np.argmax(rewards)] < self.reset_topo:
                    act = action_options[np.argmax(rewards)]
                    # Execute the topology reset action
                    (
                        g2op_obs,
                        rw,
                        terminated,
                        infos,
                    ) = self.env_g2op.step(act)
                    self.update_obs(g2op_obs)
                    self.reset_count += 1
                    # print(act)
                    return g2op_obs, rw, terminated
        return g2op_obs, 0, False


register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)

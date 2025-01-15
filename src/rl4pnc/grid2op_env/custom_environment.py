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

from rl4pnc.grid2op_env.observation_converter import ObservationConverter

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
        self.observation_converter = self.setup_obs_converter(self.env_gym, env_config)
        self.observation_space = self.observation_converter.observation_space
        print("Observation space: ", self.observation_space)
        self.cur_gym_obs = None
        self.cur_g2op_obs = None

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
        self.observation_converter.reset_obs()

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

    def setup_obs_converter(self, env_gym: GymEnv, env_config: dict) -> ObservationConverter:
        self._obs_space_in_preferred_format = True
        return ObservationConverter(env_gym, env_config)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        This function resets the environment.
        """
        # reset episode metrics
        self.reset_metrics()
        if self.prio:
            terminated = True
            while terminated:
                # use chronic priority
                g2op_obs, terminated = self.prio_reset()
        else:
            g2op_obs = self.env_g2op.reset()
        self.update_obs(g2op_obs)
        # Start with activation of the high level agent > decide to act or not to act.
        observations = {"high_level_agent": g2op_obs.rho.max().flatten()}
        chron_id = self.env_g2op.chronics_handler.get_name()
        infos = {"time serie id": chron_id}

        return observations, infos

    def update_obs(self, g2op_obs):
        self.cur_g2op_obs = g2op_obs
        # print('gen_p: ', g2op_obs.gen_p)
        self.cur_gym_obs = self.observation_converter.convert_obs(g2op_obs)
        # print('updated_obs: ', self.cur_gym_obs)

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
                observations = {"reinforcement_learning_agent": self.cur_gym_obs}
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
        # reconnect lines if needed.
        g2op_act = self.reconnect_lines(g2op_act)
        if self.reset_topo:
            # reset topo if needed.
            g2op_act = self.reset_ref_topo(g2op_act)
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
        if self.prio:
            self.step_surv += 1
            if terminated:
                self.chron_prios.update_prios(self.step_surv)
        self.update_obs(g2op_obs)
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

    def reconnect_lines(self, g2op_action: grid2op.Action):
        line_stat_s = self.cur_g2op_obs.line_status
        cooldown = self.cur_g2op_obs.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)
        if can_be_reco.any():
            (
                sim_obs,
                _,
                _,
                _,
            ) = self.cur_g2op_obs.simulate(g2op_action)
            cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
            for id_ in (can_be_reco).nonzero()[0]:
                # reconnect all lines that improve the current action
                action = g2op_action + self.env_g2op.action_space({"set_line_status": [(id_, +1)]})
                (
                    sim_obs,
                    _,
                    _,
                    _,
                ) = self.cur_g2op_obs.simulate(action)
                if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
                    self.reconnect_count += 1
                    g2op_action = action
        # return original action
        return g2op_action

    def reset_ref_topo(self, g2op_action: grid2op.Action):
        # The environment goes back to the reference topology when safe
        if (self.cur_g2op_obs.rho.max() < self.reset_topo) and (self.cur_g2op_obs.current_step < self.cur_g2op_obs.max_step-1):
            # Get all subs that are not in default topology
            subs_changed = np.unique(self.cur_g2op_obs._topo_vect_to_sub[self.cur_g2op_obs.topo_vect != 1])
            if len(subs_changed):
                sim_obs, rw, done, info = self.cur_g2op_obs.simulate(g2op_action)
                cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
                action_options = []
                max_rhos = np.zeros(len(subs_changed))
                rewards = np.zeros(len(subs_changed))
                for i, sub in enumerate(subs_changed):
                    # Simulate going back to reference topology for each substation that has changed
                    action = self.env_g2op.action_space(
                        {"set_bus": {
                            "substations_id":
                                [(sub, np.ones(self.cur_g2op_obs.sub_info[sub], dtype=int))]
                        }
                        })
                    sim_obs, rw, done, info = self.cur_g2op_obs.simulate(action)
                    max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
                    if max_rho < cur_max_rho:
                        # if result better than currently proposed (DN) action save action as option.
                        action_options.append(action)
                        max_rhos[i] = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
                        rewards[i] = rw
                if action_options:
                    # use the best revert action.
                    act = action_options[np.argmax(rewards)]
                    self.reset_count += 1
                    # print(act)
                    return act
        return g2op_action


register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)

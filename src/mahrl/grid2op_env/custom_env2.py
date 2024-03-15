"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, TypeVar
import numpy as np
import torch
import gymnasium as gym
import grid2op
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from mahrl.grid2op_env.observation_converter import ObsConverter


class RlGrid2OpEnv(CustomizedGrid2OpEnvironment):
    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        obs_features = env_config.get("input", ["p_i", "p_l", "r", "o", "d"])
        n_power_attr = len([i for i in obs_features if i.startswith("p")])
        n_feature = len(obs_features) - (n_power_attr > 1) * (n_power_attr - 1)

        n_history = env_config.get("n_history", 6)

        # re-define RLlib observationspace:
        dim_topo = self.env_glop.observation_space.dim_topo

        self.observation_space = gym.spaces.Dict({
            "feature_matrix": gym.spaces.Box(-np.inf, np.inf, shape=(dim_topo, n_feature * n_history)),
            "topology": gym.spaces.Box(0, 2, shape=(dim_topo, dim_topo)) if env_config.get("adj_matrix") else
            gym.spaces.Box(-1, 1, shape=(dim_topo,))
        })

        self.obs_converter = ObsConverter(self.env_glop, env_config.get("danger", 0.9), attr=obs_features, n_history=n_history, adj_mat=env_config.get("adj_matrix"))
        self.cur_obs = None

        # initialize training chronic sampling weights
        self.chron_prios = ChronPrioMatrix(self.env_glop)
        self.step_surv = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        self.obs_converter.reset()

        # TODO use chronic priority
        self.env_glop.set_id(
            self.chron_prios.sample_chron()
        )  # NOTE: this will take the previous chronic since with env_glop.reset() you will get the next
        g2op_obs = self.env_glop.reset()
        if self.chron_prios.cur_ffw > 0:
            self.env_glop.fast_forward_chronics(self.chron_prios.cur_ffw * self.chron_prios.ffw_size)
            g2op_obs, *_ = self.env_glop.step(self.env_glop.action_space())
        self.step_surv = 0

        self.cur_obs = self.obs_converter.get_cur_obs(g2op_obs)
        observations = {"high_level_agent": g2op_obs.rho.max().flatten()}

        chron_id = self.env_glop.chronics_handler.get_name()
        info = {"time serie id": chron_id}
        # print("chron_id: ", chron_id)
        # print('ts: ', g2op_obs.current_step)
        return observations, info

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
        truncateds = {
            "__all__": False,
        }
        if self.step_nb >= self.max_tsteps:
            # terminate when train_batch_size is collected and reset step count.
            self.step_nb = 0

        rewards: Dict[str, Any] = {}
        infos: Dict[str, Any] = {}
        observations = {}

        logging.info(f"ACTION_DICT = {action_dict}")

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
            logging.info("do_nothing_agent IS CALLED: DO NOTHING")

            # overwrite action in action_dict to nothing
            action = action_dict["do_nothing_agent"]
        elif "reinforcement_learning_agent" in action_dict.keys():
            logging.info("reinforcement_learning_agent IS CALLED: DO SOMETHING")
            action = action_dict["reinforcement_learning_agent"]
            # Increase step
            self.step_nb = self.step_nb + 1
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
        ) = self.env_glop.step(g2op_act)
        # Save current observation
        self.cur_obs = self.obs_converter.get_cur_obs(g2op_obs)
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
        return observations, rewards, terminateds, truncateds, infos



register_env("RlGrid2OpEnv", RlGrid2OpEnv)


class ChronPrioMatrix:
    def __init__(self, env: grid2op.Environment):
        self.max_ep_dur = env.max_episode_duration()
        # initialize training chronic sampling weights
        self.ffw_size = 288
        self.max_ffw = self.max_ep_dur // self.ffw_size
        avail_chron = env.chronics_handler.real_data.available_chronics()
        self.chron_scores = torch.ones(len(avail_chron), self.max_ffw) * 2.0

        self.cur_ffw = 0
        self.chronic_idx = None

    def sample_chron(self):
        # sample training chronic
        dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(self.chron_scores.flatten()))
        record_idx = dist.sample().item()
        self.chronic_idx = record_idx // self.max_ffw
        self.cur_ffw = record_idx % self.max_ffw
        return self.chronic_idx

    def update_prios(self, steps_surv):
        pieces_played = int(np.ceil(steps_surv / self.ffw_size))
        max_steps = self.max_ep_dur - self.cur_ffw * self.ffw_size
        scores = torch.ones(pieces_played) * 2.0  # scale = 2.0
        for p in range(pieces_played):
            scores[p] *= 1 - np.sqrt((steps_surv - self.ffw_size * p) / (max_steps - self.ffw_size * p))
        self.chron_scores[self.chronic_idx][self.cur_ffw: (self.cur_ffw + pieces_played)] = scores
        # print('current chronic scores: ', self.chron_scores)


class RlGrid2OpEnv2(RlGrid2OpEnv):
    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

        # re-define RLlib observationspace:
        dim_topo = self.env_glop.observation_space.dim_topo
        obs_features = env_config.get("input", ["p_i", "p_l", "r", "o", "d"])
        n_history = env_config.get("n_history", 6)
        obs_dict = {}
        if "p_i" in obs_features:
            obs_dict["p_gen"] = gym.spaces.Box(-np.inf, np.inf, shape=(dim_topo, n_history))
            obs_dict["p_load"] = gym.spaces.Box(-np.inf, np.inf, shape=(dim_topo, n_history))
        if "p_l" in obs_features:
            obs_dict["p_line"] = gym.spaces.Box(-np.inf, np.inf, shape=(dim_topo, n_history))
        if "r" in obs_features:
            obs_dict["rho"] = gym.spaces.Box(-np.inf, np.inf, shape=(dim_topo, n_history))
        if "o" in obs_features:
            obs_dict["timestep_overflow"] = gym.spaces.Box(-np.inf, np.inf, shape=(dim_topo, n_history))
        if "d" in obs_features:
            obs_dict["danger"] = gym.spaces.Box(-np.inf, np.inf, shape=(dim_topo, n_history))
        obs_dict["topology"] = gym.spaces.Box(0, 2, shape=(dim_topo, dim_topo)) if env_config.get("adj_matrix") else gym.spaces.Box(-1, 1, shape=(dim_topo,))

        self.observation_space = gym.spaces.Dict(obs_dict)

        # TODO: create new obs converter
        self.obs_converter = ObsConverter(self.env_glop, env_config.get("danger", 0.9), attr=obs_features, n_history=n_history, adj_mat=env_config.get("adj_matrix"))
        self.cur_obs = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        self.obs_converter.reset()
        #TODO use chronic priority
        g2op_obs = self.env_glop.reset()
        self.cur_obs = self.obs_converter.get_cur_obs(g2op_obs)
        observations = {"high_level_agent": g2op_obs.rho.max().flatten()}

        chron_id = self.env_glop.chronics_handler.get_id()
        info = {"time serie id": chron_id}
        return observations, info

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
        truncateds = {
            "__all__": False,
        }
        if self.step_nb >= self.max_tsteps:
            # terminate when train_batch_size is collected and reset step count.
            self.step_nb = 0

        rewards: Dict[str, Any] = {}
        infos: Dict[str, Any] = {}
        observations = {}

        logging.info(f"ACTION_DICT = {action_dict}")

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
            logging.info("do_nothing_agent IS CALLED: DO NOTHING")

            # overwrite action in action_dict to nothing
            action = action_dict["do_nothing_agent"]
        elif "reinforcement_learning_agent" in action_dict.keys():
            logging.info("reinforcement_learning_agent IS CALLED: DO SOMETHING")
            action = action_dict["reinforcement_learning_agent"]
            # Increase step
            self.step_nb = self.step_nb + 1
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
        ) = self.env_glop.step(g2op_act)
        # Save current observation
        self.cur_obs = self.obs_converter.get_cur_obs(g2op_obs)
        # Give reward to RL agent
        rewards = {"reinforcement_learning_agent": reward}
        # Let high-level agent decide to act or not
        observations = {"high_level_agent": g2op_obs.rho.max().flatten()}
        terminateds = {"__all__": terminated}
        # truncateds = {"__all__": terminated}
        infos = {}
        return observations, rewards, terminateds, truncateds, infos

register_env("RlGrid2OpEnv2", RlGrid2OpEnv2)
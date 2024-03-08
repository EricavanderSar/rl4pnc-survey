"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, TypeVar
import numpy as np
import gymnasium as gym
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
            "adjacency_matrix": gym.spaces.Box(0, 2, shape=(dim_topo, dim_topo))
        })

        self.obs_converter = ObsConverter(self.env_glop, env_config.get("danger", 0.9), attr=obs_features, n_history=n_history)
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

register_env("RlGrid2OpEnv", RlGrid2OpEnv)
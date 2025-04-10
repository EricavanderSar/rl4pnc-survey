"""
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""

from typing import Any
import numpy as np
import os
import gymnasium as gym
from ray.tune.registry import register_env

from rl4pnc.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from rl4pnc.grid2op_env.observation_converter import ObsConverter


class RlGrid2OpEnv(CustomizedGrid2OpEnvironment):
    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

    def define_obs_space(self, env_config) -> gym.Space:
        # Adjusted observation space for a Single Agent Environment
        obs_features = env_config.get("input", ["p_i", "p_l", "r", "o", "d"])
        n_power_attr = len([i for i in obs_features if i.startswith("p")])
        n_feature = len(obs_features) - (n_power_attr > 1) * (n_power_attr - 1)
        n_history = env_config.get("n_history", 6)
        dim_topo = self.env_g2op.observation_space.dim_topo

        self.obs_converter = ObsConverter(self.env_g2op, env_config.get("danger", 0.9), attr=obs_features,
                                          n_history=n_history, adj_mat=env_config.get("adj_matrix"))

        # Normalize state observations:
        if env_config.get("normalize", '') == "zscore":
            load_path = os.path.join(
                env_config["lib_dir"],
                f"data/observations_dn/{self.env_g2op.env_name}",
            )
            self.obs_converter.load_mean_std(load_path)
        elif env_config.get("normalize", '') == "maxmin":
            path = os.path.join(
                env_config["lib_dir"],
                f"data/scaling_arrays/{self.env_g2op.env_name}",
            )
            self.obs_converter.load_max_min(load_path=path)

        # re-define RLlib observationspace:
        self._obs_space_in_preferred_format = True
        return gym.spaces.Dict(
            {
                "high_level_agent": gym.spaces.Discrete(2),
                "reinforcement_learning_agent":
                    gym.spaces.Dict({
                        "feature_matrix": gym.spaces.Box(-np.inf, np.inf, shape=(dim_topo, n_feature * n_history)),
                        "topology": gym.spaces.Box(0, 2, shape=(dim_topo, dim_topo)) if env_config.get("adj_matrix")
                        else gym.spaces.Box(-1, 1, shape=(dim_topo,))
                    }),
                "do_nothing_agent": gym.spaces.Discrete(1)
            }
        )

    def update_obs(self, g2op_obs):
        self.cur_g2op_obs = g2op_obs
        self.cur_gym_obs = self.obs_converter.get_cur_obs(g2op_obs)


register_env("RlGrid2OpEnv", RlGrid2OpEnv)

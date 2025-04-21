import numpy as np
from datetime import date
import torch
import os
import gymnasium as gym
from grid2op import Observation
from grid2op import Environment
from grid2op.Parameters import Parameters
from grid2op.gym_compat import ScalerAttrConverter
from grid2op.gym_compat.gym_obs_space import GymnasiumObservationSpace
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import GymEnv

from rl4pnc.grid2op_env.utils import get_attr_list


def extend_with_history(obs_attributes: dict, history: int):
    """
    Function that extends the observation space with history.

    The result will be that each attribute will be presented as attr_i with i in [0, history-1]
    here attr_i is i time steps in the past
    """
    cur_keys = list(obs_attributes.keys())
    hist_attr = {key + f"_{i}": value for i in range(1, history) for key, value in obs_attributes.items()}
    obs_attributes.update(hist_attr)
    for key in cur_keys:
        obs_attributes[key + "_0"] = obs_attributes[key]
        del obs_attributes[key]
    return obs_attributes


class ObservationConverter:
    def __init__(self,
                 gym_env: GymEnv,
                 env_config: dict,
                 ):
        self.env_gym = gym_env
        self.cur_gym_obs = None
        self.env_name = gym_env.init_env.env_name
        # Select attributes and normalize observation space
        self.env_gym.observation_space = self.rescale_observation_space(
            env_config["lib_dir"],
            env_config.get("g2op_input", ["p_i", "p_l", "r", "o"])
        )

        # Standard attributes of g2op that are included:
        attr = dict(self.env_gym.observation_space.spaces.items())
        # print("Used as RL obs: ", attr)
        # Custom attributes added:
        custom_attr = self.custom_observation_space(input_attr=env_config.get("custom_input"))
        self.custom_attr = list(custom_attr.keys())
        # update attribute list for observation space with custom attributes
        attr.update(custom_attr)
        # initialize danger variable in case it is used.
        self.danger = env_config.get("danger", 0.9)
        self.thermal_limit_under400 = (gym_env.init_env.get_thermal_limit() < 400)
        # update history variable and extend observation space
        self.history = env_config.get("n_history", 1)
        if self.history > 1:
            attr = extend_with_history(attr, self.history)
        # print("Updated RL obs: ", attr)

        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict(
            {
                "high_level_agent": gym.spaces.Discrete(2),
                "reinforcement_learning_agent":
                    gym.spaces.Dict(attr),
                "do_nothing_agent": gym.spaces.Discrete(1),
            }
        )

    def rescale_observation_space(self,
                                  lib_dir: str,
                                  input_attr: list = ["p_i", "p_l", "r", "o"]
                                  ) -> GymnasiumObservationSpace:
        """
        Function that rescales the observation space.
        """
        # scale observations
        attr_list = get_attr_list(input_attr)
        print("Observation attributes used are: ", attr_list)
        gym_obs = self.env_gym.observation_space
        gym_obs = gym_obs.keep_only_attr(attr_list)

        if "timestep_overflow" in attr_list:
            gym_obs = gym_obs.reencode_space(
                "timestep_overflow",
                ScalerAttrConverter(
                    substract=0.0,
                    divide=Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED,  # assuming no custom params
                ),
            )
        path = os.path.join(lib_dir, f"data/scaling_arrays")
        if self.env_name in os.listdir(path):
            # underestimation_constant = 1.2  # constant to account that our max/min are underestimated
            for attr in attr_list:
                if os.path.exists(os.path.join(path, f"{self.env_name}/{attr}.npy")):
                    max_arr, min_arr = np.load(os.path.join(path, f"{self.env_name}/{attr}.npy"))
                    if np.all(max_arr - min_arr < 1e-5):
                        # if max and min are almost the same, we cannot divide by 0
                        print(f"Max and min are almost the same for {attr}. "
                              f"Thus constant value and not relevant for training -> IGNORE.")
                        gym_obs = gym_obs.reencode_space(attr, None)
                    else:
                        # values are multiplied with a constant to account that our max/min are underestimated
                        gym_obs = gym_obs.reencode_space(
                            attr,
                            ScalerAttrConverter(
                                substract=0.8 * min_arr,
                                divide=np.where(1.2 * max_arr - 0.8 * min_arr == 0, 1.0, 1.2 * max_arr - 0.8 * min_arr),
                            ),
                        )
        else:
            raise ValueError("This scaling is not yet implemented for this environment.")

        return gym_obs

    def custom_observation_space(self, input_attr: list = ["d"]) -> dict:
        custom_obs = {}
        if "d" in input_attr:
            # danger attribute
            custom_obs["danger"] = gym.spaces.MultiBinary(self.env_gym.init_env.n_line)
        if "t" in input_attr:
            # add time of day as attribute
            custom_obs["time_of_day"] = gym.spaces.Box(-1, 1, shape=(1,))
        if "y" in input_attr:
            # add time of day as attribute
            custom_obs["day_of_year"] = gym.spaces.Box(-1, 1, shape=(1,))
        return custom_obs

    def reset_obs(self):
        self.cur_gym_obs = None

    def convert_obs(self, g2op_obs: BaseObservation):
        cur_gym_obs = dict(self.env_gym.observation_space.to_gym(g2op_obs))
        # print("Current gym obs: ", cur_gym_obs)
        cur_gym_obs.update(self.convert_custom_obs(g2op_obs))
        if self.history > 1:
            cur_keys = list(cur_gym_obs.keys())
            if self.cur_gym_obs is None:
                # initialize history with current observation
                hist_obs = {key + f"_{i}": value for key,value in cur_gym_obs.items() for i in range(1, self.history)}
                cur_gym_obs.update(hist_obs)
            else:
                # add history
                for i in range(self.history-1, 0, -1):
                    for key in cur_keys:
                        cur_gym_obs[key + f"_{i}"] = self.cur_gym_obs[key + f"_{i-1}"]
            # Set the current observation as key_0
            for key in cur_keys:
                cur_gym_obs[key + "_0"] = cur_gym_obs[key]
                del cur_gym_obs[key]
            # Update self.cur_gym_obs
            self.cur_gym_obs = cur_gym_obs
        # print("Updated gym obs: ", cur_gym_obs)
        return cur_gym_obs

    def convert_custom_obs(self,
                           g2op_obs: BaseObservation) -> dict:
        custom_obs = {}
        if "danger" in self.custom_attr:
            # danger attribute
            custom_obs["danger"] = ((g2op_obs.rho >= self.danger - 0.05) & self.thermal_limit_under400 ) | \
                                   (g2op_obs.rho >= self.danger)
        if "time_of_day" in self.custom_attr:
            # time of day attribute
            min_day = 60 * g2op_obs.hour_of_day + g2op_obs.minute_of_hour
            max_val = 60*24
            # translate to cyclic pattern between [-1, 1] of 24 * 60 minutes per day
            custom_obs["time_of_day"] = np.array([np.cos(2*np.pi * min_day / max_val)])
        if "day_of_year" in self.custom_attr:
            # day of year attribute
            day = (date(g2op_obs.year, g2op_obs.month, g2op_obs.day) - date(g2op_obs.year, 1, 1)).days + 1
            total_days = 365 if g2op_obs.year % 4 != 0 else 366
            # translate to cyclic pattern between [-1, 1] of 365 days per year
            custom_obs["day_of_year"] = np.array([np.cos(2*np.pi * day / total_days)])
        return custom_obs


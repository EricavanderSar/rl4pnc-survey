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


class ObservationConverter:
    def __init__(self,
                 gym_env: GymEnv,
                 env_config: dict,
                 ):
        self.env_gym = gym_env
        self.cur_gym_obs = None
        self.cur_g2op_obs = None
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
        # initialize danger variable in case it is used.
        self.danger = env_config.get("danger", 0.9)
        self.thermal_limit_under400 = (gym_env.init_env.get_thermal_limit() < 400)
        attr.update(custom_attr)
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

    def convert_obs(self, g2op_obs: BaseObservation):
        cur_gym_obs = dict(self.env_gym.observation_space.to_gym(g2op_obs))
        # print("Current gym obs: ", cur_gym_obs)
        cur_gym_obs.update(self.convert_custom_obs(g2op_obs))
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


class ObsConverter:
    def __init__(self,
                 env: Environment,
                 danger,
                 attr=["p_i", "p_l", "r", "o", "d", "m"],
                 n_history=1,
                 adj_mat=True,):
        self.obs_space = env.observation_space
        self.act_space = env.action_space
        self.danger = danger
        self.thermal_limit_under400 = env._thermal_limit_a < 400
        self.attr = attr
        self.init_obs_converter()
        self.stacked_obs = []
        self.n_history = n_history
        self.adj_mat = adj_mat
        # add info for Normalizing params
        self.state_mean = None
        self.state_std = None
        self.normalize = False
        # For different way of normalizing
        self.max = np.ones(self.obs_space.n)
        self.min = np.zeros(self.obs_space.n)

    def load_mean_std(self, load_path):
        self.normalize = True
        # Add data DN to facilitate normalization:
        mean = torch.load(os.path.join(load_path, "mean.pt"))
        std = torch.load(os.path.join(load_path, "std.pt"))
        self.state_mean = mean.numpy().flatten()
        self.state_std = std.masked_fill(std < 1e-5, 1.0).numpy().flatten()
        # don't normalize params that are not interesting.
        last_attr = 'rho'
        pos = self.obs_space.attr_list_vect.index(last_attr)
        self.state_mean[sum(self.obs_space.shape[:pos]):] = 0
        self.state_std[sum(self.act_space.shape[:pos]):] = 1

    def load_max_min(self, load_path):
        self.normalize = True
        self.max[..., self.pp] = self.obs_space.gen_pmax
        # self.max[] = Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED # Overflow not needed already taken care of
        under_const = [1.2, 0.8] # constant to account that our max/min are underestimated
        self.max[..., self.lp], self.min[..., self.lp] = [vec*c for c, vec in zip(under_const, np.load(os.path.join(load_path, "load_p.npy")))]
        self.max[..., self.op], self.min[..., self.op] = [vec*c for c, vec in zip(under_const, np.load(os.path.join(load_path, "p_or.npy")))]
        self.max[..., self.ep], self.min[..., self.ep] = [vec*c for c, vec in zip(under_const, np.load(os.path.join(load_path, "p_ex.npy")))]

    def reset(self):
        self.stacked_obs = []

    def _get_attr_pos(self, list_attr):
        all_obs_ranges = []
        idx = self.obs_space.shape
        for attr in list_attr:
            pos = self.obs_space.attr_list_vect.index(attr)
            start = sum(idx[:pos])
            end = start + idx[pos]
            all_obs_ranges.append(np.arange(start, end))
        return all_obs_ranges

    def state_normalize(self, s):
        if self.state_std is not None:
            s = (s - self.state_mean) / self.state_std
        else:
            s = (s - self.min) / (self.max - self.min)
        return s

    def init_obs_converter(self):
        list_attr = [
            "gen_p",
            "load_p",
            "p_or",
            "p_ex",
            "rho",
            "timestep_overflow",
            "time_next_maintenance",
            "topo_vect",
        ]
        (
            self.pp,
            self.lp,
            self.op,
            self.ep,
            self.rho,
            self.over,
            self.main,
            self.topo,
        ) = self._get_attr_pos(list_attr)

    def convert_obs(self, o):
        # o.shape : (B, O)
        # output (Batch, Node, Feature)
        length = self.obs_space.dim_topo  # N

        attr_list = []
        p = False
        p_ = np.zeros(length)  # (B, N)
        if "p_i" in self.attr:
            # active power p
            p = True
            p_[..., self.obs_space.gen_pos_topo_vect] = o[..., self.pp]
            p_[..., self.obs_space.load_pos_topo_vect] = o[..., self.lp]
        if "p_l" in self.attr:
            # active power p
            p = True
            p_[..., self.obs_space.line_or_pos_topo_vect] = o[..., self.op]
            p_[..., self.obs_space.line_ex_pos_topo_vect] = o[..., self.ep]
        if p:
            attr_list.append(p_)
        if "r" in self.attr:
            # rho (powerline usage ratio)
            rho_ = np.zeros(length)
            rho_[..., self.obs_space.line_or_pos_topo_vect] = o[..., self.rho]
            rho_[..., self.obs_space.line_ex_pos_topo_vect] = o[..., self.rho]
            attr_list.append(rho_)
        if "d" in self.attr:
            # lines in danger
            danger_ = np.zeros(length)
            danger = ((o[..., self.rho] >= self.danger - 0.05) & self.thermal_limit_under400) | (
                    o[..., self.rho] >= self.danger
            )
            danger_[..., self.obs_space.line_or_pos_topo_vect] = danger.astype(float)
            danger_[..., self.obs_space.line_ex_pos_topo_vect] = danger.astype(float)
            attr_list.append(danger_)
        if "o" in self.attr:
            # overflow in a powerline
            over_ = np.zeros(length)
            over_[..., self.obs_space.line_or_pos_topo_vect] = o[..., self.over] / Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED
            over_[..., self.obs_space.line_ex_pos_topo_vect] = o[..., self.over] / Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED
            attr_list.append(over_)
        if "m" in self.attr:
            # powerline in maintenance
            main_ = np.zeros(length)
            temp = np.zeros_like(o[..., self.main])
            temp[o[..., self.main] == 0] = 1
            main_[..., self.obs_space.line_or_pos_topo_vect] = temp
            main_[..., self.obs_space.line_ex_pos_topo_vect] = temp
            attr_list.append(main_)

        # current bus assignment
        topo_ = np.clip(o[..., self.topo] - 1, -1, None)
        state = np.stack(attr_list, axis=1)  # B, N, F
        return state, topo_

    def get_cur_obs(self, obs: Observation):
        # Get the observation: Feature matrix
        obs_vect = obs.to_vect()
        if self.normalize:
            # improve performance with normalization/scaling of data
            obs_vect = self.state_normalize(obs_vect)
        obs_vect, topo = self.convert_obs(obs_vect)
        if len(self.stacked_obs) == 0:
            for _ in range(self.n_history):
                self.stacked_obs.append(obs_vect)
        else:
            self.stacked_obs.pop(0)
            self.stacked_obs.append(obs_vect)

        # Get the observation: Adjacency matrix
        # self.adj = (torch.FloatTensor(obs.connectivity_matrix())).to(self.device)
        if self.adj_mat:
            topo = obs.connectivity_matrix() + np.eye(int(obs.dim_topo))
        # TODO: Decide if to pass on topo vector or adjacency matrix
        cur_obs = {
            "feature_matrix": np.concatenate(self.stacked_obs, axis=1),
            "topology": topo
        }
        return cur_obs

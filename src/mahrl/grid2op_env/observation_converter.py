import numpy as np
import torch
from grid2op import Observation


class ObsConverter:
    def __init__(self, env, danger, attr=["p_i", "p_l", "r", "o", "d", "m"], n_history=1):
        self.obs_space = env.observation_space
        self.act_space = env.action_space
        self.danger = danger
        self.thermal_limit_under400 = env._thermal_limit_a < 400
        self.attr = attr
        self.init_obs_converter()
        self.stacked_obs = []
        self.n_history = n_history

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
            # whether each line is in danger
            danger_ = np.zeros(length)
            danger = ((o[..., self.rho] >= self.danger - 0.05) & self.thermal_limit_under400) | (
                    o[..., self.rho] >= self.danger
            )
            danger_[..., self.obs_space.line_or_pos_topo_vect] = danger.astype(float)
            danger_[..., self.obs_space.line_ex_pos_topo_vect] = danger.astype(float)
            attr_list.append(danger_)
        if "o" in self.attr:
            # whether overflow occurs in each powerline
            over_ = np.zeros(length)
            over_[..., self.obs_space.line_or_pos_topo_vect] = o[..., self.over] / 3
            over_[..., self.obs_space.line_ex_pos_topo_vect] = o[..., self.over] / 3
            attr_list.append(over_)
        if "m" in self.attr:
            # whether each powerline is in maintenance
            main_ = np.zeros(length)
            temp = np.zeros_like(o[..., self.main])
            temp[o[..., self.main] == 0] = 1
            main_[..., self.obs_space.line_or_pos_topo_vect] = temp
            main_[..., self.obs_space.line_ex_pos_topo_vect] = temp
            attr_list.append(main_)

        # current bus assignment
        topo_ = np.clip(o[..., self.topo] - 1, -1, None)

        state = np.stack(attr_list, axis=1)  # B, N, F
        return state, np.expand_dims(topo_, axis=-1)

    def get_cur_obs(self, obs: Observation):
        # Get the observation: Feature matrix
        obs_vect = obs.to_vect()
        # TODO: Normalize (or will RLlib do this for us?)
        # obs_vect = self.state_normalize(obs_vect)
        obs_vect, topo = self.convert_obs(obs_vect)
        if len(self.stacked_obs) == 0:
            for _ in range(self.n_history):
                self.stacked_obs.append(obs_vect)
        else:
            self.stacked_obs.pop(0)
            self.stacked_obs.append(obs_vect)

        # Get the observation: Adjacency matrix
        # self.adj = (torch.FloatTensor(obs.connectivity_matrix())).to(self.device)
        adj = obs.connectivity_matrix() + np.eye(int(obs.dim_topo))
        # TODO: Decide if to pass on topo vector or adjacency matrix
        cur_obs = {"feature_matrix": np.concatenate(self.stacked_obs, axis=1), "adjacency_matrix": adj}
        return cur_obs

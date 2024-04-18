import numpy as np
import torch
import os
from grid2op import Observation
from grid2op import Environment
from grid2op.Parameters import Parameters

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
        c = 1.2 # constant to account that our max/min are underestimated
        print('loading data from ', os.path.join(load_path, "load_p.npy"))
        self.max[..., self.lp], self.min[..., self.lp] = [vec*c for vec in np.load(os.path.join(load_path, "load_p.npy"))]
        self.max[..., self.op], self.min[..., self.op] = [vec*c for vec in np.load(os.path.join(load_path, "p_or.npy"))]
        self.max[..., self.ep], self.min[..., self.ep] = [vec*c for vec in np.load(os.path.join(load_path, "p_ex.npy"))]

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
            over_[..., self.obs_space.line_or_pos_topo_vect] = o[..., self.over] / Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED
            over_[..., self.obs_space.line_ex_pos_topo_vect] = o[..., self.over] / Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED
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
        cur_obs = {"feature_matrix": np.concatenate(self.stacked_obs, axis=1), "topology": topo}
        return cur_obs

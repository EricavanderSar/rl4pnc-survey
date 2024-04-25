import numpy as np
import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F

from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


def lin_block(dim_in, dim_out, *args, **kwargs):
    return nn.Sequential(nn.Linear(dim_in, dim_out, *args, **kwargs), nn.ReLU())


class LinFCN(TorchModelV2, nn.Module):
    """
    This is a simple fully connected linear neural network.
    It is similar to the one in EVDS code base marl4powergrid
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        # get the features input dimension:
        obs_feature = obs_space.original_space["feature_matrix"]
        obs_topo = obs_space.original_space["topology"]
        input_dim = int(obs_feature.shape[-1])
        hiddens = list(model_config.get("fcnet_hiddens", []))
        self.layers = nn.ModuleList()
        # put all layer dimensions together, ending with a layer that downsizes the matrix
        dims = [input_dim] + hiddens + [1]
        self._layers = nn.ModuleList([lin_block(dim_in, dim_out) for dim_in, dim_out in zip(dims, dims[1:])])
        concat_dim = obs_topo.shape[0] + obs_feature.shape[0]
        self.mu = nn.Linear(concat_dim, num_outputs)
        self._feature_matrix = None
        self._topology = None

        # define the critic value function layers
        self._vf_layers = nn.ModuleList([lin_block(dim_in, dim_out) for dim_in, dim_out in zip(dims, dims[1:])])
        self._vf_ly = nn.Linear(obs_feature.shape[0], obs_feature.shape[0] // 4)
        self._vf_out = nn.Linear(obs_feature.shape[0] // 4, 1)

    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS], self.processed_obs_space, tensorlib="torch"
            )
        x = self._feature_matrix = orig_obs['feature_matrix']
        self._topology = orig_obs['topology']
        for l in self._layers:
            x = l(x)
        x = torch.cat([x.squeeze(-1), self._topology], dim=-1)
        x = F.leaky_relu(x)
        mu = self.mu(x)
        probs = mu.softmax(dim=1)
        return probs, []

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        # the critic part of the algorithm estimating the value function
        assert self._feature_matrix is not None, "must call forward() first"
        x = self._feature_matrix
        for l in self._vf_layers:
            x = l(x)
        x = x.squeeze(-1)  # B,N
        x = F.leaky_relu(self._vf_ly(x))
        x = self._vf_out(x).squeeze(-1)
        return x

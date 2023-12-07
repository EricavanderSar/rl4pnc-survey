"""
Defines agent policies.
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from ray.actor import ActorHandle
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import (
    ModelGradients,
    ModelWeights,
    PolicyID,
    TensorStructType,
    TensorType,
)
from torch import nn


class RecordingTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )

        self.out_file = model_config["custom_model_config"]["out_file"]

        self.layer = nn.Linear(2, 2)

    def forward(self, input_dict, state, seq_lens):
        with open(self.out_file, "at", encoding="utf-8") as f:
            f.write(f"Action performed at {seq_lens}\n")

        obs = input_dict["obs_flat"]
        self.last_batch_size = obs.shape[0]

        # act = torch.argmax(obs, axis=1, keepdim=True).to(torch.long)
        act = (obs - 1) * 2
        # act = self.layer(obs)
        # self._value = act[:, 0]

        return act, state

    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self.last_batch_size,)))
        # return self._value

    def set_weights(self) -> None:
        pass

    def load_state_dict(self, state_dict) -> None:
        pass


ModelCatalog.register_custom_model("RecordingTorchModel", RecordingTorchModel)

RHO_THRESHOLD = 0.95  # TODO include in obs?


def policy_mapping_fn(agent_id: str) -> str:
    """Maps each agent to a policy."""
    if agent_id.startswith("reinforcement_learning_"):
        return "reinforcement_learning_policy"
    if agent_id.startswith("high_level_"):
        return "high_level_policy"
    if agent_id.startswith("do_nothing_"):
        return "do_nothing_policy"
    raise NotImplementedError


class DoNothingPolicy(Policy):
    """Example of a custom policy written from scratch.

    You might find it more convenient to use the `build_tf_policy` and
    `build_torch_policy` helpers instead for a real policy, which are
    described in the next sections.
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List[str]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        """Computes actions for the current policy.

        Args:
            obs_batch: Batch of observations.
            state_batches: List of RNN state input batches, if any.
            prev_action_batch: Batch of previous action values.
            prev_reward_batch: Batch of previous rewards.

        Returns:
            actions: Do nothing action. Batch of output actions, with shape like
                [BATCH_SIZE, ACTION_SHAPE].
            state_outs (List[TensorType]): List of RNN state output
                batches, if any, each with shape [BATCH_SIZE, STATE_SIZE].
            info (List[dict]): Dictionary of extra feature batches, if any,
                with shape like
                {"f1": [BATCH_SIZE, ...], "f2": [BATCH_SIZE, ...]}.
        """
        return [0], [], {}

    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""

    # def apply_gradients(self, gradients: ModelGradients) -> None:
    #     """No gradients to apply.

    #     Args:
    #         gradients: The already calculated gradients to apply to this
    #             Policy.
    #     """
    #     raise NotImplementedError

    # def compute_gradients(
    #     self, postprocessed_batch: SampleBatch
    # ) -> Tuple[ModelGradients, Dict[str, TensorType]]:
    #     """No gradient to compute.

    #     Args:
    #         postprocessed_batch: The SampleBatch object to use
    #             for calculating gradients.

    #     Returns:
    #         grads: List of gradient output values.
    #         grad_info: Extra policy-specific info values.
    #     """
    #     return [], {}

    # def loss(
    #     self, model: ModelV2, dist_class: ActionDistribution, train_batch: SampleBatch
    # ) -> Union[TensorType, List[TensorType]]:
    #     """Loss function for this Policy.

    #     Override this method in order to implement custom loss computations.

    #     Args:
    #         model: The model to calculate the loss(es).
    #         dist_class: The action distribution class to sample actions
    #             from the model's outputs.
    #         train_batch: The input batch on which to calculate the loss.

    #     Returns:
    #         Either a single loss tensor or a list of loss tensors.
    #     """
    #     raise NotImplementedError

    # def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
    #     """Perform one learning update, given `samples`.

    #     Either this method or the combination of `compute_gradients` and
    #     `apply_gradients` must be implemented by subclasses.

    #     Args:
    #         samples: The SampleBatch object to learn from.

    #     Returns:
    #         Dictionary of extra metadata from `compute_gradients()`.

    #     Examples:
    #         >>> policy, sample_batch = ... # doctest: +SKIP
    #         >>> policy.learn_on_batch(sample_batch) # doctest: +SKIP
    #     """
    #     raise NotImplementedError

    # def learn_on_batch_from_replay_buffer(
    #     self, replay_actor: ActorHandle, policy_id: PolicyID
    # ) -> Dict[str, TensorType]:
    #     """Samples a batch from given replay actor and performs an update.

    #     Args:
    #         replay_actor: The replay buffer actor to sample from.
    #         policy_id: The ID of this policy.

    #     Returns:
    #         Dictionary of extra metadata from `compute_gradients()`.
    #     """
    #     raise NotImplementedError

    # def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
    #     """Bulk-loads the given SampleBatch into the devices' memories.

    #     The data is split equally across all the Policy's devices.
    #     If the data is not evenly divisible by the batch size, excess data
    #     should be discarded.

    #     Args:
    #         batch: The SampleBatch to load.
    #         buffer_index: The index of the buffer (a MultiGPUTowerStack) to use
    #             on the devices. The number of buffers on each device depends
    #             on the value of the `num_multi_gpu_tower_stacks` config key.

    #     Returns:
    #         The number of tuples loaded per device.
    #     """
    #     raise NotImplementedError

    # def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
    #     """Returns the number of currently loaded samples in the given buffer.

    #     Args:
    #         buffer_index: The index of the buffer (a MultiGPUTowerStack)
    #             to use on the devices. The number of buffers on each device
    #             depends on the value of the `num_multi_gpu_tower_stacks` config
    #             key.

    #     Returns:
    #         The number of tuples loaded per device.
    #     """
    #     raise NotImplementedError

    # def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
    #     """Runs a single step of SGD on an already loaded data in a buffer.

    #     Runs an SGD step over a slice of the pre-loaded batch, offset by
    #     the `offset` argument (useful for performing n minibatch SGD
    #     updates repeatedly on the same, already pre-loaded data).

    #     Updates the model weights based on the averaged per-device gradients.

    #     Args:
    #         offset: Offset into the preloaded data. Used for pre-loading
    #             a train-batch once to a device, then iterating over
    #             (subsampling through) this batch n times doing minibatch SGD.
    #         buffer_index: The index of the buffer (a MultiGPUTowerStack)
    #             to take the already pre-loaded data from. The number of buffers
    #             on each device depends on the value of the
    #             `num_multi_gpu_tower_stacks` config key.

    #     Returns:
    #         The outputs of extra_ops evaluated over the batch.
    #     """
    #     raise NotImplementedError


class SelectAgentPolicy(Policy):
    """Example of a custom policy written from scratch.

    You might find it more convenient to use the `build_tf_policy` and
    `build_torch_policy` helpers instead for a real policy, which are
    described in the next sections.
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List[str]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        """Computes actions for the current policy.

        Args:
            obs_batch: Batch of observations.
            state_batches: List of RNN state input batches, if any.
            prev_action_batch: Batch of previous action values.
            prev_reward_batch: Batch of previous rewards.

        Returns:
            actions: Do nothing action. Batch of output actions, with shape like
                [BATCH_SIZE, ACTION_SHAPE].
            state_outs (List[TensorType]): List of RNN state output
                batches, if any, each with shape [BATCH_SIZE, STATE_SIZE].
            info (List[dict]): Dictionary of extra feature batches, if any,
                with shape like
                {"f1": [BATCH_SIZE, ...], "f2": [BATCH_SIZE, ...]}.
        """
        # TODO how to implement this? How to make sure batch size = 1?
        for obs in obs_batch:
            rho = obs["rho"]
            print(f"OBS: {rho}")
            if np.max(obs["rho"]) > RHO_THRESHOLD:
                # Set results for do something agent
                actions_result = 0  # ["reinforcement_learning_policy"]
                state_outs_result = []
                info_result = {}
                break  # exit the loop since we have a result
            elif np.max(obs["rho"]) <= RHO_THRESHOLD:
                # Set results for do nothing agent
                actions_result = ["do_nothing_policy"]
                state_outs_result = []
                info_result = {}
                break  # exit the loop since we have a result
            else:
                actions_result = 1  # ["do_nothing_policy"]
                break

        return actions_result, state_outs_result, info_result

    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""

    # def apply_gradients(self, gradients: ModelGradients) -> None:
    #     """No gradients to apply.

    #     Args:
    #         gradients: The already calculated gradients to apply to this
    #             Policy.
    #     """
    #     raise NotImplementedError

    # def compute_gradients(
    #     self, postprocessed_batch: SampleBatch
    # ) -> Tuple[ModelGradients, Dict[str, TensorType]]:
    #     """No gradient to compute.

    #     Args:
    #         postprocessed_batch: The SampleBatch object to use
    #             for calculating gradients.

    #     Returns:
    #         grads: List of gradient output values.
    #         grad_info: Extra policy-specific info values.
    #     """
    #     return [], {}

    # def loss(
    #     self, model: ModelV2, dist_class: ActionDistribution, train_batch: SampleBatch
    # ) -> Union[TensorType, List[TensorType]]:
    #     """Loss function for this Policy.

    #     Override this method in order to implement custom loss computations.

    #     Args:
    #         model: The model to calculate the loss(es).
    #         dist_class: The action distribution class to sample actions
    #             from the model's outputs.
    #         train_batch: The input batch on which to calculate the loss.

    #     Returns:
    #         Either a single loss tensor or a list of loss tensors.
    #     """
    #     raise NotImplementedError

    # def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
    #     """Perform one learning update, given `samples`.

    #     Either this method or the combination of `compute_gradients` and
    #     `apply_gradients` must be implemented by subclasses.

    #     Args:
    #         samples: The SampleBatch object to learn from.

    #     Returns:
    #         Dictionary of extra metadata from `compute_gradients()`.

    #     Examples:
    #         >>> policy, sample_batch = ... # doctest: +SKIP
    #         >>> policy.learn_on_batch(sample_batch) # doctest: +SKIP
    #     """
    #     raise NotImplementedError

    # def learn_on_batch_from_replay_buffer(
    #     self, replay_actor: ActorHandle, policy_id: PolicyID
    # ) -> Dict[str, TensorType]:
    #     """Samples a batch from given replay actor and performs an update.

    #     Args:
    #         replay_actor: The replay buffer actor to sample from.
    #         policy_id: The ID of this policy.

    #     Returns:
    #         Dictionary of extra metadata from `compute_gradients()`.
    #     """
    #     raise NotImplementedError

    # def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
    #     """Bulk-loads the given SampleBatch into the devices' memories.

    #     The data is split equally across all the Policy's devices.
    #     If the data is not evenly divisible by the batch size, excess data
    #     should be discarded.

    #     Args:
    #         batch: The SampleBatch to load.
    #         buffer_index: The index of the buffer (a MultiGPUTowerStack) to use
    #             on the devices. The number of buffers on each device depends
    #             on the value of the `num_multi_gpu_tower_stacks` config key.

    #     Returns:
    #         The number of tuples loaded per device.
    #     """
    #     raise NotImplementedError

    # def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
    #     """Returns the number of currently loaded samples in the given buffer.

    #     Args:
    #         buffer_index: The index of the buffer (a MultiGPUTowerStack)
    #             to use on the devices. The number of buffers on each device
    #             depends on the value of the `num_multi_gpu_tower_stacks` config
    #             key.

    #     Returns:
    #         The number of tuples loaded per device.
    #     """
    #     raise NotImplementedError

    # def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
    #     """Runs a single step of SGD on an already loaded data in a buffer.

    #     Runs an SGD step over a slice of the pre-loaded batch, offset by
    #     the `offset` argument (useful for performing n minibatch SGD
    #     updates repeatedly on the same, already pre-loaded data).

    #     Updates the model weights based on the averaged per-device gradients.

    #     Args:
    #         offset: Offset into the preloaded data. Used for pre-loading
    #             a train-batch once to a device, then iterating over
    #             (subsampling through) this batch n times doing minibatch SGD.
    #         buffer_index: The index of the buffer (a MultiGPUTowerStack)
    #             to take the already pre-loaded data from. The number of buffers
    #             on each device depends on the value of the
    #             `num_multi_gpu_tower_stacks` config key.

    #     Returns:
    #         The outputs of extra_ops evaluated over the batch.
    #     """
    #     raise NotImplementedError

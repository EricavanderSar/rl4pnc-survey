"""
Defines agent policies.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium
import numpy as np
from ray.actor import ActorHandle
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    ModelGradients,
    ModelWeights,
    PolicyID,
    TensorStructType,
    TensorType,
)

RHO_THRESHOLD = 0.9  # TODO include in obs?


def policy_mapping_fn(
    agent_id: str,
    episode: Optional[EpisodeV2] = None,
    worker: Optional[RolloutWorker] = None,
) -> str:
    """Maps each agent to a policy."""
    if agent_id.startswith("reinforcement_learning_agent"):
        return "reinforcement_learning_policy"
    if agent_id.startswith("high_level_agent"):
        return "high_level_policy"
    if agent_id.startswith("do_nothing_agent"):
        return "do_nothing_policy"
    raise NotImplementedError


class DoNothingPolicy(Policy):
    """Example of a custom policy written from scratch.

    You might find it more convenient to use the `build_tf_policy` and
    `build_torch_policy` helpers instead for a real policy, which are
    described in the next sections.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        config: AlgorithmConfigDict,
    ):
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
        info_batch: Optional[Dict[str, List[Any]]] = None,
        episodes: Optional[List[str]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs: Dict[str, Any],
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

    def apply_gradients(self, gradients: ModelGradients) -> None:
        """No gradients to apply.

        Args:
            gradients: The already calculated gradients to apply to this
                Policy.
        """
        raise NotImplementedError

    def compute_gradients(
        self, postprocessed_batch: SampleBatch
    ) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        """No gradient to compute.

        Args:
            postprocessed_batch: The SampleBatch object to use
                for calculating gradients.

        Returns:
            grads: List of gradient output values.
            grad_info: Extra policy-specific info values.
        """
        return [], {}

    def loss(
        self, model: ModelV2, dist_class: ActionDistribution, train_batch: SampleBatch
    ) -> Union[TensorType, List[TensorType]]:
        """Loss function for this Policy.

        Override this method in order to implement custom loss computations.

        Args:
            model: The model to calculate the loss(es).
            dist_class: The action distribution class to sample actions
                from the model's outputs.
            train_batch: The input batch on which to calculate the loss.

        Returns:
            Either a single loss tensor or a list of loss tensors.
        """
        raise NotImplementedError

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        """Perform one learning update, given `samples`.

        Either this method or the combination of `compute_gradients` and
        `apply_gradients` must be implemented by subclasses.

        Args:
            samples: The SampleBatch object to learn from.

        Returns:
            Dictionary of extra metadata from `compute_gradients()`.

        Examples:
            >>> policy, sample_batch = ... # doctest: +SKIP
            >>> policy.learn_on_batch(sample_batch) # doctest: +SKIP
        """
        raise NotImplementedError

    def learn_on_batch_from_replay_buffer(
        self, replay_actor: ActorHandle, policy_id: PolicyID
    ) -> Dict[str, TensorType]:
        """Samples a batch from given replay actor and performs an update.

        Args:
            replay_actor: The replay buffer actor to sample from.
            policy_id: The ID of this policy.

        Returns:
            Dictionary of extra metadata from `compute_gradients()`.
        """
        raise NotImplementedError

    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
        """Bulk-loads the given SampleBatch into the devices' memories.

        The data is split equally across all the Policy's devices.
        If the data is not evenly divisible by the batch size, excess data
        should be discarded.

        Args:
            batch: The SampleBatch to load.
            buffer_index: The index of the buffer (a MultiGPUTowerStack) to use
                on the devices. The number of buffers on each device depends
                on the value of the `num_multi_gpu_tower_stacks` config key.

        Returns:
            The number of tuples loaded per device.
        """
        raise NotImplementedError

    def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
        """Returns the number of currently loaded samples in the given buffer.

        Args:
            buffer_index: The index of the buffer (a MultiGPUTowerStack)
                to use on the devices. The number of buffers on each device
                depends on the value of the `num_multi_gpu_tower_stacks` config
                key.

        Returns:
            The number of tuples loaded per device.
        """
        raise NotImplementedError

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0) -> None:
        """Runs a single step of SGD on an already loaded data in a buffer.

        Runs an SGD step over a slice of the pre-loaded batch, offset by
        the `offset` argument (useful for performing n minibatch SGD
        updates repeatedly on the same, already pre-loaded data).

        Updates the model weights based on the averaged per-device gradients.

        Args:
            offset: Offset into the preloaded data. Used for pre-loading
                a train-batch once to a device, then iterating over
                (subsampling through) this batch n times doing minibatch SGD.
            buffer_index: The index of the buffer (a MultiGPUTowerStack)
                to take the already pre-loaded data from. The number of buffers
                on each device depends on the value of the
                `num_multi_gpu_tower_stacks` config key.

        Returns:
            The outputs of extra_ops evaluated over the batch.
        """
        raise NotImplementedError


class SelectAgentPolicy(Policy):
    """Example of a custom policy written from scratch.

    You might find it more convenient to use the `build_tf_policy` and
    `build_torch_policy` helpers instead for a real policy, which are
    described in the next sections.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        config: AlgorithmConfigDict,
    ):
        Policy.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

    def compute_actions(
        self,
        obs_batch: Union[List[Dict[str, Any]], Dict[str, Any]],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, List[Any]]] = None,
        episodes: Optional[List[str]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        state_outs_result: List[Any] = []
        info_result: Dict[str, Any] = {}

        if isinstance(obs_batch, list):
            max_rho = np.max([item["rho"] for item in obs_batch])
        elif isinstance(obs_batch, dict):
            max_rho = np.max(obs_batch["rho"])
        else:
            # Handle the case where obs_batch has an unexpected type
            raise TypeError(f"Unexpected type for obs_batch: {type(obs_batch)}")

        logging.info(f"max_rho={max_rho}")
        if np.max(max_rho) > RHO_THRESHOLD:
            # Set results for do something agent
            actions_result = [0]  # ["reinforcement_learning_policy"]
        else:
            # Set results for do nothing agent
            actions_result = [1]

        return actions_result, state_outs_result, info_result

    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""

    def apply_gradients(self, gradients: ModelGradients) -> None:
        """No gradients to apply.

        Args:
            gradients: The already calculated gradients to apply to this
                Policy.
        """
        raise NotImplementedError

    def compute_gradients(
        self, postprocessed_batch: SampleBatch
    ) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        """No gradient to compute.

        Args:
            postprocessed_batch: The SampleBatch object to use
                for calculating gradients.

        Returns:
            grads: List of gradient output values.
            grad_info: Extra policy-specific info values.
        """
        return [], {}

    def loss(
        self, model: ModelV2, dist_class: ActionDistribution, train_batch: SampleBatch
    ) -> Union[TensorType, List[TensorType]]:
        """Loss function for this Policy.

        Override this method in order to implement custom loss computations.

        Args:
            model: The model to calculate the loss(es).
            dist_class: The action distribution class to sample actions
                from the model's outputs.
            train_batch: The input batch on which to calculate the loss.

        Returns:
            Either a single loss tensor or a list of loss tensors.
        """
        raise NotImplementedError

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        """Perform one learning update, given `samples`.

        Either this method or the combination of `compute_gradients` and
        `apply_gradients` must be implemented by subclasses.

        Args:
            samples: The SampleBatch object to learn from.

        Returns:
            Dictionary of extra metadata from `compute_gradients()`.

        Examples:
            >>> policy, sample_batch = ... # doctest: +SKIP
            >>> policy.learn_on_batch(sample_batch) # doctest: +SKIP
        """
        raise NotImplementedError

    def learn_on_batch_from_replay_buffer(
        self, replay_actor: ActorHandle, policy_id: PolicyID
    ) -> Dict[str, TensorType]:
        """Samples a batch from given replay actor and performs an update.

        Args:
            replay_actor: The replay buffer actor to sample from.
            policy_id: The ID of this policy.

        Returns:
            Dictionary of extra metadata from `compute_gradients()`.
        """
        raise NotImplementedError

    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
        """Bulk-loads the given SampleBatch into the devices' memories.

        The data is split equally across all the Policy's devices.
        If the data is not evenly divisible by the batch size, excess data
        should be discarded.

        Args:
            batch: The SampleBatch to load.
            buffer_index: The index of the buffer (a MultiGPUTowerStack) to use
                on the devices. The number of buffers on each device depends
                on the value of the `num_multi_gpu_tower_stacks` config key.

        Returns:
            The number of tuples loaded per device.
        """
        raise NotImplementedError

    def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
        """Returns the number of currently loaded samples in the given buffer.

        Args:
            buffer_index: The index of the buffer (a MultiGPUTowerStack)
                to use on the devices. The number of buffers on each device
                depends on the value of the `num_multi_gpu_tower_stacks` config
                key.

        Returns:
            The number of tuples loaded per device.
        """
        raise NotImplementedError

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0) -> None:
        """Runs a single step of SGD on an already loaded data in a buffer.

        Runs an SGD step over a slice of the pre-loaded batch, offset by
        the `offset` argument (useful for performing n minibatch SGD
        updates repeatedly on the same, already pre-loaded data).

        Updates the model weights based on the averaged per-device gradients.

        Args:
            offset: Offset into the preloaded data. Used for pre-loading
                a train-batch once to a device, then iterating over
                (subsampling through) this batch n times doing minibatch SGD.
            buffer_index: The index of the buffer (a MultiGPUTowerStack)
                to take the already pre-loaded data from. The number of buffers
                on each device depends on the value of the
                `num_multi_gpu_tower_stacks` config key.

        Returns:
            The outputs of extra_ops evaluated over the batch.
        """
        raise NotImplementedError

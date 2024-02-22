"""
Defines agent policies.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import grid2op
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

from mahrl.experiments.utils import (
    calculate_action_space_asymmetry,
    calculate_action_space_medha,
    calculate_action_space_tennet,
    get_capa_substation_id,
)


def policy_mapping_fn(
    agent_id: str,
    episode: Optional[EpisodeV2] = None,
    worker: Optional[RolloutWorker] = None,
) -> str:
    """Maps each agent to a policy."""
    if agent_id.startswith("reinforcement_learning_agent"):
        # from agent_id, use re to extract the integer at the end
        id_number = re.search(r"\d+$", agent_id)
        if id_number:
            agent_number = int(id_number.group(0))
            return f"reinforcement_learning_policy_{agent_number}"
        return "reinforcement_learning_policy"
    if agent_id.startswith("high_level_agent"):
        return "high_level_policy"
    if agent_id.startswith("do_nothing_agent"):
        return "do_nothing_policy"
    if agent_id.startswith("choose_substation_agent"):
        return "choose_substation_policy"
    raise NotImplementedError


class CapaPolicy(Policy):
    """
    Policy that that returns a substation to act on based on the CAPA heuristic.
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

        env_config = config["model"]["custom_model_config"]["environment"]["env_config"]
        setup_env = grid2op.make(env_config["env_name"], **env_config["grid2op_kwargs"])

        # get changeable substations
        if env_config["action_space"] == "asymmetry":
            _, _, self.controllable_substations = calculate_action_space_asymmetry(
                setup_env
            )
        elif env_config["action_space"] == "medha":
            _, _, self.controllable_substations = calculate_action_space_medha(
                setup_env
            )
        elif env_config["action_space"] == "tennet":
            _, _, self.controllable_substations = calculate_action_space_tennet(
                setup_env
            )
        else:
            raise ValueError("No action valid space is defined.")

        self.idx = 0
        self.substation_to_act_on = None

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
        """Computes actions for the current policy."""
        state_outs_result: List[Any] = []
        info_result: Dict[str, Any] = {}

        line_info = self.config["model"]["custom_model_config"]["line_info"]

        # print(f"obs_batch: {obs_batch}")

        # if no list is created yet, do so
        if obs_batch["reset_capa_idx"][0] is True:
            self.idx = 0
            self.substation_to_act_on = get_capa_substation_id(
                line_info, obs_batch, self.controllable_substations
            )

        # find an action that is not the do nothing action by looping over the substations
        chosen_action = obs_batch["do_nothing_action"][0]
        # print(f"initial chosen action: {chosen_action}")
        while not chosen_action.as_dict() and self.idx < len(
            self.controllable_substations
        ):
            single_substation = self.substation_to_act_on[
                self.idx % len(self.controllable_substations)
            ]

            self.idx += 1
            chosen_action = obs_batch["proposed_actions"][0][single_substation]
            # print(f"loop chosen action: {chosen_action}")

            # if it's not the do nothing action, return action
            # if it's the do nothing action, continue the loop
            if chosen_action.as_dict():
                return ([chosen_action], state_outs_result, info_result)

        # grid is safe or no action is found, reset list count and return DoNothing
        self.idx = 0
        return ([chosen_action], state_outs_result, info_result)

        # print(f"obs_batch: {obs_batch}")
        # substation_to_act_on = get_capa_substation_id(
        #     line_info, obs_batch, self.controllable_substations
        # )[self.idx % 3]

        # self.idx += 1

        # find substation with max average rho
        # NOTE: When there are two equal max values, the first one is returned first
        # return (
        #     [substation_to_act_on],
        #     state_outs_result,
        #     info_result,
        # )

    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""

    def apply_gradients(self, gradients: ModelGradients) -> None:
        """No gradients to apply."""
        raise NotImplementedError

    def compute_gradients(
        self, postprocessed_batch: SampleBatch
    ) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        """No gradient to compute."""
        return [], {}

    def loss(
        self, model: ModelV2, dist_class: ActionDistribution, train_batch: SampleBatch
    ) -> Union[TensorType, List[TensorType]]:
        """No loss function"""
        raise NotImplementedError

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        """Not implemented."""
        raise NotImplementedError

    def learn_on_batch_from_replay_buffer(
        self, replay_actor: ActorHandle, policy_id: PolicyID
    ) -> Dict[str, TensorType]:
        """Not implemented."""
        raise NotImplementedError

    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
        """Not implemented."""
        raise NotImplementedError

    def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
        """Not implemented."""
        raise NotImplementedError

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0) -> None:
        """Not implemented."""
        raise NotImplementedError


class DoNothingPolicy(Policy):
    """
    Policy that always returns a do-nothing action.
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
        """Computes actions for the current policy."""
        return [0], [], {}

    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""

    def apply_gradients(self, gradients: ModelGradients) -> None:
        """No gradients to apply."""
        raise NotImplementedError

    def compute_gradients(
        self, postprocessed_batch: SampleBatch
    ) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        """No gradient to compute."""
        return [], {}

    def loss(
        self, model: ModelV2, dist_class: ActionDistribution, train_batch: SampleBatch
    ) -> Union[TensorType, List[TensorType]]:
        """No loss function"""
        raise NotImplementedError

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        """Not implemented."""
        raise NotImplementedError

    def learn_on_batch_from_replay_buffer(
        self, replay_actor: ActorHandle, policy_id: PolicyID
    ) -> Dict[str, TensorType]:
        """Not implemented."""
        raise NotImplementedError

    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
        """Not implemented."""
        raise NotImplementedError

    def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
        """Not implemented."""
        raise NotImplementedError

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0) -> None:
        """Not implemented."""
        raise NotImplementedError


class SelectAgentPolicy(Policy):
    """
    High level agent that determines whether an action is required.
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
        self.config = config

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
        if (
            np.max(max_rho)
            > self.config["model"]["custom_model_config"]["rho_threshold"]
        ):
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
        """No gradients to apply."""
        raise NotImplementedError

    def compute_gradients(
        self, postprocessed_batch: SampleBatch
    ) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        """No gradient to compute."""
        return [], {}

    def loss(
        self, model: ModelV2, dist_class: ActionDistribution, train_batch: SampleBatch
    ) -> Union[TensorType, List[TensorType]]:
        """No loss function"""
        raise NotImplementedError

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        """Not implemented."""
        raise NotImplementedError

    def learn_on_batch_from_replay_buffer(
        self, replay_actor: ActorHandle, policy_id: PolicyID
    ) -> Dict[str, TensorType]:
        """Not implemented."""
        raise NotImplementedError

    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
        """Not implemented."""
        raise NotImplementedError

    def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
        """Not implemented."""
        raise NotImplementedError

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0) -> None:
        """Not implemented."""
        raise NotImplementedError

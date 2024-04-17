"""
Defines agent policies.
"""

import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import grid2op
import gymnasium
import numpy as np
import torch
from ray.actor import ActorHandle
from ray.rllib.algorithms.ppo import PPOTorchPolicy
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
from mahrl.grid2op_env.utils import load_action_space, setup_converter


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


class ValueFunctionTorchPolicy(PPOTorchPolicy):
    def _compute_action_helper(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):
        """Shared forward pass logic (w/ and w/o trajectory view API).
        Adjusted to also return the value of the value function.

        Returns:
            A tuple consisting of a) actions and value functions, b) state_out, c) extra_fetches.
            The input_dict is modified in-place to include a numpy copy of the computed
            actions under `SampleBatch.ACTIONS`.
        """
        (actions, state_out, extra_fetches) = super()._compute_action_helper(
            input_dict, state_batches, seq_lens, explore, timestep
        )
        value = self.model.value_function()
        if not isinstance(actions[0], np.int32):
            print(f"action type: {type(actions[0])}")
            print("Instance cannot be properly found")

        if len(value) > 1 or value[0] >= 1 or value[0] < 0:
            print(f"Value={value[0]}")
            print(f"Value len={len(value)}")
            print("Value exceeds 1, does not work anymore")
            return actions, state_out, extra_fetches
        else:
            print(f"action output: {np.array(actions[0] + value[0])}")
            return np.array([actions[0], value[0]]), state_out, extra_fetches
            # return np.array([actions[0] + value[0]]), state_out, extra_fetches

    # (CustomPPO pid=51757)   File "/Users/barberademol/Documents/GitHub/mahrl_grid2op/2venv_mahrl/lib/python3.11/site-packages/ray/rllib/evaluation/env_runner_v2.py", line 1141, in _process_policy_eval_results
    # (CustomPPO pid=51757)     env_id: int = eval_data[i].env_id
    # (CustomPPO pid=51757)                   ~~~~~~~~~^^^
    # (CustomPPO pid=51757) IndexError: list index out of range

    # ValueError: Expected value argument (Tensor of shape (7,)) to be within the support (IntegerInterval(lower_bound=0, upper_bound=6)) of the distribution Categorical(logits: torch.Size([7, 7])), but found invalid values:
    # tensor([2.0084, 0.0095, 2.0092, 6.0092, 5.0092, 4.0095, 1.0068])

    def action_sampler_fn(
        self,
        model: ModelV2,
        obs_batch: torch.Tensor,
        state_batches: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        print(f"customized sampler")
        # Sample two actions using the model
        action_dist, _ = model(obs_batch, state_batches, self.config)
        actions = action_dist.sample()
        # Return two actions instead of one
        return [actions, actions], (), (), []


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
        env_config = config["model"]["custom_model_config"]["environment"]["env_config"]
        setup_env = grid2op.make(env_config["env_name"], **env_config["grid2op_kwargs"])

        path = os.path.join(
            env_config["lib_dir"],
            f"data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
        )
        self.possible_substation_actions = load_action_space(path, setup_env)

        self.converter = setup_converter(setup_env, self.possible_substation_actions)

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

        Policy.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            config=config,
        )

        # get recurrent memory in view requirments
        self._update_model_view_requirements_from_init_state()

    def get_initial_state(self) -> List[TensorType]:
        """Returns initial RNN state for the current policy.

        Returns:
            List[TensorType]: Initial RNN state for the current policy.
        """
        return [0 for _ in range(1 + len(self.controllable_substations))]

    def is_recurrent(self) -> bool:
        """Whether this Policy holds a recurrent Model.

        Returns:
            True if this Policy has-a RNN-based Model.
        """
        return True

    def compute_actions(  # pylint: disable=signature-differs
        self,
        obs_batch: Dict[str, Any],  # WAS UNION with list before,
        state_batches: List[TensorType],
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, List[Any]]] = None,
        episodes: Optional[List[str]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        """Computes actions for the current policy."""
        # setup counter
        idx = state_batches[0][0]

        # substation to act on is the remaining digits in the state
        substation_to_act_on = [sub[0] for sub in state_batches[1:]]

        # convert all gym to grid2op actions
        for gym_action in obs_batch["proposed_actions"]:
            obs_batch["proposed_actions"][gym_action] = self.converter.convert_act(
                int(gym_action)
            )

        # if no list is created yet, do so
        if obs_batch["reset_capa_idx"][0] or not any(substation_to_act_on):
            idx = 0

            substation_to_act_on = get_capa_substation_id(
                self.config["model"]["custom_model_config"]["line_info"],
                obs_batch,
                self.controllable_substations,
            )

        # find an action that is not the do nothing action by looping over the substations
        chosen_action: dict[str, Any] = {}
        while (not chosen_action) and (idx < len(self.controllable_substations)):
            action_index = idx % len(self.controllable_substations)

            idx += 1
            chosen_action = obs_batch["proposed_actions"][
                str(substation_to_act_on[action_index])
            ]

            # if it's not the do nothing action, return action index (similar to NN)
            # if it's the do nothing action, continue the loop
            if chosen_action:
                return (
                    np.array([substation_to_act_on[action_index]]),
                    [np.array([idx])]
                    + [np.array([sub]) for sub in substation_to_act_on],
                    {},
                )

        # grid is safe or no action is found, reset list count and return DoNothing
        # Communicates that a new order should be made
        return (
            np.array([-1]),
            [np.array([0]) for _ in range(len(substation_to_act_on) + 1)],
            {},
        )

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
        obs_batch: List[float],
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

        max_rho = obs_batch[0]
        if max_rho > self.config["model"]["custom_model_config"]["rho_threshold"]:
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


class RandomPolicy(Policy):
    """
    Policy that chooses a random substation.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        config: AlgorithmConfigDict,
    ):
        env_config = config["model"]["custom_model_config"]["environment"]["env_config"]
        setup_env = grid2op.make(env_config["env_name"], **env_config["grid2op_kwargs"])

        path = os.path.join(
            env_config["lib_dir"],
            f"data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
        )
        self.possible_substation_actions = load_action_space(path, setup_env)

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
        random_sub_id = random.choice(list(self.controllable_substations.keys()))
        return [random_sub_id], [], {}

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

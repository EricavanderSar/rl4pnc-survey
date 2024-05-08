"""
Defines agent policies.
"""

import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import grid2op
import gymnasium as gym
import numpy as np
import torch
from ray.actor import ActorHandle
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    ModelConfigDict,
    ModelGradients,
    ModelInputDict,
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


# pylint: disable=too-many-return-statements
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
    if agent_id.startswith("value_reinforcement_learning_agent"):
        # from agent_id, use re to extract the integer at the end
        id_number = re.search(r"\d+$", agent_id)
        if id_number:
            agent_number = int(id_number.group(0))
            return f"value_reinforcement_learning_policy_{agent_number}"
        return "value_reinforcement_learning_policy"
    if agent_id.startswith("value_function_agent"):
        # from agent_id, use re to extract the integer at the end
        id_number = re.search(r"\d+$", agent_id)
        if id_number:
            agent_number = int(id_number.group(0))
            return f"value_function_policy_{agent_number}"
        return "value_function_policy"
    if agent_id.startswith("high_level_agent"):
        return "high_level_policy"
    if agent_id.startswith("do_nothing_agent"):
        return "do_nothing_policy"
    if agent_id.startswith("choose_substation_agent"):
        return "choose_substation_policy"
    raise NotImplementedError(f"Given AgentID is {agent_id}")


class ActionFunctionTorchPolicy(PPOTorchPolicy):
    """
    A custom policy class that extends the PPOTorchPolicy class.
    This policy is used for action function torch models.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: AlgorithmConfigDict,
    ):
        # self.sub_modelaldkfj = config["model"]["custom_model_config"]["model"]

        self.model = config["model"]["custom_model_config"]["model"]
        super().__init__(observation_space, action_space, config)

        assert isinstance(self.model, ModelV2)

    def make_model(self) -> ModelV2:
        """Creates a new model for this policy."""
        return self.model

    def _compute_action_helper(
        self,
        input_dict: Union[SampleBatch, ModelInputDict],
        state_batches: List[TensorType],
        seq_lens: List[int],
        explore: bool,
        timestep: int,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        """
        Helper function to compute the action based on the input.

        Args:
            input_dict (dict): Input dictionary containing the input data.
            state_batches (List): List of state batches.
            seq_lens (List): List of sequence lengths.
            explore (bool): Whether to explore or not.
            timestep (int): The current timestep.

        Returns:
            Tuple: A tuple containing the logits, state outputs, and extra fetches.
        """
        logits, state_out, extra_fetches = super()._compute_action_helper(
            input_dict, state_batches, seq_lens, explore, timestep
        )
        return logits, state_out, extra_fetches


class OnlyValueFunctionTorchPolicy(PPOTorchPolicy):
    """
    A custom policy class that extends the PPOTorchPolicy class and implements a value-only function policy.

    Args:
        observation_space (gym.Space): The observation space of the environment.
        action_space (gym.Space): The action space of the environment.
        config (AlgorithmConfigDict): The configuration dictionary for the algorithm.

    Attributes:
        sub_modelaldkf: The sub-model used for the policy.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: AlgorithmConfigDict,
    ):
        self.model = config["model"]["custom_model_config"]["model"]
        super().__init__(observation_space, action_space, config)

        assert isinstance(self.model, ModelV2)

    def make_model(self) -> ModelV2:
        """Creates a new model for this policy."""
        return ValueOnlyModel(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=2,  # mean and std
            model_config=self.config,
            name="CustomModelV2",
            _sub_model=self.model,
        )


class CustomFCN(FullyConnectedNetwork):
    """
    Implements a custom FCN model that appends the mean and stdev for the value function.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
        )

    def __call__(
        self,
        input_dict: Union[SampleBatch, ModelInputDict],
        state: Union[List[Any], None] = None,
        seq_lens: TensorType = None,
    ) -> tuple[TensorType, List[TensorType]]:
        """
        Executes the policy for a given input.

        Args:
            input_dict (Union[SampleBatch, ModelInputDict]): The input data for the policy.
            state (Union[List[Any], None], optional): The state of the policy. Defaults to None.
            seq_lens (TensorType, optional): The sequence lengths. Defaults to None.

        Returns:
            tuple[TensorType, List[TensorType]]: The outputs of the policy and the updated state.
        """
        outputs, state_out = super().__call__(input_dict, state, seq_lens)

        return outputs, state_out

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> Any:
        """
        Overwrite deepcopy so that it returns the same actual object to both policies.

        Args:
            memo (Optional[Dict[int, Any]], optional): The memo dictionary. Defaults to None.

        Returns:
            Any: The copied object.
        """
        return self

    def import_from_h5(self, h5_file: str) -> None:
        """
        Import model weights from an HDF5 file.

        This method should be overridden by subclasses to provide actual
        functionality.

        Args:
            h5_file (str): The path to the HDF5 file.
        """
        raise NotImplementedError("This model cannot import weights from HDF5 files.")


class ValueOnlyModel(FullyConnectedNetwork):
    """
    Implements a custom FCN model that appends the mean and stdev for the value function.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        _sub_model: FullyConnectedNetwork,
    ):
        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
        )

        self._sub_model = _sub_model
        self._values = None

    def __call__(
        self,
        input_dict: Union[SampleBatch, ModelInputDict],
        state: Union[List[Any], None] = None,
        seq_lens: TensorType = None,
    ) -> tuple[TensorType, List[TensorType]]:
        logits_act, state_out = self._sub_model(
            input_dict=input_dict, state=state, seq_lens=seq_lens
        )

        # create empty shell for gaussians of Value Function
        outputs = torch.empty((logits_act.shape[0], 2))

        # store values
        self._values = self._sub_model.value_function().cpu().detach()

        # replace last two values of with the mean (vf) and stdev (symbolic stdev)
        assert self._values is not None

        outputs[:, -2:-1] = self._values.reshape(-1, 1)
        outputs[:, -1:] = -1e2

        return outputs, state_out

    def value_function(self) -> TensorType:
        """
        Returns the value function.

        Raises:
            RuntimeError: If `forward` method has not been called before.

        Returns:
            The value function as a TensorType object.
        """
        if self._values is None:
            raise RuntimeError("Need to call forward first")

        return self._values

    def import_from_h5(self, h5_file: str) -> None:
        """
        Import model weights from an HDF5 file.

        This method should be overridden by subclasses to provide actual
        functionality.

        Args:
            h5_file (str): The path to the HDF5 file.
        """
        raise NotImplementedError("This model cannot import weights from HDF5 files.")


class CapaPolicy(Policy):
    """
    Policy that that returns a substation to act on based on the CAPA heuristic.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: AlgorithmConfigDict,
    ):
        env_config = config["model"]["custom_model_config"]["environment"]["env_config"]
        setup_env = grid2op.make(env_config["env_name"], **env_config["grid2op_kwargs"])

        path = os.path.join(
            env_config["lib_dir"],
            f"data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
        )
        self.possible_substation_actions = load_action_space(path, setup_env)

        # add the do-nothing action at index 0
        do_nothing_action = setup_env.action_space({})
        self.possible_substation_actions.insert(0, do_nothing_action)

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
        # setup counter (1st digit of state)
        idx = state_batches[0][0]

        # substation to act on is the remaining digits in the state
        substation_to_act_on = [sub[0] for sub in state_batches[1:]]

        # convert all gym to grid2op actions
        for sub, gym_action in obs_batch["proposed_actions"].items():
            obs_batch["proposed_actions"][str(sub)] = self.converter.convert_act(
                int(gym_action[0])
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
            chosen_action = obs_batch["proposed_actions"][
                str(substation_to_act_on[idx % len(self.controllable_substations)])
            ]

            # if it's not the do nothing action, return action index (similar to NN)
            # if it's the do nothing action, continue the loop
            if chosen_action:
                return (
                    np.array(
                        [substation_to_act_on[idx % len(self.controllable_substations)]]
                    ),
                    [np.array([idx + 1])]
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
        observation_space: gym.Space,
        action_space: gym.Space,
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
        observation_space: gym.Space,
        action_space: gym.Space,
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
        observation_space: gym.Space,
        action_space: gym.Space,
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
        # extract the keys from the action dict from the obs batch
        if isinstance(obs_batch, dict):
            action_keys = list(obs_batch["proposed_actions"].keys())
        else:
            action_keys = list(obs_batch[0]["proposed_actions"].keys())
        random_sub_id = random.randrange(len(action_keys))

        # if the last agent is chosen, return -1 to do nothing
        if random_sub_id == len(action_keys):
            random_sub_id = -1
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


class ArgMaxPolicy(Policy):
    """
    Policy that chooses a random substation.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
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
        # find the index with the highest value in proposed_confidences
        if isinstance(obs_batch, dict):
            proposed_confidences = obs_batch["proposed_confidences"]
        else:
            proposed_confidences = obs_batch[0]["proposed_confidences"]

        sub_id = np.argmax(proposed_confidences)
        return [sub_id], [], {}

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


class SampleValuePolicy(Policy):
    """
    Policy that chooses a random substation.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
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
        # find the index with the highest value in proposed_confidences
        if isinstance(obs_batch, dict):
            proposed_confidences = obs_batch["proposed_confidences"]
        else:
            proposed_confidences = obs_batch[0]["proposed_confidences"]

        # take the sub_id based on a uniform sample of proposed_confidences
        sub_id = random.choices(
            list(proposed_confidences.keys()),
            weights=list(proposed_confidences.values()),
            k=1,
        )[0]

        return [sub_id], [], {}

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

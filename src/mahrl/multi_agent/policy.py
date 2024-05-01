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
from ray.actor import ActorHandle
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
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
from mahrl.multi_agent.value_policy import YetAnotherTorchCentralizedCriticModel


class ActionFunctionTorchPolicy(PPOTorchPolicy):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        config: AlgorithmConfigDict,
    ):
        self.model = config["model"]["custom_model"]
        config = config["model"]["custom_model_config"]

        assert isinstance(self.model, ModelV2)

        print(f"Action space: {action_space}")
        print(f"Observation space: {observation_space}")

        super().__init__(observation_space, action_space, config)

    def make_model(self) -> ModelV2:
        """Creates a new model for this policy."""
        return self.model


class OnlyValueFunctionTorchPolicy(PPOTorchPolicy):
    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        config: AlgorithmConfigDict,
    ):
        self.model = config["model"]["custom_model"]
        config = config["model"]["custom_model_config"]

        assert isinstance(self.model, ModelV2)

        print(f"Action space: {action_space}")
        print(f"Observation space: {observation_space}")

        super().__init__(observation_space, action_space, config)

    def make_model(self) -> ModelV2:
        """Creates a new model for this policy."""
        return self.model

    def _compute_action_helper(
        self,
        input_dict: Dict[str, Any],
        state_batches: List[TensorType],
        seq_lens: TensorType,
        explore: bool,
        timestep: Optional[int],
    ) -> Tuple[Dict[str, TensorType], List[TensorType], Dict[str, TensorType]]:
        """Shared forward pass logic (w/ and w/o trajectory view API).
        Adjusted to also return the value of the value function.

        Returns:
            A tuple consisting of a) actions and value functions, b) state_out, c) extra_fetches.
            The input_dict is modified in-place to include a numpy copy of the computed
            actions under `SampleBatch.ACTIONS`.
        """
        # seq_lens = [1, 4]
        # print(f"Input dict: {input_dict[SampleBatch.ACTION_DIST_INPUTS]}")

        (actions, state_out, extra_fetches) = self._compute_action_helper(
            input_dict, state_batches, seq_lens, explore, timestep
        )

        print(f"Actions={actions}")

        actions[0] = self.model.value_function().cpu().detach().numpy()
        actions[1] = 0.0

        return actions, state_out, extra_fetches


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
    if agent_id.startswith("high_level_agent"):
        return "high_level_policy"
    if agent_id.startswith("do_nothing_agent"):
        return "do_nothing_policy"
    if agent_id.startswith("choose_substation_agent"):
        return "choose_substation_policy"
    raise NotImplementedError(f"Given AgentID is {agent_id}")


# class CustomTorchModelV2(FullyConnectedNetwork):
#     """
#     Implements a custom FCN model that appends the mean and stdev for the value function.
#     """

#     def __call__(
#         self,
#         input_dict: Union[SampleBatch, ModelInputDict],
#         state: Union[List[Any], None] = None,
#         seq_lens: TensorType = None,
#     ) -> tuple[TensorType, List[TensorType]]:

#         # get the logits and state for the actions
#         outputs, state_out = super().__call__(
#             input_dict=input_dict, state=state, seq_lens=seq_lens
#         )

#         # add two columns of zeroes to the logit output to represent the value
#         outputs = torch.cat(
#             (
#                 outputs,
#                 torch.zeros(
#                     (outputs.shape[0], 2),
#                     dtype=torch.float32,
#                 ),
#             ),
#             dim=1,
#         )

#         print(f"Final outputs: {outputs}")

#         return outputs, state_out


# ModelCatalog.register_custom_model("CustomModelV2", CustomTorchModelV2)


# class ValueFunctionTorchPolicy(PPOTorchPolicy):
#     """
#     Custom Torch Policy that outputs the output of the value function
#     besides the action.
#     """

#     def __init__(
#         self,
#         observation_space: gymnasium.Space,
#         action_space: gymnasium.Space,
#         config: AlgorithmConfigDict,
#     ):
#         # action space is dict
#         assert isinstance(action_space, gymnasium.spaces.Dict)

#         # action and value in dict
#         assert "action" in action_space.spaces
#         assert "value" in action_space.spaces

#         self._policy = PPOTorchPolicy(
#             observation_space, action_space.spaces["action"], config
#         )
#         super().__init__(observation_space, action_space, config)

#     def make_model(self) -> ModelV2:
#         """
#         Creates a new model for this policy with two less dimensions
#         to account for not learning the value function.
#         """
#         _, logit_dim = ModelCatalog.get_action_dist(
#             self._policy.action_space,
#             self._policy.config["model"],
#             framework=self.framework,
#         )

#         return CustomTorchModelV2(
#             obs_space=self._policy.observation_space,
#             action_space=self._policy.action_space,
#             num_outputs=logit_dim,
#             model_config=self._policy.config,
#             name="CustomModelV2",
#         )

#     def _compute_action_helper(
#         self,
#         input_dict: Dict[str, Any],
#         state_batches: List[TensorType],
#         seq_lens: TensorType,
#         explore: bool,
#         timestep: Optional[int],
#     ) -> Tuple[Dict[str, TensorType], List[TensorType], Dict[str, TensorType]]:
#         """Shared forward pass logic (w/ and w/o trajectory view API).
#         Adjusted to also return the value of the value function.

#         Returns:
#             A tuple consisting of a) actions and value functions, b) state_out, c) extra_fetches.
#             The input_dict is modified in-place to include a numpy copy of the computed
#             actions under `SampleBatch.ACTIONS`.
#         """

#         (actions, state_out, extra_fetches) = self._policy._compute_action_helper(
#             input_dict, state_batches, seq_lens, explore, timestep
#         )

#         try:
#             value = self.model.value_function().cpu().detach().numpy()
#             value = value[:, None]
#         except AssertionError:
#             # during initialization
#             value = np.zeros_like(actions).astype(np.float32)

#         dict_act = {"action": actions, "value": value}

#         # Create an empty array with the same number of rows to account for the missing mean and stdev of value
#         empty_columns = np.empty((extra_fetches["action_dist_inputs"].shape[0], 2))

#         # Add the empty columns to the array
#         extra_fetches["action_dist_inputs"] = np.hstack(
#             (extra_fetches["action_dist_inputs"], empty_columns)
#         )

#         return dict_act, state_out, extra_fetches


class CustomTorchModelV2(FullyConnectedNetwork):
    """
    Implements a custom FCN model that appends the mean and stdev for the value function.
    """

    def __call__(
        self,
        input_dict: Union[SampleBatch, ModelInputDict],
        state: Union[List[Any], None] = None,
        seq_lens: TensorType = None,
    ) -> tuple[TensorType, List[TensorType]]:
        if "action_dist_inputs" in input_dict:
            # print(f"Input dict: {input_dict['action_dist_inputs']}")
            # replace last two digits with 0
            input_dict["action_dist_inputs"][:, -2:] = 0.0
            # print(f"New input dict: {input_dict['action_dist_inputs']}")
        outputs, state_out = super().__call__(
            input_dict=input_dict, state=state, seq_lens=seq_lens
        )

        # replace last two values of each row with +- 0
        outputs[:, -2:] = (self.value_function(),)

        self.value_function()

        return outputs, state_out

    def import_from_h5(self, h5_file: str) -> None:
        """
        Import model weights from an HDF5 file.

        This method should be overridden by subclasses to provide actual
        functionality.

        Args:
            h5_file (str): The path to the HDF5 file.
        """
        raise NotImplementedError("This model cannot import weights from HDF5 files.")


class ValueFunctionTorchPolicy(PPOTorchPolicy):
    """
    Custom Torch Policy that outputs the output of the value function
    besides the action.
    """

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        config: AlgorithmConfigDict,
    ):
        # action space is dict
        assert isinstance(action_space, gymnasium.spaces.Dict)

        # action and value in dict
        assert "action" in action_space.spaces
        assert "value" in action_space.spaces

        # self._policy = PPOTorchPolicy(
        #     observation_space, action_space.spaces["action"], config
        # )
        super().__init__(observation_space, action_space, config)

        # self.dist_class = CustomActionDistribution

    def make_model(self) -> ModelV2:
        """Creates a new model for this policy."""
        _, logit_dim = ModelCatalog.get_action_dist(
            self.action_space, self.config["model"], framework=self.framework
        )
        return CustomTorchModelV2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=logit_dim,
            model_config=self.config,
            name="CustomModelV2",
        )

    def _compute_action_helper(
        self,
        input_dict: Dict[str, Any],
        state_batches: List[TensorType],
        seq_lens: TensorType,
        explore: bool,
        timestep: Optional[int],
    ) -> Tuple[Dict[str, TensorType], List[TensorType], Dict[str, TensorType]]:
        """Shared forward pass logic (w/ and w/o trajectory view API).
        Adjusted to also return the value of the value function.

        Returns:
            A tuple consisting of a) actions and value functions, b) state_out, c) extra_fetches.
            The input_dict is modified in-place to include a numpy copy of the computed
            actions under `SampleBatch.ACTIONS`.
        """
        # seq_lens = [1, 4]
        # print(f"Input dict: {input_dict[SampleBatch.ACTION_DIST_INPUTS]}")

        (actions, state_out, extra_fetches) = self._compute_action_helper(
            input_dict, state_batches, seq_lens, explore, timestep
        )

        try:
            value = self.model.value_function().cpu().detach().numpy()
            value = value[:, None]
            # print(f"Value: {value}")
        except AssertionError:
            # during initialization
            value = np.zeros_like(actions).astype(np.float32)

            # add a small value to each item of the array for numerical stability TODO does this matter
        # value += 1e-3
        # value += 1

        # if len(value.shape) < 2:
        #     print("In this if-statemet")
        #     value = value[:, None]
        dict_act = {"action": actions, "value": value}

        # Create an empty array with the same number of rows to account for the missing mean and stdev of value
        empty_columns = np.empty((extra_fetches["action_dist_inputs"].shape[0], 2))
        # empty_columns += 1e-3
        # TODO: Check if numerical stability breaks here with extra fetches

        # Add the empty columns to the array
        extra_fetches["action_dist_inputs"] = np.hstack(
            (extra_fetches["action_dist_inputs"], empty_columns)
        )

        print(extra_fetches)

        return dict_act, state_out, extra_fetches


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

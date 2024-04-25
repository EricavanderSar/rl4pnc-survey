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

# from ray.rllib.models.catalog import ModelCatalog
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
    if agent_id.startswith("reinforcement_learning_agent") or agent_id.startswith(
        "value_reinforcement_learning_agent"
    ):
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
    raise NotImplementedError(f"Given AgentID is {agent_id}")


# class CustomTorchModelV2(FullyConnectedNetwork):
#     """
#     Implements a custom FCN model that appends the mean and stdev for the value function.
#     """

#     # TODO investigate constructor for [1,2] shape
#     def __call__(
#         self,
#         input_dict: Union[SampleBatch, ModelInputDict],
#         state: Union[List[Any], None] = None,
#         seq_lens: TensorType = None,
#     ) -> tuple[TensorType, List[TensorType]]:
#         # return super().__call__(input_dict=input_dict, state=state, seq_lens=seq_lens)
#         # TODO: append value as mean and stdev as 0
#         # print(f"Seq_lens in model: {seq_lens}")
#         # print(
#         #     f"CALL: {super().__call__(input_dict=input_dict, state=state, seq_lens=seq_lens)}"
#         # )
#         # seq_lens = torch.tensor([3], dtype=torch.int32)

#         outputs, state_out = super().__call__(
#             input_dict=input_dict, state=state, seq_lens=seq_lens
#         )

#         # add a small value to each outputs tensor for numeric stability
#         # outputs += 1e-6

#         # print(
#         #     f"SHAPE : {outputs.shape} consists of {outputs.shape[0]} + out {state_out}"
#         # )

#         # if during initialzation
#         # if outputs.shape[0] != 1 and outputs.shape[0] != 32:
#         #     # add two columns to the logits output
#         #     value_function_tensor = torch.zeros((outputs.shape[0], 2))
#         #     new_outputs = torch.cat((outputs, value_function_tensor), dim=1)

#         #     print(f"NEW SHAPE : {new_outputs.shape} + out {state_out}")
#         #     return new_outputs, state_out  # if len(state_out) > 0 else (state or [])
#         # else:
#         return outputs, state_out  # if len(state_out) > 0 else (state or [])

# if outputs.shape[0] > 1:
#     # add two columns to the output
#     tensor_b = torch.zeros((outputs.shape[0], 2))
#     new_outputs = torch.cat((outputs, tensor_b), dim=1)

#     # tensor_b = torch.zeros((2, outputs.shape[1]))
#     # new_outputs = torch.cat((outputs, tensor_b), dim=0)

#     print(f"NEW SHAPE : {new_outputs.shape} + out {state_out}")
#     return new_outputs, state_out  # if len(state_out) > 0 else (state or [])
# else:
#     print(f"KEEP OLD")
#     return outputs, state_out

# def __call__(
#     self,
#     input_dict: Union[SampleBatch, ModelInputDict],
#     state: List[Any] = None,
#     seq_lens: TensorType = None,
# ) -> (TensorType, List[TensorType]):
#     """Call the model with the given input tensors and state.

#     This is the method used by RLlib to execute the forward pass. It calls
#     forward() internally after unpacking nested observation tensors.

#     Custom models should override forward() instead of __call__.

#     Args:
#         input_dict: Dictionary of input tensors.
#         state: list of state tensors with sizes matching those
#             returned by get_initial_state + the batch dimension
#         seq_lens: 1D tensor holding input sequence lengths.

#     Returns:
#         A tuple consisting of the model output tensor of size
#             [BATCH, output_spec.size] or a list of tensors corresponding to
#             output_spec.shape_list, and a list of state tensors of
#             [BATCH, state_size_i] if any.
#     """

#     # Original observations will be stored in "obs".
#     # Flattened (preprocessed) obs will be stored in "obs_flat".

#     # SampleBatch case: Models can now be called directly with a
#     # SampleBatch (which also includes tracking-dict case (deprecated now),
#     # where tensors get automatically converted).
#     if isinstance(input_dict, SampleBatch):
#         restored = input_dict.copy(shallow=True)
#     else:
#         restored = input_dict.copy()

#     # Backward compatibility.
#     if not state:
#         state = []
#         i = 0
#         while "state_in_{}".format(i) in input_dict:
#             state.append(input_dict["state_in_{}".format(i)])
#             i += 1
#     if seq_lens is None:
#         seq_lens = input_dict.get(SampleBatch.SEQ_LENS)

#     # No Preprocessor used: `config._disable_preprocessor_api`=True.
#     # TODO: This is unnecessary for when no preprocessor is used.
#     #  Obs are not flat then anymore. However, we'll keep this
#     #  here for backward-compatibility until Preprocessors have
#     #  been fully deprecated.
#     if self.model_config.get("_disable_preprocessor_api"):
#         restored["obs_flat"] = input_dict["obs"]
#     # Input to this Model went through a Preprocessor.
#     # Generate extra keys: "obs_flat" (vs "obs", which will hold the
#     # original obs).
#     else:
#         restored["obs"] = restore_original_dimensions(
#             input_dict["obs"], self.obs_space, self.framework
#         )
#         try:
#             if len(input_dict["obs"].shape) > 2:
#                 restored["obs_flat"] = flatten(input_dict["obs"], self.framework)
#             else:
#                 restored["obs_flat"] = input_dict["obs"]
#         except AttributeError:
#             restored["obs_flat"] = input_dict["obs"]

#     print(f"Restored: {restored['obs_flat']}, shape: {restored['obs_flat'].shape}")
#     with self.context():
#         res = self.forward(restored, state or [], seq_lens)

#     if isinstance(input_dict, SampleBatch):
#         input_dict.accessed_keys = restored.accessed_keys - {"obs_flat"}
#         input_dict.deleted_keys = restored.deleted_keys
#         input_dict.added_keys = restored.added_keys - {"obs_flat"}

#     if (not isinstance(res, list) and not isinstance(res, tuple)) or len(res) != 2:
#         raise ValueError(
#             "forward() must return a tuple of (output, state) tensors, "
#             "got {}".format(res)
#         )
#     outputs, state_out = res

#     if not isinstance(state_out, list):
#         raise ValueError("State output is not a list: {}".format(state_out))

#     self._last_output = outputs
#     # breakpoint()
#     return outputs, state_out if len(state_out) > 0 else (state or [])


# ModelCatalog.register_custom_model("CustomModelV2", CustomTorchModelV2)


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

        self._policy = PPOTorchPolicy(
            observation_space, action_space.spaces["action"], config
        )
        super().__init__(observation_space, action_space, config)

    # def make_model(self) -> ModelV2:
    #     """Creates a new model for this policy."""
    #     _, logit_dim = ModelCatalog.get_action_dist(
    #         self.action_space, self.config["model"], framework=self.framework
    #     )
    #     return CustomTorchModelV2(
    #         obs_space=self.observation_space,
    #         action_space=self.action_space,
    #         num_outputs=logit_dim,
    #         model_config=self.config,
    #         name="CustomModelV2",
    #     )

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
        # print(f"Input dict: {input_dict}")

        (actions, state_out, extra_fetches) = self._policy._compute_action_helper(
            input_dict, state_batches, seq_lens, explore, timestep
        )

        try:
            value = self.model.value_function().cpu().detach().numpy()
        except AssertionError:
            # during initialization
            value = np.zeros_like(actions).astype(np.float32)

            # add a small value to each item of the array for numerical stability TODO does this matter
            value += 1e-3
        # value += 1

        # check if value ever contains nan or inf
        if np.isnan(value).any() or np.isinf(value).any():
            raise ValueError("Value contains nan or inf")

        if len(value.shape) < 2:
            value = value[:, None]
        dict_act = {"action": actions, "value": value}

        # Create an empty array with the same number of rows to account for the missing mean and stdev of value
        empty_columns = np.empty((extra_fetches["action_dist_inputs"].shape[0], 2))

        # TODO: Check if numerical stability breaks here with extra fetches

        # Add the empty columns to the array
        extra_fetches["action_dist_inputs"] = np.hstack(
            (extra_fetches["action_dist_inputs"], empty_columns)
        )

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

"""
Implements yaml config loading.
"""
import os
from typing import Any, Callable, Union

import yaml
from grid2op.Action import BaseAction, PowerlineSetAction
from grid2op.Opponent import (
    BaseActionBudget,
    BaseOpponent,
    OpponentSpace,
    RandomLineOpponent,
)
from gymnasium.spaces import Discrete
from ray import tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec

from yaml.loader import FullLoader, Loader, UnsafeLoader
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

from rl4pnc.experiments.callback import CustomMetricsCallback
from rl4pnc.experiments.opponent import ReconnectingOpponentSpace
from rl4pnc.experiments.rewards import (
    LossReward,
    ScaledL2RPNReward,
    AlphaZeroRW,
    LossRewardRescaled2,
    LossRewardNew,
    RewardRho
)
from grid2op.Reward import L2RPNReward,LinesCapacityReward
from rl4pnc.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from rl4pnc.multi_agent.policy import (
    DoNothingPolicy,
    SelectAgentPolicy,
    policy_mapping_fn,
)


def discrete_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: ScalarNode
) -> Discrete:
    """Custom constructor for Discrete"""
    return Discrete(int(loader.construct_scalar(node) or 0))


def algorithm_config_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> AlgorithmConfig:
    """Custom constructor for AlgorithmConfig"""
    loader.construct_mapping(node)
    return AlgorithmConfig()


def policy_spec_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> PolicySpec:
    """Custom constructor for PolicySpec"""
    loader.construct_mapping(node)
    return PolicySpec()


def customized_environment_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> CustomizedGrid2OpEnvironment:
    """Custom constructor for CustomizedGrid2OpEnvironment"""
    fields = {str(k): v for k, v in loader.construct_mapping(node).items()}
    env_config = fields.get("env_config", {})  # Extract env_config explicitly
    fields["env_config"] = env_config
    return CustomizedGrid2OpEnvironment(**fields)


def loss_reward_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> LossReward:
    """Custom constructor for LossReward"""
    return LossReward()


def loss_rw_rescaled_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> LossRewardRescaled2:
    """Custom constructor for LossReward"""
    return LossRewardRescaled2()


def loss_rw_new_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> LossRewardNew:
    """Custom constructor for LossReward"""
    return LossRewardNew()


def scaled_reward_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> ScaledL2RPNReward:
    """Custom constructor for ScaledL2RPNReward"""
    return ScaledL2RPNReward()


def l2rpn_reward_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> L2RPNReward:
    """Custom constructor for L2RPNReward"""
    return L2RPNReward()


def linecap_reward_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> LinesCapacityReward:
    """Custom constructor for L2RPNReward"""
    return LinesCapacityReward()


def binbin_reward_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> RewardRho:
    """Custom constructor for L2RPNReward"""
    return RewardRho()


def alphazero_reward_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> AlphaZeroRW:
    """Custom constructor for L2RPNReward"""
    return AlphaZeroRW()


def policy_mapping_fn_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> Callable[[str, EpisodeV2, RolloutWorker], str]:
    """Custom constructor for policy_mapping_fn"""
    return policy_mapping_fn


def custom_metrics_callback_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> DefaultCallbacks:
    """Custom constructor for CustomMetricsCallback"""
    return CustomMetricsCallback


def select_agent_policy_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> SelectAgentPolicy:
    """Custom constructor for SelectAgentPolicy"""
    fields = {str(k): v for k, v in loader.construct_mapping(node).items()}
    return SelectAgentPolicy(**fields)


def do_nothing_policy_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> DoNothingPolicy:
    """Custom constructor for DoNothingPolicy"""
    fields = {str(k): v for k, v in loader.construct_mapping(node).items()}
    return DoNothingPolicy(**fields)


def float_to_integer(float_value: float) -> Union[int, float]:
    """
    Turns a float into an int if appropriate. Otherwise keep int.
    """
    if float_value.is_integer():
        return int(float_value)
    return float_value


def tune_search_quniform_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: SequenceNode
) -> Any:
    """
    Constructor for tune uniform float sampling

    """
    vals = loader.construct_sequence(node)
    if all(isinstance(val, int) for val in vals):
        return tune.qrandint(vals[0], vals[1], vals[2])
    return tune.quniform(vals[0], vals[1], vals[2])

def tune_search_qloguniform_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> Any:
    """
    Constructor for tune uniform float sampling

    """
    vals = loader.construct_sequence(node)
    return tune.qloguniform(vals[0], vals[1], vals[2])

def tune_search_grid_search_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> Any:
    """
    Constructor for tune grid search.
    """
    vals = []
    for sub_node in node.value:
        if isinstance(sub_node, yaml.SequenceNode):
            val = loader.construct_sequence(sub_node)
        elif isinstance(sub_node, yaml.ScalarNode):
            try:
                val = float_to_integer(float(sub_node.value))
            except ValueError:
                val = sub_node.value
        vals.append(val)
    return tune.grid_search(vals)


def tune_choice_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> Any:
    """
    Constructor for tune grid search.

    """
    vals = []
    for sub_node in node.value:
        if sub_node.value == "True":
            val = True
        elif sub_node.value == "False":
            val = False
        else:
            if isinstance(sub_node, yaml.SequenceNode):
                val = loader.construct_sequence(sub_node)
            elif isinstance(sub_node, yaml.ScalarNode):
                try:
                    val = float_to_integer(float(sub_node.value))
                except ValueError:
                    val = sub_node.value
        vals.append(val)
    return tune.choice(vals)


def powerline_action_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> BaseAction:
    """Custom constructor for PowerlineSetAction"""
    return PowerlineSetAction


def randomline_opponent_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> BaseOpponent:
    """Custom constructor for RandomLineOpponent"""
    return RandomLineOpponent


def baseaction_budget_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> BaseActionBudget:
    """Custom constructor for BaseActionBudget"""
    return BaseActionBudget


def reconnecting_opponent_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> OpponentSpace:
    """Custom constructor for ReconnectingOpponentSpace"""
    return ReconnectingOpponentSpace


def path_workdir_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> str:
    """Adjust working directory"""
    import os
    print(os.getcwd())
    return os.getcwd()


def add_constructors() -> None:
    """Add the constructors to the yaml loader"""
    yaml.FullLoader.add_constructor(
        "!CustomizedGrid2OpEnvironment", customized_environment_constructor
    )
    yaml.FullLoader.add_constructor("!LossReward", loss_reward_constructor)
    yaml.FullLoader.add_constructor("!LossRwRescaled2", loss_rw_rescaled_constructor)
    yaml.FullLoader.add_constructor("!LossRwNew", loss_rw_new_constructor)
    yaml.FullLoader.add_constructor("!ScaledL2RPNReward", scaled_reward_constructor)
    yaml.FullLoader.add_constructor("!L2RPNReward", l2rpn_reward_constructor)
    yaml.FullLoader.add_constructor("!AlphaZeroRW", alphazero_reward_constructor)
    yaml.FullLoader.add_constructor("!LinesCapacityReward", linecap_reward_constructor)
    yaml.FullLoader.add_constructor("!RewardRho", binbin_reward_constructor)
    yaml.FullLoader.add_constructor("!policy_mapping_fn", policy_mapping_fn_constructor)
    yaml.FullLoader.add_constructor(
        "!CustomMetricsCallback", custom_metrics_callback_constructor
    )
    yaml.FullLoader.add_constructor(
        "!SelectAgentPolicy", select_agent_policy_constructor
    )
    yaml.FullLoader.add_constructor("!DoNothingPolicy", do_nothing_policy_constructor)
    yaml.FullLoader.add_constructor("!Discrete", discrete_constructor)
    yaml.FullLoader.add_constructor("!AlgorithmConfig", algorithm_config_constructor)
    yaml.FullLoader.add_constructor("!PolicySpec", policy_spec_constructor)
    yaml.FullLoader.add_constructor("!quniform", tune_search_quniform_constructor)
    yaml.FullLoader.add_constructor("!qloguniform", tune_search_qloguniform_constructor)
    yaml.FullLoader.add_constructor("!grid_search", tune_search_grid_search_constructor)
    yaml.FullLoader.add_constructor("!choice", tune_choice_constructor)
    yaml.FullLoader.add_constructor("!PowerlineSetAction", powerline_action_constructor)
    yaml.FullLoader.add_constructor(
        "!RandomLineOpponent", randomline_opponent_constructor
    )
    yaml.FullLoader.add_constructor("!BaseActionBudget", baseaction_budget_constructor)
    yaml.FullLoader.add_constructor(
        "!ReconnectingOpponentSpace", reconnecting_opponent_constructor
    )
    yaml.FullLoader.add_constructor(
        "!workdir", path_workdir_constructor
    )


def load_config(path: str) -> Any:
    """Adds constructors and returns config."""
    add_constructors()

    with open(path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

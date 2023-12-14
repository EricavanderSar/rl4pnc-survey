"""
Implements yaml config loading.
"""

from typing import Any, Callable, Union

import yaml
from gymnasium.spaces import Discrete
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import PolicySpec
from yaml.constructor import BaseConstructor
from yaml.loader import FullLoader, Loader, UnsafeLoader
from yaml.nodes import Node

from mahrl.experiments.callback import CustomMetricsCallback
from mahrl.experiments.rewards import LossReward
from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from mahrl.multi_agent.policy import (
    DoNothingPolicy,
    SelectAgentPolicy,
    policy_mapping_fn,
)


def discrete_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: Node
) -> Discrete:
    """Custom constructor for Discrete"""
    return Discrete(int(loader.construct_object(node) or 0))


def algorithm_config_constructor(
    loader: BaseConstructor, node: Node
) -> AlgorithmConfig:
    """Custom constructor for AlgorithmConfig"""
    loader.construct_object(node)
    return AlgorithmConfig()


def policy_spec_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: Node
) -> PolicySpec:
    """Custom constructor for PolicySpec"""
    loader.construct_object(node)
    return PolicySpec()


def customized_environment_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: Node
) -> CustomizedGrid2OpEnvironment:
    """Custom constructor for CustomizedGrid2OpEnvironment"""
    fields = loader.construct_object(node, deep=True)
    env_config = fields.get("env_config", {})  # Extract env_config explicitly
    fields["env_config"] = env_config
    return CustomizedGrid2OpEnvironment(**fields)


def loss_reward_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: Node
) -> LossReward:
    """Custom constructor for LossReward"""
    return LossReward()


def policy_mapping_fn_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: Node
) -> Callable[[str, EpisodeV2, RolloutWorker], str]:
    """Custom constructor for policy_mapping_fn"""
    return policy_mapping_fn


def custom_metrics_callback_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: Node
) -> DefaultCallbacks:
    """Custom constructor for CustomMetricsCallback"""
    return CustomMetricsCallback


def select_agent_policy_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: Node
) -> SelectAgentPolicy:
    """Custom constructor for SelectAgentPolicy"""
    fields = loader.construct_object(node)
    return SelectAgentPolicy(**fields)


def do_nothing_policy_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: Node
) -> DoNothingPolicy:
    """Custom constructor for DoNothingPolicy"""
    fields = loader.construct_object(node)
    return DoNothingPolicy(**fields)


def add_constructors() -> None:
    """Add the constructors to the yaml loader"""
    yaml.add_constructor(
        "!CustomizedGrid2OpEnvironment", customized_environment_constructor
    )
    yaml.add_constructor("!LossReward", loss_reward_constructor)
    yaml.add_constructor("!policy_mapping_fn", policy_mapping_fn_constructor)
    yaml.add_constructor("!CustomMetricsCallback", custom_metrics_callback_constructor)
    yaml.add_constructor("!SelectAgentPolicy", select_agent_policy_constructor)
    yaml.add_constructor("!DoNothingPolicy", do_nothing_policy_constructor)
    yaml.add_constructor("!Discrete", discrete_constructor)
    yaml.add_constructor("!AlgorithmConfig", algorithm_config_constructor)
    yaml.add_constructor("!PolicySpec", policy_spec_constructor)


def load_config(path: str) -> Any:  # TODO change to dict?
    """Adds constructors and returns config."""
    add_constructors()

    with open(path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

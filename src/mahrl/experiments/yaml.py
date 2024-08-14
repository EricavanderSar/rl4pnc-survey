"""
Implements yaml config loading.
"""

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
from ray.tune.search.sample import Integer, randint
from yaml.loader import FullLoader, Loader, UnsafeLoader
from yaml.nodes import MappingNode, ScalarNode

from mahrl.experiments.callback import (
    CustomMetricsCallback,
    CustomPPOMetricsCallback,
    SingleAgentCallback,
)
from mahrl.experiments.opponent import ReconnectingOpponentSpace
from mahrl.experiments.rewards import LossReward, ScaledL2RPNReward
from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from mahrl.multi_agent.policy import (
    DoNothingPolicy,
    SelectAgentPolicy,
    policy_mapping_fn,
)


def discrete_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: ScalarNode
) -> Discrete:
    """
    Custom constructor for Discrete.

    This function is used as a custom constructor for the Discrete class. It takes a YAML loader
    and a ScalarNode as input and returns a Discrete object.

    Parameters:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader used to load the scalar value.
        node (ScalarNode): The ScalarNode containing the scalar value to be loaded.

    Returns:
        Discrete: The constructed Discrete object.

    """
    return Discrete(int(loader.construct_scalar(node) or 0))


def algorithm_config_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> AlgorithmConfig:
    """
    Custom constructor for AlgorithmConfig.

    Args:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader.
        node (MappingNode): The YAML node.

    Returns:
        AlgorithmConfig: An instance of AlgorithmConfig.
    """
    loader.construct_mapping(node)
    return AlgorithmConfig()


def policy_spec_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> PolicySpec:
    """
    Custom constructor for PolicySpec.

    Args:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader.
        node (MappingNode): The YAML node.

    Returns:
        PolicySpec: An instance of PolicySpec.
    """
    loader.construct_mapping(node)
    return PolicySpec()


def customized_environment_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> CustomizedGrid2OpEnvironment:
    """
    Custom constructor for CustomizedGrid2OpEnvironment.

    Args:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader.
        node (MappingNode): The YAML node.

    Returns:
        CustomizedGrid2OpEnvironment: An instance of CustomizedGrid2OpEnvironment.

    """
    fields = {str(k): v for k, v in loader.construct_mapping(node).items()}
    env_config = fields.get("env_config", {})  # Extract env_config explicitly
    fields["env_config"] = env_config
    return CustomizedGrid2OpEnvironment(**fields)


def loss_reward_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> LossReward:
    """
    Custom constructor for LossReward.

    Args:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader.
        node (MappingNode): The YAML node.

    Returns:
        LossReward: An instance of the LossReward class.
    """
    return LossReward()


def scaled_reward_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> LossReward:
    """
    Custom constructor for ScaledL2RPNReward.

    This function is used as a custom constructor for creating an instance of ScaledL2RPNReward.
    It takes a loader and a node as input arguments and returns an instance of LossReward.

    Parameters:
    - loader: The loader object used for loading the YAML file.
    - node: The mapping node representing the YAML node.

    Returns:
    - An instance of LossReward representing the ScaledL2RPNReward.

    """
    return ScaledL2RPNReward()


def policy_mapping_fn_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> Callable[[str, EpisodeV2, RolloutWorker], str]:
    """
    Custom constructor for policy_mapping_fn.

    Args:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader.
        node (MappingNode): The YAML node.

    Returns:
        Callable[[str, EpisodeV2, RolloutWorker], str]: The constructed policy_mapping_fn.
    """
    return policy_mapping_fn


def custom_metrics_callback_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> DefaultCallbacks:
    """
    Custom constructor for CustomMetricsCallback.

    This function is used as a constructor for the CustomMetricsCallback class.
    It takes a loader and a node as input arguments and returns an instance of the CustomMetricsCallback class.

    Parameters:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader.
        node (MappingNode): The YAML node.

    Returns:
        DefaultCallbacks: An instance of the CustomMetricsCallback class.
    """
    return CustomMetricsCallback


def custom_ppo_metrics_callback_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> DefaultCallbacks:
    """
    Custom constructor for CustomPPOMetricsCallback.

    This function is used as a constructor for the CustomPPOMetricsCallback class.
    It takes a loader and a node as input parameters and returns an instance of the
    CustomPPOMetricsCallback class.

    Parameters:
    - loader: A loader object that is used to load the YAML file.
    - node: A MappingNode object that represents the YAML node.

    Returns:
    - An instance of the CustomPPOMetricsCallback class.

    """
    return CustomPPOMetricsCallback


def custom_single_metrics_callback_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> DefaultCallbacks:
    """
    Custom constructor for CustomPPOMetricsCallback.

    This function is used as a constructor for the CustomPPOMetricsCallback class.
    It takes a loader and a node as input arguments and returns an instance of the DefaultCallbacks class.

    Parameters:
    - loader: A loader object used for loading YAML files.
    - node: A mapping node representing the YAML node.

    Returns:
    - An instance of the DefaultCallbacks class.

    """
    return SingleAgentCallback


def select_agent_policy_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> SelectAgentPolicy:
    """Custom constructor for SelectAgentPolicy.

    Args:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader.
        node (MappingNode): The YAML node.

    Returns:
        SelectAgentPolicy: The constructed SelectAgentPolicy object.
    """
    fields = {str(k): v for k, v in loader.construct_mapping(node).items()}
    return SelectAgentPolicy(**fields)


def do_nothing_policy_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> DoNothingPolicy:
    """Custom constructor for DoNothingPolicy.

    Args:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader.
        node (MappingNode): The YAML node.

    Returns:
        DoNothingPolicy: The constructed DoNothingPolicy object.
    """
    fields = {str(k): v for k, v in loader.construct_mapping(node).items()}
    return DoNothingPolicy(**fields)


def float_to_integer(float_value: float) -> Union[int, float]:
    """
    Turns a float into an int if appropriate. Otherwise keep it as a float.

    Parameters:
        float_value (float): The input float value.

    Returns:
        Union[int, float]: The converted value. If the float is an integer, it is converted to an int.
                           Otherwise, it is returned as a float.
    """
    if float_value.is_integer():
        return int(float_value)
    return float_value


def tune_search_quniform_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> Any:
    """
    Constructor for tune quantified uniform float sampling

    This function constructs a tune quantified uniform float sampling using the given loader and node.
    It takes a loader and a node as input parameters and returns a tune quantified uniform float sampling.

    Parameters:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The loader object used to load the YAML file.
        node (MappingNode): The node object representing the YAML mapping node.

    Returns:
        Any: The tune quantified uniform float sampling.
    """
    vals = []
    for scalar_node in node.value:
        val = float_to_integer(float(scalar_node.value))
        vals.append(val)
    return tune.quniform(vals[0], vals[1], vals[2])


def tune_search_uniform_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> Any:
    """
    Constructor for tune uniform float sampling

    This function constructs a uniform float sampling for the tune library.
    It takes a loader and a node as input parameters and returns a uniform sampling object.

    Parameters:
    - loader: A loader object used for loading the YAML file.
    - node: A mapping node representing the YAML node.

    Returns:
    - A uniform sampling object for the tune library.

    """
    vals = []
    for scalar_node in node.value:
        val = float_to_integer(float(scalar_node.value))
        vals.append(val)
    return tune.uniform(vals[0], vals[1])


def tune_search_loguniform_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> Any:
    """
    Constructor for tune loguniform float sampling

    This function is used as a constructor for tuning loguniform float sampling in the YAML configuration file.
    It takes a loader object and a mapping node as input and returns a loguniform distribution for tuning.

    Parameters:
    - loader: The loader object used to load the YAML configuration file.
    - node: The mapping node containing the values for tuning.

    Returns:
    - A loguniform distribution for tuning.

    """
    vals = []
    for scalar_node in node.value:
        val = float_to_integer(float(scalar_node.value))
        vals.append(val)
    return tune.loguniform(vals[0], vals[1])


def tune_search_grid_search_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> Any:
    """
    Constructor for tune grid search.

    This function constructs a grid search object for the Tune library.
    It takes a YAML node and converts its values into a list of values
    to be used for grid search.

    Parameters:
    - loader: The YAML loader object.
    - node: The YAML node containing the values for grid search.

    Returns:
    - A grid search object for the Tune library.
    """
    vals = []
    for scalar_node in node.value:
        # check if val is a float
        value = scalar_node.value
        if isinstance(value, str):
            value = float_to_integer(float(value))
        vals.append(value)
    return tune.grid_search(vals)


def tune_choice_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> Any:
    """
    Constructor for tune grid search.

    This function constructs a tune grid search by converting the values specified in the YAML file
    into a list of choices. The values can be integers, booleans, or floats.

    Parameters:
    - loader: The YAML loader object.
    - node: The YAML mapping node.

    Returns:
    - A tune choice object containing the converted values.

    """
    vals = []
    for scalar_node in node.value:
        val: Union[int, bool, float]
        if scalar_node.value == "True":
            val = True
        elif scalar_node.value == "False":
            val = False
        else:
            val = float_to_integer(float(scalar_node.value))
        vals.append(val)
    return tune.choice(vals)


def randint_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> Integer:
    """
    Constructor for randint

    This function takes a loader and a node as input and constructs a random integer
    using the values specified in the node. The node should be a MappingNode containing
    two scalar nodes representing the minimum and maximum values for the random integer.

    Parameters:
    - loader: The loader object used to load the YAML file.
    - node: The MappingNode containing the minimum and maximum values.

    Returns:
    - Integer: A random integer between the minimum and maximum values.

    """
    vals = []
    for scalar_node in node.value:
        val = float_to_integer(float(scalar_node.value))
        vals.append(val)
    return randint(vals[0], vals[1])


def powerline_action_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> BaseAction:
    """
    Custom constructor for PowerlineSetAction.

    This function is used as a custom constructor for creating instances of the PowerlineSetAction class.
    It takes a YAML loader and a YAML node as input and returns an instance of the PowerlineSetAction class.

    Parameters:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader used to load the YAML file.
        node (MappingNode): The YAML node representing the PowerlineSetAction object.

    Returns:
        BaseAction: An instance of the PowerlineSetAction class.

    """
    return PowerlineSetAction


def randomline_opponent_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> BaseOpponent:
    """
    Custom constructor for RandomLineOpponent.

    Args:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader.
        node (MappingNode): The YAML node.

    Returns:
        BaseOpponent: An instance of RandomLineOpponent.
    """
    return RandomLineOpponent


def baseaction_budget_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> BaseActionBudget:
    """
    Custom constructor for BaseActionBudget.

    Args:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader.
        node (MappingNode): The YAML node.

    Returns:
        BaseActionBudget: An instance of the BaseActionBudget class.
    """
    return BaseActionBudget


def reconnecting_opponent_constructor(
    loader: Union[Loader, FullLoader, UnsafeLoader], node: MappingNode
) -> OpponentSpace:
    """
    Custom constructor for ReconnectingOpponentSpace.

    This function is used as a custom constructor for the ReconnectingOpponentSpace class.
    It takes a loader and a node as input arguments and returns an instance of the OpponentSpace class.

    Parameters:
        loader (Union[Loader, FullLoader, UnsafeLoader]): The YAML loader used to load the YAML file.
        node (MappingNode): The YAML node representing the ReconnectingOpponentSpace object.

    Returns:
        OpponentSpace: An instance of the OpponentSpace class.

    """
    return ReconnectingOpponentSpace


def add_constructors() -> None:
    """
    Add the constructors to the yaml loader.

    This function adds various constructors to the yaml loader for parsing custom objects
    and types when loading YAML files.

    The constructors added include:
    - CustomizedGrid2OpEnvironment
    - LossReward
    - ScaledL2RPNReward
    - policy_mapping_fn
    - CustomMetricsCallback
    - CustomPPOMetricsCallback
    - SingleAgentCallback
    - SelectAgentPolicy
    - DoNothingPolicy
    - Discrete
    - AlgorithmConfig
    - PolicySpec
    - quniform
    - uniform
    - loguniform
    - grid_search
    - choice
    - randint
    - PowerlineSetAction
    - RandomLineOpponent
    - BaseActionBudget
    - ReconnectingOpponentSpace
    """
    yaml.FullLoader.add_constructor(
        "!CustomizedGrid2OpEnvironment", customized_environment_constructor
    )
    yaml.FullLoader.add_constructor("!LossReward", loss_reward_constructor)
    yaml.FullLoader.add_constructor("!ScaledL2RPNReward", scaled_reward_constructor)
    yaml.FullLoader.add_constructor("!policy_mapping_fn", policy_mapping_fn_constructor)
    yaml.FullLoader.add_constructor(
        "!CustomMetricsCallback", custom_metrics_callback_constructor
    )
    yaml.FullLoader.add_constructor(
        "!CustomPPOMetricsCallback", custom_ppo_metrics_callback_constructor
    )
    yaml.FullLoader.add_constructor(
        "!SingleAgentCallback", custom_single_metrics_callback_constructor
    )
    yaml.FullLoader.add_constructor(
        "!SelectAgentPolicy", select_agent_policy_constructor
    )
    yaml.FullLoader.add_constructor("!DoNothingPolicy", do_nothing_policy_constructor)
    yaml.FullLoader.add_constructor("!Discrete", discrete_constructor)
    yaml.FullLoader.add_constructor("!AlgorithmConfig", algorithm_config_constructor)
    yaml.FullLoader.add_constructor("!PolicySpec", policy_spec_constructor)
    yaml.FullLoader.add_constructor("!quniform", tune_search_quniform_constructor)
    yaml.FullLoader.add_constructor("!uniform", tune_search_uniform_constructor)
    yaml.FullLoader.add_constructor("!loguniform", tune_search_loguniform_constructor)
    yaml.FullLoader.add_constructor("!grid_search", tune_search_grid_search_constructor)
    yaml.FullLoader.add_constructor("!choice", tune_choice_constructor)
    yaml.FullLoader.add_constructor("!randint", randint_constructor)
    yaml.FullLoader.add_constructor("!PowerlineSetAction", powerline_action_constructor)
    yaml.FullLoader.add_constructor(
        "!RandomLineOpponent", randomline_opponent_constructor
    )
    yaml.FullLoader.add_constructor("!BaseActionBudget", baseaction_budget_constructor)
    yaml.FullLoader.add_constructor(
        "!ReconnectingOpponentSpace", reconnecting_opponent_constructor
    )


def load_config(path: str) -> Any:
    """Adds constructors and returns config."""
    add_constructors()

    with open(path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

"""
Utils for evaluation of agents.
"""
from typing import Any
import importlib
from grid2op.Reward import BaseReward


def instantiate_reward_class(class_name: str) -> Any:
    """
    Instantiates the Reward class from json string.
    """
    # Split the class name into module and class
    package_name = __name__.split(".", 1)[0]
    module_name, class_name = class_name.rsplit(".", 1)
    module_name = f"{package_name}." + module_name.split(".", 1)[1]
    class_name = class_name.split(" ", 1)[0]
    # Import the module dynamically
    module = importlib.import_module(module_name)
    # Get the class from the module
    reward_class: BaseReward = getattr(module, class_name)
    # Instantiate the class
    if reward_class:
        return reward_class()
    raise ValueError("Problem instantiating reward class for evaluation.")
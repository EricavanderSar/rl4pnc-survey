from typing import Any, Dict, List, Optional, Tuple, TypeVar
import numpy as np
import os
import gymnasium as gym
import grid2op
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from rl4pnc.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from rl4pnc.grid2op_env.observation_converter import ObsConverter
from rl4pnc.evaluation.evaluation_agents import get_actions_per_substation


class MultiAgentG2OpEnv(CustomizedGrid2OpEnvironment):
    """
    High level decision: act or do nothing
    Mid level decision: where to act
    Low level decision: action to execute
    """
    def __init__(self, env_config: dict[str, Any]):
        super().__init__(env_config)

    def define_action_space(self, env_config: dict) -> gym.Space:
        # Define action space for each agent in the multi-agent environment

        # Actions per low-level agent
        actions_per_substations = get_actions_per_substation(
            possible_substation_actions=self.possible_substation_actions,
        )



        pass

    def define_obs_space(self, env_config: dict) -> gym.Space:
        # Define observation space for each agent in the multi-agent environment
        pass
"""
DEPRICATED/UNUSED. 
Class that defines the custom Grid2op to gym environment with the set observation and action spaces.
"""

import os
from typing import Any, TypeVar

import gymnasium as gym
from ray.tune.registry import register_env

from mahrl.grid2op_env.custom_environment import ReconnectingGymEnv
from mahrl.grid2op_env.utils import (
    CustomDiscreteActions,
    load_action_space,
    make_g2op_env,
    rescale_observation_space,
    setup_converter,
)

RENDERFRAME = TypeVar("RENDERFRAME")


class SingleAgentGrid2OpEnvironment(gym.Env):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        # create the grid2op environment
        self.grid2op_env = make_g2op_env(env_config)

        # create the gym environment
        if env_config["shuffle_scenarios"]:
            self.env_gym = ReconnectingGymEnv(self.grid2op_env, shuffle_chronics=True)
        else:  # ensure the evaluation chronics are not shuffled
            self.env_gym = ReconnectingGymEnv(self.grid2op_env, shuffle_chronics=False)

        # setting up custom action space
        path = os.path.join(
            env_config["lib_dir"],
            f"data/action_spaces/{env_config['env_name']}/{env_config['action_space']}.json",
        )
        self.possible_substation_actions = load_action_space(path, self.grid2op_env)

        # insert do-nothing action at index 0
        do_nothing_action = self.grid2op_env.action_space({})
        self.possible_substation_actions.insert(0, do_nothing_action)

        # create converter
        converter = setup_converter(self.grid2op_env, self.possible_substation_actions)

        # set gym action space to discrete
        self.env_gym.action_space = CustomDiscreteActions(converter)

        # specific to rllib
        self.action_space = gym.spaces.Discrete(len(self.possible_substation_actions))

        # customize observation space
        self.env_gym.observation_space = self.env_gym.observation_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect", "p_or", "p_ex", "timestep_overflow"]
        )

        # rescale observation space
        self.env_gym.observation_space = rescale_observation_space(
            self.env_gym.observation_space, self.grid2op_env, env_config
        )

        # specific to rllib
        self.observation_space = gym.spaces.Dict(
            dict(self.env_gym.observation_space.spaces.items())
        )

        # setup shared parameters
        self.rho_threshold = env_config["rho_threshold"]
        self.steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:  # type: ignore
        """
        This function resets the environment.
        """
        done = True
        while done:
            obs = self.env_gym.reset()

            if obs is not None:
                obs = obs[0]  # remove timeseries ID
            else:
                raise ValueError("Observation is None.")

            # find first step that surpasses threshold
            done = False
            self.steps = 0
            while (max(obs["rho"]) < self.rho_threshold) and (not done):
                obs, _, done, _, _ = self.env_gym.step(0)
                self.steps += 1

        return obs, {}

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any] | None, float, bool, bool, dict[str, Any]]:
        """
        This function performs a single step in the environment.
        """
        cum_reward: float = 0.0
        obs: dict[str, Any]
        # obs, reward, done, truncated, info = self.env_gym.step(action)
        obs, reward, done, truncated, info = self.env_gym.step(action)
        self.steps += 1
        cum_reward += reward
        while (max(obs["rho"]) < self.rho_threshold) and (not done):
            # obs, reward, done, truncated, _ = self.env_gym.step(0)
            obs, reward, done, truncated, _ = self.env_gym.step(0)
            # obs, reward, done, _ = self.env_gym.step(self.do_nothing_actions[0])
            self.steps += 1
            cum_reward += reward

        if done:
            info["steps"] = self.steps
        return obs, cum_reward, done, truncated, info

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError


register_env("SingleAgentGrid2OpEnvironment", SingleAgentGrid2OpEnvironment)

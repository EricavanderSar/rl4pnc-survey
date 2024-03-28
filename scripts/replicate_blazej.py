"""
Aims to replicate Blazej's single-agent PPO one-on-one.
"""

# import packages
import json
import os
import random
from typing import Any

import grid2op
import gymnasium as gym
import numpy as np
import ray
import torch
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.dtypes import dt_float
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import GymEnv, ScalerAttrConverter
from grid2op.Parameters import Parameters
from grid2op.Reward import L2RPNReward
from gymnasium.spaces import Discrete
from lightsim2grid import LightSimBackend  # pylint: disable=wrong-import-order
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune import CLIReporter

from mahrl.models.mlp import SimpleMlp


class SingleAgentCallback(DefaultCallbacks):
    """Implements custom callbacks metric for single agent."""

    def on_episode_end(
        self,
        *,
        episode: Any,
        worker: Any = None,
        base_env: BaseEnv = None,
        policies: Any = None,
        env_index: Any = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Callback function called at the end of each episode.

        Args:
            episode (Episode): The episode object containing information about the episode.
            worker (RolloutWorker): The worker object responsible for executing the episode.
            base_env (Environment): The base environment used for the episode.
            policies (Dict[str, Policy]): A dictionary of policies used for the episode.
            env_index (int): The index of the environment in the multi-environment setup.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # Make sure this episode is really done.
        episode.custom_metrics["num_env_steps"] = episode.last_info_for()["steps"]
        # print("num_env_steps:", episode.custom_metrics["num_env_steps"])


class CustomDiscreteActions(Discrete):
    """
    Class that customizes the action space.

    Example usage:

    import grid2op
    from grid2op.Converter import IdToAct

    env = grid2op.make("rte_case14_realistic")

    all_actions = # a list of of desired actions
    converter = IdToAct(env.action_space)
    converter.init_converter(all_actions=all_actions)


    env.action_space = ChooseDiscreteActions(converter=converter)


    """

    def __init__(self, converter: Any):
        """init"""
        self.converter = converter
        Discrete.__init__(self, n=converter.n)

    def from_gym(self, gym_action: int) -> BaseAction:
        """from_gym"""
        return self.converter.convert_act(gym_action)

    def close(self) -> None:
        """close"""


class ScaledL2RPNReward(L2RPNReward):
    """
    Scaled version of L2RPNReward such that the reward falls between 0 and 1.
    Additionally -0.5 is awarded for illegal actions.
    """

    def __init__(self, logger: Any = None):
        super().__init__()
        self.reward_min: float = -0.5
        self.reward_illegal: float = -0.5
        self.reward_max: float = 1.0
        self.num_lines: int

    def initialize(self, env: BaseEnv) -> None:
        """
        inits
        """
        self.num_lines = env.backend.n_line

    def __call__(
        self,
        action: BaseAction,
        env: BaseEnv,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Calculate the reward based on the given parameters.

        Parameters:
        - action: The action taken in the environment.
        - env: The environment object.
        - has_error: Flag indicating if there was an error in the environment.
        - is_done: Flag indicating if the episode is done.
        - is_illegal: Flag indicating if the action is illegal.
        - is_ambiguous: Flag indicating if the action is ambiguous.

        Returns:
        - res: The calculated reward.
        """
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            res = np.sum(line_cap) / self.num_lines
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        # print(f"\t env.backend.get_line_flow(): {env.backend.get_line_flow()}")
        return res

    @staticmethod
    def __get_lines_capacity_usage(env: BaseEnv) -> Any:
        """
        Calculate the lines capacity usage score for the given environment.

        Parameters:
        - env: The environment object.

        Returns:
        - lines_capacity_usage_score: The lines capacity usage score as a numpy array.
        """
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        min_flow = np.minimum(relative_flow, dt_float(1.0))
        lines_capacity_usage_score = np.maximum(
            dt_float(1.0) - min_flow**2, np.zeros(min_flow.shape, dtype=dt_float)
        )
        return lines_capacity_usage_score


# create custom environment from scratch
class CustomGymEnv(GymEnv):
    """Implements a grid2op env in gym."""

    def __init__(self, env: BaseEnv):
        super().__init__(env)
        self.idx = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """resets"""
        g2op_obs = self.init_env.reset()
        return self.observation_space.to_gym(g2op_obs)

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """steps"""
        g2op_act = self.action_space.from_gym(action)

        g2op_obs, reward, done, info = self.init_env.step(g2op_act)

        gym_obs = self.observation_space.to_gym(g2op_obs)
        return gym_obs, reward, done, info


class GridGym(gym.Env):
    """
    loads the grid2op env into gym
    """

    def __init__(self, env_config: dict[str, Any]):
        # super().__init__(env_config)
        # create gym env
        self.grid2op_env = grid2op.make(
            "rte_case14_realistic",
            reward_class=ScaledL2RPNReward,
            backend=LightSimBackend(),
        )

        thermal_limits = [
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            1000,
            760,
            450,
            760,
            380,
            380,
            760,
            380,
            760,
            380,
            380,
            380,
            2000,
            2000,
        ]
        self.grid2op_env.set_thermal_limit(thermal_limits)

        self.env_gym = CustomGymEnv(self.grid2op_env)

        # define observations
        self.env_gym.observation_space = self.env_gym.observation_space.keep_only_attr(
            [
                "rho",
                "gen_p",
                "load_p",
                "topo_vect",
                "p_or",
                "p_ex",
                "timestep_overflow",
                # "maintenance",  # NOTE: Maintenance is added
            ]
        )

        # scale observations
        self.env_gym.observation_space = self.env_gym.observation_space.reencode_space(
            "gen_p",
            ScalerAttrConverter(substract=0.0, divide=self.grid2op_env.gen_pmax),
        )
        self.env_gym.observation_space = self.env_gym.observation_space.reencode_space(
            "timestep_overflow",
            ScalerAttrConverter(
                substract=0.0,
                divide=Parameters().NB_TIMESTEP_OVERFLOW_ALLOWED,  # assuming no custom params
            ),
        )

        for attr in ["p_ex", "p_or", "load_p"]:
            underestimation_constant = (
                1.2  # constant to account that our max/min are underestimated
            )
            max_arr, min_arr = np.load(
                os.path.join(
                    "/Users/barberademol/Documents/GitHub/mahrl_grid2op/",
                    "data/scaling_arrays",
                    "rte_case14_realistic",
                    f"{attr}.npy",
                )
            )

            self.env_gym.observation_space = (
                self.env_gym.observation_space.reencode_space(
                    attr,
                    ScalerAttrConverter(
                        substract=underestimation_constant * min_arr,
                        divide=underestimation_constant * (max_arr - min_arr),
                    ),
                )
            )

        self.observation_space = gym.spaces.Dict(
            dict(self.env_gym.observation_space.items())
        )

        # define actions from medha
        path = os.path.join(
            "/Users/barberademol/Documents/GitHub/mahrl_grid2op/",
            "data",
            "action_spaces",
            "rte_case14_realistic",
            "medha.json",
        )
        with open(path, "rt", encoding="utf-8") as action_set_file:
            self.all_actions = list(
                (
                    self.grid2op_env.action_space(action_dict)
                    for action_dict in json.load(action_set_file)
                )
            )

        # add do nothing action
        do_nothing_action = self.grid2op_env.action_space({})
        self.all_actions.insert(0, do_nothing_action)

        converter = IdToAct(
            self.grid2op_env.action_space
        )  # initialize with regular the environment of the regular action space
        converter.init_converter(all_actions=self.all_actions)

        self.env_gym.action_space = CustomDiscreteActions(converter=converter)
        self.action_space = gym.spaces.Discrete(self.env_gym.action_space.n)

        # set parameters
        self.steps = 0
        self.rho_threshold = env_config["rho_threshold"]

    def reset(
        self, *, seed: int | None = None, options: None | dict[str, Any] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Resets the environment and returns the initial observation.

        Args:
            seed (int, optional): Random seed for the environment. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the initial observation and an empty dictionary.
        """
        obs = self.env_gym.reset()

        # additional loop for if it completes it immediately # NOTE: Latest addition
        done = True
        while done:
            # find first step that surpasses threshold
            done = False
            self.steps = 0

            while (max(obs["rho"]) < self.rho_threshold) and (not done):
                obs, _, done, _ = self.env_gym.step(0)
                self.steps += 1
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """steps"""
        cum_reward: float = 0.0
        obs, reward, done, info = self.env_gym.step(action)
        self.steps += 1
        cum_reward += reward
        while (max(obs["rho"]) < self.rho_threshold) and (not done):
            obs, reward, done, _ = self.env_gym.step(0)
            self.steps += 1
            cum_reward += reward

        if done:
            info["steps"] = self.steps
        return obs, cum_reward, done, False, info

    def render(self) -> None:
        """renders"""
        raise NotImplementedError("render not implemented")


if __name__ == "__main__":
    # set seeds # NOTE env_seed now always same to check
    random.seed(2137)
    np.random.seed(2137)
    torch.manual_seed(2137)

    env_seed = random.choice(
        [
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
        ]
    )

    # print(f"env_seed: {env_seed}")
    # create custom environment
    # grid2op_env = grid2op.make("rte_case14_realistic")
    # custom_env = CustomEnv(grid2op_env)
    ModelCatalog.register_custom_model("fcn", SimpleMlp)

    ray.shutdown()
    ray.init(ignore_reinit_error=False)

    ppo_config = ppo.PPOConfig().to_dict()

    # config from paper
    custom_config = {
        "training": {
            "lr": 0.0001,
            "kl_coeff": 0.2,
            "clip_param": 0.3,
            "sgd_minibatch_size": 256,
            "train_batch_size": 1024,
            "num_sgd_iter": 5,
            "entropy_coeff": 0,
            "_enable_learner_api": False,
        },
        "framework": {"framework": "torch"},
        "rl_module": {
            "_enable_rl_module_api": False,
            "model": {
                "fcnet_hiddens": [256, 256, 256],
                "fcnet_activation": "relu",
                "custom_model": "fcn",
                "custom_model_config": {
                    "use_parametric": False,
                    "env_obs_name": "grid",
                },
            },
        },
        "environment": {
            "env_config": {
                "env_name": "rte_case14_realistic_blazej_train",
                "rho_threshold": 0.95,
            }
        },
        "callbacks": {"callbacks": SingleAgentCallback},
        "evaluation": {
            "evaluation_interval": 2,
            "evaluation_num_episodes": 200,
            "evaluation_config": {
                "env": GridGym,
                "env_config": {
                    "env_name": "rte_case14_realistic_blazej_val",
                    "rho_threshold": 0.95,
                    "explore": True,
                },
            },
        },
    }

    # grid search
    # custom_config = {
    #     "debugging": {"seed": env_seed},
    #     "training": {
    #         "lr": tune.grid_search([0.001, 0.01]),
    #         "kl_coeff": 0.2,
    #         "clip_param": 0.3,
    #         "rollout_fragment_length": 128,
    #         "sgd_minibatch_size": tune.grid_search([128, 256]),
    #         "train_batch_size": tune.grid_search([1024, 2048]),
    #         "num_sgd_iter": 5,
    #         "entropy_coeff": 0,
    #         "evaluation_interval": 2,
    #         "evaluation_num_episodes": 200,
    #         "_enable_learner_api": False,
    #     },
    #     "framework": {"framework": "torch"},
    #     "rl_module": {
    #         "_enable_rl_module_api": False,
    #         "model": {
    #             "fcnet_hiddens": [256, 256, 256],
    #             "fcnet_activation": "relu",
    #             "custom_model": "fcn",
    #             "custom_model_config": {
    #                 "use_parametric": False,
    #                 "env_obs_name": "grid",
    #             },
    #         },
    #     },
    #     "environment": {
    #         "env_config": {
    #             "env_name": "rte_case14_realistic_blazej_train",
    #             "rho_threshold": 0.95,
    #         }
    #     },
    #     "callbacks": {"callbacks": SingleAgentCallback},
    #     "evaluation": {
    #         "evaluation_config": AlgorithmConfig.overrides(explore=False),
    #         "evaluation_interval": 2,
    #         "evaluation_num_episodes": 200,
    #         "evaluation_config": {
    #             "env": Grid_Gym,
    #             "env_config": {
    #                 "env_name": "rte_case14_realistic_blazej_val",
    #                 "rho_threshold": 0.95,
    #             },
    #         },
    #     },
    # }

    # config from .yaml
    # custom_config = {
    #     "training": {
    #         "lr": 0.0001,
    #         "kl_coeff": 0.2,
    #         "lambda": 0.95,
    #         "vf_loss_coeff": 0.9,
    #         "vf_clip_param": 900,
    #         "rollout_fragment_length": 128,
    #         "sgd_minibatch_size": 256,
    #         "train_batch_size": 1024,
    #         "num_sgd_iter": 15,
    #         "entropy_coeff": 0.01,
    #         "evaluation_interval": 2,
    #         "evaluation_num_episodes": 100,
    #         "_enable_learner_api": False,
    #     },
    #     "framework": {"framework": "torch"},
    #     "rl_module": {
    #         "_enable_rl_module_api": False,
    #         "model": {
    #             "fcnet_hiddens": [256, 256, 256],
    #             "fcnet_activation": "relu",
    #             "custom_model": "fcn",
    #             "custom_model_config": {
    #                 "use_parametric": False,
    #                 "env_obs_name": "grid",
    #             },
    #         },
    #     },
    #     "environment": {
    #         "env_config": {
    #             "env_name": "rte_case14_realistic_blazej_train",
    #             "rho_threshold": 0.95,
    #         }
    #     },
    #     "callbacks": {"callbacks": SingleAgentCallback},
    #     "evaluation": {
    #         "evaluation_config": AlgorithmConfig.overrides(explore=False),
    #         "evaluation_interval": 2,
    #         "evaluation_num_episodes": 200,
    #         "evaluation_config": {
    #             "env": Grid_Gym,
    #             "env_config": {
    #                 "env_name": "rte_case14_realistic_blazej_val",
    #                 "rho_threshold": 0.95,
    #             },
    #         },
    #     },
    # }

    ppo_config.update(custom_config["training"])
    ppo_config.update(custom_config["framework"])
    ppo_config.update(custom_config["rl_module"])
    ppo_config.update(custom_config["environment"])
    ppo_config.update(custom_config["callbacks"])
    ppo_config.update({"env": GridGym})

    # tune
    PPOTrainer = ppo.PPO  # NOTE: Blazej used PPOTrainer
    reporter = CLIReporter()

    analysis = tune.run(
        PPOTrainer,
        progress_reporter=reporter,
        config=ppo_config,
        stop={"timesteps_total": 100000},
        local_dir="/Users/barberademol/Documents/GitHub/mahrl_grid2op/runs/",
        checkpoint_freq=10,
        checkpoint_at_end=True,
        num_samples=2,
        keep_checkpoints_num=5,
        checkpoint_score_attr="evaluation/episode_reward_mean",
        verbose=1,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=os.path.join(
                    "/Users/barberademol/Documents/GitHub/mahrl_grid2op/runs/", "mlruns"
                ),
                experiment_name="blazej_file/grid_search",
                save_artifact=True,
            )
        ],
    )
    ray.shutdown()

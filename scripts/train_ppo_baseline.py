"""
Trains PPO baseline agent.
"""
from typing import Any, TypeVar

import grid2op
import gymnasium
import numpy as np
import ray
from grid2op import Reward
from grid2op.gym_compat import GymEnv
from ray import air, tune
from ray.rllib.algorithms import ppo  # import the type of agents

from mahrl.grid2op_env.utils import (
    CustomDiscreteActions,
    get_possible_topologies,
    setup_converter,
)

ENV_NAME = "rte_case5_example"
ENV_IS_TEST = True
LIB_DIR = "/Users/barberademol/Documents/GitHub/mahrl_grid2op/"
# LIB_DIR = "/home/daddabarba/VirtualEnvs/mahrl/lib/python3.10/site-packages/grid2op/data"
RHO_THRESHOLD = 0.95
CHANGEABLE_SUBSTATIONS = [0, 2, 3]
NB_TSTEPS = 100000
CHECKPOINT_FREQ = 1000
VERBOSE = 1

OBSTYPE = TypeVar("OBSTYPE")
ACTTYPE = TypeVar("ACTTYPE")
RENDERFRAME = TypeVar("RENDERFRAME")


class CustomizedGrid2OpEnvironment(gymnasium.Env):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        # 1. create the grid2op environment
        if "env_name" not in env_config:
            raise RuntimeError(
                "The configuration for RLLIB should provide the env name"
            )
        nm_env = env_config.pop("env_name", None)
        self.env_glop = grid2op.make(nm_env, **env_config["grid2op_kwargs"])

        # 1.a. Setting up custom action space
        possible_substation_actions = get_possible_topologies(
            self.env_glop, CHANGEABLE_SUBSTATIONS
        )

        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_glop)
        self.env_gym.reset()

        # 3. customize action and observation space space to only change bus
        # create converter
        converter = setup_converter(self.env_glop, possible_substation_actions)

        # set gym action space to discrete
        self.env_gym.action_space = CustomDiscreteActions(
            converter, self.env_glop.action_space()
        )

        # customize observation space
        ob_space = self.env_gym.observation_space
        ob_space = ob_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect", "p_or", "p_ex", "timestep_overflow"]
        )

        self.env_gym.observation_space = ob_space

        # 4. specific to rllib
        self.action_space = gymnasium.spaces.Discrete(len(possible_substation_actions))
        self.observation_space = gymnasium.spaces.Dict(
            dict(self.env_gym.observation_space.spaces.items())
        )

        self.last_rho = 0  # below threshold TODO

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[OBSTYPE, dict[str, Any]]:  # type: ignore
        """
        This function resets the environment.
        """
        obs, info = self.env_gym.reset()
        self.last_rho = np.max(obs["rho"])
        return obs, info

    def step(self, action: int) -> tuple[OBSTYPE, float, bool, bool, dict[str, Any]]:
        """
        This function performs a single step in the environment.
        """
        # for the first action or whenever the lines are not near overloading, do nothing
        if self.last_rho < RHO_THRESHOLD:
            action = -1

        obs, reward, done, truncated, info = self.env_gym.step(action)
        self.last_rho = np.max(obs["rho"])
        return obs, reward, done, truncated, info

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError


def run_training(config: dict[str, Any]) -> None:
    """
    Function that runs the training script.
    """
    # Create tuner
    tuner = tune.Tuner(
        ppo.PPO,
        param_space=config,
        run_config=air.RunConfig(
            stop={"timesteps_total": NB_TSTEPS},
            storage_path="/Users/barberademol/Documents/GitHub/mahrl_grid2op/runs/",
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=CHECKPOINT_FREQ,
                checkpoint_at_end=True,
                checkpoint_score_attribute="evaluation/episode_reward_mean",
            ),
            verbose=VERBOSE,
        ),
    )

    # Launch tuning
    try:
        tuner.fit()
    finally:
        # Close ray instance
        ray.shutdown()


if __name__ == "__main__":
    # make_train_test_val_split(
    #     os.path.join(LIB_DIR, "environments"), ENV_NAME, 5.0, 5.0, Reward.L2RPNReward
    # )
    ppo_config = ppo.PPOConfig()
    ppo_config = ppo_config.training(
        gamma=0.95,
        lr=0.003,
        vf_loss_coeff=0.5,
        entropy_coeff=0.01,
        clip_param=0.2,
        lambda_=0.95,
        # sgd_minibatch_size=4,
        # train_batch_size=32,
        # seed=14,
    )
    ppo_config = ppo_config.environment(
        env=CustomizedGrid2OpEnvironment,
        env_config={
            "env_name": ENV_NAME,
            "grid2op_kwargs": {
                "test": ENV_IS_TEST,
                "reward_class": Reward.L2RPNReward,
            },
        },
    )

    # config = load_config(os.path.join(LIB_DIR, "experiments", "ppo_baseline.yaml"))
    # print(config)
    # run_training(config["tune_config"])
    run_training(ppo_config)

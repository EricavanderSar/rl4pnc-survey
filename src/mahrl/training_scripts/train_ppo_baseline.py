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
from ray import train, tune
from ray.rllib.algorithms import ppo  # import the type of agents

from mahrl.grid2op_env import utils

ENV_NAME = "rte_case5_example"
LIB_DIR = "/Users/barberademol/Documents/GitHub/mahrl_grid2op/venv_mahrl/lib/python3.10/site-packages/grid2op/data/"
NB_STEP_TRAIN = 10
RHO_THRESHOLD = 0.95
CHANGEABLE_SUBSTATIONS = [0, 2, 3]

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
        self.env_glop = grid2op.make(nm_env, **env_config)

        # 1.a. Setting up custom action space
        possible_substation_actions = utils.get_possible_topologies(
            self.env_glop, CHANGEABLE_SUBSTATIONS
        )

        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_glop)
        _, _ = self.env_gym.reset()

        # 3. customize action and observation space space to only change bus
        # create converter
        converter = utils.setup_converter(self.env_glop, possible_substation_actions)

        # set gym action space to discrete
        self.env_gym.action_space = utils.CustomDiscreteActions(
            converter, self.env_glop.action_space()
        )

        # customize observation space
        ob_space = self.env_gym.observation_space
        ob_space = ob_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect", "p_or", "p_ex", "timestep_overflow"]
        )

        self.env_gym.observation_space = ob_space

        # 4. specific to rllib
        # self.action_space = gym.spaces.Discrete(converter.n)
        self.action_space = self.env_gym.action_space
        self.observation_space = self.env_gym.observation_space

        self.last_rho = 0  # below threshold TODO

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[OBSTYPE, dict[str, Any]]:  # type: ignore
        obs, info = self.env_gym.reset()
        self.last_rho = np.max(obs["rho"])
        return obs, info

    def step(self, action: int) -> tuple[OBSTYPE, float, bool, bool, dict[str, Any]]:
        # for the first action or whenever the lines are not near overloading, do nothing
        if self.last_rho < RHO_THRESHOLD:
            action = -1

        obs, reward, done, truncated, info = self.env_gym.step(action)
        self.last_rho = np.max(obs["rho"])
        return obs, reward, done, truncated, info

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        raise NotImplementedError


utils.make_train_test_val_split(LIB_DIR, ENV_NAME, 5.0, 5.0, Reward.L2RPNReward)
env = CustomizedGrid2OpEnvironment({"env_name": LIB_DIR + ENV_NAME + "_train"})
config = ppo.PPOConfig()
config = config.training(
    gamma=0.95,
    lr=0.003,
    vf_loss_coeff=0.5,
    entropy_coeff=0.01,
    clip_param=0.2,
    lambda_=0.95,
    sgd_minibatch_size=4,
    train_batch_size=32,
)
config = config.environment(
    env=CustomizedGrid2OpEnvironment,
    env_config={
        "env_name": LIB_DIR + ENV_NAME + "_train",
        "reward_class": Reward.L2RPNReward,
    },
)

if NB_STEP_TRAIN:
    try:
        analysis = tune.run(
            ppo.PPO,
            config=config.to_dict(),
            stop={"timesteps_total": 10000},
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=1000, checkpoint_at_end=True
            ),
            verbose=1,
            local_dir="/Users/barberademol/Documents/GitHub/mahrl_grid2op/notebooks/results",
        )
    finally:
        # shutdown ray
        ray.shutdown()

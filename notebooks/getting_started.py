# %% [markdown]
# # Importing packages

# %%
import os

import grid2op
import gymnasium as gym
import ray
from grid2op.gym_compat import GymEnv, ScalerAttrConverter, MultiToTupleConverter
from ray.rllib.algorithms import ppo  # import the type of agents
from ray import tune
from typing import Any, OrderedDict


# %% [markdown]
# # Global settings

# %%
ENV_NAME = "rte_case5_example"
LIBRARY_DIRECTORY = "/Users/barberademol/Documents/GitHub/mahrl_grid2op/venv_mahrl/lib/python3.10/site-packages/grid2op/data/"
NB_STEP_TRAIN = 10

# %% [markdown]
# # Only run first time to set-up

# %%
if not os.path.exists(LIBRARY_DIRECTORY + ENV_NAME + "_train"):
    # env = grid2op.make(ENV_NAME, test=True)
    env = grid2op.make(LIBRARY_DIRECTORY + ENV_NAME)

    # extract 10% of the "chronics" to be used in the validation environment, 10% for testing,
    # 80% for training
    nm_env_train, nm_env_val, nm_env_test = env.train_val_split_random(
        pct_val=10.0, pct_test=10.0, add_for_test="test"
    )
    # and now you can use the training set only to train your agent:
    print(f"The name of the training environment is {nm_env_train}")
    print(f"The name of the validation environment is {nm_env_val}")
    print(f"The name of the test environment is {nm_env_test}")


# %% [markdown]
# # Speeding up

# %%
# TODO: The grid2op documentation is full of details to "optimize" the number of steps you can do
# per seconds. This number can rise from a few dozen per seconds to around a thousands per seconds
# with proper care. We strongly encouraged you to leverage all the possibilities which includes
# (but are not limited to):
# - using "lightsim2grid" as a backend for a 10-15x speed up in the "env.step(...)" function
# - using "MultifolderWithCache"/"env.chronics_handler.set_chunk(...)" for faster "env.reset(...)"
#   see https://grid2op.readthedocs.io/en/latest/environment.html#optimize-the-data-pipeline
# - using "SingleEnvMultiProcess" for parrallel computation

# %% [markdown]
# # Define environment


# %%
# MyEnv class, and train a Proximal Policy Optimisation based agent
class MyEnv(gym.Env):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        # 1. create the grid2op environment
        if not "env_name" in env_config:
            raise RuntimeError(
                "The configuration for RLLIB should provide the env name"
            )
        nm_env: str = env_config["env_name"]
        del env_config["env_name"]
        self.env_glop = grid2op.make(nm_env, **env_config)

        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_glop)
        obs_gym = self.env_gym.reset()

        # 3. customize action space to only change bus and set line status (needed with disconnections)
        self.env_gym.action_space = self.env_gym.action_space.ignore_attr(
            "set_bus"
        ).ignore_attr("set_line_status")
        self.env_gym.action_space = self.env_gym.action_space.reencode_space(
            "change_bus", MultiToTupleConverter()
        )
        self.env_gym.action_space = self.env_gym.action_space.reencode_space(
            "change_line_status", MultiToTupleConverter()
        )

        ## customize observation space
        ob_space: dict[str, Any] = self.env_gym.observation_space
        ob_space = ob_space.keep_only_attr(
            [
                "rho",
                "gen_p",
                "load_p",
                "topo_vect",
                "actual_dispatch",
                "p_or",
                "p_ex",
                "timestep_overflow",
            ]
        )
        ob_space = ob_space.reencode_space(
            "actual_dispatch",
            ScalerAttrConverter(substract=0.0, divide=self.env_glop.gen_pmax),
        )
        ob_space = ob_space.reencode_space(
            "gen_p", ScalerAttrConverter(substract=0.0, divide=self.env_glop.gen_pmax)
        )
        ob_space = ob_space.reencode_space(
            "load_p",
            ScalerAttrConverter(
                substract=obs_gym[0]["load_p"], divide=0.5 * obs_gym[0]["load_p"]
            ),
        )
        ob_space = ob_space.reencode_space(
            "p_or", ScalerAttrConverter(substract=0.0, divide=0.5 * obs_gym[0]["p_or"])
        )
        ob_space = ob_space.reencode_space(
            "p_ex", ScalerAttrConverter(substract=0.0, divide=0.5 * obs_gym[0]["p_ex"])
        )

        self.env_gym.observation_space = ob_space

        # 4. specific to rllib
        self.action_space = self.env_gym.action_space
        self.observation_space = self.env_gym.observation_space

        # 4. build the action space and observation space directly from the spaces class.
        d: dict[str, Any] = {
            k: v for k, v in self.env_gym.observation_space.spaces.items()
        }
        self.observation_space = gym.spaces.Dict(d)
        print(self.observation_space)
        a: dict[str, Any] = {k: v for k, v in self.env_gym.action_space.items()}
        self.action_space = gym.spaces.Dict(a)
        print(self.action_space)

    def reset(
        self, seed: int = None, options: dict[str, Any] = None
    ) -> tuple[OrderedDict[str, Any], dict[str, str]]:
        obs: tuple[OrderedDict[str, Any], dict[str, str]] = self.env_gym.reset()
        return obs

    def step(self, action):
        obs: tuple[OrderedDict[str, Any], dict[str, str]]
        obs, reward, done, truncated, info = self.env_gym.step(action)
        return obs, reward, done, truncated, info

    def get_env(self):
        return self.env_glop


env = MyEnv({"env_name": LIBRARY_DIRECTORY + ENV_NAME + "_train"})
env.reset()

# %% [markdown]
# # # Train agent

# %%
config = ppo.PPOConfig()
config = config.training(
    gamma=0.99,
    lr=0.001,
    vf_loss_coeff=0.5,
)
config = config.environment(
    env=MyEnv, env_config={"env_name": LIBRARY_DIRECTORY + ENV_NAME + "_train"}
)
# config = {
#     "env": MyEnv,
#     "env_config": {
#         "env_name": LIBRARY_DIRECTORY + ENV_NAME + "_train"},
#     "framework": "torch",
# }


if NB_STEP_TRAIN:
    try:
        analysis = tune.run(
            ppo.PPO,
            config=config.to_dict(),
            stop={"timesteps_total": 10000},  # Adjust the stopping criterion
            verbose=1,
            local_dir="/Users/barberademol/Documents/GitHub/mahrl_grid2op/notebooks/results",
        )
    finally:
        # shutdown ray
        ray.shutdown()

# # %%
env = MyEnv({"env_name": LIBRARY_DIRECTORY + ENV_NAME + "_train"}).get_env()

print(
    "Total maximum number of timesteps possible: {}".format(
        env.chronics_handler.max_timestep()
    )
)

# # %%
# print(ppo.PPOConfig().to_dict())

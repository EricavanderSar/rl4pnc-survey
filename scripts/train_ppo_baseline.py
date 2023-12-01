"""
Trains PPO baseline agent.
"""
from collections import OrderedDict
from typing import Any, Optional, Tuple, TypeVar

import grid2op
import gymnasium
from gymnasium.spaces import Discrete
import ray
from grid2op import Reward
from grid2op.gym_compat import GymEnv
from ray import air, tune
from ray.rllib.algorithms import ppo  # import the type of agents
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import (
    MultiAgentDict,
)

from mahrl.grid2op_env.utils import (
    CustomDiscreteActions,
    get_possible_topologies,
    setup_converter,
)
from mahrl.multi_agent.policy import (
    DoNothingPolicy,
    SelectAgentPolicy,
    policy_mapping_fn,
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


class CustomizedGrid2OpEnvironment(MultiAgentEnv):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        MultiAgentEnv.__init__(self)

        self._agent_ids = [
            "high_level_policy",
            "reinforcement_learning_policy",
            "do_nothing_policy",
        ]

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

        self.previous_obs = OrderedDict()  # TODO: How to initalize?

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        This function resets the environment.
        """
        self.previous_obs, infos = self.env_gym.reset()
        observations = {"high_level_policy": self.previous_obs}
        return observations, infos

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """
        This function performs a single step in the environment.
        """
        if "high_level_policy" in action_dict.keys():
            # TODO change this to inside agent policy?
            action = action_dict["high_level_policy"]
            if action == "do_nothing_policy":
                # if np.max(self.previous_obs["rho"]) < RHO_THRESHOLD:
                # do nothing
                observations = {"do_nothing_policy": self.previous_obs}
                return observations, {}, {}, {}, {}
            if action == "reinforcement_learning_policy":
                # else:
                # do something
                observations = {"reinforcement_learning_policy": self.previous_obs}
                return observations, {}, {}, {}, {}
            raise NotImplementedError
        if "do_nothing_policy" in action_dict.keys():
            # do nothing
            action = action_dict["reinforcement_learning_policy"]
            (
                self.previous_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_gym.step(action)
            # give reward to RL agent
            rewards = {"reinforcement_learning_policy": reward}
            observations = {"high_level_policy": self.previous_obs}
            terminateds = {"do_nothing_policy": terminated}
            truncateds = {"do_nothing_policy": truncated}
            infos = {"do_nothing_policy": info}
            return observations, rewards, terminateds, truncateds, infos
        if "reinforcement_learning_policy" in action_dict.keys():
            # perform action
            action = action_dict["reinforcement_learning_policy"]
            (
                self.previous_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_gym.step(action)
            # give reward to RL agent
            rewards = {"reinforcement_learning_policy": reward}
            observations = {"high_level_policy": self.previous_obs}
            terminateds = {"reinforcement_learning_policy": terminated}
            truncateds = {"reinforcement_learning_policy": truncated}
            infos = {"reinforcement_learning_policy": info}
            return observations, rewards, terminateds, truncateds, infos
            return observations, rewards, terminateds, truncateds, infos
        # raise NotImplementedError
        observations = {"high_level_policy": self.previous_obs}
        return observations, {}, {}, {}, {}

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

    policies = {
        "high_level_policy": PolicySpec(  # chooses RL or do-nothing agent
            policy_class=SelectAgentPolicy,  # infer automatically from Algorithm --TODO not actually needed
            observation_space=None,  # infer automatically from env --TODO only rho needed
            action_space=Discrete(2),  # choose one of agents
        ),
        "reinforcement_learning_policy": PolicySpec(  # performs RL topology
            policy_class=ppo.PPO,  # use PPO
            observation_space=None,  # infer automatically from env
            action_space=None,  # infer automatically from env
        ),
        "do_nothing_policy": PolicySpec(  # performs do-nothing action
            policy_class=DoNothingPolicy,  # infer automatically from Algorithm --TODO not actually needed
            observation_space=None,  # infer automatically from env --TODO not actually needed
            action_space=Discrete(1),  # only perform do-nothing
        ),
    }
    ppo_config = ppo.PPOConfig()
    ppo_config = ppo_config.training(
        _enable_learner_api=False,
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
            "num_agents": len(policies),
            "grid2op_kwargs": {
                "test": ENV_IS_TEST,
                "reward_class": Reward.L2RPNReward,
            },
        },
    )

    ppo_config = ppo_config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["reinforcement_learning_policy"],
    )

    ppo_config = ppo_config.framework(framework="torch")
    ppo_config = ppo_config.rl_module(_enable_rl_module_api=False)

    run_training(ppo_config)

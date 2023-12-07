"""
Trains PPO baseline agent.
"""
from typing import Any

import ray
from grid2op import Reward
from gymnasium.spaces import Discrete
from ray import air, tune
from ray.rllib.algorithms import ppo  # import the type of agents
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec

from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
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
NB_TSTEPS = 100000
CHECKPOINT_FREQ = 1000
VERBOSE = 1


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
            # policy_class=ppo.PPO,  # infer automatically from Algorithm --TODO not actually needed
            policy_class=SelectAgentPolicy,  # infer automatically from Algorithm --TODO not actually needed
            observation_space=None,  # infer automatically from env --TODO only rho needed
            action_space=Discrete(2),  # choose one of agents
            # action_space=None,  # choose one of agents
            config=(
                AlgorithmConfig()
                .training(_enable_learner_api=False)
                .rl_module(_enable_rl_module_api=False)
                .exploration(
                    exploration_config={
                        "type": "EpsilonGreedy",
                    }
                )
                .rollouts(preprocessor_pref=None)
            ),
        ),
        "reinforcement_learning_policy": PolicySpec(  # performs RL topology
            policy_class=None,  # use default policy PPO
            observation_space=None,  # infer automatically from env
            action_space=None,  # infer automatically from env
            config=(
                AlgorithmConfig()
                .training(_enable_learner_api=False)
                .rl_module(_enable_rl_module_api=False)
                .exploration(
                    exploration_config={
                        "type": "EpsilonGreedy",
                    }
                )
            ),
        ),
        "do_nothing_policy": PolicySpec(  # performs do-nothing action
            # policy_class=ppo.PPO,  # infer automatically from Algorithm --TODO not actually needed
            policy_class=DoNothingPolicy,  # infer automatically from Algorithm --TODO not actually needed
            observation_space=None,  # infer automatically from env --TODO not actually needed
            action_space=Discrete(1),  # only perform do-nothing
            # action_space=None,  # only perform do-nothing
            config=(
                AlgorithmConfig()
                .training(_enable_learner_api=False)
                .rl_module(_enable_rl_module_api=False)
                .exploration(
                    exploration_config={
                        "type": "EpsilonGreedy",
                    }
                )
            ),
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
            # AlgorithmConfig(),
            "env_name": ENV_NAME,
            "num_agents": len(policies),
            "grid2op_kwargs": {
                "test": ENV_IS_TEST,
                "reward_class": Reward.L2RPNReward,
            },
        },
    )

    ppo_config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["reinforcement_learning_policy"],
    )

    ppo_config.framework(framework="torch")
    ppo_config.rl_module(_enable_rl_module_api=False)
    ppo_config.exploration(
        exploration_config={
            "type": "EpsilonGreedy",
        }
    )

    run_training(ppo_config)

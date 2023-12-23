"""
Trains PPO baseline agent.
"""
import os
from typing import Any

import ray
from gymnasium.spaces import Discrete
from ray import air, tune
from ray.rllib.algorithms import ppo  # import the type of agents
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec

from mahrl.experiments.yaml import load_config
from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from mahrl.multi_agent.policy import DoNothingPolicy, SelectAgentPolicy

LIB_DIR = "/Users/barberademol/Documents/GitHub/mahrl_grid2op/"
# LIB_DIR = "/home/daddabarba/VirtualEnvs/mahrl/"


def run_training(config: dict[str, Any]) -> None:
    """
    Function that runs the training script.
    """
    # Create tuner
    tuner = tune.Tuner(
        ppo.PPO,
        param_space=config,
        run_config=air.RunConfig(
            stop={"timesteps_total": config["nb_timesteps"]},
            storage_path="/Users/barberademol/Documents/GitHub/mahrl_grid2op/runs/",
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=config["checkpoint_freq"],
                checkpoint_at_end=True,
                checkpoint_score_attribute="evaluation/episode_reward_mean",
            ),
            verbose=config["verbose"],
        ),
    )

    # Launch tuning
    try:
        tuner.fit()
    finally:
        # Close ray instance
        ray.shutdown()


if __name__ == "__main__":  # load base PPO config and load in hyperparameters
    ppo_config = ppo.PPOConfig().to_dict()
    custom_config = load_config(
        os.path.join(LIB_DIR, "experiments/configurations/ppo_baseline.yaml")
    )
    ppo_config.update(custom_config)

    policies = {
        "high_level_policy": PolicySpec(  # chooses RL or do-nothing agent
            policy_class=SelectAgentPolicy,
            observation_space=None,  # infer automatically from env
            action_space=Discrete(2),  # choose one of agents
            config=(
                AlgorithmConfig()
                .training(
                    _enable_learner_api=False,
                    model={
                        "custom_model_config": {
                            "rho_threshold": custom_config["env_config"][
                                "rho_threshold"
                            ]
                        }
                    },
                )
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
            policy_class=None,  # use default policy of PPO
            observation_space=None,  # infer automatically from env
            action_space=None,  # infer automatically from env
            config=(
                AlgorithmConfig()
                .training(
                    _enable_learner_api=False,
                )
                .rl_module(_enable_rl_module_api=False)
                .exploration(
                    exploration_config={
                        "type": "EpsilonGreedy",
                    }
                )
            ),
        ),
        "do_nothing_policy": PolicySpec(  # performs do-nothing action
            policy_class=DoNothingPolicy,
            observation_space=None,  # infer automatically from env --TODO not actually needed
            action_space=Discrete(1),  # only perform do-nothing
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

    # load environment and agents manually
    ppo_config.update({"env": CustomizedGrid2OpEnvironment})
    ppo_config.update({"policies": policies})

    run_training(ppo_config)

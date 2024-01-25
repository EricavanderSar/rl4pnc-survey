"""
Trains PPO baseline agent.
"""
import argparse
import logging
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


def run_training(config: dict[str, Any], setup: dict[str, Any]) -> None:
    """
    Function that runs the training script.
    """
    # init ray
    ray.init()

    # Create tuner
    tuner = tune.Tuner(
        ppo.PPO,
        param_space=config,
        run_config=air.RunConfig(
            stop={"timesteps_total": setup["nb_timesteps"]},
            storage_path=os.path.abspath(setup["storage_path"]),
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=setup["checkpoint_freq"],
                checkpoint_at_end=True,
                checkpoint_score_attribute="evaluation/episode_reward_mean",
            ),
            verbose=setup["verbose"],
        ),
    )

    # Launch tuning
    try:
        tuner.fit()
    finally:
        # Close ray instance
        ray.shutdown()


def setup_config(config_path: str) -> None:
    """
    Loads the json as config and sets it up for training.
    """
    # load base PPO config and load in hyperparameters
    ppo_config = ppo.PPOConfig().to_dict()
    custom_config = load_config(config_path)
    ppo_config.update(custom_config["training"])
    ppo_config.update(custom_config["debugging"])
    ppo_config.update(custom_config["framework"])
    ppo_config.update(custom_config["rl_module"])
    ppo_config.update(custom_config["explore"])
    ppo_config.update(custom_config["callbacks"])
    ppo_config.update(custom_config["environment"])
    ppo_config.update(custom_config["multi_agent"])

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
                            "rho_threshold": custom_config["environment"]["env_config"][
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
                ppo.PPOConfig()
                .training(**custom_config["training"])
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
    ppo_config.update({"policies": policies})
    ppo_config.update({"env": CustomizedGrid2OpEnvironment})

    run_training(ppo_config, custom_config["setup"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        help="Path to the config file.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    input_file_path = args.file_path

    if input_file_path:
        setup_config(input_file_path)
    else:
        parser.print_help()
        logging.error("\nError: --file_path is required to specify config location.")

"""
Trains PPO baseline agent.
"""

import argparse
import logging

import gymnasium
import numpy as np
from ray.rllib.algorithms import ppo  # import the type of agents
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import PolicySpec

from mahrl.algorithms.custom_ppo import CustomPPO
from mahrl.experiments.utils import run_training
from mahrl.experiments.yaml import load_config
from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from mahrl.multi_agent.policy import DoNothingPolicy, SelectAgentPolicy


def setup_config(config_path: str, checkpoint_path: str | None) -> None:
    """
    Loads the json as config and sets it up for training.
    """
    # load base PPO config and load in hyperparameters
    ppo_config = ppo.PPOConfig().to_dict()
    custom_config = load_config(config_path)
    if checkpoint_path:
        custom_config["setup"]["checkpoint_path"] = checkpoint_path

    for key in custom_config.keys():
        if key != "setup":
            ppo_config.update(custom_config[key])

    policies = {
        "high_level_policy": PolicySpec(  # chooses RL or do-nothing agent
            policy_class=SelectAgentPolicy,
            observation_space=gymnasium.spaces.Box(-np.inf, np.inf),  # only the max rho
            action_space=gymnasium.spaces.Discrete(2),  # choose one of agents
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
                .rollouts(preprocessor_pref=None)
            ),
        ),
        "reinforcement_learning_policy": PolicySpec(  # performs RL topology
            policy_class=None,
            observation_space=None,  # infer automatically from env
            action_space=None,
            config=None,
        ),
        "do_nothing_policy": PolicySpec(  # performs do-nothing action
            policy_class=DoNothingPolicy,
            observation_space=gymnasium.spaces.Discrete(1),  # no observation space
            action_space=gymnasium.spaces.Discrete(1),  # only perform do-nothing
            config=(
                AlgorithmConfig()
                .training(_enable_learner_api=False)
                .rl_module(_enable_rl_module_api=False)
            ),
        ),
    }

    # load environment and agents manually
    ppo_config.update({"policies": policies})
    ppo_config.update({"env": CustomizedGrid2OpEnvironment})

    run_training(ppo_config, custom_config["setup"], CustomPPO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the config file.",
    )

    parser.add_argument(
        "-r",
        "--retrain",
        type=str,
        help="Path to the checkpoint file for retraining.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.config:
        if args.retrain:
            setup_config(args.config, args.retrain)
        else:
            setup_config(args.config, None)
    else:
        parser.print_help()
        logging.error("\nError: --config is required to specify config location.")

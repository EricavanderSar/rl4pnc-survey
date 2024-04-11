"""
Trains PPO baseline agent.
"""

import argparse
import logging

from ray.rllib.algorithms import ppo  # import the type of agents

from mahrl.experiments.utils import run_training
from mahrl.experiments.yaml import load_config
from mahrl.grid2op_env.custom_environment import SingleAgentGrid2OpEnvironment


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
    ppo_config.update({"env": SingleAgentGrid2OpEnvironment})

    run_training(ppo_config, custom_config["setup"], ppo.PPO)


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

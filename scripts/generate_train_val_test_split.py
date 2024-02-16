"""
Script that splits the scenarios into train, test and validation sets.
"""

import argparse

from mahrl.grid2op_env.utils import make_train_test_val_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-e",
        "--env_name",
        type=str,
        help="Environment name. Default in /home/data_grid2op.",
    )

    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="/Users/barberademol/data_grid2op/",
        help="Path where the environment is stored.",
    )

    parser.add_argument(
        "-t",
        "--test",
        type=int,
        default=10,
        help="Percentage of scenarios to be used for testing. Default = 10%.",
    )

    parser.add_argument(
        "-v",
        "--validation",
        type=int,
        default=10,
        help="Percentage of scenarios to be used for validation. Default = 10%.",
    )

    args = parser.parse_args()

    input_env_name = args.env_name
    input_path = args.path
    input_test = args.test
    input_validation = args.validation

    if input_env_name:
        make_train_test_val_split(
            input_path, input_env_name, input_validation, input_test
        )

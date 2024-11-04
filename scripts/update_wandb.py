
import argparse
import wandb
from tqdm import tqdm


def update_rw_config(run):
    # This function creates a new reward parameter that can be used for grouping
    # print(run)
    # update reward label
    reward_label = run.config["env_config"]["grid2op_kwargs"]["reward_class"].split(".")[-1].split(" ")[0]
    # print(reward_label)
    run.config["reward_fun"] = reward_label


def update_old_env_history(run):
    if run.config["env_config"]["env_type"] == "old_env":
        # set n_history to 1 (since no history is possible for old_env
        if run.config["env_config"]["n_history"] != 1:
            run.config["env_config"]["n_history"] = 1


def update_wandb(project):
    # Update parameters in WandB that can be used for grouping and filtering
    # wandb.login(key="6f33124a4b4139d3e5e7700cf9a9f6739e69c247")
    api = wandb.Api()
    wandb_path = f"rl4pnc4grid2op/{project}"
    print("Updating parameters in WandB Project: ", wandb_path)
    runs = api.runs(f"rl4pnc4grid2op/{project}")
    print(f"Matching runs: {len(runs)}")
    for run in tqdm(runs, total=len(runs)):
        update_rw_config(run)
        update_old_env_history(run)
        run.update()
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-p",
        "--project_id",
        type=str,
        default="Case14_Enhancements",
        help="Project name of the Weights and Biases project for which some parameters need to be adjusted.",
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    res = update_wandb(args.project_id)

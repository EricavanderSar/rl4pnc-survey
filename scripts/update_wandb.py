import argparse
import wandb

PROJECT_NAME = "Case14_Enhancements"

def update_wandb_rw_config(project):
    # wandb.login(key="6f33124a4b4139d3e5e7700cf9a9f6739e69c247")
    api = wandb.Api()
    wandb_path = f"mahrl4grid2op/{project}"
    print("Wandb path", wandb_path)
    runs = api.runs(f"mahrl4grid2op/{project}")
    print(f"matching runs: {len(runs)}")
    for run in runs:
        print(run)
        reward_label = run.config["env_config"]["grid2op_kwargs"]["reward_class"].split(".")[-1].split(" ")[0]
        print(reward_label)
        run.config["reward_fun"] = reward_label
        run.update()


if __name__ == "__main__":
    update_wandb_rw_config(PROJECT_NAME)

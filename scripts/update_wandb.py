import wandb

# Name of project for which you want to update name of parameters used.
PROJECT_NAME = "Case14_Enhancements"


def update_wandb_rw_config(project):
    # This function creates a new reward parameter that can be used for grouping
    # potentially there might be other parameters that can be valuable in the future.
    # wandb.login(key="6f33124a4b4139d3e5e7700cf9a9f6739e69c247")
    api = wandb.Api()
    wandb_path = f"mahrl4grid2op/{project}"
    print("Wandb path", wandb_path)
    runs = api.runs(f"mahrl4grid2op/{project}")
    print(f"matching runs: {len(runs)}")
    for run in runs:
        print(run)
        # update reward label
        reward_label = run.config["env_config"]["grid2op_kwargs"]["reward_class"].split(".")[-1].split(" ")[0]
        print(reward_label)
        run.config["reward_fun"] = reward_label
        run.update()


def update_wandb_old_env_history(project):
    api = wandb.Api()
    wandb_path = f"mahrl4grid2op/{project}"
    print("Wandb path", wandb_path)
    runs = api.runs(f"mahrl4grid2op/{project}")
    print(f"matching runs: {len(runs)}")
    for run in runs:
        print(run)
        print("n_history: ", run.config["env_config"]["n_history"])
        if run.config["env_config"]["env_type"] == "old_env":
            # set n_history to 1 (since no history is possible for old_env
            if run.config["env_config"]["n_history"] != 1:
                run.config["env_config"]["n_history"] = 1
        # if run.config["env_config"]["env_type"] == "new_env":
        #     job_id = int(run.name.split("_")[3])
        #     if 7062409 <= job_id <= 7062414:
        #         run.config["env_config"]["n_history"] = 6
        #     elif 7062296 <= job_id <= 7062300:
        #         run.config["env_config"]["n_history"] = 3
                print("new n_history: ", run.config["env_config"]["n_history"])
                run.update()
    return runs


if __name__ == "__main__":
    update_wandb_rw_config(PROJECT_NAME)
    # runs = update_wandb_old_env_history(PROJECT_NAME)

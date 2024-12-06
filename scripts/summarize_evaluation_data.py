import os
import numpy as np
import matplotlib
import pandas as pd
import json
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scienceplots

file_name_ep_data = "survival.csv"
file_name_action_data = "line_action_topo_data.csv"
file_name_env_config = 'env_config.json'


def get_summarized_data(path, size_case=14, max_env_steps=8064):
    # Initialization of data lists
    overview_data = []
    plot_data = []

    for dir in os.listdir(path):
        agent_dir = os.path.join(path, dir)
        if os.path.isdir(agent_dir) and (not dir.startswith('.')):
            if len(os.listdir(agent_dir)) == 1:
                agent_dir = os.path.join(agent_dir, os.listdir(agent_dir)[0])
            if dir.startswith("CustomPPO"):
                # get the folder where the results of the evaluation are stored inside the agent folder
                for sub_dir in os.listdir(agent_dir):
                    if sub_dir.startswith("evaluation_episodes"):
                        eval_dir = os.path.join(agent_dir, sub_dir)
                        overview_data, plot_data = extend_data(
                            eval_dir, overview_data, plot_data, size_case, max_env_steps
                        )
            else:
                # For the heuristic agents the results of the evaluation are simply stored inside the
                overview_data, plot_data = extend_data(agent_dir, overview_data, plot_data, size_case, max_env_steps)
    return overview_data, plot_data


def extend_data(agent_dir, summarized_data, boxplot_data, size_case, max_env_steps):
    print(f"Collecting data of dir: {agent_dir}")
    path_ep_data = os.path.join(agent_dir, file_name_ep_data)
    path_action_data = os.path.join(agent_dir, file_name_action_data)
    if os.path.exists(path_ep_data) and os.path.exists(path_action_data):
        ep_data = pd.read_csv(path_ep_data)
        # print(ep_data)
        action_data = pd.read_csv(path_action_data)
        # print(action_data)
        with open(os.path.join(agent_dir, file_name_env_config), 'r') as file:
            env_config = json.load(file)
        # print(env_config)

        # Extract relevant data from env_config
        agent_type = env_config.get('agent_type', "Unknown")
        agent_id = env_config.get('agent_id', "xxx")
        action_space = env_config['action_space']
        rules = env_config['rules']
        opponent = ("kwargs_opponent" in env_config["grid2op_kwargs"])
        reward_func = env_config["grid2op_kwargs"]["reward_class"].split(".")[-1].split(" ")[0]

        # Compute mean values from episode data
        mean_ts_survived = ep_data['survived'].mean()
        completed_ep = (ep_data['survived'] == max_env_steps).sum()/len(ep_data['survived'])
        mean_max_exec_time = ep_data['max exec time'].mean()
        mean_exec_time = ep_data['mean exec time'].mean()

        # get data from action_data (which is all ts for which rho.max() > activation_threshold = 0.95)
        action_exec_time = action_data['agent_exec_time'].mean()
        ts_overloaded = len(action_data[action_data['rho'] > 1.0]) / len(ep_data)  # mean ts_overloaded/episode
        unique_actions = len(action_data[['action_sub', 'action_topo']].drop_duplicates())
        unique_line_danger = len(action_data.line_danger.unique())
        subs_changed = action_data.action_sub.unique()
        subs_changed = np.delete(subs_changed, np.where(subs_changed > size_case))
        n_subs_changed = len(subs_changed)
        max_topo_depth = action_data['sub_topo_depth'].max()

        # Add a row to the list
        summarized_data.append({
            'agent_type': agent_type,
            'agent_id': agent_id,
            'opponent': opponent,
            'train_opponent': env_config.get("train_opponent", False),
            'action space': action_space,
            'observation space': env_config.get('input', ""),
            'AT': rules['activation_threshold'],
            'line_reco': rules['line_reco'],
            'line_disc': rules['line_disc'],
            'reset_topo': rules['reset_topo'],
            'train reward': reward_func,
            'train AT': env_config.get('rho_threshold', ""),
            'train line_reco': env_config.get('line_reco', True),
            'train line_disc': env_config.get('line_disc', False),
            'train reset_topo': env_config.get('reset_topo', 0),
            'steps survived': round(mean_ts_survived / max_env_steps * 100, 2),
            'completed episodes': round(completed_ep * 100, 2),
            'steps overloaded': round(ts_overloaded / mean_ts_survived * 100, 3),
            'execution time [ms]': round(mean_exec_time * 1000, 3),
            'agent execution time [ms]': round(action_exec_time * 1000, 3),
            'maximum topology depth': max_topo_depth,
            'unique actions': unique_actions,
            'unique lines in danger': unique_line_danger,
            'unqique subs changed': n_subs_changed,
            'substations changed': subs_changed,
            # 'max execution time [ms]': round(mean_max_exec_time*1000, 3)
        })

        # For boxplots:
        rule_label = (f"threshold={rules['activation_threshold']}, "
                      f"reco={rules['line_reco']}, "
                      f"disc={rules['line_disc']}, "
                      f"reset={rules['reset_topo']}")

        # Add data to boxplot_data
        for survived_value in ep_data['survived']:
            boxplot_data.append({
                'agent_type': agent_type,
                'agent_id': agent_id,
                'opponent': opponent,
                'action space': action_space,
                'rules': rule_label,
                'survived': survived_value
            })
    else:
        print(f"Data file {path_ep_data} and/or {path_action_data} cant be found in {agent_dir}. "
              f"This data will be skipped.")
    return summarized_data, boxplot_data


def save_summary_table(path, data):
    print(f"Saving summary table at {path}/summarized_data.csv")
    # Create a new DataFrame from the collected rows
    summary_table = pd.DataFrame(data)
    summary_table = summary_table.sort_values(['opponent', 'agent_type', 'line_reco', 'line_disc', 'reset_topo'],
                                              ascending=[True, True, True, True, False])
    summary_table.to_csv(os.path.join(path, "summarized_data.csv"), index=False)


def save_boxplot_table(path, boxplot_data):
    print(f"Saving boxplot table at {path}/boxplot_data.csv")
    # Convert boxplot_data into a DataFrame
    boxplot_df = pd.DataFrame(boxplot_data)
    boxplot_df = boxplot_df.sort_values(['opponent', 'agent_type'])
    boxplot_df.to_csv(os.path.join(path, "boxplot_data.csv"), index=False)
    return boxplot_df


def make_boxplot(path, boxplot_df, action_space=None):
    print("Making boxplot of survived time steps per agent...")
    # Filter only specific action space
    if action_space is not None:
        boxplot_df = boxplot_df[boxplot_df['action space'] == action_space]

    # Set up a color palette
    palette = sns.color_palette("Set2", len(boxplot_df['rules'].unique()))
    style = ['science', 'grid', 'no-latex']
    plt.style.use(style) # use scientific plotting style.
    # # Create a new column combining agent_type and opponent for separate subplots
    boxplot_df['agent_opponent_actspace'] = boxplot_df.apply(
        lambda row: f"{row['agent_type']} - "
                    f"{row['action space']}" + (" with opponent" if row['opponent'] else f""), axis=1
    )
    # display(boxplot_df)

    # Create a combined column for unique groups
    boxplot_df['group'] = boxplot_df.apply(
        lambda row: f"{row['agent_type']} | {row['action space']} | "
                    f"{row['rules']} | Opponent={row['opponent']}", axis=1
    )

    # Create the boxplots
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Boxplot for survived by agent_type
    boxplot = sns.boxplot(
        data=boxplot_df,
        x='agent_opponent_actspace',
        y='survived',
        hue='rules',
        palette=palette,
        ax=ax
    )

    # Apply hatching based on opponent status
    patches = [patch for patch in boxplot.patches if type(patch) == mpatches.PathPatch]
    for patch, (_, row) in zip(patches, boxplot_df[['group', 'opponent']].drop_duplicates().iterrows()):
        if row['opponent']:  # Apply hatching if opponent=True
            patch.set_hatch('//')

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    hatch_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Opponent=True")
    plain_patch = mpatches.Patch(facecolor="white", edgecolor="black", label="Opponent=False")
    ax.legend(
        handles=handles + [hatch_patch, plain_patch],
        labels=labels + ["Opponent=True", "Opponent=False"],
        title="Rules and Opponent",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    # Customize the plot
    plt.title("Steps survived per agent", fontsize=14)
    plt.xlabel("Agent Type", fontsize=12)
    plt.ylabel("Survived", fontsize=12)
    x_labels = [f"{row['agent_type']} - {'(N-0)' if row['action space'] == 'medha' else '(N-1)'}" for (_, row) in
                boxplot_df[['agent_opponent_actspace', 'agent_type', 'action space']].drop_duplicates().iterrows()]
    plt.xticks(rotation=45)
    boxplot.set_xticks([], minor=True) # remove extra ticks on x-axis.
    boxplot.set_xticklabels(x_labels)
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(path, f'box_plots_agents_rules_opponent.svg'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")
    parser.add_argument(
        "-p",
        "--path",
        # default="/Users/ericavandersar/Documents/Python_Projects/Research/Rl4Pnc/results/",
        default="/Users/ericavandersar/surfdrive/Documents/Research/TestEval/",
        type=str,
        help="The path where the evaluation results have been stored.",
    )
    parser.add_argument(
        "-c",
        "--case",
        type=int,
        default=14,
        help="The size of the case",
    )
    parser.add_argument(
        "-m",
        "--max_env_steps",
        type=int,
        default=8064,
        help="Maximum number of ts for an episode in this environment",
    )

    args = parser.parse_args()
    path = args.path
    size_case = args.case
    max_env_steps = args.max_env_steps

    summ_data, data_to_plot = get_summarized_data(path, size_case, max_env_steps)
    save_summary_table(path, summ_data)
    df_boxplot = save_boxplot_table(path, data_to_plot)
    make_boxplot(path, df_boxplot)

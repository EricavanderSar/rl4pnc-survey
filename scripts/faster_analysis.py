import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from grid2op.Episode import EpisodeData
from tqdm import tqdm  # for easy progress bar

from mahrl.experiments.yaml import load_config
from mahrl.grid2op_env.utils import make_g2op_env


def get_action_data(env, env_config, this_episode, input_data=None):
    # get data lines in overflow
    idx = env.observation_space.shape
    pos = env.observation_space.attr_list_vect.index("rho")
    start = sum(idx[:pos])
    end = start + idx[pos]
    rho_values = this_episode.get_observations()[
        0 : this_episode.meta["nb_timestep_played"]
    ][..., np.arange(start, end)]
    ts_danger, line_danger = np.where(rho_values > env_config["rho_threshold"])
    rho = rho_values[rho_values > env_config["rho_threshold"]]

    # get actions
    idx = env.action_space.shape
    pos = env.action_space.attr_list_vect.index("_set_topo_vect")
    start = sum(idx[:pos])
    end = start + idx[pos]
    actions = this_episode.get_actions()[ts_danger][..., np.arange(start, end)]
    action_sub = [
        env._topo_vect_to_sub[act != 0][0] if any(act != 0) else 15 for act in actions
    ]
    action_topo = [
        list(act[act != 0].astype(int)) if any(act != 0) else [0] for act in actions
    ]

    # get new topology and topological distances
    idx = env.observation_space.shape
    pos = env.observation_space.attr_list_vect.index("topo_vect")
    start = sum(idx[:pos])
    end = start + idx[pos]

    # check current topo (before changed)
    cur_topos = this_episode.get_observations()[ts_danger][..., np.arange(start, end)]

    # check the new topos (ts_danger+1)
    topos = this_episode.get_observations()[ts_danger + 1][..., np.arange(start, end)]

    implicit_DNs = []
    # check that action is not implicit do nothing
    for idx, topo in enumerate(topos):
        if np.array_equal(topo, cur_topos[idx]):
            implicit_DNs.append(idx)
            print(f"implicit do nothing at {ts_danger[idx]}")
        # if action_topo[idx] == [0]:
        #     implicit_DNs.append(idx)
        #     print(f"explicit do nothing at {ts_danger[idx]}")

    # remove item from topos and cur_topos
    for idx in sorted(implicit_DNs, reverse=True):
        topos = np.delete(topos, idx, 0)
        del action_topo[idx]  # = np.delete(action_topo, idx, 0)
        del action_sub[idx]  # = np.delete(action_sub, idx, 0)
        rho = np.delete(rho, idx, 0)
        line_danger = np.delete(line_danger, idx, 0)
        ts_danger = np.delete(ts_danger, idx, 0)

    subs_changed = [
        np.unique(env._topo_vect_to_sub[topo == 2]) if any(topo == 2) else [0]
        for topo in topos
    ]
    sub_topo_depth = [len(changed) for changed in subs_changed]
    el_changed = [np.nonzero(topo == 2)[0] for topo in topos]
    el_topo_depth = [len(changed) for changed in el_changed]
    # chronic ids
    chron_id = [this_episode.name for _ in ts_danger]

    if input_data == None:
        data = {}
        data["chron_id"] = chron_id
        data["ts_danger"] = ts_danger
        data["line_danger"] = line_danger
        data["rho"] = rho
        data["action_sub"] = action_sub
        data["action_topo"] = action_topo
        data["subs_changed"] = subs_changed
        data["sub_topo_depth"] = sub_topo_depth
        data["el_changed"] = el_changed
        data["el_topo_depth"] = el_topo_depth
        # data["topo"] = topos
    else:
        data = input_data
        data["chron_id"].extend(chron_id)
        data["ts_danger"] = np.append(data["ts_danger"], ts_danger)
        data["line_danger"] = np.append(data["line_danger"], line_danger)
        data["rho"] = np.append(data["rho"], rho)
        data["action_sub"].extend(action_sub)
        data["action_topo"].extend(action_topo)
        data["subs_changed"].extend(subs_changed)
        data["sub_topo_depth"].extend(sub_topo_depth)
        data["el_changed"].extend(el_changed)
        data["el_topo_depth"].extend(el_topo_depth)
        # data["topo"] = np.append(data["topo"], topos)
    return data


def collect_episode_data(env, env_config, store_trajectories_folder, li_episode):
    print(" Start collecting episode data ... ")
    act_data = None
    store_actdata_path = os.path.join(
        store_trajectories_folder, "line_action_topo_data.csv"
    )
    store_surv_path = os.path.join(store_trajectories_folder, "survival.csv")
    df_act, df_sur = None, None
    chron = []
    surv = []
    rw = []

    if os.path.exists(store_actdata_path):
        # check if part of the data was already collected...
        df_act = pd.read_csv(store_actdata_path)
        n_episode_evaluated = len(df_act.chron_id.unique())
        if os.path.exists(store_surv_path):
            df_sur = pd.read_csv(store_surv_path)
            if n_episode_evaluated == len(df_sur.chron_id.unique()):
                li_episode = li_episode[n_episode_evaluated:]
            else:
                df_act, df_sur = None, None
        else:
            df_act, df_sur = None, None

    # start collecting the data by going through the played episodes
    for ep in tqdm(li_episode, total=len(li_episode)):
        full_path, episode_studied = ep
        this_episode = EpisodeData.from_disk(store_trajectories_folder, episode_studied)
        act_data = get_action_data(env, env_config, this_episode, act_data)
        # save chronic data
        chron.append(
            os.path.basename(os.path.normpath(this_episode.meta["chronics_path"]))
        )
        surv.append(this_episode.meta["nb_timestep_played"])
        rw.append(np.round(this_episode.meta["cumulative_reward"], decimals=2))

    # Save action data in data frame
    if df_act is not None:
        df_act = df_act.append(pd.DataFrame(act_data))
    else:
        df_act = pd.DataFrame(act_data)
    print(df_act.head())
    df_act.to_csv(
        os.path.join(store_trajectories_folder, "line_action_topo_data.csv"),
        index=False,
    )

    # Save chronic data in data frame
    chron_data = {"chron": chron, "survived": surv, "cum reward": rw}
    if df_sur is not None:
        df_sur = df_sur.append(pd.DataFrame(chron_data))
    else:
        df_sur = pd.DataFrame(chron_data)
    print(
        f"Survival stats: mean {df_sur.survived.mean()} with std: {df_sur.survived.std()}"
    )
    df_sur.to_csv(os.path.join(store_trajectories_folder, "survival.csv"), index=False)

    return act_data, df_act


def print_measures(df):
    print("\n Frequency substations activations")
    print(df.action_sub.value_counts())
    print(
        f"\n Frequency topology actions: {df.action_topo.value_counts().sum()} of which unique: {len(df.action_topo.unique())}"
    )
    print(df.action_topo.value_counts())
    print("\n Frequency lines in danger: ")
    print(df.line_danger.value_counts())
    print(f"\n # unique topologies: {len(df.el_changed.unique())}")
    max_topo_depth = df.sub_topo_depth.unique().max()
    # if max_topo_depth == 14:  # TODO make not hardcoded
    #     max_topo_depth = df.sub_topo_depth.unique()[-2]
    print(
        f"\n Average topology depth {df.sub_topo_depth.mean()} with std {df.sub_topo_depth.std()} and max {df.sub_topo_depth.unique().max()}"
    )
    return max_topo_depth


def quick_overview(data_folder):
    df = pd.read_csv(os.path.join(data_folder, "line_action_topo_data.csv"))
    # print overview
    max_topo_depth = print_measures(df)

    # Create a pivot table to count occurrences of 'action topo' for each 'action sub'
    pivot_table = df.pivot_table(
        index=["action_sub"], columns="action_topo", aggfunc="size", fill_value=0
    )
    # Plotting the data
    ax = pivot_table.plot(kind="bar", figsize=(10, 6), width=0.8)
    # Customize the plot
    ax.set_title("Frequency actions at substations")
    ax.set_xlabel("Substation")
    ax.set_ylabel("Frequency")
    ax.legend(title="Topology")
    plt.savefig(os.path.join(data_folder, "actions_at_substations.png"))
    # plt.show()

    #
    pivot_table = df.pivot_table(
        index=["action_sub", "action_topo"],
        columns="line_danger",
        aggfunc="size",
        fill_value=0,
    )
    ax = pivot_table.plot(kind="bar", figsize=(10, 6), width=0.8)
    ax.set_title("Frequency actions per line in danger")
    ax.set_xlabel("Action")
    ax.set_ylabel("Frequency")
    ax.legend(title="Line in danger")
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, "actions_per_line_danger.png"))
    # plt.show()

    # Plat data
    for i in range(1, max_topo_depth + 1):
        pivot_table = df[df["sub_topo_depth"] == i].pivot_table(
            index="action_sub", columns="action_topo", aggfunc="size", fill_value=0
        )
        ax = pivot_table.plot(kind="bar", figsize=(10, 6), width=0.8)
        ax.set_title(f"Frequency of {i}th topology action")
        ax.set_xlabel("Substation")
        ax.set_ylabel("Frequency")
        ax.legend(title="Topology")
        plt.savefig(os.path.join(data_folder, f"topology_action_{i}.png"))
        # plt.show()

    # Create a pivot table
    pivot_table = df.pivot_table(
        index=["line_danger"], columns="subs_changed", aggfunc="size", fill_value=0
    )

    ax = pivot_table.plot(kind="bar", figsize=(10, 6), width=0.8)
    ax.set_title("Frequency different changed subs")
    ax.set_xlabel("Line in danger")
    ax.set_ylabel("Frequency")
    ax.legend(title="Subs changed")
    plt.savefig(os.path.join(data_folder, f"subs_changed_frequency.png"))
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        help="The location of the evaluation files",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Name of the config file.",
    )
    args = parser.parse_args()

    # Get location of studied agent
    store_trajectories_folder = args.file_path
    config = args.config
    lib_dir = "/Users/barberademol/Documents/GitHub/mahrl_grid2op/"

    # load config file
    config = load_config(args.config)
    env_config = config["environment"]["env_config"]
    env = make_g2op_env(env_config)

    li_episode = EpisodeData.list_episode(store_trajectories_folder)
    all_data, df = collect_episode_data(
        env, env_config, store_trajectories_folder, li_episode
    )
    quick_overview(store_trajectories_folder)

"""
Script to analyse agent behaviour. The script collects data from the played episodes and saves the data in CSV files.
"""

import argparse
import logging
import os
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from grid2op.Environment import BaseEnv
from grid2op.Episode import EpisodeData
from tqdm import tqdm  # for easy progress bar

from mahrl.experiments.yaml import load_config
from mahrl.grid2op_env.utils import make_g2op_env


def get_action_data(  # pylint: disable=too-many-locals,too-many-statements
    env: BaseEnv,
    env_config: dict[str, Any],
    this_episode: EpisodeData,
    input_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Get action data from the environment.

    Args:
        env (object): The environment object.
        env_config (dict): Configuration parameters for the environment.
        this_episode (object): The episode object.
        input_data (dict, optional): Existing data to append to. Defaults to None.

    Returns:
        dict: A dictionary containing the action data.

    """
    # get data lines in overflow
    idx = env.observation_space.shape
    pos = env.observation_space.attr_list_vect.index("rho")
    start = sum(idx[:pos])
    end = start + idx[pos]
    rho_values = this_episode.get_observations()[
        0 : this_episode.meta["nb_timestep_played"]  # noqa: E203
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

    implicit_dns = []
    # check that action is not implicit do nothing
    for idx, topo in enumerate(topos):
        if np.array_equal(topo, cur_topos[idx]):
            implicit_dns.append(idx)
            logging.info(f"implicit do nothing at {ts_danger[idx]}")

    # remove item from topos and cur_topos
    for idx in sorted(implicit_dns, reverse=True):
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

    if input_data is None:
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


def collect_episode_data(
    env: BaseEnv,
    env_config: dict[str, Any],
    store_trajectories_folder: str,
    li_episode: list[EpisodeData],
) -> Tuple[Optional[dict[str, Any]], pd.DataFrame]:
    """
    Collects episode data by going through the played episodes and saves the data in CSV files.

    Args:
        env (object): The environment object.
        env_config (object): The environment configuration object.
        store_trajectories_folder (str): The path to the folder where the episode data will be stored.
        li_episode (list): A list of tuples containing the full path and episode number.

    Returns:
        tuple: A tuple containing the action data and the data frame containing the action data.

    """
    logging.info(" Start collecting episode data ... ")
    act_data = None
    df_act, df_sur = None, None
    chron = []
    surv = []
    reward = []

    if os.path.exists(
        os.path.join(store_trajectories_folder, "line_action_topo_data.csv")
    ):
        # check if part of the data was already collected...
        df_act = pd.read_csv(
            os.path.join(store_trajectories_folder, "line_action_topo_data.csv")
        )
        n_episode_evaluated = len(df_act.chron_id.unique())
        if os.path.exists(os.path.join(store_trajectories_folder, "survival.csv")):
            df_sur = pd.read_csv(
                os.path.join(store_trajectories_folder, "survival.csv")
            )
            if n_episode_evaluated == len(df_sur.chron_id.unique()):
                li_episode = li_episode[n_episode_evaluated:]
            else:
                df_act, df_sur = None, None
        else:
            df_act, df_sur = None, None

    # start collecting the data by going through the played episodes
    for episode in tqdm(li_episode, total=len(li_episode)):
        _, episode_studied = episode
        this_episode = EpisodeData.from_disk(store_trajectories_folder, episode_studied)
        act_data = get_action_data(env, env_config, this_episode, act_data)
        # save chronic data
        chron.append(
            os.path.basename(os.path.normpath(this_episode.meta["chronics_path"]))
        )
        surv.append(this_episode.meta["nb_timestep_played"])
        reward.append(np.round(this_episode.meta["cumulative_reward"], decimals=2))

    # Save action data in data frame
    if df_act is not None:
        df_act = df_act.append(pd.DataFrame(act_data))
    else:
        df_act = pd.DataFrame(act_data)
    logging.info(df_act.head())
    df_act.to_csv(
        os.path.join(store_trajectories_folder, "line_action_topo_data.csv"),
        index=False,
    )

    # Save chronic data in data frame
    chron_data = {"chron": chron, "survived": surv, "cum reward": reward}
    if df_sur is not None:
        df_sur = df_sur.append(pd.DataFrame(chron_data))
    else:
        df_sur = pd.DataFrame(chron_data)
    logging.info(
        f"Survival stats: mean {df_sur.survived.mean()} with std: {df_sur.survived.std()}"
    )
    df_sur.to_csv(os.path.join(store_trajectories_folder, "survival.csv"), index=False)

    return act_data, df_act


def print_dataframe(dataframe: pd.DataFrame) -> int:
    """
    Prints various statistics and information about the given dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe containing the data to be analyzed.

    Returns:
    int: The maximum topology depth found in the dataframe.
    """
    logging.info("\n Frequency substations activations")
    logging.info(dataframe.action_sub.value_counts())
    logging.info(
        f"\n Frequency topology actions: {dataframe.action_topo.value_counts().sum()} \
            of which unique: {len(dataframe.action_topo.unique())}"
    )
    logging.info(dataframe.action_topo.value_counts())
    logging.info("\n Frequency lines in danger: ")
    logging.info(dataframe.line_danger.value_counts())
    logging.info(f"\n # unique topologies: {len(dataframe.el_changed.unique())}")
    max_topo_depth = int(dataframe.sub_topo_depth.unique().max())
    # if max_topo_depth == 14:  # TODO make not hardcoded
    #     max_topo_depth = df.sub_topo_depth.unique()[-2]
    logging.info(
        f"\n Average topology depth {dataframe.sub_topo_depth.mean()} with std {dataframe.sub_topo_depth.std()} \
            and max {dataframe.sub_topo_depth.unique().max()}"
    )
    return max_topo_depth


def quick_overview(data_folder: str) -> None:
    """
    Generate a quick overview of the data by creating various plots based on the input data.

    Parameters:
    - data_folder (str): The path to the folder containing the data.

    Returns:
    None
    """
    dataframe = pd.read_csv(os.path.join(data_folder, "line_action_topo_data.csv"))
    # logging.log overview
    max_topo_depth = print_dataframe(dataframe)

    # Create a pivot table to count occurrences of 'action topo' for each 'action sub'
    pivot_table = dataframe.pivot_table(
        index=["action_sub"], columns="action_topo", aggfunc="size", fill_value=0
    )
    # Plotting the data
    axis = pivot_table.plot(kind="bar", figsize=(10, 6), width=0.8)
    # Customize the plot
    axis.set_title("Frequency actions at substations")
    axis.set_xlabel("Substation")
    axis.set_ylabel("Frequency")
    axis.legend(title="Topology")
    plt.savefig(os.path.join(data_folder, "actions_at_substations.png"))
    # plt.show()

    #
    pivot_table = dataframe.pivot_table(
        index=["action_sub", "action_topo"],
        columns="line_danger",
        aggfunc="size",
        fill_value=0,
    )
    axis = pivot_table.plot(kind="bar", figsize=(10, 6), width=0.8)
    axis.set_title("Frequency actions per line in danger")
    axis.set_xlabel("Action")
    axis.set_ylabel("Frequency")
    axis.legend(title="Line in danger")
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, "actions_per_line_danger.png"))
    # plt.show()

    # Plat data
    for i in range(1, max_topo_depth + 1):
        pivot_table = dataframe[dataframe["sub_topo_depth"] == i].pivot_table(
            index="action_sub", columns="action_topo", aggfunc="size", fill_value=0
        )
        axis = pivot_table.plot(kind="bar", figsize=(10, 6), width=0.8)
        axis.set_title(f"Frequency of {i}th topology action")
        axis.set_xlabel("Substation")
        axis.set_ylabel("Frequency")
        axis.legend(title="Topology")
        plt.savefig(os.path.join(data_folder, f"topology_action_{i}.png"))
        # plt.show()

    # Create a pivot table
    pivot_table = dataframe.pivot_table(
        index=["line_danger"], columns="subs_changed", aggfunc="size", fill_value=0
    )

    axis = pivot_table.plot(kind="bar", figsize=(10, 6), width=0.8)
    axis.set_title("Frequency different changed subs")
    axis.set_xlabel("Line in danger")
    axis.set_ylabel("Frequency")
    axis.legend(title="Subs changed")
    plt.savefig(os.path.join(data_folder, "subs_changed_frequency.png"))
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

    # load config file
    config = load_config(args.config)
    setup_env_config = config["environment"]["env_config"]
    setup_env = make_g2op_env(setup_env_config)

    setup_li_episode = EpisodeData.list_episode(args.file_path)
    _, _ = collect_episode_data(
        setup_env, setup_env_config, args.file_path, setup_li_episode
    )
    quick_overview(args.file_path)

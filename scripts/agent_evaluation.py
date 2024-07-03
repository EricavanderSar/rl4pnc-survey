import json
import os
import grid2op
import numpy as np
import argparse
import re
from grid2op.PlotGrid import PlotMatplot
from lightsim2grid import LightSimBackend
from grid2op.Agent import DoNothingAgent
from grid2op.Runner import Runner
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm  # for easy progress bar
import pandas as pd
from grid2op.Episode import EpisodeData
from ray.rllib.algorithms import ppo
from mahrl.evaluation.evaluation_agents import RllibAgent
from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from mahrl.grid2op_env.custom_env2 import RlGrid2OpEnv
from mahrl.experiments.yaml import load_config
from mahrl.evaluation.utils import instantiate_reward_class


def get_latest_checkpoint(agent_path):
    def extract_number(f):
        s = re.findall("\d+$", f)
        return (int(s[0]) if s else -1, f)
    # Get latest checkpoint:
    checkpoints = [name for name in os.listdir(agent_path) if name.startswith("checkpoint")]
    return max(checkpoints, key=extract_number)


def get_env_config(studie_path, test_case, reset_topo, libdir):
    agent_path = os.path.join(studie_path, test_case)
    # load config
    config_path = os.path.join(agent_path, "params.json")
    config = load_config(config_path)
    env_config = config["env_config"]
    # adjust lib_dir:
    env_config["lib_dir"] = libdir
    # Overwrite file params
    config["env_config"] = env_config
    with open(config_path, "w") as outfile:
        json.dump(config, outfile, indent=4)

    # change the env_name from _train to _test
    env_config["env_name"] = env_config["env_name"].replace("_train", "")
    # adjust reward class
    env_config["grid2op_kwargs"]["reward_class"] = instantiate_reward_class(
        env_config["grid2op_kwargs"]["reward_class"]
    )
    env_config["reset_topo"] = reset_topo

    return env_config, agent_path


def run_evaluation(agent_path,
                   checkpoint,
                   env_config,
                   env_type,
                   chronics,
                   nb_episodes):
    folder_name = "evaluation_episodes_testreset" if env_config["reset_topo"] else "evaluation_episodes"
    store_trajectories_folder = os.path.join(agent_path, folder_name)
    try:
        env = grid2op.make(env_config["env_name"], backend=LightSimBackend())
    except:
        # try without LightSimBackend()
        env = grid2op.make(env_config["env_name"])
    env.chronics_handler.set_chunk_size(100)
    li_episode = EpisodeData.list_episode(store_trajectories_folder) if \
        os.path.exists(store_trajectories_folder) else []

    if len(li_episode) < nb_episodes:
        print(f">> Start running evaluation of {nb_episodes-len(li_episode)} episodes")
        agent = RllibAgent(
            action_space=env.action_space,
            env_config=env_config,
            file_path=agent_path,
            policy_name="reinforcement_learning_policy",
            algorithm=ppo.PPO,
            checkpoint_name=checkpoint,
            gym_wrapper=env_type(env_config),
        )
        runner = Runner(
            **env.get_params_for_runner(),
            agentClass=None,
            agentInstance=agent
        )
        res = runner.run(
            nb_episode=nb_episodes-len(li_episode),
            # pbar=True,
            episode_id=chronics[len(li_episode):nb_episodes],
            path_save=os.path.abspath(store_trajectories_folder),
        )

        ts_surv = []

        res_txt = f"{store_trajectories_folder}/results_{checkpoint}.txt"
        for _, chron_id, cum_reward, nb_time_step, max_ts in res:
            msg_tmp = "\tFor chronics with id {}\n".format(chron_id)
            msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
            msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)
            with open(
                    res_txt,
                    "a",
                    encoding="utf-8",
            ) as file:
                file.write(
                    f"\n\tFor chronics with id {chron_id}\n"
                    + f"\t\t - number of time steps completed: {nb_time_step:.0f} / {max_ts:.0f}"
                )
            ts_surv.append(nb_time_step)

        with open(
                res_txt,
                "a",
                encoding="utf-8",
        ) as file:
            file.write(
                f"\nAverage timesteps survived: {np.mean(ts_surv)}\n{ts_surv}\n"
                f"\nCompleted: {np.count_nonzero(np.array(ts_surv)==8064)}/{nb_episodes-len(li_episode)}"
            )
        print(f"\nAverage timesteps survived: {np.mean(ts_surv)}\n{ts_surv}\n")

    return store_trajectories_folder, env


def get_action_data(env, env_config, this_episode, input_data=None):
    # get data lines in overflow
    idx = env.observation_space.shape
    pos = env.observation_space.attr_list_vect.index('rho')
    start = sum(idx[:pos])
    end = start + idx[pos]
    rho_values = this_episode.get_observations()[0:this_episode.meta['nb_timestep_played']][..., np.arange(start, end)]
    ts_danger, line_danger = np.where(rho_values > env_config["rho_threshold"])
    rho = rho_values[rho_values > env_config["rho_threshold"]]

    # get actions
    idx = env.action_space.shape
    pos = env.action_space.attr_list_vect.index('_set_topo_vect')
    start = sum(idx[:pos])
    end = start + idx[pos]
    actions = this_episode.get_actions()[ts_danger][..., np.arange(start, end)]
    action_sub = [env._topo_vect_to_sub[act != 0][0] if any(act != 0) else 15 for act in actions]
    action_topo = [list(act[act != 0].astype(int)) if any(act != 0) else [0] for act in actions]

    # get new topology and topological distances
    idx = env.observation_space.shape
    pos = env.observation_space.attr_list_vect.index('topo_vect')
    start = sum(idx[:pos])
    end = start + idx[pos]

    # # check current topo (before changed)
    # topos = this_episode.get_observations()[ts_danger][..., np.arange(start, end)]
    # subs_curr = [np.unique(env._topo_vect_to_sub[topo != 1]) if any(topo != 1) else [0] for topo in topos]

    # check the new topos (ts_danger+1)
    topos = this_episode.get_observations()[ts_danger + 1][..., np.arange(start, end)]
    subs_changed = [np.unique(env._topo_vect_to_sub[topo != 1]) if any(topo != 1) else [0] for topo in topos]
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
    store_actdata_path = os.path.join(store_trajectories_folder, "line_action_topo_data.csv")
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
            df_sur = pd.read_csv(store_actdata_path)
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
        chron.append(os.path.basename(os.path.normpath(this_episode.meta['chronics_path'])))
        surv.append(this_episode.meta['nb_timestep_played'])
        rw.append(np.round(this_episode.meta['cumulative_reward'], decimals=2))

    # Save action data in data frame
    if df_act is not None:
        df_act = df_act.append(pd.DataFrame(act_data))
    else:
        df_act = pd.DataFrame(act_data)
    print(df_act.head())
    df_act.to_csv(os.path.join(store_trajectories_folder, "line_action_topo_data.csv"), index=False)

    # Save chronic data in data frame
    chron_data = {'chron': chron, 'survived': surv, 'cum reward': rw}
    if df_sur is not None:
        df_sur = df_sur.append(pd.DataFrame(chron_data))
    else:
        df_sur = pd.DataFrame(chron_data)
    print(df_sur.head())
    df_sur.to_csv(os.path.join(store_trajectories_folder, "survival.csv"), index=False)

    return act_data, df_act


def print_measures(df):
    print("\n Frequency substations activations")
    print(df.action_sub.value_counts())
    print("\n Frequency topology actions")
    print(df.action_topo.value_counts())
    print("\n Frequency lines in danger")
    print(df.line_danger.value_counts())
    print(f"\n # unique topologies: {len(df.el_changed.unique())}")
    max_topo_depth = df.sub_topo_depth.unique().max()
    if max_topo_depth == 14: # TODO make not hardcoded
        max_topo_depth = df.sub_topo_depth.unique()[-2]
    print(f"\n Maximum topology depth {max_topo_depth}")
    return max_topo_depth


def quick_overview(data_folder):
    df = pd.read_csv(os.path.join(data_folder, "line_action_topo_data.csv"))
    # print overview
    max_topo_depth = print_measures(df)

    # Create a pivot table to count occurrences of 'action topo' for each 'action sub'
    pivot_table = df.pivot_table(index=['action_sub'], columns='action_topo', aggfunc='size',
                                 fill_value=0)
    # Plotting the data
    ax = pivot_table.plot(kind='bar', figsize=(10, 6), width=0.8)
    # Customize the plot
    ax.set_title('Frequency actions at substations')
    ax.set_xlabel('Substation')
    ax.set_ylabel('Frequency')
    ax.legend(title='Topology')
    plt.savefig(os.path.join(data_folder, 'actions_at_substations.png'))
    # plt.show()

    #
    pivot_table = df.pivot_table(index=['action_sub', 'action_topo'], columns='line_danger', aggfunc='size',
                                 fill_value=0)
    ax = pivot_table.plot(kind='bar', figsize=(10, 6), width=0.8)
    ax.set_title('Frequency actions per line in danger')
    ax.set_xlabel('Action')
    ax.set_ylabel('Frequency')
    ax.legend(title='Line in danger')
    plt.tight_layout()
    plt.savefig(os.path.join(data_folder, 'actions_per_line_danger.png'))
    # plt.show()

    # Plat data
    for i in range(1, max_topo_depth+1):
        pivot_table = df[df["sub_topo_depth"] == i].pivot_table(index='action_sub', columns='action_topo',
                                                                aggfunc='size', fill_value=0)
        ax = pivot_table.plot(kind='bar', figsize=(10, 6), width=0.8)
        ax.set_title(f'Frequency of {i}th topology action')
        ax.set_xlabel('Substation')
        ax.set_ylabel('Frequency')
        ax.legend(title='Topology')
        plt.savefig(os.path.join(data_folder, f'topology_action_{i}.png'))
        # plt.show()

    # Create a pivot table
    pivot_table = df.pivot_table(index=['line_danger'], columns='subs_changed', aggfunc='size', fill_value=0)

    ax = pivot_table.plot(kind='bar', figsize=(10, 6), width=0.8)
    ax.set_title('Frequency different changed subs')
    ax.set_xlabel('Line in danger')
    ax.set_ylabel('Frequency')
    ax.legend(title='Subs changed')
    plt.savefig(os.path.join(data_folder, f'subs_changed_frequency.png'))
    # plt.show()


def eval_single_agent(test_case,
                      studie_path,
                      reset_topo_todefault,
                      lib_dir,
                      test_chronics
                      ):
    # Get environment configuration
    env_config, agent_path = get_env_config(studie_path, test_case, reset_topo_todefault, lib_dir)
    ENV_TYPE = RlGrid2OpEnv if env_config["env_type"] == "new_env" else CustomizedGrid2OpEnvironment
    checkpoint_name = get_latest_checkpoint(agent_path)

    print(f"Studying agent at {agent_path}")
    print(f"Environment configuration {env_config}")

    NB_EPISODE = len(test_chronics)
    store_trajectories_folder, env = run_evaluation(agent_path,
                                                    checkpoint_name,
                                                    env_config,
                                                    ENV_TYPE,
                                                    test_chronics,
                                                    NB_EPISODE)

    li_episode = EpisodeData.list_episode(store_trajectories_folder)
    all_data, df = collect_episode_data(env, env_config, store_trajectories_folder, li_episode)
    quick_overview(store_trajectories_folder)
    return all_data, df, env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-p",
        "--path",
        default="/Users/ericavandersar/surfdrive/Documents/Research/Result/Case14_Sandbox_ActSpaces/",
        type=str,
        help="The location of studied agent",
    )
    parser.add_argument(
        "-t",
        "--test_case",
        default="CustomPPO_CustomizedGrid2OpEnvironment_cc933_00000_0_2024-05-06_10-15-28",
        type=str,
        help="Name of the agent you want to evaluate.",
    )
    parser.add_argument('-r', '--reset_topo', default=False, action='store_true',
                        help="reset topology when the environment is in a safe state"
                        )
    args = parser.parse_args()

    test_case = args.test_case
    # Get location of studied agent
    studie_path = args.path
    lib_dir = "/Users/ericavandersar/Documents/Python_Projects/Research/mahrl_grid2op/"

    # chronics copied from test set in Snellius
    chronics = "0020  0047  0076  0129  0154  0164  0196  0230  0287  0332  0360  0391  0454  0504  0516  0539  0580  0614  0721  0770  0842  0868  0879  0925  0986 0023  0065  0103  0141  0156  0172  0206  0267  0292  0341  0369  0401  0474  0505  0529  0545  0595  0628  0757  0774  0844  0869  0891  0950  0993 0026  0066  0110  0144  0157  0179  0222  0274  0303  0348  0381  0417  0481  0511  0531  0547  0610  0636  0763  0779  0845  0870  0895  0954  0995 0030  0075  0128  0153  0162  0192  0228  0286  0319  0355  0387  0418  0486  0513  0533  0565  0612  0703  0766  0812  0852  0871  0924  0962  1000"
    test_chronics = chronics.split()
    reset_topo_todefault = args.reset_topo

    all_data, df, env = eval_single_agent(test_case, studie_path, reset_topo_todefault, lib_dir, test_chronics)
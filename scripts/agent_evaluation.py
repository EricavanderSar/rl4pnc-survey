import json
import os
import grid2op
import numpy as np
import argparse
import re
import json
from grid2op.PlotGrid import PlotMatplot
from lightsim2grid import LightSimBackend
from grid2op.Agent import DoNothingAgent, RecoPowerlineAgent
from grid2op.Runner import Runner
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm  # for easy progress bar
import pandas as pd
from grid2op.Episode import EpisodeData
from ray.rllib.algorithms import ppo
from rl4pnc.evaluation.evaluation_agents import RllibAgent, RhoGreedyAgent, HeuristicsAgent
from rl4pnc.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment
from rl4pnc.grid2op_env.custom_env2 import RlGrid2OpEnv
from rl4pnc.grid2op_env.utils import load_actions
from rl4pnc.experiments.yaml import load_config
from rl4pnc.evaluation.utils import instantiate_reward_class


def get_latest_checkpoint(agent_path):
    def extract_number(f):
        s = re.findall("\d+$", f)
        return (int(s[0]) if s else -1, f)
    # Get latest checkpoint:
    checkpoints = [name for name in os.listdir(agent_path) if name.startswith("checkpoint")]
    return max(checkpoints, key=extract_number)


# Custom function to handle non-serializable objects
def default_serializer(obj):
    try:
        return str(obj)  # Convert the object to a string representation
    except TypeError:
        return None  # Return None if conversion to string fails


def get_env_config(studie_path, test_case, rules, libdir, opponent=False):
    agent_path = os.path.join(studie_path, test_case)
    # load config
    config_path = os.path.join(agent_path, "params.json")
    config = load_config(config_path)
    env_config: dict = config["env_config"]
    # adjust lib_dir:
    env_config["lib_dir"] = libdir
    # Overwrite file params
    config["env_config"] = env_config
    with open(config_path, "w") as outfile:
        json.dump(config, outfile, indent=4)

    # change the env_name from "_train" to "" such that we can collect data for specified chronics.
    env_config["env_name"] = env_config["env_name"].replace("_train", "")
    # adjust reward class
    env_config["grid2op_kwargs"]["reward_class"] = instantiate_reward_class(
        env_config["grid2op_kwargs"]["reward_class"]
    )

    # Add rule based part:
    env_config["rules"] = rules
    print("Rules used for env are: ", rules)
    # Add agent_type:
    env_config["agent_type"] = test_case.split("_")[0]
    env_config["agent_id"] = "_".join(test_case.split("_")[:-2])
    env_config["train_opponent"] = ("kwargs_opponent" in env_config["grid2op_kwargs"])

    return env_config, agent_path


def run_agent(agent,
              store_trajectories_folder,
              env_config,
              chronics_name,
              libdir,
              opponent,
              ):
    if not os.path.exists(store_trajectories_folder):
        os.makedirs(store_trajectories_folder)
    print(f"Store results at: {store_trajectories_folder}.")
    # add opponent kwargs
    if opponent:
        print("Running with opponent.")
        opponent_path = os.path.join(libdir, f"configs/{env_config['env_name']}/opponent.yaml")
        opponent_kwargs = load_config(opponent_path)
    else:
        # Get kwargs for no opponent
        print("Running without opponent.")
        opponent_kwargs = grid2op.Opponent.get_kwargs_no_opponent()
    env_config["grid2op_kwargs"].update(opponent_kwargs)
    # save env_config to file
    with open(os.path.join(store_trajectories_folder, "env_config.json"), "w") as outfile:
        outfile.write(json.dumps(env_config, indent=4, default=default_serializer))

    # Select the chronics to be tested
    if chronics_name:
        env_chron = grid2op.make(f"{env_config['env_name']}_{chronics_name}")
    else:
        env_chron = grid2op.make(f"{env_config['env_name']}")
    chronics = [os.path.basename(d) for d in env_chron.chronics_handler.real_data.available_chronics()]

    # # case 14 chronics copied from test set in Snellius > FOR TESTING A limited set.
    # chronics = "0020  0047  0076  0129  0154  0164  "  # 0196  0230  0287  0332  0360  0391  0454  0504  0516  0539  0580  0614  0721  0770  0842  0868  0879  0925  0986 0023  0065  0103  0141  0156  0172  0206  0267  0292  0341  0369  0401  0474  0505  0529  0545  0595  0628  0757  0774  0844  0869  0891  0950  0993 0026  0066  0110  0144  0157  0179  0222  0274  0303  0348  0381  0417  0481  0511  0531  0547  0610  0636  0763  0779  0845  0870  0895  0954  0995 0030  0075  0128  0153  0162  0192  0228  0286  0319  0355  0387  0418  0486  0513  0533  0565  0612  0703  0766  0812  0852  0871  0924  0962  1000"
    # chronics = chronics.split()
    nb_episodes = len(chronics)
    del env_chron

    try:
        env = grid2op.make(env_config["env_name"], backend=LightSimBackend(), **env_config["grid2op_kwargs"])
    except:
        # try without LightSimBackend()
        print("NOTE: Not using LightSimBackend!")
        env = grid2op.make(env_config["env_name"], **env_config["grid2op_kwargs"])
    env.chronics_handler.set_chunk_size(100)
    li_episode = EpisodeData.list_episode(store_trajectories_folder) if \
        os.path.exists(store_trajectories_folder) else []

    if len(li_episode) < nb_episodes:
        print(f">> Start running evaluation of {nb_episodes-len(li_episode)} episodes")
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

        res_txt = f"{store_trajectories_folder}/results.txt"
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


def get_action_data(env, env_config, this_episode: EpisodeData, exec_times, input_data=None):
    # get data lines in overflow
    idx = env.observation_space.shape
    pos = env.observation_space.attr_list_vect.index('rho')
    start = sum(idx[:pos])
    end = start + idx[pos]
    rho_values = this_episode.get_observations()[0:this_episode.meta['nb_timestep_played']][..., np.arange(start, end)]
    ts_danger, line_danger = np.where(rho_values > env_config["activation_threshold"])
    rho = rho_values[rho_values > env_config["activation_threshold"]]

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
    subs_changed = [np.unique(env._topo_vect_to_sub[topo == 2]) if any(topo == 2) else [0] for topo in topos]
    sub_topo_depth = [len(changed) for changed in subs_changed]
    el_changed = [np.nonzero(topo == 2)[0] for topo in topos]
    el_topo_depth = [len(changed) for changed in el_changed]

    # line disconnections
    idx = env.observation_space.shape
    pos = env.observation_space.attr_list_vect.index('line_status')
    start = sum(idx[:pos])
    end = start + idx[pos]
    line_status = this_episode.get_observations()[ts_danger + 1][..., np.arange(start, end)]
    disc_lines = [np.where(status==False)[0] for status in line_status]

    # chronic ids
    chron_id = [this_episode.name for _ in ts_danger]

    # agent step duration
    agent_execution_times = exec_times[ts_danger]

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
        data["agent_exec_time"] = agent_execution_times
        data["lines_disc"] = disc_lines
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
        data["agent_exec_time"] = np.append(data["agent_exec_time"], agent_execution_times)
        data["lines_disc"].extend(disc_lines)
        # data["topo"] = np.append(data["topo"], topos)
    return data


def collect_episode_data(env, rules, store_trajectories_folder):
    print(" Start collecting episode data ... ")
    li_episode = EpisodeData.list_episode(store_trajectories_folder)
    act_data = None
    store_actdata_path = os.path.join(store_trajectories_folder, "line_action_topo_data.csv")
    store_surv_path = os.path.join(store_trajectories_folder, "survival.csv")
    df_act, df_sur = None, None
    chron = []
    surv = []
    rw = []
    mean_exec_times = []
    max_exec_times = []

    if os.path.exists(store_actdata_path):
        # check if part of the data was already collected...
        df_act = pd.read_csv(store_actdata_path)
        n_episode_evaluated = len(df_act.chron_id.unique())
        if os.path.exists(store_surv_path):
            df_sur = pd.read_csv(store_surv_path)
            if n_episode_evaluated == len(df_sur.chron.unique()):
                li_episode = li_episode[n_episode_evaluated:]
            else:
                df_act, df_sur = None, None
        else:
            df_act, df_sur = None, None

    # start collecting the data by going through the played episodes
    for ep in tqdm(li_episode, total=len(li_episode)):
        full_path, episode_studied = ep
        this_episode = EpisodeData.from_disk(store_trajectories_folder, episode_studied)
        agent_exec_times = np.array(np.load(os.path.join(store_trajectories_folder,
                                                episode_studied,
                                                this_episode.AG_EXEC_TIMES))["data"])
        # print("shape data: ", agent_exec_times.shape)
        # print(agent_exec_times)
        act_data = get_action_data(env, rules, this_episode, exec_times=agent_exec_times, input_data=act_data)
        # save chronic data
        chron.append(os.path.basename(os.path.normpath(this_episode.meta['chronics_path'])))
        surv.append(this_episode.meta['nb_timestep_played'])
        rw.append(np.round(this_episode.meta['cumulative_reward'], decimals=2))
        # Include mean and max execution time agent
        mean_exec_times.append(np.nanmean(agent_exec_times))
        max_exec_times.append(np.nanmax(agent_exec_times))


    # Save action data in data frame
    if df_act is not None:
        df_act = df_act.append(pd.DataFrame(act_data))
    else:
        df_act = pd.DataFrame(act_data)
    print(df_act.head())
    df_act.to_csv(os.path.join(store_trajectories_folder, "line_action_topo_data.csv"), index=False)

    # Save chronic data in data frame
    chron_data = {'chron': chron,
                  'survived': surv,
                  'cum reward': rw,
                  'mean exec time': mean_exec_times,
                  'max exec time': max_exec_times,
                  }
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


def quick_overview(data_folder, plot_topo_actions=True):
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

    # Plot overview of actions used. Not needed for DN agents
    if plot_topo_actions:
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


def eval_single_rlagent(test_case,
                        studie_path: str,
                        rules: dict,
                        opponent: bool,
                        lib_dir,
                        chronics,
                        unique_id,
                        ):
    # Get environment configuration from agent studied
    env_config, agent_path = get_env_config(studie_path, test_case, rules, lib_dir)
    ENV_TYPE = RlGrid2OpEnv if env_config["env_type"] == "new_env" else CustomizedGrid2OpEnvironment
    checkpoint_name = get_latest_checkpoint(agent_path)

    print(f"Studying agent at {agent_path}")
    print(f"Environment configuration {env_config}")
    # Create agent:
    env = grid2op.make(env_config["env_name"])
    agent = RllibAgent(
        action_space=env.action_space,
        env_config=env_config,
        file_path=agent_path,
        policy_name="reinforcement_learning_policy",
        checkpoint_name=checkpoint_name,
        gym_wrapper=ENV_TYPE(env_config),
    )
    # location of data to store:
    folder_name = "evaluation_episodes" + f"_{checkpoint_name}_{unique_id}"
    store_trajectories_folder = os.path.join(agent_path, folder_name)

    run_agent(agent,
              store_trajectories_folder,
              env_config,
              chronics,
              libdir=lib_dir,
              opponent=opponent)

    all_data, df = collect_episode_data(env, rules, store_trajectories_folder)
    quick_overview(store_trajectories_folder)
    return all_data, df, env


def eval_heuristic_agent(
        res_path: str,
        lib_dir: str,
        test_chronics: list,
        rules: dict,
        opponent: bool,
        unique_id: str,
):
    topo_actions = False
    # Load config
    config = load_config(os.path.join(lib_dir, "configs/l2rpn_case14_sandbox/heuristics_configs.yaml"))
    res_path = os.path.join(res_path, f"{config['agent_type']}_{unique_id}")
    # Add rule based part:
    config["rules"] = rules
    print("Rules used for env are: ", rules)
    # create agent
    env = grid2op.make(config["env_name"])
    if config["agent_type"] == "DoNothing":
        print("Evaluate DoNothingAgent")
        agent = HeuristicsAgent(env.action_space, config["rules"])
    elif config["agent_type"] == "RhoGreedy":
        topo_actions = True
        # res_path += f"/{config['action_space']}"
        print("Evaluate RhoGreedyAgent")
        actions_path = os.path.abspath(
            f"{lib_dir}/data/action_spaces/{config['env_name']}/{config['action_space']}.json",
        )
        possible_actions = load_actions(actions_path, env)
        agent = RhoGreedyAgent(
            env.action_space,
            config,
            possible_actions,
        )
    else:
        raise Exception(f"The agent_type {config['agent_type']} is not defined. Choose either")

    # run agent
    run_agent(
        agent,
        res_path,
        config,
        test_chronics,
        libdir=lib_dir,
        opponent=opponent,
    )

    # collect and save data
    all_data, df = collect_episode_data(env, rules, res_path)
    quick_overview(res_path, plot_topo_actions=topo_actions)
    return all_data, df, env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument("-a", "--agent_type", default="rl", choices=["heur", "rl"],
                        help="Agent type can be either heuristic (heur) or RL-based (rl).")

    parser.add_argument("-o", "--opponent", default=False, action='store_true')
    parser.add_argument("-c", "--chronics", default="test", choices=["test", "train", "val", ""],
                        help="Which part of the data set should be used for evaluation? "
                             "if empty  all data will be evaluated")

    parser.add_argument(
        "-p",
        "--path",
        # default="/Users/ericavandersar/Documents/Python_Projects/Research/Rl4Pnc/results/",
        default="/Users/ericavandersar/surfdrive/Documents/Research/Result/Case14_Sandbox_ActSpaces/",
        type=str,
        help="The path where the results will be stored AND in case of an RL-based agent "
             "the location of studied agent.",
    )

    parser.add_argument(
        "-l",
        "--lib_dir",
        default="/Users/ericavandersar/Documents/Python_Projects/Research/Rl4Pnc/",
        type=str,
        help="Path of the config file, for Greedy and DoNothing agents. N.A. for RL agents.",
    )

    parser.add_argument(
        "-t",
        "--test_case",
        default="CustomPPO_RlGrid2OpEnv_4e144_00000_0_2024-06-07_11-14-12",
        type=str,
        help="Name of the agent you want to evaluate. N.A. when using a heuristic like Do Nothing or Greedy",
    )

    parser.add_argument(
        "-j",
        "--job_id",
        type=str,
        default="Test_Reset",
        help="job_id of this evaluation, this way each evaluation gets an extra unique identifier.",
    )

    # rule based part:
    parser.add_argument("-at", "--activation_threshold", type=float, default=0.95)
    parser.add_argument('-lr', '--line_reco', default=False, action='store_true')
    parser.add_argument('-ld', '--line_disc', default=False, action='store_true')
    parser.add_argument("-rt", "--reset_topo", type=float, default=0.0,
                        help="Threshold to revert to reference topology. If max_rho<RT revert to ref topo.")
    parser.add_argument("-s", "--simulate", default=False, action='store_true',
                        help="If simulate is True: Simulate the proposed topo action before executing it.")

    args = parser.parse_args()

    test_case = args.test_case
    # Get location of studied agent
    studie_path = args.path
    lib_dir = args.lib_dir

    # Rule based configuration:
    rules = {"activation_threshold": args.activation_threshold,
             "line_reco": args.line_reco,
             "line_disc": args.line_disc,
             "reset_topo": args.reset_topo,
             "simulate": args.simulate}

    if args.agent_type == "rl":
        all_data, df, env = eval_single_rlagent(test_case,
                                                studie_path,
                                                rules,
                                                args.opponent,
                                                lib_dir,
                                                args.chronics,
                                                args.job_id,)
    else:
        all_data, df, env = eval_heuristic_agent(studie_path,
                                                 lib_dir,
                                                 args.chronics,
                                                 rules,
                                                 args.opponent,
                                                 args.job_id,
                                                 )

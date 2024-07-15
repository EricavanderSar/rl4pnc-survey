import json
import os

import grid2op
import pandas as pd
from tqdm import tqdm
import argparse
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool

from agent_evaluation import eval_single_agent
from mahrl.grid2op_env.utils import load_actions


def actions_per_agent(env_name, agents, main_folder, lib_dir, reset_topo=False):
    data_dict = {}
    for agent_name in agents:
        # Get agent code
        agent_code = agent_name.split("_")[2]

        # Get action space
        agent_path = os.path.join(main_folder, agent_name)
        with open(os.path.join(agent_path ,'params.json')) as json_file:
            agent_pars = json.load(json_file)
        act_space = agent_pars["env_config"]["action_space"]
        data_dict[agent_code] = {"action_space": act_space}

        # Timesteps survived
        if reset_topo:
            data_folder = os.path.join(agent_path, "evaluation_episodes_testreset")
        else:
            data_folder = os.path.join(agent_path, "evaluation_episodes")
        if os.path.exists(os.path.join(data_folder, "survival.csv")) and os.path.exists(os.path.join(data_folder, "line_action_topo_data.csv")):
            df_surv = pd.read_csv(os.path.join(data_folder, "survival.csv"))
            data_dict[agent_code]["mean_surv_ts"] = df_surv.survived.mean()


            # Size action space
            env = grid2op.make(env_name)
            path = os.path.join(
                lib_dir,
                f"data/action_spaces/{env_name}/{act_space}.json",
            )
            actions = load_actions(path, env)
            data_dict[agent_code]["n_actions"] = len(actions)

            # action frequency
            df = pd.read_csv(os.path.join(data_folder, "line_action_topo_data.csv"))
            df_sub_act = df.value_counts(["action_sub", "action_topo"]).reset_index(name='frequency')
            print(df_sub_act)
            for idx, row in df_sub_act.iterrows():
                data_dict[agent_code][f'sub{row["action_sub"]}_{row["action_topo"]}'] = row["frequency"]
            df_line_danger = df.value_counts("line_danger").reset_index(name='frequency')
            for idx, row in df_line_danger.iterrows():
                data_dict[agent_code][f'line_{row["line_danger"]}'] = row["frequency"]
            print(agent_name, data_dict[agent_code])
        else:
            print(f"Agent {agent_name} does not have required data. "
                  f"Either 'survival.csv' or 'line_action_topo_data.csv' is missing")
    new_df = pd.DataFrame.from_dict(data_dict, orient='index')
    print(new_df)
    # order columns:
    new_df = new_df[
        ['action_space', 'mean_surv_ts', 'n_actions']
        + sorted([col for col in new_df if col.startswith('sub')])
        + [col for col in new_df if col.startswith('line')]
        ]
    print(new_df)
    new_df.to_csv(os.path.join(main_folder, "agents_overview.csv"), index=False)
    return new_df


def eval_all_agents(env_name: str,
                    path: str,
                    lib_dir: str,
                    chron_list: list,
                    nb_workers: int,
                    reset_topo=False,
                    run_eval=False):
    # Get all agents in current directory
    agent_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    print("Collecting data for the following agents: ", agent_list)

    if run_eval:
        print("Run evaluation episodes first... ")
        with Pool(nb_workers) as pool:
            worker = partial(eval_single_agent,
                             studie_path=path,
                             reset_topo_todefault=reset_topo,
                             lib_dir=lib_dir,
                             test_chronics=chron_list)
            results = list(tqdm(pool.imap(worker, agent_list), total=len(agent_list), desc="Running evaluation for all agents"))
            for res in results:
                # Collect all data before continueing, hopefully avoiding error
                _, _, _ = res
    else:
        print("Save results evaluation in table")
        # Create overview table showing actions used per agent & survival of the agent.
        res_df = actions_per_agent(env_name, agent_list, path, lib_dir, reset_topo=reset_topo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-e",
        "--environment",
        default="l2rpn_case14_sandbox",
        # "l2rpn_wcci_2022", # "l2rpn_neurips_2020_track1_small", #"l2rpn_icaps_2021_small", #
        type=str,
        help="Name of the environment to be used.",
    )

    parser.add_argument(
        "-p",
        "--path",
        default="/Users/ericavandersar/surfdrive/Documents/Research/Result/Case14_Sandbox_ActSpaces/",
        type=str,
        help="The location of studied agents",
    )

    parser.add_argument(
        "-l",
        "--lib_dir",
        default="/Users/ericavandersar/Documents/Python_Projects/Research/mahrl_grid2op/",
        type=str,
        help="The directory of the python libary - to find the action spaces etc.",
    )

    parser.add_argument(
        "-w",
        "--nb_workers",
        default=8,
        type=int,
        help="Number of workers used to reduce the action space.",
    )

    parser.add_argument('-r', '--reset_topo', default=False, action='store_true',
                        help="reset topology when the environment is in a safe state"
                        )

    parser.add_argument('-re', '--run_evaluation', default=False, action='store_true',
                        help="run evaluation to collect the data if True, else put all data in one table"
                        )

    args = parser.parse_args()

    # Get location of studied agent
    studie_path = args.path

    env = grid2op.make(f"{args.environment}_test")
    # # chronics copied from test set in Snellius
    # chronics = "0020  0047  0076  0129  0154  0164  0196  0230  0287  0332  0360  0391  0454  0504  0516  0539  0580  0614  0721  0770  0842  0868  0879  0925  0986 0023  0065  0103  0141  0156  0172  0206  0267  0292  0341  0369  0401  0474  0505  0529  0545  0595  0628  0757  0774  0844  0869  0891  0950  0993 0026  0066  0110  0144  0157  0179  0222  0274  0303  0348  0381  0417  0481  0511  0531  0547  0610  0636  0763  0779  0845  0870  0895  0954  0995 0030  0075  0128  0153  0162  0192  0228  0286  0319  0355  0387  0418  0486  0513  0533  0565  0612  0703  0766  0812  0852  0871  0924  0962  1000"
    # test_chronics = chronics.split()
    test_chronics = [os.path.basename(d) for d in env.chronics_handler.real_data.available_chronics()]
    NB_EPISODE = len(test_chronics)
    eval_all_agents(args.environment,
                    args.path,
                    args.lib_dir,
                    test_chronics,
                    args.nb_workers,
                    reset_topo= args.reset_topo,
                    run_eval=args.run_evaluation)


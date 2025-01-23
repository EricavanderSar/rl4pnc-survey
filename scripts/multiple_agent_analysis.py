"""
This script is to test trained rl agents that are saved in the same folder.
"""

import json
import os

import grid2op
import pandas as pd
from tqdm import tqdm
import argparse
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool

from agent_evaluation import eval_single_rlagent
from rl4pnc.grid2op_env.utils import load_actions


def actions_per_agent(env_name, agents, main_folder, lib_dir, reset_topo=False):
    data_dict = {}
    for agent_name in agents:
        # Get agent code
        agent_code = agent_name.split("_")[2]

        # Get action space
        agent_path = os.path.join(main_folder, agent_name)
        with open(os.path.join(agent_path, 'params.json')) as json_file:
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
    new_df.to_csv(os.path.join(main_folder, "agents_overview.csv"), index=True)
    return new_df


def eval_all_agents(path: str,
                    input_rules: dict,
                    input_opponent: bool,
                    lib_dir: str,
                    chronics: list,
                    nb_workers: int,
                    job_id: str,
                    filter_name:str="",
                    best_checkpoint: bool = True):
    # Get all agents in current directory
    agent_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    # TODO REMOVE AFTER succesfully executed.
    agent_list = [
        agent for agent in agent_list
        if "evaluation_episodes_checkpoint_000009_9546143" not in os.listdir(os.path.join(path, agent))
    ]
    ############################################
    if filter_name:
        print(f"Filter only agents with *{filter_name}* in the name")
        agent_list = [name for name in agent_list if filter_name in name]
    print("Run evaluation for the following agents: ", agent_list)

    # if run_eval:
    print("Run evaluation episodes... ")
    with Pool(nb_workers) as pool:
        worker = partial(eval_single_rlagent,
                         studie_path=path,
                         rules=input_rules,
                         opponent=input_opponent,
                         lib_dir=lib_dir,
                         chronics=chronics,
                         unique_id=job_id,
                         best_checkpoint=best_checkpoint)
        results = list(tqdm(pool.imap(worker, agent_list), total=len(agent_list), desc="Running evaluation for all agents"))
        for res in results:
            # Collect all data before continueing, hopefully avoiding error
            _, _, _ = res
    # else:
    #     print("Save results evaluation in table")
    #     # Create overview table showing actions used per agent & survival of the agent.
    #     res_df = actions_per_agent(env_name, agent_list, path, lib_dir, reset_topo=reset_topo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process possible variables.")

    parser.add_argument(
        "-p",
        "--path",
        default="/Users/ericavandersar/surfdrive/Documents/Research/TestEval/",
        type=str,
        help="The location of studied agents",
    )

    parser.add_argument("-c", "--chronics", default="test", choices=["test", "train", "val", ""],
                        help="Which part of the data set should be used for evaluation? "
                             "if empty  all data will be evaluated")

    parser.add_argument(
        "-l",
        "--lib_dir",
        default="/Users/ericavandersar/Documents/Python_Projects/Research/Rl4Pnc/",
        type=str,
        help="The directory of the python libary - to find the action spaces etc.",
    )

    parser.add_argument(
        "-b",
        "--best_checkpoint",
        default=False,
        action='store_true',
        help="If True the best checkpoint is used for evaluation, otherwise the latest checkpoint is used.",
    )

    parser.add_argument(
        "-w",
        "--nb_workers",
        default=16,
        type=int,
        help="Number of workers used to reduce the action space.",
    )

    parser.add_argument(
        "-j",
        "--job_id",
        type=str,
        default="Test_multiple2",
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

    parser.add_argument("-o", "--opponent", default=False, action='store_true')
    parser.add_argument("-f", "--filter_name", type=str, default="",
                        help="optional filter for the agents, if *filter_name* is in agent folder name than "
                             "this agent will be included in analysis.")

    args = parser.parse_args()

    # Rule based configuration:
    rules = {"activation_threshold": args.activation_threshold,
             "line_reco": args.line_reco,
             "line_disc": args.line_disc,
             "reset_topo": args.reset_topo,
             "simulate": args.simulate}

    # Get location of studied agent
    studie_path = args.path

    # # chronics copied from test set in Snellius
    # chronics = "0020  0047  0076  0129  0154  0164  0196  0230  0287  0332  0360  0391  0454  0504  0516  0539  0580  0614  0721  0770  0842  0868  0879  0925  0986 0023  0065  0103  0141  0156  0172  0206  0267  0292  0341  0369  0401  0474  0505  0529  0545  0595  0628  0757  0774  0844  0869  0891  0950  0993 0026  0066  0110  0144  0157  0179  0222  0274  0303  0348  0381  0417  0481  0511  0531  0547  0610  0636  0763  0779  0845  0870  0895  0954  0995 0030  0075  0128  0153  0162  0192  0228  0286  0319  0355  0387  0418  0486  0513  0533  0565  0612  0703  0766  0812  0852  0871  0924  0962  1000"
    # test_chronics = chronics.split()
    eval_all_agents(args.path,
                    rules,
                    args.opponent,
                    args.lib_dir,
                    args.chronics,
                    args.nb_workers,
                    job_id=args.job_id,
                    filter_name=args.filter_name,
                    best_checkpoint=args.best_checkpoint
                    )

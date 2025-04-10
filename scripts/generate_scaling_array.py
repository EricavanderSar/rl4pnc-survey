import os
import argparse
import numpy as np
import time
import multiprocessing as mp

from lightsim2grid import LightSimBackend
import grid2op
from grid2op.Agent import DoNothingAgent
from grid2op.Runner import Runner
from grid2op.Parameters import Parameters

ENV_CASE = {
    '5': 'rte_case5_example',
    '14': 'l2rpn_case14_sandbox',
    '14_rel': 'rte_case14_realistic'
}

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str,
                    help="change the directory used for storing environment + data",
                    default="../data/scaling_arrays")
parser.add_argument("-a", "--all", default=True, action="store_true",
                    help="Collect data of ALL episodes")
parser.add_argument("-t", "--test", default=True, action="store_true",
                    help="Use test environments")
parser.add_argument("-e", "--env", type=str, default='14', choices=['14', '5', '14_rel'])
args = parser.parse_args()


class GatheredData:
    def __init__(self):
        # initialize data with empty dict
        self.data_dict = {}

    def gather_data(self, data_input, nb_time_step):
        new_data = {
            "p_or": np.array([data_input.observations[step].p_or for step in range(nb_time_step)]),
            "p_ex": np.array([data_input.observations[step].p_ex for step in range(nb_time_step)]),
            "load_p": np.array([data_input.observations[step].load_p for step in range(nb_time_step)]),
            "q_or": np.array([data_input.observations[step].q_or for step in range(nb_time_step)]),
            "q_ex": np.array([data_input.observations[step].q_ex for step in range(nb_time_step)]),
            "load_q": np.array([data_input.observations[step].load_q for step in range(nb_time_step)]),
            "gen_q": np.array([data_input.observations[step].gen_q for step in range(nb_time_step)]),
            "a_or": np.array([data_input.observations[step].a_or for step in range(nb_time_step)]),
            "a_ex": np.array([data_input.observations[step].a_ex for step in range(nb_time_step)]),
            "v_or": np.array([data_input.observations[step].v_or for step in range(nb_time_step)]),
            "v_ex": np.array([data_input.observations[step].v_ex for step in range(nb_time_step)]),
            "load_v": np.array([data_input.observations[step].load_v for step in range(nb_time_step)]),
            "gen_v": np.array([data_input.observations[step].gen_v for step in range(nb_time_step)]),
            "theta_ex": np.array([data_input.observations[step].theta_ex for step in range(nb_time_step)]),
            "theta_or": np.array([data_input.observations[step].theta_or for step in range(nb_time_step)]),
            "load_theta": np.array([data_input.observations[step].load_theta for step in range(nb_time_step)]),
            "gen_theta": np.array([data_input.observations[step].gen_theta for step in range(nb_time_step)]),
        }
        # p_ex = np.array([data_input.observations[step].p_ex for step in range(nb_time_step)])
        # load_p = np.array([data_input.observations[step].load_p for step in range(nb_time_step)])
        if not self.data_dict:
            for attr in new_data.keys():
                self.data_dict[attr] = {}
                self.data_dict[attr]["mean"] = np.mean(new_data[attr], axis=0)
                self.data_dict[attr]["max"] = np.max(new_data[attr], axis=0)
                self.data_dict[attr]["min"] = np.min(new_data[attr], axis=0)
        else:
            for attr in new_data.keys():
                self.data_dict[attr]["mean"] = np.vstack([self.data_dict[attr]["mean"],
                                                          np.mean(new_data[attr], axis=0)])
                self.data_dict[attr]["max"] = np.vstack([self.data_dict[attr]["max"],
                                                         np.max(new_data[attr], axis=0)])
                self.data_dict[attr]["min"] = np.vstack([self.data_dict[attr]["min"],
                                                         np.min(new_data[attr], axis=0)])

    def load_from_path(self, path):
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                if filename.startswith("all_"):
                    attr = filename.split("_")[1]
                    par = filename.split(".")[0].split("_")[-1]
                    file = os.path.join(root, filename)
                    self.data_dict[attr] = {}
                    self.data_dict[attr][par] = np.load(file)
        # return the number of episodes already done
        # print('Loaded data: ', self.data_dict)
        if "p_ex" in self.data_dict.keys():
            return len(self.data_dict["p_ex"]["mean"])
        else:
            return 0

    def save_data(self, path):
        for attr in self.data_dict.keys():
            for par in ["mean", "max", "min"]:
                # Save all mean, min and max PER EPISODE
                if os.path.exists(os.path.join(path, f"all_{attr}_{par}.npy")):
                    os.remove(os.path.join(path, f"all_{attr}_{par}.npy"))
                np.save(os.path.join(path, f"all_{attr}_{par}.npy"), self.data_dict[attr][par])

            # Save overall mean, min and max
            mean_all = np.mean(self.data_dict[attr]["mean"], axis=0)
            max_all = np.max(self.data_dict[attr]["max"], axis=0)
            min_all = np.min(self.data_dict[attr]["min"], axis=0)
            np.save(os.path.join(path, f"{attr}_mean.npy"), mean_all)
            np.save(os.path.join(path, f"{attr}.npy"), np.vstack([max_all, min_all]))
            print(f"Overall mean {attr}: {mean_all}")
            print(f"Overall max {attr}: {max_all}")
            print(f"Overall min {attr}: {min_all}")


if __name__ == "__main__":
    env_name = ENV_CASE[args.env]
    print("The current local directory where the environment are downloaded is \n{}"
          "".format(grid2op.get_current_local_dir()))
    path_save = os.path.join((args.dir if args.dir else "."), env_name)
    print("data will be saved in {}".format(path_save))

    if args.test:
        env_name += "_test"
    env = grid2op.make(env_name, backend=LightSimBackend())
    # Save the scaling arrays for gen_p (No need to collect data)
    np.save(os.path.join(path_save, "gen_p.npy"), np.vstack([env.gen_pmax / 1.2, env.gen_pmin]))

    # Make the object to save all gathered data
    gathered_data = GatheredData()
    # If some data has already been collected load the data to start with.
    if os.path.exists(path_save):
        EP_START = gathered_data.load_from_path(path_save)
    else:
        os.makedirs(path_save)
        EP_START = 0
    print(f"Starting from episode: {EP_START}")

    # Init multiprocessing.Pool()
    print("Using %d cores" % mp.cpu_count())
    tic = time.perf_counter()
    NB_CORE = mp.cpu_count()
    # Number of episodes in one iteration - This should not be too many otherwise program crashes
    # because of too much data in res
    NB_EP_IT = 5
    NB_EPISODE = len(env.chronics_handler.available_chronics()) if args.all else 5 + EP_START

    # Disable disconnection of lines (No game over) to collect all data
    new_param = Parameters()
    new_param.NO_OVERFLOW_DISCONNECTION = True
    env.change_parameters(new_param)
    obs = env.reset()

    # Make a runner for the agent
    runner = Runner(**env.get_params_for_runner(),
                    agentClass=DoNothingAgent
                    )
    # Path to store runner data:
    store_path = f"{grid2op.get_current_local_dir()}/{env_name}/all_ep_data/"
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    for i in range(EP_START, NB_EPISODE, NB_EP_IT):
        # define range of episodes to test
        ep_range = [chron_path.split("/")[-1] for chron_path in
                    env.chronics_handler.subpaths[i: min(i + NB_EP_IT, NB_EPISODE)]]
        # ep_range = ['%04d' % ep_i for ep_i in range(i, min(i+NB_EP_IT, NB_EPISODE))]

        res = runner.run(
            nb_episode=len(ep_range),
            nb_process=NB_CORE,
            episode_id=ep_range,
            pbar=True,
            add_detailed_output=True,
            path_save=store_path,
        )

        for _, chron_id, cum_reward, nb_time_step, max_ts, data in res:
            msg_tmp = "\n\tFor chronics with id {}\n".format(chron_id)
            msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
            msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
            print(msg_tmp)
            gathered_data.gather_data(data, nb_time_step)

    gathered_data.save_data(path_save)

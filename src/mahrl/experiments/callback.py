"""
Implements callbacks.
"""

from typing import Any, Dict, Optional
from tabulate import tabulate
import os
import numpy as np

# from grid2op.Environment import BaseEnv
from ray.rllib.env import BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

from mahrl.grid2op_env.custom_env2 import RlGrid2OpEnv

class Style:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class CustomMetricsCallback(DefaultCallbacks):
    """Implements custom callbacks metric"""

    # def on_episode_start(self,
    #     *,
    #     worker: "RolloutWorker",
    #     base_env: BaseEnv,
    #     policies: Dict[str, Policy],
    #     episode: EpisodeV2,
    #     env_index: Optional[int] = None,
    #     **kwargs,
    # ) -> None:
    #     print('sub env [0]', base_env.get_sub_environments()[0])
    #     print('len sub envs: ', len(base_env.get_sub_environments()))
    #     env = base_env.get_sub_environments()[0]
    #     print('current start step is :', env.env_glop.current_obs.current_step)
    #     episode.custom_metrics["start_step"] = env.env_glop.current_obs.current_step

    def on_episode_end(
        self,
        *,
        episode: EpisodeV2,
        worker: Optional[RolloutWorker] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Policy] = None,
        env_index: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Halfs the episode length as rllib counts double.
        """
        agents_steps = {k: len(v) for k, v in episode._agent_reward_history.items()}

        episode.custom_metrics["corrected_ep_len"] = agents_steps["high_level_agent"]
        envs = base_env.get_sub_environments()
        if type(envs[0]) == RlGrid2OpEnv:
            grid2op_end = np.array([env.env_glop.current_obs.current_step for env in envs]).mean()
            # print('chron ID:', envs[0].env_glop.chronics_handler.get_id())
            chron_id = envs[0].env_glop.chronics_handler.get_name()
        else:
            grid2op_end = np.array([env.env_gym.init_env.current_obs.current_step for env in envs]).mean()
            # print('chron ID:', envs[0].env_glop.chronics_handler.get_id())
            chron_id = envs[0].env_gym.init_env.chronics_handler.get_name()

        episode.custom_metrics["grid2op_end"] = grid2op_end
        episode.media["chronic_id"] = chron_id

    def on_evaluate_end(
            self,
            *,
            algorithm: "Algorithm",
            evaluation_metrics: dict,
            **kwargs,
    ) -> None:
        print(Style.BOLD + " ----- EVALUATION METRICS -------- " + Style.END)
        # print(evaluation_metrics)
        data = evaluation_metrics["evaluation"]["sampler_results"]
        trial_id = "_".join(os.path.basename(algorithm._logdir).split('_')[:-3])
        head_len = 5 # only show the first #head_len chronics
        print(f" Showing results for the first {head_len} evaluated chronics:")
        overview = {
            "chronic_id": data["episode_media"]["chronic_id"][:head_len],
            "grid2op_end": data["custom_metrics"]["grid2op_end"][:head_len],
            "reward": data["hist_stats"]["episode_reward"][:head_len]}
        print(tabulate(overview, headers="keys",  tablefmt="rounded_grid"))

        data["custom_metrics"]["grid2op_end_min"] = np.int(np.min(overview["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_mean"] = np.int(np.mean(overview["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_max"] = np.int(np.max(overview["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_std"] = np.std(overview["grid2op_end"])
        del data["custom_metrics"]["grid2op_end"]
        del data["episode_media"]["chronic_id"]
        del data["custom_metrics"]["corrected_ep_len"]

        rw_mean = data["episode_reward_mean"]
        # print table
        headers = ["trial_id", "grid2op_end_mean", "grid2op_end_max", "grid2op_end_min", "reward"]
        table = [[trial_id, data["custom_metrics"]["grid2op_end_mean"], data["custom_metrics"]["grid2op_end_max"],
                  data["custom_metrics"]["grid2op_end_min"], rw_mean]]
        print(tabulate(table, headers, tablefmt="rounded_grid", floatfmt=".3f"))


    # def on_learn_on_batch(
    #         self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    #     ) -> None:
    #     breakpoint()

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict,
        **kwargs,
    ) -> None:
        # print(f' TRIAL RESOURCES {algorithm.trial_resources}')
        # print(f'ALL METRICS {result}')
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

        print(Style.BOLD + " ------ TRAIN METRICS -------" + Style.END)
        mean_grid2op_end = np.mean(result["custom_metrics"]["grid2op_end"])
        std_grid2op_end = np.var(result["custom_metrics"]["grid2op_end"])
        mean_episode_duration = np.mean(result["custom_metrics"]["corrected_ep_len"])
        result["custom_metrics"]["grid2op_end_mean"] = mean_grid2op_end
        result["custom_metrics"]["grid2op_end_std"] = std_grid2op_end
        result["custom_metrics"]["corrected_ep_len_mean"] = mean_episode_duration
        # Delete irrelevant data
        del result["custom_metrics"]["grid2op_end"]
        del result["custom_metrics"]["corrected_ep_len"]
        del result["episode_media"]["chronic_id"]
        trial_id = "_".join(os.path.basename(algorithm._logdir).split('_')[:-3])
        seconds = result["time_total_s"]

        # Print the table
        table = [[trial_id, algorithm.iteration, '%dmin %02ds' % (seconds / 60, seconds % 60),
                  result["timesteps_total"], np.int(mean_grid2op_end), np.int(mean_episode_duration),
                  result["episodes_this_iter"], result["sampler_results"]["episode_reward_mean"]]]
        headers = ["Trial name", "iter", "total time", "ts", "Mean Grid2Op End", "Mean Duration", "episodes_this_iter", "reward_mean"]
        print(tabulate(table, headers, tablefmt="rounded_grid", floatfmt=".3f"))

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
        episode.custom_metrics["grid2op_end"] = np.array([env.env_glop.current_obs.current_step for env in envs]).mean()
        # print('chron ID:', envs[0].env_glop.chronics_handler.get_id())
        if envs[0].evaluate:
            chron_id = envs[0].env_glop.chronics_handler.get_name()
            episode.media["chronic_id"] = chron_id
        # if "reinforcement_learning_agent" in agents_steps:
        #     episode.custom_metrics["RL_ep_len_pct"] = (
        #         agents_steps["reinforcement_learning_agent"]
        #         / agents_steps["high_level_agent"]
        #     )
        # else:
        #     episode.custom_metrics["RL_ep_len_pct"] = 0.0

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
        chronic_id = data["episode_media"]["chronic_id"]
        grid2op_end = data["custom_metrics"]["grid2op_end_mean"]
        ep_len_mean = data["custom_metrics"]["corrected_ep_len_mean"]
        rw_mean = data["episode_reward_mean"]
        # print table
        headers = ["trial_id", "chronic_id_list", "grid2op_end_mean", "corrected_ep_len_mean", "reward"]
        table = [[trial_id, chronic_id, grid2op_end, ep_len_mean, rw_mean]]
        print(tabulate(table, headers, tablefmt="rounded_grid"))


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

        print(Style.BOLD + " ------ TRAIN METRICS -------" + Style.END)
        mean_grid2op_end = result["custom_metrics"]["grid2op_end_mean"]
        mean_episode_duration = result["custom_metrics"]["corrected_ep_len_mean"]
        trial_id = "_".join(os.path.basename(algorithm._logdir).split('_')[:-3])
        seconds = result["time_total_s"]

        # Print the table
        table = [[trial_id, algorithm.iteration, '%dmin %02ds' % (seconds / 60, seconds % 60),
                  result["timesteps_total"], mean_grid2op_end, mean_episode_duration,
                  result["episodes_this_iter"], result["sampler_results"]["episode_reward_mean"]]]
        headers = ["Trial name", "iter", "total time", "ts", "Mean Grid2Op End", "Mean Duration", "episodes_this_iter", "reward_mean"]
        print(tabulate(table, headers, tablefmt="rounded_grid"))

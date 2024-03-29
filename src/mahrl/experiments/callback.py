"""
Implements callbacks.
"""

from typing import Any, Dict, Optional, List, Union
from tabulate import tabulate
import os
import numpy as np
import time

# from grid2op.Environment import BaseEnv
import ray
from ray.tune.experimental.output import TuneReporterBase, get_air_verbosity, _get_time_str, AIR_TABULATE_TABLEFMT, _get_trial_table_data
from ray.tune.experiment import Trial
from ray.air.integrations.wandb import WandbLoggerCallback

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
    # def __init__(self, log_level: int = 0):
    #     super().__init__()
    #     self.log_level = log_level

    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        self.log_level = algorithm.my_log_level

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
            grid2op_end = np.array([env.env_g2op.current_obs.current_step for env in envs]).mean()
            # print('chron ID:', envs[0].env_glop.chronics_handler.get_id())
            chron_id = envs[0].env_g2op.chronics_handler.get_name()
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
        data = evaluation_metrics["evaluation"]["sampler_results"]
        # Save summarized data
        data["custom_metrics"]["grid2op_end_min"] = np.int(np.min(data["custom_metrics"]["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_mean"] = np.int(np.mean(data["custom_metrics"]["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_max"] = np.int(np.max(data["custom_metrics"]["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_std"] = np.std(data["custom_metrics"]["grid2op_end"])
        # # Print specified logging level
        # if self.log_level:
        #     print(Style.BOLD + " ----- EVALUATION METRICS -------- " + Style.END)
        #     # print(evaluation_metrics)
        #     trial_id = "_".join(os.path.basename(algorithm._logdir).split('_')[:-3])
        #     rw_mean = data["episode_reward_mean"]
        #     # print table
        #     headers = ["trial_id", "grid2op_end_mean", "grid2op_end_max", "grid2op_end_min", "reward"]
        #     table = [[trial_id, data["custom_metrics"]["grid2op_end_mean"], data["custom_metrics"]["grid2op_end_max"],
        #               data["custom_metrics"]["grid2op_end_min"], rw_mean]]
        #     print(tabulate(table, headers, tablefmt="rounded_grid", floatfmt=".3f"))
        if self.log_level > 1:
            head_len = self.log_level # only show the first #head_len chronics
            print(f" Showing results for the first {head_len} evaluated chronics:")
            overview = {
                "chronic_id": data["episode_media"]["chronic_id"][:head_len],
                "grid2op_end": data["custom_metrics"]["grid2op_end"][:head_len],
                "reward": data["hist_stats"]["episode_reward"][:head_len]}
            print(tabulate(overview, headers="keys", tablefmt="rounded_grid"))
        # Delete irrelevant data
        del data["custom_metrics"]["grid2op_end"]
        del data["episode_media"]["chronic_id"]
        del data["custom_metrics"]["corrected_ep_len"]

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict,
        **kwargs,
    ) -> None:
        # print(f'ALL METRICS {result}')
        mean_grid2op_end = np.int(np.mean(result["custom_metrics"]["grid2op_end"]))
        std_grid2op_end = np.var(result["custom_metrics"]["grid2op_end"])
        mean_episode_duration = np.int(np.mean(result["custom_metrics"]["corrected_ep_len"]))
        result["custom_metrics"]["grid2op_end_mean"] = mean_grid2op_end
        result["custom_metrics"]["grid2op_end_std"] = std_grid2op_end
        result["custom_metrics"]["corrected_ep_len_mean"] = mean_episode_duration
        # Delete irrelevant data
        del result["custom_metrics"]["grid2op_end"]
        del result["custom_metrics"]["corrected_ep_len"]
        del result["episode_media"]["chronic_id"]
        del result["sampler_results"]["custom_metrics"]
        del result["evaluation"]["sampler_results"]["custom_metrics"]


class TuneCallback(TuneReporterBase):
    def __init__(
            self,
            log_level
    ):
        super().__init__(get_air_verbosity(0))
        self._start_end_verbosity = 0
        self.log_level = log_level

    def print_heartbeat(self, trials, *args, force: bool = False):
        if force or time.time() - self._last_heartbeat_time >= self._heartbeat_freq:
            self._print_heartbeat(trials, *args, force=force)
            self._last_heartbeat_time = time.time()

    def _print_heartbeat(self, trials, *sys_args, force: bool = False):
        result = list()
        # Trial status: 1 RUNNING | 7 PENDING
        result.append(self._get_overall_trial_progress_str(trials))
        # Current time: 2023-02-24 12:35:39 (running for 00:00:37.40)
        result.append(self._time_heartbeat_str)
        # Logical resource usage: 8.0/64 CPUs, 0/0 GPUs
        result.extend(sys_args)
        for line in result:
            print(line)

    def on_trial_result(
        self,
        iteration: int,
        trials: List[Trial],
        trial: Trial,
        result: Dict,
        **info,
    ):
        if self.log_level:
            print(Style.BOLD + " ------ TRAIL RESULTS -------" + Style.END)
            self._start_block(f"trial_{trial}_result_{result['training_iteration']}")
            curr_time_str, running_time_str = _get_time_str(self._start_time, time.time())
            print(
                f"{self._addressing_tmpl.format(trial)} "
                f"finished iteration {result['training_iteration']} "
                f"at {curr_time_str}. Total running time: " + running_time_str
            )
            # print intermediate results for trial:
            self._print_result(trial, result)

    def _print_result(self, trial, result: Optional[Dict] = None, force: bool = False):
        result = result or trial.last_result
        # skip for now since this is already printed after tuning... Perhaps move?
        trial_id = str(trial)
        seconds = result["time_total_s"]
        eval_res = result["evaluation"]
        train_res = result["custom_metrics"]
        # Print the table
        headers = ["trial name",
                   "iter",
                   "total time",
                   "ts",
                   "EVAL g2op end",
                   "EVAL reward",
                   "TRAIN g2op end",
                   "TRAIN ep duration",
                   "TRAIN reward",
                   "episodes_this_iter"]
        table = [[trial_id,
                  result['training_iteration'],
                  '%dmin %02ds' % (seconds / 60, seconds % 60),
                  result["timesteps_total"],
                  eval_res["custom_metrics"]["grid2op_end_mean"],
                  eval_res["episode_reward_mean"],
                  train_res["grid2op_end_mean"],
                  train_res["corrected_ep_len_mean"],
                  result["sampler_results"]["episode_reward_mean"],
                  result["episodes_this_iter"]]]
        print(tabulate(table, headers, tablefmt="rounded_grid", floatfmt=".3f"))

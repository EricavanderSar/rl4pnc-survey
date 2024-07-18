"""
Implements callbacks.
"""

from typing import Any, Dict, Optional, List, Union, Tuple
from tabulate import tabulate
import numpy as np
import time

# from grid2op.Environment import BaseEnv
from ray.tune.experimental.output import (
    TuneReporterBase,
    get_air_verbosity,
    _get_time_str,
    _current_best_trial,
)
from ray._private.dict import unflattened_lookup
from ray.tune.experiment import Trial

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
        Collect extra metrics such as:
         - grid2op episode length - RLlib counts extra steps because of high level agent.
         - chronic id.
        """
        agents_steps = {k: len(v) for k, v in episode._agent_reward_history.items()}

        episode.custom_metrics["corrected_ep_len"] = agents_steps["high_level_agent"]
        envs = base_env.get_sub_environments()
        grid2op_end = np.array([env.env_g2op.current_obs.current_step for env in envs]).mean()
        # print('chron ID:', envs[0].env_glop.chronics_handler.get_id())
        chron_id = envs[0].env_g2op.chronics_handler.get_name()
        episode.custom_metrics["grid2op_end"] = grid2op_end
        episode.media["chronic_id"] = chron_id

        # New extra metrics:
        interact_count = np.array([env.interact_count for env in envs]).mean()
        active_dn_count = np.array([env.active_dn_count for env in envs]).mean()
        reconnect_count = np.array([env.reconnect_count for env in envs]).mean()
        reset_count = np.array([env.reset_count for env in envs]).mean()

        episode.custom_metrics["interact_count"] = interact_count
        episode.custom_metrics["active_dn_count"] = active_dn_count
        episode.custom_metrics["reconnect_count"] = reconnect_count
        episode.custom_metrics["reset_count"] = reset_count

    def on_evaluate_end(
            self,
            *,
            algorithm: "Algorithm",
            evaluation_metrics: dict,
            **kwargs,
    ) -> None:
        data = evaluation_metrics["evaluation"]
        # Save summarized data
        data["custom_metrics"]["grid2op_end_min"] = int(np.min(data["custom_metrics"]["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_mean"] = int(np.mean(data["custom_metrics"]["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_max"] = int(np.max(data["custom_metrics"]["grid2op_end"]))
        data["custom_metrics"]["grid2op_end_std"] = np.std(data["custom_metrics"]["grid2op_end"])
        # Extra metrics:
        data["custom_metrics"]["interact_count"] = int(np.mean(data["custom_metrics"]["interact_count"])) 
        data["custom_metrics"]["active_dn_count"] = int(np.mean(data["custom_metrics"]["active_dn_count"]))  
        data["custom_metrics"]["reconnect_count"] = int(np.mean(data["custom_metrics"]["reconnect_count"]))  
        data["custom_metrics"]["reset_count"] = int(np.mean(data["custom_metrics"]["reset_count"]))     
        
        # data["custom_metrics"]["mean_agent_interact"] = np.mean(data["custom_metrics"]["agent_interactions"])
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
        del data["sampler_results"]

        del data["custom_metrics"]["interact_count"]
        del data["custom_metrics"]["active_dn_count"]
        del data["custom_metrics"]["reconnect_count"]
        del data["custom_metrics"]["reset_count"]
        # del data["custom_metrics"]["agent_interactions"]

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict,
        **kwargs,
    ) -> None:
        # print(f'ALL METRICS {result}')
        mean_grid2op_end = int(np.mean(result["custom_metrics"]["grid2op_end"]))
        std_grid2op_end = np.var(result["custom_metrics"]["grid2op_end"])
        mean_episode_duration = int(np.mean(result["custom_metrics"]["corrected_ep_len"]))
        result["custom_metrics"]["grid2op_end_mean"] = mean_grid2op_end
        result["custom_metrics"]["grid2op_end_std"] = std_grid2op_end
        result["custom_metrics"]["corrected_ep_len_mean"] = mean_episode_duration
        
        # Extra metrics:
        result["custom_metrics"]["mean_interact_count"] = int(np.mean(result["custom_metrics"]["interact_count"]))
        result["custom_metrics"]["mean_active_dn_count"] = int(np.mean(result["custom_metrics"]["active_dn_count"]))
        result["custom_metrics"]["mean_reconnect_count"] = int(np.mean(result["custom_metrics"]["reconnect_count"]))
        result["custom_metrics"]["mean_reset_count"] = int(np.mean(result["custom_metrics"]["reset_count"]))

        # Delete irrelevant data
        del result["custom_metrics"]["grid2op_end"]
        del result["custom_metrics"]["corrected_ep_len"]
        del result["episode_media"]["chronic_id"]
        # del result["custom_metrics"]["agent_interactions"]
        del result["sampler_results"]
        del result["custom_metrics"]["interact_count"]
        del result["custom_metrics"]["active_dn_count"]
        del result["custom_metrics"]["reconnect_count"]
        del result["custom_metrics"]["reset_count"]

    # def on_learn_on_batch(
    #     self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    # ) -> None:
    #     print(f"Policy {train_batch.as_multi_agent().policy_batches.keys()} batch size: {train_batch.count}")

    # def on_postprocess_trajectory(
    #         self,
    #         *,
    #         worker: "EnvRunner",
    #         episode: EpisodeV2,
    #         agent_id: "AgentID",
    #         policy_id: "PolicyID",
    #         policies: Dict["PolicyID", Policy],
    #         postprocessed_batch: SampleBatch,
    #         original_batches: Dict["AgentID", Tuple["PolicyID", Policy, SampleBatch]],
    #         **kwargs,
    # ) -> None:
    #     if "rein" in policy_id:
    #         print(f'agent id {agent_id}')
    #         pid, policy, batch = original_batches[agent_id]
    #         print(f'batch size : {batch.count}')


class TuneCallback(TuneReporterBase):
    def __init__(
            self,
            log_level: int,
            metric: str,
            mode: str = "max",
            heartbeat_freq: int = 30,
            eval_freq: int = 1,
    ):
        super().__init__(get_air_verbosity(0))
        self._start_end_verbosity = 1
        self._heartbeat_freq = heartbeat_freq
        self.log_level = log_level
        self._last_res_it = 0
        self._eval_freq = eval_freq
        self._metric = metric
        self._mode = mode
        self._best_trial = None

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
        # *** Current BEST TRIAL: 6c81141f  ***  | with SCORE: 267 found at TIMESTEP: 529
        current_best_trial, metric_val = _current_best_trial(
            trials, self._metric, self._mode
        )
        if current_best_trial:
            best_trial_str = f" *** Current BEST TRIAL: {current_best_trial.trial_id}  ***  |" \
                             f" with SCORE: " \
                             f"{unflattened_lookup(self._metric, current_best_trial.last_result)} " \
                             f" at TIMESTEP: {current_best_trial.last_result['timesteps_total']} "
            result.append(best_trial_str)
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
            # start printing after first evaluation
            if result['training_iteration'] % self._eval_freq == 0:
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
                self._last_res_it = result['training_iteration']

    def _print_result(self, trial: Trial, result: Optional[Dict] = None, force: bool = False):
        # print(f'ALL TRIAL METRICS {result}')
        result = result or trial.last_result
        # skip for now since this is already printed after tuning... Perhaps move?
        trial_id = str(trial)
        eval_res = result["evaluation"]
        train_res = result["custom_metrics"]
        # Print the table
        headers = ["trial name",
                   "iter",
                   "total time",
                   "ts",
                   # "agent_interactions",
                   "EVAL g2op_end",
                   "EVAL reward",
                   "TRAIN g2op_end",
                   "TRAIN ep_duration",
                   "TRAIN reward",
                   "episodes_this_iter"]
        table = [[trial_id,
                  result['training_iteration'],
                  _get_time_str(self._start_time, time.time())[1],
                  result["timesteps_total"],
                  # result["custom_metrics"]["total_agent_interact"],
                  eval_res["custom_metrics"]["grid2op_end_mean"],
                  eval_res["episode_reward_mean"],
                  train_res["grid2op_end_mean"],
                  train_res["corrected_ep_len_mean"],
                  result["episode_reward_mean"],
                  result["episodes_this_iter"]]]
        print(tabulate(table, headers, tablefmt="rounded_grid", floatfmt=".3f"))

"""
Implements callbacks.
"""
from typing import Any, Dict, Optional

from grid2op.Environment import BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class CustomMetricsCallback(DefaultCallbacks):
    """Implements custom callbacks metric"""

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
        # episode.custom_metrics["RL_ep_len_pct"] = (
        #     agents_steps["reinforcement_learning_agent"]
        #     / agents_steps["high_level_agent"]
        # )

    # def on_learn_on_batch(
    #         self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    #     ) -> None:
    #     breakpoint()
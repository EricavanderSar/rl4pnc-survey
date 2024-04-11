import logging
from typing import List, Optional, Union
import numpy as np
import os

from ray.rllib.evaluation.worker_set import WorkerSet
from ray.util.debug import log_once
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.ppo import LEARNER_RESULTS_KL_KEY
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
    SAMPLE_TIMER,
    ALL_MODULES,
)
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.utils.typing import ResultDict

from ray.rllib.policy.sample_batch import concat_samples, MultiAgentBatch
from ray.rllib.utils.typing import SampleBatchType

logger = logging.getLogger(__name__)


def custom_synchronous_parallel_sample(
    *,
    worker_set: WorkerSet,
    max_agent_steps: Optional[int] = None,
    max_env_steps: Optional[int] = None,
    concat: bool = True,
) -> Union[List[SampleBatchType], SampleBatchType]:
    """ ******* ADJUSTED VERSION OF RLlib BY EVDS **********
    In this version only the steps of the trainable agents are saved
    in the batches.

    ********************************************************
    Runs parallel and synchronous rollouts on all remote workers.

    Waits for all workers to return from the remote calls.

    If no remote workers exist (num_workers == 0), use the local worker
    for sampling.

    Alternatively to calling `worker.sample.remote()`, the user can provide a
    `remote_fn()`, which will be applied to the worker(s) instead.

    Args:
        worker_set: The WorkerSet to use for sampling.
        remote_fn: If provided, use `worker.apply.remote(remote_fn)` instead
            of `worker.sample.remote()` to generate the requests.
        max_agent_steps: Optional number of agent steps to be included in the
            final batch.
        max_env_steps: Optional number of environment steps to be included in the
            final batch.
        concat: Whether to concat all resulting batches at the end and return the
            concat'd batch.

    Returns:
        The list of collected sample batch types (one for each parallel
        rollout worker in the given `worker_set`).

    Examples:
        >>> # Define an RLlib Algorithm.
        >>> algorithm = ... # doctest: +SKIP
        >>> # 2 remote workers (num_workers=2):
        >>> batches = synchronous_parallel_sample(algorithm.workers) # doctest: +SKIP
        >>> print(len(batches)) # doctest: +SKIP
        2
        >>> print(batches[0]) # doctest: +SKIP
        SampleBatch(16: ['obs', 'actions', 'rewards', 'terminateds', 'truncateds'])
        >>> # 0 remote workers (num_workers=0): Using the local worker.
        >>> batches = synchronous_parallel_sample(algorithm.workers) # doctest: +SKIP
        >>> print(len(batches)) # doctest: +SKIP
        1
    """
    # Only allow one of `max_agent_steps` or `max_env_steps` to be defined.
    assert not (max_agent_steps is not None and max_env_steps is not None)
    agent_or_env_steps = 0
    max_agent_or_env_steps = max_agent_steps or max_env_steps or None
    all_sample_batches = []

    policies_to_train = worker_set.local_worker().get_policies_to_train()
    worker_set.local_worker()
    # Stop collecting batches as soon as one criterium is met.
    while (max_agent_or_env_steps is None and agent_or_env_steps == 0) or (
        max_agent_or_env_steps is not None
        and agent_or_env_steps < max_agent_or_env_steps
    ):
        # No remote workers in the set -> Use local worker for collecting
        # samples.
        if worker_set.num_remote_workers() <= 0:
            sample_batches = [worker_set.local_worker().sample()]
        # Loop over remote workers' `sample()` method in parallel.
        else:
            sample_batches = worker_set.foreach_worker(
                lambda w: w.sample(), local_worker=False, healthy_only=True
            )
            if worker_set.num_healthy_remote_workers() <= 0:
                # There is no point staying in this loop, since we will not be able to
                # get any new samples if we don't have any healthy remote workers left.
                break
        # Update our counters for the stopping criterion of the while loop.
        if max_agent_steps:
            new_batches = []
            for b in sample_batches:
                filtered_policy_batches = {pid: batch for pid, batch in b.policy_batches.items() if pid in policies_to_train}
                new_batches.append(MultiAgentBatch.wrap_as_needed(filtered_policy_batches, b.env_steps()))
            sample_batches = new_batches
            # print('sample_batches: ', sample_batches)

        for b in sample_batches:
            if max_agent_steps:
                agent_or_env_steps += b.agent_steps()
            else:
                agent_or_env_steps += b.env_steps()
        all_sample_batches.extend(sample_batches)
        # print("agent or env steps:", agent_or_env_steps)

    if concat is True:
        full_batch = concat_samples(all_sample_batches)
        # Discard collected incomplete episodes in episode mode.
        # if max_episodes is not None and episodes >= max_episodes:
        #    last_complete_ep_idx = len(full_batch) - full_batch[
        #        SampleBatch.DONES
        #    ].reverse().index(1)
        #    full_batch = full_batch.slice(0, last_complete_ep_idx)
        return full_batch
    else:
        return all_sample_batches


class CustomPPO(PPO):
    def __init__(
            self,
            config: Optional["AlgorithmConfig"] = None,
            env=None,  # deprecated arg
            logger_creator: Optional["Callable[[], Logger]"] = None,
            **kwargs,
    ):
        print("my_log_level: ", config["my_log_level"])
        self.my_log_level = config["my_log_level"]
        super().__init__(config, env, logger_creator, **kwargs)

    def training_step(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        with self._timers[SAMPLE_TIMER]:
            if self.config.count_steps_by == "agent_steps":
                train_batch = custom_synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_agent_steps=self.config.train_batch_size,
                )
            else:
                train_batch = custom_synchronous_parallel_sample(
                    worker_set=self.workers, max_env_steps=self.config.train_batch_size
                )
        # print("policies", train_batch.policy_batches.keys())
        # print("train_batch_size: ", train_batch.count)
        # print("agent_steps: ", train_batch.agent_steps())
        # print("env_steps : ", train_batch.env_steps())
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])
        # Train
        if self.config._enable_learner_api:
            # TODO (Kourosh) Clearly define what train_batch_size
            #  vs. sgd_minibatch_size and num_sgd_iter is in the config.
            # TODO (Kourosh) Do this inside the Learner so that we don't have to do
            #  this back and forth communication between driver and the remote
            #  learner actors.
            is_module_trainable = self.workers.local_worker().is_policy_to_train
            self.learner_group.set_is_module_trainable(is_module_trainable)
            train_results = self.learner_group.update(
                train_batch,
                minibatch_size=self.config.sgd_minibatch_size,
                num_iters=self.config.num_sgd_iter,
            )

        elif self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        if self.config._enable_learner_api:
            # The train results's loss keys are pids to their loss values. But we also
            # return a total_loss key at the same level as the pid keys. So we need to
            # subtract that to get the total set of pids to update.
            # TODO (Kourosh): We should also not be using train_results as a message
            #  passing medium to infer which policies to update. We could use
            #  policies_to_train variable that is given by the user to infer this.
            policies_to_update = set(train_results.keys()) - {ALL_MODULES}
        else:
            policies_to_update = list(train_results.keys())

        # TODO (Kourosh): num_grad_updates per each policy should be accessible via
        # train_results
        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: self.workers.local_worker().policy_map[pid].num_grad_updates
                for pid in policies_to_update
            },
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if self.workers.num_remote_workers() > 0:
                from_worker_or_learner_group = None
                if self.config._enable_learner_api:
                    # sync weights from learner_group to all rollout workers
                    from_worker_or_learner_group = self.learner_group
                self.workers.sync_weights(
                    from_worker_or_learner_group=from_worker_or_learner_group,
                    policies=policies_to_update,
                    global_vars=global_vars,
                )
            elif self.config._enable_learner_api:
                weights = self.learner_group.get_weights()
                self.workers.local_worker().set_weights(weights)

        if self.config._enable_learner_api:

            kl_dict = {}
            if self.config.use_kl_loss:
                for pid in policies_to_update:
                    kl = train_results[pid][LEARNER_RESULTS_KL_KEY]
                    kl_dict[pid] = kl
                    if np.isnan(kl):
                        logger.warning(
                            f"KL divergence for Module {pid} is non-finite, this will "
                            "likely destabilize your model and the training process. "
                            "Action(s) in a specific state have near-zero probability. "
                            "This can happen naturally in deterministic environments "
                            "where the optimal policy has zero mass for a specific "
                            "action. To fix this issue, consider setting `kl_coeff` to "
                            "0.0 or increasing `entropy_coeff` in your config."
                        )

            # triggers a special update method on RLOptimizer to update the KL values.
            additional_results = self.learner_group.additional_update(
                module_ids_to_update=policies_to_update,
                sampled_kl_values=kl_dict,
                timestep=self._counters[NUM_AGENT_STEPS_SAMPLED],
            )
            for pid, res in additional_results.items():
                train_results[pid].update(res)

            return train_results

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results

    @override(Algorithm)
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        # Checkpoint is provided as a local directory.
        # Restore from the checkpoint file or dir.

        checkpoint_info = get_checkpoint_info(checkpoint_dir)
        #EVDS: BugFix: When you use checkpoint trainable policies only you need to specify policy_ids to be only
        # the checkpointed policies.
        checkpoint_data = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info,
                                                                        policy_ids=checkpoint_info['policy_ids'])
        self.__setstate__(checkpoint_data)
        if self.config._enable_new_api_stack:
            learner_state_dir = os.path.join(checkpoint_dir, "learner")
            self.learner_group.load_state(learner_state_dir)

        # Call the `on_checkpoint_loaded` callback.
        self.callbacks.on_checkpoint_loaded(algorithm=self)

    # def postprocess_trajectory(self, sample_batch, other_agent_batches, agent_id, **kwargs):
    #     # Filter out samples that are not from the trainable policy
    #     print('PRE Post Process SAMPLE BATCH: ', sample_batch)
    #     policies_to_train = self.workers.local_worker().get_policies_to_train()
    #     sample_batch = sample_batch[sample_batch["policy_id"] in policies_to_train]
    #
    #     print('POST Post Process SAMPLE BATCH: ', sample_batch)
    #
    #     # Continue with the rest of the postprocessing logic
    #     return super().postprocess_trajectory(sample_batch, other_agent_batches, agent_id, **kwargs)
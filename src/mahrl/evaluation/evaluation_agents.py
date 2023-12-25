"""
Describes classes of agents that can be evaluated.
"""


import os
from collections import OrderedDict
from typing import Any

import numpy as np
from grid2op.Action import ActionSpace, BaseAction
from grid2op.Agent import BaseAgent, GreedyAgent
from grid2op.dtypes import dt_float
from grid2op.Observation import BaseObservation
from grid2op.Reward import BaseReward
from ray.rllib.algorithms import Algorithm

from mahrl.grid2op_env.custom_environment import CustomizedGrid2OpEnvironment


class RllibAgent(BaseAgent):
    """
    Class that runs a RLlib model in the Grid2Op environment.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        env_config: dict[str, Any],
        file_path: str,
        policy_name: str,
        algorithm: Algorithm,
        checkpoint_name: str,
    ):
        BaseAgent.__init__(self, action_space)

        # load PPO
        checkpoint_path = os.path.join(file_path, checkpoint_name)
        self._rllib_agent = algorithm.from_checkpoint(
            checkpoint_path, policy_ids=[policy_name]
        )

        # setup env
        self.gym_wrapper = CustomizedGrid2OpEnvironment(env_config)

        # setup threshold
        self.threshold = env_config["rho_threshold"]

    def act(
        self, observation: BaseObservation, reward: BaseReward, done: bool = False
    ) -> BaseAction:
        """
        Returns a grid2op action based on a RLlib observation.
        """
        # Grid2Op to RLlib observation
        gym_obs = self.gym_wrapper.env_gym.observation_space.to_gym(observation)
        gym_obs = OrderedDict(
            (k, gym_obs[k]) for k in self.gym_wrapper.observation_space.spaces
        )

        if np.max(gym_obs["rho"]) > self.threshold:
            # get action as int
            gym_act = self._rllib_agent.compute_single_action(
                gym_obs, policy_id="reinforcement_learning_policy"
            )
        else:
            gym_act = 0

        # convert Grid2Op action to RLlib
        return self.gym_wrapper.env_gym.action_space.from_gym(gym_act)


class TopologyGreedyAgent(GreedyAgent):
    """
    Defines the behaviour of a Greedy Agent that can perform topology changes based
    on a provided set of possible actions.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        env_config: dict[str, Any],
        possible_actions: list[BaseAction],
    ):
        GreedyAgent.__init__(self, action_space)
        self.tested_action: list[BaseAction] = []
        self.action_space = action_space
        self.possible_actions = possible_actions

        # setup threshold
        self.threshold = env_config["rho_threshold"]

    def act(
        self, observation: BaseObservation, reward: BaseReward, done: bool = False
    ) -> BaseAction:
        """
        By definition, all "greedy" agents are acting the same way. The only thing that can differentiate multiple
        agents is the actions that are tested.

        These actions are defined in the method :func:`._get_tested_action`. This :func:`.act` method implements the
        greedy logic: take the actions that maximizes the instantaneous reward on the simulated action.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by the bot / controller / agent.

        """
        # get all possible actions to be tested
        self.tested_action = self._get_tested_action(observation)

        # if the threshold is exceeded, act
        if np.max(observation.to_dict()["rho"]) > self.threshold:
            # simulate all possible actions and choose the best
            if len(self.tested_action) > 1:
                self.resulting_rewards = np.full(
                    shape=len(self.tested_action), fill_value=np.NaN, dtype=dt_float
                )
                for i, action in enumerate(self.tested_action):
                    (
                        _,
                        simul_reward,
                        _,
                        _,
                    ) = observation.simulate(action)
                    self.resulting_rewards[i] = simul_reward
                reward_idx = int(
                    np.argmax(self.resulting_rewards)
                )  # rewards.index(max(rewards))
                best_action = self.tested_action[reward_idx]
            else:
                best_action = self.tested_action[0]
        # if the threshold is not exceeded, do nothing
        else:
            best_action = self.tested_action[0]
        return best_action

    def _get_tested_action(self, observation: BaseObservation) -> list[BaseAction]:
        """
        Adds all possible actions to be tested.
        """
        if not self.tested_action:
            # add the do nothing
            res = [self.action_space({})]
            # add all possible actions
            res += self.possible_actions
            self.tested_action = res
        return self.tested_action

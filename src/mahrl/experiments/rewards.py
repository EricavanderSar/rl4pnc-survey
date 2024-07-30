"""
Specifies reward functions for experiments.
"""

import logging
from typing import Any, Optional

import numpy as np
from grid2op.Action.baseAction import BaseAction
from grid2op.dtypes import dt_float
from grid2op.Environment.baseEnv import BaseEnv
from grid2op.Reward import L2RPNReward
from grid2op.Reward.baseReward import BaseReward


class LossReward(BaseReward):
    """
    Taken from van der Sar, Zocca, and Bhulai (2023).
    Linear reward function that computes reward based on energy loss in the network.
    When > 10% loss neg. reward will be given. Otherwise positive.
    reward possible between -0.9 and 0.1
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        BaseReward.__init__(self, logger=None)
        self.reward_min = -1.0
        self.reward_illegal = -0.5
        self.reward_max = 1.0

    def initialize(self, env: BaseEnv) -> None:
        """
        Initializes reward, not implemented.
        """

    def __call__(
        self,
        action: BaseAction,
        env: BaseEnv,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Calls reward.
        """
        reward: float
        if has_error:
            if is_illegal or is_ambiguous:
                return self.reward_illegal
            if is_done:
                return self.reward_min
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        reward = (load_p.sum() / gen_p.sum() * 10.0 - 9.0) * 0.1  # avg ~ 0.01
        return reward


class LossRewardRescaled2(LossReward):
    """
    Taken from van der Sar, Zocca, and Bhulai (2023).
    With a small adjustment (not multiplied with 0.1 but devided if <0)
    Rewards possible between -1 and 1.
    """

    def __call__(
        self,
        action: BaseAction,
        env: BaseEnv,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Calls reward.
        """
        reward: float
        if has_error or is_done:
            return self.reward_min
        if is_illegal or is_ambiguous:
            return self.reward_illegal
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        reward = (load_p.sum() / gen_p.sum() * 10.0 - 9.0)
        if reward < 0:
            reward /= 9
        return reward


class LossRewardNew(LossReward):
    """
    Adjusted reward function also based on loss of power in the network.
    """

    def __call__(
        self,
        action: BaseAction,
        env: BaseEnv,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Calls reward.
        """
        reward: float
        if has_error or is_done:
            return self.reward_min
        if is_illegal or is_ambiguous:
            return self.reward_illegal
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        loss_ratio = load_p.sum() / gen_p.sum()
        reward = loss_ratio * 10.0 - 9.0
        if reward > 0:
            reward = 1 - (1 - (loss_ratio * 10 - 9)) ** 2
        else:
            reward = loss_ratio ** 2 - 1
        return reward


class ScaledL2RPNReward(L2RPNReward):
    """
    Scaled version of L2RPNReward such that the reward falls between 0 and 1.
    Additionally -0.5 is awarded for illegal actions. Taken from Manczak, Viebahn and Van Hoof (2023).
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        L2RPNReward.__init__(self, logger=None)
        self.reward_min: float = -0.5
        self.reward_illegal: float = -0.5
        self.reward_max: float = 1.0
        self.num_lines: int | None = None

    def initialize(self, env: BaseEnv) -> None:
        """
        Initializes the reward values and other variables.

        Args:
            env (Environment): The environment object.

        Returns:
            None
        """
        self.reward_min = -0.5
        self.reward_illegal = -0.5
        self.reward_max = 1.0
        self.num_lines = env.backend.n_line

    def __call__(
        self,
        action: BaseAction,
        env: BaseEnv,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        """
        Calculate the reward for a given action in the environment.

        Parameters:
        - action: The action taken in the environment.
        - env: The environment object.
        - has_error: Flag indicating if there was an error during the action execution.
        - is_done: Flag indicating if the episode is done.
        - is_illegal: Flag indicating if the action is illegal.
        - is_ambiguous: Flag indicating if the action is ambiguous.

        Returns:
        - res: The calculated reward value.
        """
        if not is_done and not has_error:
            line_cap = self.__get_lines_capacity_usage(env)
            if self.num_lines is not None:
                res = np.sum(line_cap) / self.num_lines
            else:
                raise ValueError("Number of lines is not set.")
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        # print(f"\t env.backend.get_line_flow(): {env.backend.get_line_flow()}")
        return res

    @staticmethod
    def __get_lines_capacity_usage(env: BaseEnv) -> Any:
        """
        Calculate the lines capacity usage score for the given environment.

        Parameters:
        env (object): The environment object.

        Returns:
        numpy.ndarray: The lines capacity usage score.
        """
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-1  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)

        min_value = np.minimum(relative_flow, dt_float(1.0))
        lines_capacity_usage_score = np.maximum(
            dt_float(1.0) - min_value**2, np.zeros(min_value.shape, dtype=dt_float)
        )
        return lines_capacity_usage_score


class AlphaZeroRW(BaseReward):
    """
        Implemented as described in M. Dorfer et al. (2022) - Power Grid Congestion Management via Topology
        Optimization with AlphaZero:
        "... a shaped reward based on the cumulative sum of all overflowing line loads, which the agent aims
        to minimize..."
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        BaseReward.__init__(self, logger=None)
        self.reward_min = -1.0
        self.reward_illegal = -0.5
        self.reward_max = 1.0

    def initialize(self, env: BaseEnv) -> None:
        """
        Initializes reward, not implemented.
        """

    def __call__(
            self,
            action: BaseAction,
            env: BaseEnv,
            has_error: bool,
            is_done: bool,
            is_illegal: bool,
            is_ambiguous: bool,
    ) -> float:
        """
        Calls reward.
        """
        rho_values = env.current_obs.rho
        rho_max = np.max(rho_values)
        if rho_max <= 1.0:
            # If ρ_max ≤ 1, i.e., there is currently no overflow, and line loads of all lines are within the
            # allowed bounds, u is calculated as
            u = max(rho_max - 0.5, 0)
        else:
            # In case of an overflow, i.e., when ρ_max > 1, u is computed as
            u = np.sum(rho_values[rho_values > 1] - 0.5)
        # r = exp (−u − 0.5 · n_offline), where n_offline is the number of lines which are currently offline as
        # a result of an overflow or agent’s actions
        reward = np.exp(-u - 0.5 * (env.n_line - np.sum(env.current_obs.line_status)) )
        return reward

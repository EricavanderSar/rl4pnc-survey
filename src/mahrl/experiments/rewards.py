"""
Specifies reward functions for experiments.
"""

import logging
from typing import Optional

from grid2op.Action.baseAction import BaseAction
from grid2op.Environment.baseEnv import BaseEnv
from grid2op.Reward.baseReward import BaseReward


class LossReward(BaseReward):
    """
    Taken from van der Sar, Zocca, and Bhulai (2023).
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

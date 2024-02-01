"""
Extending grid2op's opponent space to reconnect the line after attack is finished.
Copied from Davide Barbieri's GridMind.
"""

from typing import Any

from grid2op.Action import ActionSpace as BaseActionSpace
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Observation.baseObservation import BaseObservation
from grid2op.Opponent.BaseOpponent import BaseOpponent
from grid2op.Opponent.OpponentSpace import OpponentSpace


class ReconnectingOpponentSpace(OpponentSpace):
    """
    Extends grid2op.Opponent.OpponentSpace.OpponentSpace class
    to reconnect the lines that have been disconnected after
    action has finished running
    """

    # TODO: Not plugged into the rest of the code yet

    def __init__(
        self,
        compute_budget: float,
        init_budget: float,
        opponent: BaseOpponent,
        attack_duration: int,
        attack_cooldown: int,
        budget_per_timestep: float = 0.0,
        action_space: BaseActionSpace | None = None,
    ) -> None:
        """
        Extend constructor to initialize remedial actions for line reconnection
        """

        # Call OpponentSpace's constructor
        super().__init__(
            compute_budget,
            init_budget,
            opponent,
            attack_duration,
            attack_cooldown,
            budget_per_timestep,
            action_space,
        )

        # Set attack to not running
        self._attacked_line_id: None | int = None

        # Set remdial action to be used in the next iteration
        self._remedial_action: None | BaseAction = None

        # Initialize list of remedial actions
        self._remedial_actions: dict[int, BaseAction] = {}
        # print(f"{self}: OVERWRITE REMEDIALS from __init__")

        # if self.opponent._lines_ids is None:
        #     print(f"Opp: None: {self}.")
        # elif len(self.opponent._lines_ids) < 1:
        #     print("Opp: No lines in opponent config.")
        # elif len(self._remedial_actions) < 1:
        #     print("Opp: No lines in remedial actions.")

    def init_opponent(self, partial_env: BaseEnv, **kwargs: dict[Any, Any]) -> None:
        """
        Populate repair action map
        """
        # Call OpponentSpace's init method
        super().init_opponent(partial_env, **kwargs)

        # Create remedial actions
        # TODO this does not work for all opponents
        # print(f"lines_ids={self.opponent._lines_ids}")
        self._remedial_actions = {
            line_id: self.action_space({"set_line_status": (line_id, 1)})
            for line_id in self.opponent._lines_ids  # pylint: disable=protected-access
        }
        # print(
        #     f"{self}: init_opponent gets called: init remedial {self._remedial_actions}"
        # )

    def reset(self) -> None:
        """
        Reset attack running and remedial action
        """
        # print(f"{self}: RESET: {self._remedial_actions}")
        # Call OpponentSpace's reset
        super().reset()

        # Reset attack running
        self._attacked_line_id = None

        # Reset remdial action
        self._remedial_action = None

        # print(f"AFTER RESET: init remedial {self._remedial_actions}")

    def attack(
        self,
        observation: BaseObservation,
        agent_action: BaseAction,
        env_action: BaseAction,
    ) -> BaseAction:
        """
        Override attack to immediately reconnect attacked line after
        the attack has finished
        """

        # Run super method
        attack, duration = super().attack(observation, agent_action, env_action)
        # print(f"duration: {duration}")

        # Check if attack is running
        if attack is not None and duration > 0:
            # Extract line id from attack action
            self._attacked_line_id = int(attack.set_line_status.argmin())
            # print(
            #     f"{self}: IDattack={self._attacked_line_id}, RemActsOptions={self._remedial_actions}"
            # )
        # else:
        #     print(f"{self}: No attack running, RemActsOptions={self._remedial_actions}")

        # Check if attack has just stopped
        if self._attacked_line_id is not None and duration < 2:
            # Set remedial action to be used in the next time-step
            # print(f"{self}: IDattack={self._attacked_line_id}")
            # print(f"{self}: possible={self._remedial_actions}")
            self._remedial_action = self._remedial_actions[self._attacked_line_id]

            # Set attack as not running anymore
            self._attacked_line_id = None

        # Check if remedial action is available
        if attack is None and self._remedial_action is not None:
            # Return remedial action
            attack = self._remedial_action
            duration = 1

            # Reset remedial action
            self._remedial_action = None

        return attack, duration

    def _get_state(
        self,
    ) -> tuple[
        tuple[
            float,
            bool,
            int,
            int,
            BaseAction,
            BaseAction | None,
            dict[int, BaseAction],
            int | None,
        ],
        None,
    ]:
        """
        Gets state for simulation and deep copy.
        """
        # used for simulate
        state_me = (
            self.budget,
            self.previous_fails,
            self.current_attack_duration,
            self.current_attack_cooldown,
            self.last_attack,
            self._remedial_action,
            self._remedial_actions,
            self._attacked_line_id,
        )

        state_opp = self.opponent.get_state()
        return state_me, state_opp

    def _set_state(
        self,
        my_state: tuple[
            float,
            bool,
            int,
            int,
            BaseAction,
            BaseAction | None,
            dict[int, BaseAction],
            int | None,
        ],
        opp_state: None = None,
    ) -> None:
        """
        Sets state for simulation and deep copy.
        """

        # used for simulate (and for deep copy)
        if opp_state is not None:
            self.opponent.set_state(opp_state)
        (
            budget,
            previous_fails,
            current_attack_duration,
            current_attack_cooldown,
            last_attack,
            remedial_action,
            remedial_actions,
            attacked_line_id,
        ) = my_state
        self.budget = budget
        self.previous_fails = previous_fails
        self.current_attack_duration = current_attack_duration
        self.current_attack_cooldown = current_attack_cooldown
        self.last_attack = last_attack
        self._remedial_action = remedial_action
        self._remedial_actions = remedial_actions
        self._attacked_line_id = attacked_line_id

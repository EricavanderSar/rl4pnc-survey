"""
Extending grid2op's opponent space to reconnect the line after attack is finished.
Copied from Davide Barbieri's GridMind.
"""

from typing import Any

from grid2op.Action import ActionSpace as BaseActionSpace
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Observation.baseObservation import BaseObservation
from grid2op.Opponent.baseOpponent import BaseOpponent
from grid2op.Opponent.opponentSpace import OpponentSpace


class ReconnectingOpponentSpace(OpponentSpace):
    """
    Extends grid2op.Opponent.OpponentSpace.OpponentSpace class
    to reconnect the lines that have been disconnected after
    action has finished running
    """

    def __init__(
        self,
        compute_budget: float,
        init_budget: float,
        opponent: BaseOpponent,
        attack_duration: int,
        attack_cooldown: int,
        budget_per_timestep: float = 0.0,
        action_space: BaseActionSpace | None = None,
        _local_dir_cls=None,
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

    def init_opponent(self, partial_env: BaseEnv, **kwargs: dict[Any, Any]) -> None:
        """
        Initialize the opponent agent.

        This method populates the repair action map and initializes the opponent agent.

        Args:
            partial_env (BaseEnv): The partial environment.
            **kwargs (dict): Additional keyword arguments.

        Returns:
            None
        """
        # Call OpponentSpace's init method
        super().init_opponent(partial_env, **kwargs)

        # Create remedial actions
        self._remedial_actions = {
            line_id: self.action_space({"set_line_status": (line_id, 1)})
            for line_id in self.opponent._lines_ids  # pylint: disable=protected-access
        }

    def reset(self) -> None:
        """
        Reset attack running and remedial action

        This method resets the attack running and remedial action of the opponent agent.
        It calls the `reset` method of the parent class `OpponentSpace` to reset the
        opponent's state. It also resets the attacked line ID and the remedial action.

        Parameters:
            None

        Returns:
            None
        """
        # Call OpponentSpace's reset
        super().reset()

        # Reset attack running
        self._attacked_line_id = None

        # Reset remedial action
        self._remedial_action = None

    def attack(
        self,
        observation: BaseObservation,
        agent_action: BaseAction,
        env_action: BaseAction,
    ) -> BaseAction:
        """
        Override attack to immediately reconnect attacked line after
        the attack has finished

        Args:
            observation (BaseObservation): The observation of the environment.
            agent_action (BaseAction): The action taken by the agent.
            env_action (BaseAction): The action taken by the environment.

        Returns:
            Tuple[BaseAction, int]: A tuple containing the attack action and its duration.
        """

        # Run super method
        attack, duration = super().attack(observation, agent_action, env_action)
        # print(f"duration: {duration}")

        # Check if attack is running
        if attack is not None and duration > 0:
            # Extract line id from attack action
            self._attacked_line_id = int(attack.set_line_status.argmin())

        # Check if attack has just stopped
        if self._attacked_line_id is not None and duration < 2:
            # Set remedial action to be used in the next time-step
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
        Gets the state for simulation and deep copy.

        Returns:
            tuple: A tuple containing the state information for the agent and the opponent.
                The agent's state is represented by a tuple with the following elements:
                    - budget (float): The current budget of the agent.
                    - previous_fails (bool): Indicates whether the agent had previous failures.
                    - current_attack_duration (int): The duration of the current attack.
                    - current_attack_cooldown (int): The cooldown period after the current attack.
                    - last_attack (BaseAction): The last attack performed by the agent.
                    - _remedial_action (BaseAction | None): The remedial action taken by the agent.
                    - _remedial_actions (dict[int, BaseAction]): Dictionary mapping line IDs to remedial actions.
                    - _attacked_line_id (int | None): The ID of the line being attacked by the agent.

                The opponent's state is obtained by calling the `get_state()` method of the opponent object.

            None: This method does not return any value for the opponent's state.

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
        Sets the state for the simulation and performs a deep copy.

        Args:
            my_state (tuple): A tuple containing the state information for the agent.
                The tuple should have the following elements:
                - budget (float): The budget of the agent.
                - previous_fails (bool): A flag indicating whether the agent has previous fails.
                - current_attack_duration (int): The duration of the current attack.
                - current_attack_cooldown (int): The cooldown of the current attack.
                - last_attack (BaseAction): The last attack performed by the agent.
                - remedial_action (BaseAction | None): The remedial action taken by the agent.
                - remedial_actions (dict[int, BaseAction]): A dictionary mapping line IDs to remedial actions.
                - attacked_line_id (int | None): The ID of the line being attacked.

            opp_state (None, optional): The state of the opponent agent. Defaults to None.
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
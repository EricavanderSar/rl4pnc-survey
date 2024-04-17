"""
Describes classes of agents that can be evaluated.
"""

import os
import random
from collections import Counter, OrderedDict
from typing import Any, Optional

import grid2op
import numpy as np
from grid2op.Action import ActionSpace, BaseAction
from grid2op.Agent import BaseAgent, GreedyAgent
from grid2op.dtypes import dt_float
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import GymEnv
from grid2op.Observation import BaseObservation
from grid2op.Reward import BaseReward
from ray.rllib.algorithms import Algorithm

from mahrl.experiments.utils import (
    calculate_action_space_asymmetry,
    calculate_action_space_medha,
    calculate_action_space_tennet,
    find_list_of_agents,
    find_substation_per_lines,
    get_capa_substation_id,
)


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
        gym_wrapper: GymEnv,
    ):
        BaseAgent.__init__(self, action_space)

        # load PPO
        checkpoint_path = os.path.join(file_path, checkpoint_name)
        self._rllib_agent = algorithm.from_checkpoint(
            checkpoint_path, policy_ids=[policy_name]
        )

        # setup env
        self.gym_wrapper = gym_wrapper

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
            action = self._rllib_agent.compute_single_action(
                gym_obs, policy_id="default_policy"
            )
            # convert Rllib action to grid2op
            return self.gym_wrapper.env_gym.action_space.from_gym(action)
        # else
        action = self.gym_wrapper.grid2op_env.action_space({})
        return action


class LargeTopologyGreedyAgent(GreedyAgent):
    """
    Defines the behaviour of a Greedy Agent that can perform topology changes based
    on a provided set of possible actions. Optimized for networks with a large hub.
    It does not search through all actions in the hub, but rather looks at all actions
    outside the hub. If any match the criteria (5% decrease or below threshold), that is
    taken, otherwise hub actions are randomized and tested with the same criteria.
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

        # Assuming self.tested_action is your list of dictionaries
        counts = Counter(
            action.as_dict()["set_bus_vect"]["modif_subs_id"][-1]
            for action in possible_actions
        )

        # Get a list of numbers that occur more than 1000 times
        # Split the list into two: one with frequent numbers and one without
        self.hub_actions: list[BaseAction] = []
        self.other_actions: list[BaseAction] = []

        for action in possible_actions:
            if counts[action.as_dict()["set_bus_vect"]["modif_subs_id"][-1]] > 1000:
                self.hub_actions.append(action)
            else:
                self.other_actions.append(action)

        # create the action space of all non-hub actions to be tested
        self.tested_action = [self.action_space({})] + self.other_actions

        # setup threshold
        self.threshold = env_config["rho_threshold"]

        random.seed(env_config["seed"])

        # self.timesteps_saved = 0

    def act(
        self,
        observation: BaseObservation,
        reward: Optional[BaseReward],
        done: Optional[bool] = False,
    ) -> BaseAction | None:
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
        # if the threshold is exceeded, act
        if np.max(observation.to_dict()["rho"]) > self.threshold:
            # first find the best action in all non-hub actions
            if len(self.tested_action) > 1:
                min_other_rho = np.inf
                best_other_action_idx = 0

                for idx, action in enumerate(self.tested_action):
                    simul_observation, _, _, simul_info = observation.simulate(action)
                    rho: float = np.max(simul_observation.to_dict()["rho"])
                    if not simul_info["exception"] and rho < min_other_rho:
                        min_other_rho = rho
                        best_other_action_idx = idx

                # if this best action passes the criteria, execute.
                # otherwise, randomize the hub actions and loop through those
                if (
                    min_other_rho < self.threshold
                    or np.max(observation.to_dict()["rho"]) - min_other_rho > 0.05
                ):
                    return self.tested_action[best_other_action_idx]

            # simulate randomized hub actions
            if len(self.hub_actions) > 1:
                min_hub_rho, best_hub_action_idx, action_found = self.check_hub_actions(
                    observation
                )

                if action_found is not None:
                    return action_found

                # Compare the minimum values and print the result
                if min_other_rho < min_hub_rho:
                    # Best action was found in non-hub actions
                    return self.tested_action[best_other_action_idx]

                # Best action was found in hub actions
                return self.hub_actions[best_hub_action_idx]

        # if the threshold is not exceeded, do nothing
        return self.tested_action[0]

    def check_hub_actions(
        self, observation: BaseObservation
    ) -> tuple[float, int, BaseAction | None]:
        """
        Returns the best hub action that was found, and executes it immediately when it's good enough.
        """
        min_hub_rho = np.inf
        best_hub_action_idx = 0

        # randomize tested_actions every time
        # create a list of indices
        indices = list(range(len(self.hub_actions)))

        # shuffle the indices
        random.shuffle(list(range(len(self.hub_actions))))

        # create a new list of shuffled actions
        shuffled_actions = [self.hub_actions[i] for i in indices]

        for idx, action in enumerate(shuffled_actions):
            (
                simul_observation,
                _,
                _,
                simul_info,
            ) = observation.simulate(action)
            rho: float = np.max(simul_observation.to_dict()["rho"])
            if not simul_info["exception"] and rho < min_hub_rho:
                min_hub_rho = rho
                best_hub_action_idx = indices[idx]

            # early stopping
            if (min_hub_rho < self.threshold) or (
                np.max(observation.to_dict()["rho"]) - min_hub_rho > 0.05
            ):  # 5% less load predicted or load below threshold
                return 0.0, 0, self.hub_actions[best_hub_action_idx]

        return min_hub_rho, best_hub_action_idx, None

    def _get_tested_action(self, observation: BaseObservation) -> list[BaseAction]:
        """
        Adds all possible actions to be tested.
        """
        raise NotImplementedError("Not implemented.")


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

        random.seed(env_config["seed"])

        self.timesteps_saved = 0

    def act(
        self,
        observation: BaseObservation,
        reward: Optional[BaseReward],
        done: Optional[bool] = False,
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
                resulting_rho_observations = np.full(
                    shape=len(self.tested_action), fill_value=np.NaN, dtype=dt_float
                )

                for i, action in enumerate(self.tested_action):
                    (
                        simul_observation,
                        _,
                        _,
                        simul_info,
                    ) = observation.simulate(action)
                    resulting_rho_observations[i] = np.max(
                        simul_observation.to_dict()["rho"]
                    )
                    # Include extra safeguard to prevent exception actions with converging powerflow
                    if simul_info["exception"]:
                        resulting_rho_observations[i] = 999999

                rho_idx = int(np.argmin(resulting_rho_observations))
                best_action = self.tested_action[rho_idx]

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


class CapaAndGreedyAgent(GreedyAgent):
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

        setup_env = grid2op.make(env_config["env_name"], **env_config["grid2op_kwargs"])

        # get changeable substations
        if env_config["action_space"] == "asymmetry":
            _, _, self.controllable_substations = calculate_action_space_asymmetry(
                setup_env
            )
        elif env_config["action_space"] == "medha":
            _, _, self.controllable_substations = calculate_action_space_medha(
                setup_env
            )
        elif env_config["action_space"] == "tennet":
            _, _, self.controllable_substations = calculate_action_space_tennet(
                setup_env
            )
        else:
            raise ValueError("No action valid space is defined.")

        # set up greedy agents
        self.agents = create_greedy_agent_per_substation(
            setup_env, env_config, self.controllable_substations, possible_actions
        )

        # extract line and substation information
        self.line_info = find_substation_per_lines(
            setup_env,
            list(find_list_of_agents(setup_env, env_config["action_space"]).keys()),
        )

        self.idx = 0
        self.substation_to_act_on: list[int] = []

    def act(
        self,
        observation: BaseObservation,
        reward: Optional[BaseReward],
        done: Optional[bool] = False,
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
        obs_batch = observation.to_dict()
        if np.max(obs_batch["rho"]) > self.threshold:
            # if no list is created yet, do so
            if self.idx == 0:
                self.substation_to_act_on = get_capa_substation_id(
                    self.line_info, obs_batch, self.controllable_substations
                )

            # find an action that is not the do nothing action by looping over the substations
            chosen_action = self.action_space({})
            while not chosen_action.as_dict() and self.idx < len(
                self.controllable_substations
            ):
                single_substation = self.substation_to_act_on[
                    self.idx % len(self.controllable_substations)
                ]

                self.idx += 1
                chosen_action = self.agents[single_substation].act(
                    observation, reward=None
                )

                # if it's not the do nothing action, return action
                # if it's the do nothing action, continue the loop
                if chosen_action.as_dict():
                    return chosen_action

        # grid is safe or no action is found, reset list count and return DoNothing
        self.idx = 0
        return self.action_space({})

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


class RandomAndGreedyAgent(GreedyAgent):
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

        setup_env = grid2op.make(env_config["env_name"], **env_config["grid2op_kwargs"])

        # get changeable substations
        if env_config["action_space"] == "asymmetry":
            _, _, controllable_substations = calculate_action_space_asymmetry(setup_env)
        elif env_config["action_space"] == "medha":
            _, _, controllable_substations = calculate_action_space_medha(setup_env)
        elif env_config["action_space"] == "tennet":
            _, _, controllable_substations = calculate_action_space_tennet(setup_env)
        else:
            raise ValueError("No action valid space is defined.")

        # set up greedy agents
        self.agents = create_greedy_agent_per_substation(
            setup_env, env_config, controllable_substations, possible_actions
        )

        # get changeable substations
        if env_config["action_space"] == "asymmetry":
            _, _, self.controllable_substations = calculate_action_space_asymmetry(
                setup_env
            )
        elif env_config["action_space"] == "medha":
            _, _, self.controllable_substations = calculate_action_space_medha(
                setup_env
            )
        elif env_config["action_space"] == "tennet":
            _, _, self.controllable_substations = calculate_action_space_tennet(
                setup_env
            )
        else:
            raise ValueError("No action valid space is defined.")

    def act(
        self,
        observation: BaseObservation,
        reward: Optional[BaseReward],
        done: Optional[bool] = False,
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
        substation_to_act_on = np.random.choice(self.controllable_substations)

        return self.agents[substation_to_act_on].act(observation, reward=None)

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


def get_actions_per_substation(
    controllable_substations: dict[int, int],
    possible_substation_actions: list[BaseAction],
) -> dict[int, list[BaseAction]]:
    """
    Get the actions per substation.
    """
    actions_per_substation: dict[int, list[BaseAction]] = {
        substation: [] for substation in list(controllable_substations.keys())
    }

    print(f"empty actions_per_substation {actions_per_substation}")

    # get possible actions related to that substation actions_per_substation
    for action in possible_substation_actions[1:]:  # exclude the DoNothing action
        sub_id = int(action.as_dict()["set_bus_vect"]["modif_subs_id"][-1])
        actions_per_substation[sub_id].append(action)

    print(f"action per substations")
    return actions_per_substation


def create_greedy_agent_per_substation(
    env: BaseEnv,
    env_config: dict[str, Any],
    controllable_substations: dict[int, int],
    possible_substation_actions: list[BaseAction],
) -> dict[int, TopologyGreedyAgent]:
    """
    Create a greedy agent for each substation.
    """
    actions_per_substation = get_actions_per_substation(
        controllable_substations, possible_substation_actions
    )

    # initialize greedy agents for all controllable substations
    agents = {}
    for sub_id in list(controllable_substations.keys()):
        agents[sub_id] = TopologyGreedyAgent(
            env.action_space, env_config, actions_per_substation[sub_id]
        )

    return agents

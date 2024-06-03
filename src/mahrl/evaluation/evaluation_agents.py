"""
Describes classes of agents that can be evaluated.
"""

# pylint: disable=all TODO
# flake8: noqa

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
from grid2op.gym_compat.gymenv import GymEnv_Modern
from grid2op.Observation import BaseObservation
from grid2op.Reward import BaseReward
from ray.rllib.algorithms import Algorithm

from mahrl.experiments.utils import (
    find_list_of_agents,
    find_substation_per_lines,
    get_capa_substation_id,
)
from mahrl.multi_agent.utils import argmax_logic, sample_logic


class SingleRllibAgent(BaseAgent):
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
        gym_wrapper: GymEnv_Modern,
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


class MultiRllibAgents(BaseAgent):
    """
    Class that runs a multi-agent RLlib model in the Grid2Op environment.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        env_config: dict[str, Any],
        file_path: str,
        lower_policy_name: str,
        middle_policy_name: str,
        algorithm: Algorithm,
        checkpoint_name: str,
        gym_wrapper: GymEnv_Modern,
    ):
        BaseAgent.__init__(self, action_space)

        self.lower_policy_name = lower_policy_name
        self.middle_policy_name = middle_policy_name
        self.policy_names: list[str] = []

        # load PPO
        checkpoint_path = os.path.join(file_path, checkpoint_name)

        if self.lower_policy_name == "rl":
            # find all folders in checkpoint path that start with reinforcement learning
            self.policy_names = [
                name
                for name in os.listdir(os.path.join(checkpoint_path, "policies"))
                if name.startswith("reinforcement_learning")
            ]
        elif self.lower_policy_name == "rl_v":
            self.policy_names = [
                name
                for name in os.listdir(os.path.join(checkpoint_path, "policies"))
                if name.startswith("value_reinforcement_learning")
                or name.startswith("value_function")
            ]

        if self.middle_policy_name not in ("capa", "random", "argmax", "sample"):
            # learned middle agent, also load this
            self.policy_names.append("choose_substation_policy")

        # load all the agents
        self._rllib_agents = algorithm.from_checkpoint(
            checkpoint_path, policy_ids=self.policy_names
        )

        # setup env
        self.gym_wrapper = gym_wrapper

        # setup threshold
        self.threshold = env_config["rho_threshold"]
        self.reconnect_line: list[BaseAction] = []

    def act(
        self, observation: BaseObservation, reward: BaseReward, done: bool = False
    ) -> BaseAction:
        """
        Returns a grid2op action based on a RLlib observation.
        """
        # TODO: Implement env extras such as automatic reconnection

        # Grid2Op to RLlib observation
        gym_obs = self.gym_wrapper.env_gym.observation_space.to_gym(observation)
        gym_obs = OrderedDict(
            (k, gym_obs[k]) for k in self.gym_wrapper.observation_space.spaces
        )

        # setup environment loop
        if np.max(gym_obs["rho"]) > self.threshold:
            # collect proposed actions
            proposed_actions = self.get_proposed_actions(gym_obs)

            # determine which agent has to act
            if self.middle_policy_name == "capa":
                assert self.lower_policy_name != "rl_v"
                # TODO: Implement CAPA based on CapaGreedy Agent
                raise NotImplementedError("Not implemented.")
            elif self.middle_policy_name == "random":
                assert self.lower_policy_name != "rl_v"
                # take a random item in self.policy_names and extract the last int
                sub_id = random.choice(self.policy_names).split("_")[-1]
            elif self.middle_policy_name == "rl":
                assert self.lower_policy_name != "rl_v"
                # give appropriate observation
                mid_rl_obs = OrderedDict(
                    {
                        "proposed_actions": proposed_actions,
                    }
                )
                sub_id = self._rllib_agents.compute_single_action(
                    mid_rl_obs, policy_id="choose_substation_policy"
                )

                # map Sub_ID to global
                sub_id = self.gym_wrapper.middle_to_substation_map[str(sub_id)]
            elif self.middle_policy_name == "rl_v":
                assert self.lower_policy_name == "rl_v"
                # collect proposed confidences as well
                proposed_confidences = self.get_proposed_confidences(gym_obs)

                # give appropriate observation
                mid_rlv_obs = OrderedDict(
                    {
                        "proposed_actions": proposed_actions,
                        "proposed_confidences": proposed_confidences,
                    }
                )
                sub_id = self._rllib_agents.compute_single_action(
                    mid_rlv_obs, policy_id="choose_substation_policy"
                )

                # map Sub_ID to global
                sub_id = self.gym_wrapper.middle_to_substation_map[str(sub_id)]

            elif self.middle_policy_name == "argmax":
                assert self.lower_policy_name == "rl_v"

                # collect proposed confidences
                proposed_confidences = self.get_proposed_confidences(gym_obs)

                # select sub_id with argmax
                sub_id = argmax_logic(proposed_confidences)
            elif self.middle_policy_name == "sample":
                assert self.lower_policy_name == "rl_v"
                # collect proposed confidences
                proposed_confidences = self.get_proposed_confidences(gym_obs)

                # sample sub_id
                sub_id = sample_logic(proposed_confidences)
            else:
                raise ValueError("Invalid middle policy name.")

            # call correct agent, get action as int
            policy_id = [
                name
                for name in self.policy_names
                if name.endswith(sub_id)
                and (
                    name.startswith("reinforcement_learning")
                    or name.startswith("value_reinforcement_learning")
                )
            ]
            assert len(policy_id) == 1

            # convert local action to global action through mapping
            action = self.gym_wrapper.local_to_global_action_map[sub_id][
                proposed_actions[sub_id]
            ]

            # convert Rllib action to grid2op
            g2op_action = self.gym_wrapper.env_gym.action_space.from_gym(action)
        else:
            # stable, do nothing
            g2op_action = self.gym_wrapper.grid2op_env.action_space({})

        # add reconnecting action, if needed
        if self.reconnect_line:
            for line in self.reconnect_line:
                g2op_action = g2op_action + line
            self.reconnect_line = []

        # find which lines to reconnect next iteration
        to_reco = ~observation.line_status
        if np.any(to_reco):  # TODO: Check if there's a difference between info and obs
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                reconnect_act = self.gym_wrapper.grid2op_env.action_space(
                    {"set_line_status": [(line_id, +1)]}
                )
                self.reconnect_line.append(reconnect_act)

        # print(f"Action: {g2op_action.to_dict()}")

        return g2op_action

    def get_proposed_confidences(self, gym_obs: dict[str, Any]) -> dict[str, float]:
        """
        Get the proposed confidences for each policy.

        Args:
            gym_obs (dict[str, Any]): The observation from the OpenAI Gym environment.

        Returns:
            dict[str, float]: A dictionary mapping policy names to their proposed confidences.
        """
        # collect proposed confidences
        proposed_confidences: dict[str, float] = {}
        for name in self.policy_names:
            if name.startswith("value_function"):
                proposed_confidences[name.split("_")[-1]] = float(
                    self._rllib_agents.compute_single_action(gym_obs, policy_id=name)
                )
        return proposed_confidences

    def get_proposed_actions(self, gym_obs: dict[str, Any]) -> dict[str, int]:
        """
        Get the proposed actions for the given gym observation.

        Args:
            gym_obs (dict[str, Any]): The gym observation.

        Returns:
            dict[str, int]: A dictionary mapping policy names to proposed actions.

        """
        # collect proposed actions
        proposed_actions: dict[str, int] = {}
        for name in self.policy_names:
            if name.startswith("reinforcement_learning") or name.startswith(
                "value_reinforcement_learning"
            ):
                proposed_actions[
                    name.split("_")[-1]
                ] = self._rllib_agents.compute_single_action(gym_obs, policy_id=name)
        return proposed_actions


# TODO: Make GreedyAgent with RL middle agent


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

        if "seed" in env_config:
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

        if "seed" in env_config:
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

        self.controllable_substations = find_list_of_agents(
            setup_env,
            env_config["action_space"],
        )

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
        self.substation_to_act_on: list[str] = []

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
            # TODO: Check if this still matches the policy implementation one-to-one

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
                chosen_action = self.agents[str(single_substation)].act(
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

        self.controllable_substations = find_list_of_agents(
            setup_env,
            env_config["action_space"],
        )

        # set up greedy agents
        self.agents = create_greedy_agent_per_substation(
            setup_env, env_config, self.controllable_substations, possible_actions
        )

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
        substation_to_act_on = np.random.choice(
            list(self.controllable_substations.keys())
        )

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
    possible_substation_actions: list[BaseAction],
    agent_per_substation: dict[str, int],
    add_dn_action_per_agent: bool = False,
) -> dict[str, list[BaseAction]]:
    """
    Get the actions per substation.
    """

    actions_per_substation: dict[str, list[BaseAction]] = {}

    # determine possible hubs
    hubs = {}

    # enumerate over dict to find the hub agent (over 1000 possible configurations)
    for sub_idx, num_actions in agent_per_substation.items():
        if num_actions > 1000:  # there exists a hub
            # determine how many agents and their average action size
            max_sub_actions = max(v for v in agent_per_substation.values() if v <= 1000)
            num_agents = num_actions // max_sub_actions
            avg_actions_per_hub = int(num_actions / num_agents)
            leftover_actions = num_actions % num_agents
            hubs[sub_idx] = {
                "avg_num_actions": avg_actions_per_hub,
                "num_agents": num_agents,
                "num_agents_extra_action": leftover_actions,
            }

    hub_count = 0
    action_count = 0
    # get possible actions related to that substation actions_per_substation
    for action in possible_substation_actions[1:]:  # exclude the DoNothing action
        sub_id = str(action.as_dict()["set_bus_vect"]["modif_subs_id"][-1])
        if sub_id in hubs:
            # move to next sub-agent, give some an extra action
            if hub_count < hubs[sub_id]["num_agents_extra_action"]:
                if action_count == hubs[sub_id]["avg_num_actions"] + 1:
                    hub_count += 1
                    action_count = 0
            else:
                if action_count == hubs[sub_id]["avg_num_actions"]:
                    hub_count += 1
                    action_count = 0

            if f"{sub_id}_{hub_count}" not in actions_per_substation:
                if add_dn_action_per_agent:
                    actions_per_substation[f"{sub_id}_{hub_count}"] = [
                        possible_substation_actions[0]
                    ]
                else:
                    actions_per_substation[f"{sub_id}_{hub_count}"] = []
            actions_per_substation[f"{sub_id}_{hub_count}"].append(action)
            action_count += 1
        else:  # treat normally
            if str(sub_id) not in actions_per_substation:
                if add_dn_action_per_agent:
                    actions_per_substation[str(sub_id)] = [
                        possible_substation_actions[0]
                    ]
                else:
                    actions_per_substation[str(sub_id)] = []

            actions_per_substation[str(sub_id)].append(action)

    return actions_per_substation


def create_greedy_agent_per_substation(
    env: BaseEnv,
    env_config: dict[str, Any],
    agent_per_substation: dict[str, int],
    possible_substation_actions: list[BaseAction],
) -> dict[str, TopologyGreedyAgent]:
    """
    Create a greedy agent for each substation.
    """
    actions_per_substation = get_actions_per_substation(
        possible_substation_actions, agent_per_substation
    )

    # initialize greedy agents for all controllable substations
    agents = {}
    for sub_id in list(agent_per_substation.keys()):
        agents[sub_id] = TopologyGreedyAgent(
            env.action_space, env_config, actions_per_substation[sub_id]
        )

    return agents

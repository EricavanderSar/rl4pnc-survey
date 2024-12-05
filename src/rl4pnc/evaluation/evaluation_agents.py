"""
Describes classes of agents that can be evaluated.
"""

import os
import random
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Optional
import pickle
from datetime import date, datetime

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
from ray.rllib.policy.policy import Policy

from rl4pnc.experiments.utils import (
    calculate_action_space_asymmetry,
    calculate_action_space_medha,
    calculate_action_space_tennet,
    find_list_of_agents,
    find_substation_per_lines,
    get_capa_substation_id,
)


class HeuristicsAgent(BaseAgent):
    """
    This agent executes heuristic rules based on the rule_config given.

    rule_config can contain:
    -   rho_threshold: float value. Activation threshold of the lower level agent.
    -   line_reco: Boolean value. If True: attempt to reconnect all disconnected power lines if
        this is beneficial for the max rho value.
    -   line_disc: Boolean value. If True: manually disconnect a line during sustained periods of
        overflow in order to avoid permanent damage. Reconnect the line back soon after the
        cooldown period ends.
    -   reset_topo: float value. Revert Threshold. If the max load rho < reset_topo, the agent
        will execute actions to revert to the reference topology.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        rule_config: dict,
    ):
        BaseAgent.__init__(self, action_space)
        self.activation_thresh = rule_config.get("activation_threshold", 0.95)
        self.line_reco = rule_config.get("line_reco", False)
        self.line_disc = rule_config.get("line_disc", False)
        self.ts_overflow_lines = np.zeros(action_space.n_line)
        self.reset_topo = rule_config.get("reset_topo", 0)
        self.simulate = rule_config.get("simulate", False)
        self.rho_max = 0

    def activate_agent(self, observation: BaseObservation):
        return self.rho_max > self.activation_thresh

    def act(self, observation: BaseObservation, reward: float, done : bool=False) -> BaseAction:
        current_action = self.action_space({})
        self.rho_max = (observation.rho.max() if observation.rho.max() > 0 else 2)
        if self.line_reco:
            current_action = self.reconnection_rule(observation, current_action)
        if self.reset_topo:
            current_action = self.revert_to_reference_topo(observation, current_action)
        if self.line_disc:
            current_action = self.disconnection_rule(observation, current_action=current_action)
        return current_action

    def reconnection_rule(self, observation: BaseObservation, current_action: BaseAction) -> BaseAction:
        """
        This methods reconnects all disconnected lines if this improves the current rho max values based on simulation.
        """
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        can_be_reco = ~line_stat_s & (cooldown == 0)
        if can_be_reco.any():
            (
                sim_obs,
                _,
                _,
                _,
            ) = observation.simulate(current_action)
            cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
            for id_ in (can_be_reco).nonzero()[0]:
                # reconnect all lines that improve the current action
                action = current_action + self.action_space({"set_line_status": [(id_, +1)]})
                (
                    sim_obs,
                    _,
                    _,
                    _,
                ) = observation.simulate(action)
                if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
                    current_action = action
        return current_action

    def revert_to_reference_topo(self, observation: BaseObservation, current_action: BaseAction) -> BaseAction:
        if (self.rho_max < self.reset_topo) and (observation.current_step < observation.max_step-1):
            # Get all subs that are not in default topology
            subs_changed = np.unique(observation._topo_vect_to_sub[observation.topo_vect != 1])
            if len(subs_changed):
                (
                    sim_obs,
                    _,
                    _,
                    _,
                ) = observation.simulate(current_action)
                cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
                # Simulate going back to reference topology for each substation that has changed
                action_options = []
                max_rhos = np.zeros(len(subs_changed))
                rewards = np.zeros(len(subs_changed))
                for i, sub in enumerate(subs_changed):
                    action = self.action_space(
                        {"set_bus": {
                            "substations_id":
                                [(sub, np.ones(observation.sub_info[sub], dtype=int))]
                        }
                        })
                    action_options.append(action)
                    sim_obs, rw, done, info = observation.simulate(current_action+action)
                    max_rhos[i] = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
                    rewards[i] = rw
                if max_rhos[np.argmax(rewards)] < cur_max_rho:
                    # add the best revert action.
                    current_action += action_options[np.argmax(rewards)]
                    # print(current_action)
        return current_action

    def disconnection_rule(self, observation: BaseObservation, current_action: BaseAction) -> BaseAction:
        # This method manually disconnect a line during sustained periods of overflow in order to avoid permanent
        # damage. Reconnect the line back soon after the cooldown period ends.
        # This can help when parameters.NB_TIMESTEP_RECONNECTION > parameters.NB_TIMESTEP_COOLDOWN_LINE
        if any(observation.rho>1.0):
            self.ts_overflow_lines= np.where(observation.rho>1.0, self.ts_overflow_lines + 1, 0)
            # print("Testing disconnection rule, ts overflowlines: ", self.ts_overflow_lines)
            # print("obs rho: ", observation.rho)
            if any(self.ts_overflow_lines>1):
                # Manually disconnet lines that are overflowed for more then 1 time step.
                id_ = np.argmax(self.ts_overflow_lines)
                current_action += self.action_space({"set_line_status": [(id_, -1)]})
                # print(current_action)
        else:
            self.ts_overflow_lines = np.zeros(self.action_space.n_line)

        return current_action

    def simulate_combinations(self,
                              observation: BaseObservation,
                              topo_action: BaseAction,
                              rb_action: BaseAction) -> BaseAction:
        comb_action = rb_action + topo_action
        if self.simulate:
            # Test if the proposed topo_action improves the result.
            sim_obs, _, _, _ = observation.simulate(rb_action)
            cur_max_rho = (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2)
            sim_obs, _, _, _ = observation.simulate(comb_action)
            if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
                # combined with the rule based action.
                action = comb_action
            else:
                # or excl the rule based action.
                sim_obs, _, _, _ = observation.simulate(topo_action)
                if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
                    action = topo_action
                else:
                    # Proposed topo_action is not better than rb_action only -> Take rule based action.
                    action = rb_action
        else:
            action = comb_action
        return action


class RenameUnpickler(pickle.Unpickler):
    # Make old agents compatable with evaluation function.
    # Rename module mahrl to rl4pnc
    def find_class(self, module, name):
        renamed_module = module
        # print(f"module: {module}")
        # print(f"name: {name}")
        if "mahrl" in module:
            renamed_module = "rl4pnc." + module.split(".",1)[1]
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


class RllibAgent(HeuristicsAgent):
    """
    Class that runs a RLlib model in the Grid2Op environment.

    """

    def __init__(
        self,
        action_space: ActionSpace,
        env_config: dict[str, Any],
        file_path: str,
        policy_name: str,
        checkpoint_name: str,
        gym_wrapper: GymEnv,
    ):
        HeuristicsAgent.__init__(self, action_space, env_config["rules"])

        # load neural network of (eg) PPO agent.
        checkpoint_path = os.path.join(file_path, checkpoint_name, "policies", policy_name)
        date_executed = datetime.strptime(file_path.rsplit("_", 2)[1], '%Y-%m-%d').date()
        if date_executed < date(2024, 11, 4):
            # update pickle file when old module was used
            pklfile = os.path.join(checkpoint_path, "policy_state.pkl")
            with open(pklfile, 'rb') as f:
                data = renamed_load(f)
            with open(pklfile, 'wb') as f:  # open a text file
                pickle.dump(data, f)  # serialize the list

        self._rllib_agent = Policy.from_checkpoint(checkpoint_path)
        # print("agent observation space is : ", self._rllib_agent.observation_space)
        # self._rllib_agent.observation_space =
        # self._rllib_agent = algorithm.from_checkpoint(
        #     checkpoint_path, policy_ids=[policy_name]
        # )

        # setup env
        self.gym_wrapper = gym_wrapper

    def act(
        self, observation: BaseObservation, reward: float, done: bool = False
    ) -> BaseAction:
        """
        Returns a grid2op action based on a RLlib observation.
        """

        # Grid2Op to RLlib observation
        self.gym_wrapper.update_obs(observation)
        # First do rule based part of the agent, line reconnections, disconnections and reverrt topo if needed.
        rb_action = HeuristicsAgent.act(self, observation, reward, done)

        if HeuristicsAgent.activate_agent(self, observation):
            # Get action from trained RL-agent when in danger.
            if not any(len(obs_el.shape) > 1 for obs_el in self.gym_wrapper.cur_gym_obs.values()):
                # Convert the observation dictionary to a NumPy array
                observation_array = np.concatenate([self.gym_wrapper.cur_gym_obs[key] for key in self.gym_wrapper.cur_gym_obs])
            else:
                observation_array = self.gym_wrapper.cur_gym_obs
            # print("current obs:", observation_array)
            # get action as int
            # compute_single_action returns:
            #   - Tuple consisting of the action,
            #   - the list of RNN state outputs (if any), and
            #   - a dictionary of extra features (if any).
            gym_action, state_out, info = self._rllib_agent.compute_single_action(
                observation_array,
                policy_id="reinforcement_learning_policy"
            )
            # convert Rllib action to grid2op
            topo_action = self.gym_wrapper.env_gym.action_space.from_gym(gym_action)
            action = HeuristicsAgent.simulate_combinations(self, observation, topo_action, rb_action)
        else:
            action = rb_action

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


class RhoGreedyAgent(HeuristicsAgent):
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
        HeuristicsAgent.__init__(self, action_space, env_config["rules"])
        self.tested_action: list[BaseAction] = []
        self.action_space = action_space
        self.possible_actions = possible_actions
        self.simulate = True

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
        # First do rule based part of the agent, line reconnections, disconnections and reverrt topo if needed.
        rb_action = HeuristicsAgent.act(self, observation, reward, done)
        # if the threshold is exceeded, act
        if HeuristicsAgent.activate_agent(self, observation):
            # get all possible actions to be tested
            self.tested_action = self._get_tested_action(observation)
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
                    ) = observation.simulate(action + rb_action)
                    resulting_rho_observations[i] = np.max(
                        simul_observation.to_dict()["rho"]
                    )
                    # Include extra safeguard to prevent exception actions with converging powerflow
                    if simul_info["exception"]:
                        resulting_rho_observations[i] = 999999
                # Get action with the lowest simulated rho_max(t+1)
                rho_idx = int(np.argmin(resulting_rho_observations))
                topo_action = self.tested_action[rho_idx]

                # Combine Greedy topology action with rb_action
                action = HeuristicsAgent.simulate_combinations(self, observation, topo_action, rb_action)
            else:
                action = rb_action
        # if the threshold is not exceeded, do nothing / no topology action.
        else:
            action = rb_action
        return action

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
    possible_substation_actions: list[BaseAction],
) -> defaultdict[int, list[BaseAction]]:
    """
    Get the actions per substation.
    """
    actions_per_substation = defaultdict(list)

    # get possible actions related to that substation actions_per_substation
    for action in possible_substation_actions:  # exclude the DoNothing action
        act_dict = action.as_dict()
        if "set_bus_vect" in act_dict.keys():
            sub_id = int(act_dict["set_bus_vect"]["modif_subs_id"][-1])
            actions_per_substation[sub_id].append(action)
        elif "change_bus_vect" in act_dict.keys():
            sub_id = int(act_dict["change_bus_vect"]["modif_subs_id"][-1])
            actions_per_substation[sub_id].append(action)

    # print(f"action per substations {[(key, len(item)) for key,item in actions_per_substation.items()]}")
    return actions_per_substation


def create_greedy_agent_per_substation(
    env: BaseEnv,
    env_config: dict[str, Any],
    controllable_substations: dict[int, int],
    possible_substation_actions: list[BaseAction],
) -> dict[int, RhoGreedyAgent]:
    """
    Create a greedy agent for each substation.
    """
    actions_per_substation = get_actions_per_substation(possible_substation_actions)

    # initialize greedy agents for all controllable substations
    agents = {}
    for sub_id in list(controllable_substations.keys()):
        agents[sub_id] = RhoGreedyAgent(
            env.action_space, env_config, actions_per_substation[sub_id]
        )

    return agents

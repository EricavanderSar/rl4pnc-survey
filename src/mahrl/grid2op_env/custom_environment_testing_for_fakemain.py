import random
from collections import OrderedDict
from typing import Any, TypeVar

import grid2op
import gymnasium
from grid2op.gym_compat import GymEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

from mahrl.grid2op_env.utils import (
    CustomDiscreteActions,
    get_possible_topologies,
    setup_converter,
)

CHANGEABLE_SUBSTATIONS = [0, 2, 3]

OBSTYPE = TypeVar("OBSTYPE")
ACTTYPE = TypeVar("ACTTYPE")
RENDERFRAME = TypeVar("RENDERFRAME")


class CustomizedGrid2OpEnvironment(MultiAgentEnv):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__()

        self._agents_list = [f"agent_{i}" for i in range(3)]
        self.agents = set(self._agents_list)
        self._agent_ids = [f"agent_{i}" for i in range(3)]

        # self._agent_ids = [
        #     "high_level_policy",
        #     "reinforcement_learning_policy",
        #     "do_nothing_policy",
        # ]
        # self._agents_list = [
        #     f"agent_{i}" for i in range(len(self._agent_ids))
        # ]  # NOTE ADDED
        # self.agents = set(self._agents_list)  # NOTE ADDED
        # self.reward_space = gymnasium.spaces.Box(
        #     low=0.0, high=1.0, shape=(), dtype=np.float32
        # )  # NOTE DUMMY ADDED

        # 1. create the grid2op environment
        if "env_name" not in env_config:
            raise RuntimeError(
                "The configuration for RLLIB should provide the env name"
            )
        # nm_env = env_config.pop("env_name", None)
        nm_env = env_config["env_name"]
        self.env_glop = grid2op.make(nm_env, **env_config["grid2op_kwargs"])

        # 1.a. Setting up custom action space
        possible_substation_actions = get_possible_topologies(
            self.env_glop, CHANGEABLE_SUBSTATIONS
        )

        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_glop)
        self.env_gym.reset()

        # 3. customize action and observation space space to only change bus
        # create converter
        converter = setup_converter(self.env_glop, possible_substation_actions)

        # set gym action space to discrete
        self.env_gym.action_space = CustomDiscreteActions(
            converter, self.env_glop.action_space()
        )

        # customize observation space
        ob_space = self.env_gym.observation_space
        ob_space = ob_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect", "p_or", "p_ex", "timestep_overflow"]
        )

        self.env_gym.observation_space = ob_space

        # 4. specific to rllib
        self.action_space = gymnasium.spaces.Discrete(len(possible_substation_actions))
        self.observation_space = gymnasium.spaces.Dict(
            dict(self.env_gym.observation_space.spaces.items())
        )
        # print(self.observation_space)
        # print(self.action_space)
        # print(env_config)

        self.previous_obs = OrderedDict()  # TODO: How to initalize?

    # def reset(
    #     self,
    #     *,
    #     seed: Optional[int] = None,
    #     options: Optional[dict] = None,
    # ) -> Tuple[MultiAgentDict, MultiAgentDict]:
    #     """
    #     This function resets the environment.
    #     """
    #     self.previous_obs, infos = self.env_gym.reset()
    #     observations = {"high_level_policy": self.previous_obs}
    #     return observations, infos
    def reset(self, seed=420, options=None):
        self.step_nb = 1
        obs = {agent_id: 1 for agent_id in self._agents_list}

        self.previous_obs, infos = self.env_gym.reset()  # NOTE ADDED
        return {"agent_1": self.previous_obs}, {}
        # return obs, {}

    # def step(
    #     self, action_dict: MultiAgentDict
    # ) -> Tuple[
    #     MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    # ]:
    #     """
    #     This function performs a single step in the environment.
    #     """
    #     if "high_level_policy" in action_dict.keys():
    #         # TODO change this to inside agent policy?
    #         action = action_dict["high_level_policy"]
    #         if action == "do_nothing_policy":
    #             # if np.max(self.previous_obs["rho"]) < RHO_THRESHOLD:
    #             # do nothing
    #             observations = {"do_nothing_policy": self.previous_obs}
    #             return observations, {}, {}, {}, {}
    #         if action == "reinforcement_learning_policy":
    #             # else:
    #             # do something
    #             observations = {"reinforcement_learning_policy": self.previous_obs}
    #             return observations, {}, {}, {}, {}
    #         raise NotImplementedError
    #     if "do_nothing_policy" in action_dict.keys():
    #         # do nothing
    #         action = action_dict["reinforcement_learning_policy"]
    #         (
    #             self.previous_obs,
    #             reward,
    #             terminated,
    #             truncated,
    #             info,
    #         ) = self.env_gym.step(action)
    #         # give reward to RL agent
    #         rewards = {"reinforcement_learning_policy": reward}
    #         observations = {"high_level_policy": self.previous_obs}
    #         terminateds = {"do_nothing_policy": terminated}
    #         truncateds = {"do_nothing_policy": truncated}
    #         infos = {"do_nothing_policy": info}
    #         return observations, rewards, terminateds, truncateds, infos
    #     if "reinforcement_learning_policy" in action_dict.keys():
    #         # perform action
    #         action = action_dict["reinforcement_learning_policy"]
    #         (
    #             self.previous_obs,
    #             reward,
    #             terminated,
    #             truncated,
    #             info,
    #         ) = self.env_gym.step(action)
    #         # give reward to RL agent
    #         rewards = {"reinforcement_learning_policy": reward}
    #         observations = {"high_level_policy": self.previous_obs}
    #         terminateds = {"reinforcement_learning_policy": terminated}
    #         truncateds = {"reinforcement_learning_policy": truncated}
    #         infos = {"reinforcement_learning_policy": info}
    #         return observations, rewards, terminateds, truncateds, infos
    #     # raise NotImplementedError
    #     observations = {"high_level_policy": self.previous_obs}
    #     return observations, {}, {}, {}, {}
    def step(self, action_dict):
        # Check which agent is going to act
        agent_mask = [
            (self.step_nb % (agent_idx + 1) == 0)
            for agent_idx in range(len(self._agents_list))
        ]

        # Construct info dict
        info = {
            "step": self.step_nb,
            "agent_mask": agent_mask,
        }
        infos = {
            agent_id: info
            for agent_can_act, agent_id in zip(agent_mask, self._agents_list)
            if agent_can_act
        }

        # Make each agent act
        rewards = {
            agent_id: 1.0
            for agent_can_act, agent_id in zip(agent_mask, self._agents_list)
            if agent_can_act
        }

        obs = {
            agent_id: 1
            for agent_can_act, agent_id in zip(agent_mask, self._agents_list)
            if agent_can_act
        }

        # Build termination dict
        terminateds = {
            "__all__": self.step_nb >= 1000,
        }

        truncateds = {
            "__all__": False,
        }

        # Increase step
        self.step_nb = self.step_nb + 1

        return (
            {"agent_1": self.previous_obs},
            {"agent_1": random.randint(0, 5)},
            terminateds,
            truncateds,
            {},
        )
        # return obs, rewards, terminateds, truncateds, infos

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError


register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)

from collections import OrderedDict
from typing import Any, Optional, Tuple, TypeVar

import grid2op
import gymnasium
from grid2op.gym_compat import GymEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from mahrl.grid2op_env.utils import (
    CustomDiscreteActions,
    get_possible_topologies,
    setup_converter,
)

# TODO: Investigate why all agents are always called

CHANGEABLE_SUBSTATIONS = [0, 2, 3]

OBSTYPE = TypeVar("OBSTYPE")
ACTTYPE = TypeVar("ACTTYPE")
RENDERFRAME = TypeVar("RENDERFRAME")


class CustomizedGrid2OpEnvironment(MultiAgentEnv):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__()

        self._agent_ids = [
            "high_level_agent",
            "reinforcement_learning_agent",
            "do_nothing_agent",
        ]

        # 1. create the grid2op environment
        if "env_name" not in env_config:
            raise RuntimeError(
                "The configuration for RLLIB should provide the env name"
            )
        nm_env = env_config["env_name"]
        self.env_glop = grid2op.make(nm_env, **env_config["grid2op_kwargs"])

        # 1.a. Setting up custom action space
        possible_substation_actions = get_possible_topologies(
            self.env_glop, CHANGEABLE_SUBSTATIONS
        )
        print(f"LEN ACTIONS={len(possible_substation_actions)}")
        # Add the do-nothing action at index 0
        do_nothing_action = self.env_glop.action_space({})
        possible_substation_actions.insert(0, do_nothing_action)

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

        self.previous_obs = OrderedDict()  # TODO: How to initalize?
        self.step_nb = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """
        This function resets the environment.
        """
        self.previous_obs, infos = self.env_gym.reset()
        observations = {"high_level_agent": self.previous_obs}
        return observations, infos

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """
        This function performs a single step in the environment.
        """
        # Increase step
        self.step_nb = self.step_nb + 1

        # Build termination dict
        terminateds = {
            "__all__": self.step_nb >= 100000,  # TODO ADJUST
        }

        truncateds = {
            "__all__": False,
        }

        print(f"ACTION_DICT = {action_dict}")

        if "high_level_agent" in action_dict.keys():
            action = action_dict["high_level_agent"]
            if action == 0:
                # do something
                print("high_level_agent SAYS: DO SOMETHING")
                observations = {"reinforcement_learning_agent": self.previous_obs}
                # rewards = {"reinforcement_learning_agent": 0}
                rewards = {}
                infos = {}
            elif action == 1:
                # if np.max(self.previous_obs["rho"]) < RHO_THRESHOLD:
                # do nothing
                print("AGENT 1 SAYS: DO NOTHING")
                observations = {"do_nothing_agent": self.previous_obs}
                # rewards = {"reinforcement_learning_agent": 0}
                rewards = {}
                infos = {}
            else:
                raise ValueError("A invalid agent is selected by the policy in step().")
        elif "do_nothing_agent" in action_dict.keys():
            # do nothing
            print("do_nothing_agent IS CALLED: DO NOTHING")
            # overwrite action in action_dict to nothing
            action = action_dict["do_nothing_agent"]
            (
                self.previous_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_gym.step(action)
            # give reward to RL agent
            rewards = {"reinforcement_learning_agent": reward}
            # rewards = {}
            observations = {"high_level_agent": self.previous_obs}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {}
        elif "reinforcement_learning_agent" in action_dict.keys():
            # perform action
            print("reinforcement_learning_agent IS CALLED: DO SOMETHING")
            action = action_dict["reinforcement_learning_agent"]
            (
                self.previous_obs,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env_gym.step(action)
            # give reward to RL agent
            rewards = {"reinforcement_learning_agent": reward}
            # rewards = {}
            observations = {"high_level_agent": self.previous_obs}
            terminateds = {"__all__": terminated}
            truncateds = {"__all__": truncated}
            infos = {}
        elif bool(action_dict) is False:
            print("Caution: Empty action dictionary!")
            # rewards = {"reinforcement_learning_agent": 0}
            rewards = {}
            # observations = {"high_level_agent": self.previous_obs}
            observations = {}
            infos = {}
        else:
            print(f"ACTION_DICT={action_dict}")
            raise ValueError("No agent found in action dictionary in step().")
        return observations, rewards, terminateds, truncateds, infos

    def render(self) -> RENDERFRAME | list[RENDERFRAME] | None:
        """
        Not implemented.
        """
        raise NotImplementedError


register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)

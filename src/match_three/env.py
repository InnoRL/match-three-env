"""JAX compatible version of the bandit environment from bsuite."""

from typing import Any, Dict, Optional, Tuple, Union


import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces

from game_grid import MatchThreeGameGridStruct

@struct.dataclass
class EnvState(environment.EnvState):
    rewards: Union[chex.Array, float]
    game_grid: MatchThreeGameGridStruct
    time: Union[float, chex.Array]


@struct.dataclass
class EnvParams(environment.EnvParams):
    grid_size: Tuple[int, int] = (9, 9)
    number_of_colors: int = 4
    max_steps_in_episode: int = 100

def calc_number_of_actions(env_params:EnvParams):
    g_h = env_params.grid_height
    g_w = env_params.grid_width
    
    # TODO: implement a proper function to calculate the number of actions
    return g_h * g_w

class MatchThree(environment.Environment[EnvState, EnvParams]):
    """
    JAX enviromnent for match three game.
    """

    def __init__(self):
        super().__init__()
        self.n_actions = calc_number_of_actions(self.default_params())

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        reward = state.rewards[action]
        state = EnvState(
            state.rewards,
            state.total_regret + params.optimal_return - reward,
            state.time + 1,
        )

        # Check game condition & no. steps for termination condition
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, Any]:  # dict]:
        """Reset environment state by sampling initial position."""
        action_mask = jax.random.choice(
            key,
            jnp.arange(self.num_actions),
            shape=(self.num_actions,),
            replace=False,
        )
        rewards = jnp.linspace(0, 1, self.num_actions)[action_mask]

        state = EnvState(rewards, 0.0, 0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state trafo."""
        return jnp.ones(shape=(1, 1), dtype=jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Episode always terminates after single step - Do not reset though!
        return jnp.array(True)

    @property
    def name(self) -> str:
        """Environment name."""
        return "SimpleBandit-bsuite"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(1, 1, (1, 1))

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "rewards": spaces.Box(0, 1, (self.num_actions,)),
                "total_regret": spaces.Box(0, params.max_steps_in_episode, ()),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

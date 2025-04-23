from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces

from match_three.game_grid import (
    MatchThreeGameGridFunctions,
    MatchThreeGameGridParams,
    MatchThreeGameGridStruct,
)


class GridSpace(spaces.Space):
    def __init__(
        self, grid_size: Tuple[int, int], grid_mask: chex.Array, n_colors: int
    ):
        assert 4 <= n_colors <= 7
        self.grid_size = grid_size
        self.grid_mask = grid_mask
        self.n_colors = n_colors

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        grid = jax.random.randint(
            key, self.grid_size, minval=1, maxval=self.n_colors + 1
        )
        grid = grid * self.grid_mask + jnp.full_like(grid) * jnp.logical_not(
            self.grid_mask
        )
        return grid

    def contains(self, grid: chex.Array) -> Any:
        grid_vals = grid * self.grid_mask
        grid_hols = grid * jnp.logical_not(self.grid_mask)

        # All cells with values must contain valid values representing colors.
        vals_correct = jnp.sum(0 < grid_vals <= self.n_colors) == jnp.sum(
            self.grid_mask
        )

        # All cells with holes must contain -1s.
        hols_correct = jnp.sum(grid_hols == -1) == (
            self.grid_size[0] * self.grid_size[1] - jnp.sum(self.grid_mask)
        )
        return jnp.logical_and(vals_correct, hols_correct)


@struct.dataclass
class EnvState(environment.EnvState):
    grid: MatchThreeGameGridStruct = MatchThreeGameGridStruct()
    time: int = 0


@struct.dataclass
class EnvParams(environment.EnvParams):
    grid_params: MatchThreeGameGridParams = MatchThreeGameGridParams()
    grid_size: Tuple[int, int] = (9, 9)
    max_steps_in_episode: int = 100


def get_action_space(params: EnvParams):
    grid_h = params.grid_size[0]
    grid_w = params.grid_size[1]
    return 2 * grid_h * grid_w - grid_h - grid_w


class MatchThree(environment.Environment[EnvState, EnvParams]):
    """JAX enviromnent for match three game."""

    def __init__(self, params: EnvParams = None):
        super().__init__()
        if params is None:
            params = EnvParams()
        self.n_actions = get_action_space(params)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        grid, matches = MatchThreeGameGridFunctions.apply_swap(key=key, state=state.grid, params=params.grid_params, grid_cell=action[:2], direction=action[2])

        reward = jnp.sum(matches)

        state = EnvState(grid, state.time + 1)
        done = self.is_terminal(state, params)
        # info = {"discount": self.discount(state, params)}
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            # info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by generating new grid."""
        key, grid = MatchThreeGameGridFunctions.generate_game_grid(
            key, params.grid_params
        )
        state = EnvState(grid=grid, time=0)
        return self.get_obs(state, params, key), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state."""
        return state.grid

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        return jnp.array(params.max_steps_in_episode < state.time)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Match-3"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_actions

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.n_actions)

    def observation_space(self, params: EnvParams) -> GridSpace:
        """Observation space of the environment."""
        return (GridSpace(params.grid_size, params.grid_mask, params.n_colors),)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "grid": GridSpace(params.grid_size, params.grid_mask, params.n_colors),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

from typing import Any, Dict, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax
from jax.tree_util import register_pytree_node_class

from match3_env.game_grid import (
    K_MAX,
    K_MIN,
    MatchThreeGameGridFunctions,
    MatchThreeGameGridParams,
    MatchThreeGameGridStruct,
)
from match3_env.utils import (
    conv_action_to_swap_continuous_jit,
    conv_action_to_swap_jit,
)

REWARD_MULTIPLIER = 1


class GridSpace(spaces.Space):
    def __init__(
        self, grid_size: Tuple[int, int], grid_mask: chex.Array, n_colors: int
    ):
        assert K_MIN <= n_colors <= K_MAX
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
        grid_cell, direction = conv_action_to_swap_jit(params.grid_size, action)
        grid, matches = MatchThreeGameGridFunctions.apply_swap(
            key=key,
            state=state.grid,
            params=params.grid_params,
            grid_cell=grid_cell,
            direction=direction,
        )

        reward = self._compute_reward(matches.matches)

        state = EnvState(grid=grid, time=state.time + 1)
        done = self.is_terminal(state, params)
        info = {
            "discount": self.discount(state, params),
            "matches": matches,
        }
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by generating new grid."""
        key, state_grid = MatchThreeGameGridFunctions.generate_game_grid(
            key, params.grid_params
        )
        state = EnvState(grid=state_grid, time=0)
        return self.get_obs(state, params, key), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state."""
        # return EnvObservation(
        #     grid=state.grid.grid, time=jnp.array([state.time], dtype=jnp.float32)
        # )
        return state.grid.grid

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

    @staticmethod
    def _compute_reward(matches: chex.Array) -> chex.Array:
        """Compute reward = sum(matches[i] * c * ln(i + 1)) for each i."""
        indices = jnp.arange(len(matches)) + 1  # 1-based indexing (avoid ln(0))
        log_weights = jnp.log2(indices)  # log2 or ln
        weighted_matches = matches * REWARD_MULTIPLIER * log_weights
        return jnp.sum(weighted_matches)


class MatchThreeContinuous(MatchThree):
    """Custom version of MatchThree with modified step_env behavior."""

    def __init__(self, params: EnvParams = None):
        super().__init__()
        if params is None:
            params = EnvParams()
        self.n_actions = 3  # vertical coordinate, horizontal coordinate, direction

    @property
    def name(self) -> str:
        """Environment name."""
        return "Match-3-Continuous"

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Box(low=0, high=1, shape=(self.n_actions,))

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        # Your custom implementation here
        # For example, let's modify the reward calculation

        # First copy the original step logic
        grid_cell, direction = conv_action_to_swap_continuous_jit(
            params.grid_size, action
        )
        grid, matches = MatchThreeGameGridFunctions.apply_swap(
            key=key,
            state=state.grid,
            params=params.grid_params,
            grid_cell=grid_cell,
            direction=direction,
        )

        # Custom reward calculation - example: simple count of matches
        reward = jnp.sum(matches.matches)  # Instead of the weighted sum

        state = EnvState(grid=grid, time=state.time + 1)
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

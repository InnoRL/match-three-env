from typing import Tuple, Union
import chex
from flax import struct
import jax
from jax import numpy as jnp

SWAP_DIRECTIONS = {
    0: (-1, 0),  # Swap with left neighbor
    1: (0, -1),  # Swap with top neighbor
    2: (1, 0),  # Swap with right neighbor
    3: (0, 1),  # Swap with bottom neighbor
    # Note: for that the coordinates are (y, x)
    # y is the vertical axis and x is the horizontal
    # The environment will use only directions 0 and 1
}

# TODO: properly set max depth for cascades
CASCADE_MAX_DEPTH = 10
MIN_MATCH_LENGTH = 3

# TODO: board should not have matches upon generating it
# TODO: when adding new elements to a board, the added elements should not have matches

# TODO: think about power-ups

# TODO: there should a second, non-accelerated version that can render the physics frame-by-frame
# This implies that there should be an interface which both
# jax version and renderer version implement
# the agents should be compatible with both versions.
# this probably implies that the env should also have two versions.


@struct.dataclass
class MatchResults:
    """
    Match Result is a 3D matrix where:

    - Dimension 0: Depth (number of match cascades)
    - Dimension 1: Number of matches at that depth
    - Dimension 2: Symbol type (e.g., which game pieces matched)

    The data type is `int`, compatible with JAX.
    """

    matches: chex.Array  # Shape: (depth, num_matches_per_depth, symbol_type)

    def initialize(grid_shape: Tuple, num_symbols: int) -> "MatchResults":
        max_possible_matches = jnp.prod(grid_shape)
        return MatchResults(
            matches=jnp.zeros(
                shape=(CASCADE_MAX_DEPTH, max_possible_matches, num_symbols),
                dtype=jnp.int32,
            )
        )


@struct.dataclass
class MatchThreeGameGridStruct:
    # a matrix representing the grid itself
    grid: chex.Array
    num_symbols: int


class MatchThreeGameGridFunctions:
    """A stateless functional class for Match-Three game grid operations."""

    # method to initialize a struct based on input
    @staticmethod
    def generate_game_grid(
        key: chex.PRNGKey, grid_size: Tuple[int, int], num_symbols: int
    ) -> Tuple[chex.PRNGKey, MatchThreeGameGridStruct]:
        key, subkey = jax.random.split(key)

    # method to swap based on (y,x) and swap direction -> new game board (can be the same if the swap was illeral)
    def apply_swap(
        self,
        key: chex.PRNGKey,
        state: MatchThreeGameGridStruct,
        grid_cell: chex.Array,
        direction: int,
    ) -> Tuple[chex.PRNGKey, MatchThreeGameGridStruct, MatchResults]:
        # check if swap is legal
        dy, dx = SWAP_DIRECTIONS[direction]
        grid_shape = state.grid.shape

        new_y, new_x = grid_cell[0] + dy, grid_cell[1] + dx

        move_is_legal = (
            (new_y >= 0)
            & (new_y < grid_shape[0])
            & (new_x >= 0)
            & (new_x < grid_shape[1])
        )

        # if true then apply swap and process physics and return MatchResults (this will take exactly CASCADE_MAX_DEPTH steps)
        # def _true_f_move_is_legal(operand):
        #     key, state, match_results = operand
        #     return self.__process_physics(
        #         key=key, state=state, match_results=match_results
        #     )

        match_results = MatchResults.initialize(
            grid_shape=grid_shape, num_symbols=state.num_symbols
        )
        key, state, match_results = jax.lax.cond(
            move_is_legal,
            lambda x: self.__process_physics(*x),
            lambda x: x,
            (key, state, match_results),
        )

        return key, state, match_results

    # step ticker to update "physics" when a move is doing
    def __process_physics(
        self,
        key: chex.PRNGKey,
        state: MatchThreeGameGridStruct,
        match_results: MatchResults,
    ) -> Tuple[chex.PRNGKey, MatchThreeGameGridStruct, MatchResults]:
        # TODO: consider using a jax while loop. an example is provided below
        for i in range(CASCADE_MAX_DEPTH):
            state, match_results = self.__process_match(state, match_results, i)
            state = self.__process_gravity(state)
            key, state = self.__generate_falling_symbols(key, state)
        return key, state, match_results

    @staticmethod
    def __process_match(
        state: MatchThreeGameGridStruct, match_results: MatchResults, cur_depth: int
    ) -> Tuple[MatchThreeGameGridStruct, MatchResults]: ...
    @staticmethod
    def __process_gravity(state: MatchThreeGameGridStruct) -> MatchThreeGameGridStruct:
        # apply gravity and put new random symbols on top until full
        ...

    @staticmethod
    def __generate_falling_symbols(
        key: chex.PRNGKey,
        state: MatchThreeGameGridStruct,
    ) -> Tuple[chex.PRNGKey, MatchThreeGameGridStruct]: ...


# ---
# Example of jax while loop


@jax.jit
def update_until_stable(
    state: jnp.ndarray, max_iterations: int = 100, tolerance: float = 1e-5
):
    # Condition function to check if state has stopped changing
    def cond_fun(carry):
        prev_state, state, i = carry
        change = jnp.linalg.norm(state - prev_state)  # Measure the change in state
        return (i < max_iterations) & (
            change > tolerance
        )  # Loop until change is small or max iterations reached

    # Loop body: Update the state (e.g., some function of state)
    def body_fun(carry):
        prev_state, state, i = carry
        # Simulate an update (replace with actual computation)
        new_state = state * 0.99  # Example: Apply some transformation to the state
        return (
            new_state,
            state,
            i + 1,
        )  # Update state, previous state, and iteration counter

    # Initial previous state is the same as the current state
    prev_state = state
    # Initial iteration counter
    i = 0

    # Run the while loop
    final_state, _, _ = jax.lax.while_loop(cond_fun, body_fun, (prev_state, state, i))
    return final_state

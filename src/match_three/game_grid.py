from typing import Optional, Tuple, Union
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

ROLL_DIRECTIONS = [
    (-1, 0),  # roll up
    (1, 0),  # roll down
    (-1, 1),  # roll left
    (1, 1),  # roll right
]

CASCADE_MAX_DEPTH = 10
GRID_SIZE = 9
MAX_GENERATION_ATTEMPTS = 25

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
    - Dimension 1: Type of matches at that depth (matching 3 in a row, 4, cross etc)
    - Dimension 2: Symbol type (e.g., which game pieces matched)

    The data type is `int`, compatible with JAX.
    """

    matches: chex.Array  # Shape: (depth)

    def initialize() -> "MatchResults":
        # TODO: calculate max max_possible_matches properly
        return MatchResults(
            matches=jnp.zeros(
                shape=(CASCADE_MAX_DEPTH),
                dtype=jnp.int32,
            )
        )


# TODO: this can be removed. Separate mask and num_symbols into env parameters
@struct.dataclass
class MatchThreeGameGridStruct:
    # a matrix representing the grid itself
    grid: chex.Array
    mask: chex.Array
    num_symbols: int


class MatchThreeGameGridFunctions:
    """A stateless functional class for Match-Three game grid operations."""

    # method to initialize a struct based on input
    def generate_game_grid(
        self, key: chex.PRNGKey, num_symbols: int, mask: chex.Array
    ) -> Tuple[chex.PRNGKey, MatchThreeGameGridStruct]:
        def cond_fn(carry):
            (
                _,
                _,
                matches,
                i,
            ) = carry
            return jnp.logical_and(jnp.sum(matches) > 0, i < MAX_GENERATION_ATTEMPTS)

        def body_fn(carry):
            key, grid, matches, i = carry
            key, subkey = jax.random.split(key)
            key, random_grid = self.__generate_random_grid(subkey, num_symbols, mask)
            grid = jnp.where(matches, random_grid, grid)
            new_matches = self.__find_matches(grid)
            return key, grid, new_matches, i + 1

        key, initial_grid = self.__generate_random_grid(key, num_symbols, mask)
        matches = self.__find_matches(initial_grid)
        key, final_grid, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (key, initial_grid, matches, 0)
        )
        return key, MatchThreeGameGridStruct(
            grid=final_grid, mask=mask, num_symbols=num_symbols
        )

    # method to swap based on (y,x) and swap direction -> new game board (can be the same if the swap was illeral)
    def apply_swap(
        self,
        key: chex.PRNGKey,
        state: MatchThreeGameGridStruct,
        grid_cell: chex.Array,
        direction: int,
    ) -> Tuple[chex.PRNGKey, MatchThreeGameGridStruct, MatchResults]:
        # TODO: WE NEED TO ASSUME THAT THE SWAP IS LEGAL. BUT THIS SHOULD BE DISCUSSED FURTHER.
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

        def cond_fn(carry):
            _, _, match_results, i = carry
            return jnp.logical_and(match_results[i - 1] > 0, i < CASCADE_MAX_DEPTH)

        def body_fn(carry):
            key, prev_matches, state, match_results, i = carry

            # remove matches
            grid = self.__remove_matches(state.grid, prev_matches)
            # collapse grid
            grid = self.__collapse_grid(grid)
            # refill grid
            key, grid = self.__refill_grid(key, grid)
            # generate new grid if there are no valid moves
            key, grid = self.__generate_new_grid_if_no_moves(key, grid)

            # new matches
            new_matches = self.__find_matches(grid)
            num_matches = jnp.sum(new_matches)
            match_results = match_results.at[i].set(num_matches)
            return (
                key,
                new_matches,
                MatchThreeGameGridStruct(
                    grid=grid, mask=state.mask, num_symbols=state.num_symbols
                ),
                match_results,
                i + 1,
            )

        init_matches = self.__find_matches(state.grid)
        key, _, _, match_results, _ = jax.lax.while_loop(
            cond_fn, body_fn, (key, init_matches, state, match_results, 0)
        )

        return key, state, match_results

    def __generate_random_grid(
        self, key: chex.PRNGKey, num_symbols: int, mask: chex.Array
    ) -> Tuple[chex.PRNGKey, chex.Array]:
        key, subkey = jax.random.split(key)
        grid = jax.random.randint(subkey, (GRID_SIZE, GRID_SIZE), 1, num_symbols + 1)
        grid = jnp.where(mask == 0, grid, -1)
        return key, grid

    def __translate_grid(self, grid, direction) -> chex.Array:
        def mask_vertical_fn(carry):
            grid, p = carry
            return grid.at[p, :].set(-1)

        def mask_horizontal_fn(carry):
            grid, p = carry
            return grid.at[:, p].set(-1)

        translated_grid = jnp.roll(grid, direction[0], axis=direction[1])

        p = jax.lax.cond(direction[0] == 1, lambda: 0, lambda: -1)
        grid = jax.lax.cond(
            direction[1] == 0,
            mask_vertical_fn,
            mask_horizontal_fn,
            (translated_grid, p),
        )
        return grid

    def __find_matches(self, grid) -> chex.Array:
        grid_roll_up = self.__translate_grid(grid, ROLL_DIRECTIONS[0])
        grid_roll_down = self.__translate_grid(grid, ROLL_DIRECTIONS[1])
        grid_roll_left = self.__translate_grid(grid, ROLL_DIRECTIONS[2])
        grid_roll_right = self.__translate_grid(grid, ROLL_DIRECTIONS[3])

        vertical_matches = jnp.equal(grid, grid_roll_up) & jnp.equal(
            grid, grid_roll_down
        )
        vertical_matches = jnp.logical_or(
            vertical_matches,
            jnp.logical_or(
                jnp.roll(vertical_matches, shift=-1, axis=0),
                jnp.roll(vertical_matches, shift=1, axis=0),
            ),
        )
        horizontal_matches = jnp.equal(grid, grid_roll_left) & jnp.equal(
            grid, grid_roll_right
        )
        horizontal_matches = jnp.logical_or(
            horizontal_matches,
            jnp.logical_or(
                jnp.roll(horizontal_matches, shift=-1, axis=1),
                jnp.roll(horizontal_matches, shift=1, axis=1),
            ),
        )
        # print(vertical_matches)

        all_matches = jnp.logical_or(vertical_matches, horizontal_matches)
        all_matches = jnp.where(grid == -1, False, all_matches)

        return all_matches

    def __remove_matches(self, grid, matches) -> chex.Array:
        return jnp.where(matches, 0, grid)

    def __collapse_grid(self, grid) -> chex.Array:
        def process_column(col_v):
            col_i = jnp.arange(0, 9)
            col_w = jnp.astype(col_v == -1, jnp.int4) - jnp.astype(col_v == 0, jnp.int4)

            order = jnp.argsort(col_w, stable=True)
            col_i = col_i[order]
            col_v = col_v[order]

            l_mask = col_v != -1
            r_mask = col_v == -1

            col_i = (
                jnp.sort(col_i * l_mask + GRID_SIZE * r_mask) * l_mask + col_i * r_mask
            )

            order = jnp.argsort(col_i, stable=True)
            return col_v[order]

        # Vectorize over columns
        processed_cols = jax.vmap(process_column)(grid.T)
        return processed_cols.T

    def __refill_grid(
        self, key: chex.PRNGKey, state: MatchThreeGameGridStruct
    ) -> Tuple[chex.PRNGKey, chex.Array]:
        key, subkey = jax.random.split(key)
        random_grid = self.__generate_random_grid(subkey, state.num_symbols, state.mask)
        return key, jnp.where(state.grid == 0, random_grid, state.grid)

    def __check_no_valid_moves(self, state: MatchThreeGameGridStruct) -> bool:
        def __translate_row(grid, row_idx, direction) -> chex.Array:
            row_v = grid.at[row_idx, :].get()
            row_v = jnp.roll(row_v, direction)
            row_v = jax.lax.cond(
                direction > 0, lambda: row_v.at[0].set(-1), lambda: row_v.at[-1].set(-1)
            )
            return grid.at[row_idx, :].set(row_v)

        def __translate_column(grid, col_idx, direction: int) -> chex.Array:
            col_v = grid.at[:, col_idx].get()
            col_v = jnp.roll(col_v, direction)
            col_v = jax.lax.cond(
                direction > 0, lambda: col_v.at[0].set(-1), lambda: col_v.at[-1].set(-1)
            )
            return grid.at[:, col_idx].set(col_v)

        def __reapply_mask(grid, mask) -> chex.Array:
            return jnp.where(mask == 0, grid, -1)

        def __get_translated_grids_tuple(
            grid, mask
        ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
            grid_new_down = jax.vmap(__translate_column, in_axes=(None, 0, None))(
                grid, jnp.arange(GRID_SIZE), -1
            )
            grid_new_up = jax.vmap(__translate_column, in_axes=(None, 0, None))(
                grid, jnp.arange(GRID_SIZE), 1
            )
            grid_new_right = jax.vmap(__translate_row, in_axes=(None, 0, None))(
                grid, jnp.arange(GRID_SIZE), -1
            )
            grid_new_left = jax.vmap(__translate_row, in_axes=(None, 0, None))(
                grid, jnp.arange(GRID_SIZE), 1
            )

            return grid_new_down, grid_new_up, grid_new_right, grid_new_left

        grids = jnp.concatenate(
            __get_translated_grids_tuple(state.grid, state.mask), axis=0
        )
        grids = jax.vmap(__reapply_mask, in_axes=(0, None))(grids, state.mask)
        matches = jax.vmap(self.__find_matches)(grids)
        return jnp.all(jnp.sum(matches, axis=(1, 2)) == 0)

    def __generate_new_grid_if_no_moves(
        self, key: chex.PRNGKey, state: MatchThreeGameGridStruct
    ) -> Tuple[chex.PRNGKey, MatchThreeGameGridStruct]:
        key, subkey = jax.random.split(key)
        grid = jax.lax.cond(
            self.__check_no_valid_moves(state),
            lambda: self.generate_game_grid(subkey, state.num_symbols, state.mask)[
                1
            ].grid,
            lambda: state.grid,
        )
        return key, MatchThreeGameGridStruct(
            grid=grid, mask=state.mask, num_symbols=state.num_symbols
        )

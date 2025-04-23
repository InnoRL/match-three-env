from typing import Optional, Tuple, Union
import chex
from flax import struct
import jax
from jax import numpy as jnp

def get_swap_direction(direction: int) -> Tuple[int, int]:
    return jax.lax.switch(
        direction,
        [
            lambda: (-1, 0),  # swap with top neighbor
            lambda: (0, -1),  # swap with left neighbor
            lambda: (1, 0),  # swap with bottom neighbor
            lambda: (0, 1),  # swap with right neighbor
        ],
    )


CASCADE_MAX_DEPTH = 100
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

    @staticmethod
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
    grid: chex.Array = jnp.zeros((GRID_SIZE, GRID_SIZE))


@struct.dataclass
class MatchThreeGameGridParams:
    num_symbols: int = 4
    mask: chex.Array = jnp.zeros((GRID_SIZE, GRID_SIZE))


class MatchThreeGameGridFunctions:
    """A stateless functional class for Match-Three game grid operations."""

    # method to initialize a struct based on input
    @staticmethod
    def generate_game_grid(
        key: chex.PRNGKey, params: MatchThreeGameGridParams
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
            key, random_grid = MatchThreeGameGridFunctions.__generate_random_grid(
                subkey, params
            )
            grid = jnp.where(matches, random_grid, grid)
            new_matches = MatchThreeGameGridFunctions.__find_matches(grid)
            return (key, grid, new_matches, i + 1)

        key, initial_grid = MatchThreeGameGridFunctions.__generate_random_grid(
            key, params
        )
        matches = MatchThreeGameGridFunctions.__find_matches(initial_grid)
        key, final_grid, _, _ = jax.lax.while_loop(
            cond_fn, body_fn, (key, initial_grid, matches, 0)
        )
        return key, MatchThreeGameGridStruct(grid=final_grid)

    # method to swap based on (y,x) and swap direction -> new game board (can be the same if the swap was illeral)
    @staticmethod
    def apply_swap(
        key: chex.PRNGKey,
        state: MatchThreeGameGridStruct,
        grid_cell: chex.Array,
        direction: int,
        params: MatchThreeGameGridParams,
    ) -> Tuple[MatchThreeGameGridStruct, MatchResults]:
        # TODO: WE NEED TO ASSUME THAT THE SWAP IS LEGAL. BUT THIS SHOULD BE DISCUSSED FURTHER.
        dy, dx = get_swap_direction(direction)
        grid_shape = state.grid.shape

        new_y, new_x = grid_cell[0] + dy, grid_cell[1] + dx

        move_is_legal = (
            (new_y >= 0)
            & (new_y < grid_shape[0])
            & (new_x >= 0)
            & (new_x < grid_shape[1])
        )

        points = (grid_cell, jnp.array((new_y, new_x), dtype=jnp.int32))
        match_results = MatchResults.initialize()
        key, state, match_results = jax.lax.cond(
            move_is_legal,
            lambda x: MatchThreeGameGridFunctions.__process_physics(*x),
            lambda x: (x[0], x[2], x[3]),
            (key, points, state, match_results, params),
        )

        return state, match_results

    # step ticker to update "physics" when a move is doing
    @staticmethod
    def __process_physics(
        key: chex.PRNGKey,
        points: Tuple[chex.Array, chex.Array],
        state: MatchThreeGameGridStruct,
        match_results: MatchResults,
        params: MatchThreeGameGridParams,
    ) -> Tuple[chex.PRNGKey, MatchThreeGameGridStruct, MatchResults]:
        def cond_fn(carry):
            _, _, _, match_results, i = carry
            return jnp.logical_and(
                match_results.matches[i - 1] > 0, i <= CASCADE_MAX_DEPTH
            )

        def body_fn(carry):
            key, prev_matches, state, match_results, i = carry

            # remove matches
            grid = MatchThreeGameGridFunctions.__remove_matches(
                state.grid, prev_matches
            )
            # collapse grid
            grid = MatchThreeGameGridFunctions.__collapse_grid(grid)
            # refill grid
            key, grid = MatchThreeGameGridFunctions.__refill_grid(key, grid, params)
            # generate new grid if there are no valid moves
            key, grid = MatchThreeGameGridFunctions.__generate_new_grid_if_no_moves(
                key, grid, params
            )

            # new matches
            new_matches = MatchThreeGameGridFunctions.__find_matches(grid)
            num_matches = jnp.sum(new_matches)
            match_results = MatchResults(match_results.matches.at[i].set(num_matches))

            new_state = MatchThreeGameGridStruct(grid=grid)
            return (
                key,
                new_matches,
                new_state,
                match_results,
                i + 1,
            )

        # swap the grid cells
        grid_cell = points[0]
        new_cell = points[1]
        temp_value = state.grid.at[grid_cell[0], grid_cell[1]].get()
        swapped_grid = state.grid.at[grid_cell[0], grid_cell[1]].set(
            state.grid.at[new_cell[0], new_cell[1]].get()
        )
        swapped_grid = swapped_grid.at[new_cell[0], new_cell[1]].set(temp_value)
        swapped_state = MatchThreeGameGridStruct(grid=swapped_grid)

        # process physics
        init_matches = MatchThreeGameGridFunctions.__find_matches(swapped_grid)
        match_results = MatchResults(
            matches=match_results.matches.at[0].set(jnp.sum(init_matches))
        )

        key, state, match_results = jax.lax.cond(
            match_results.matches[0] > 0,
            lambda x: (
                loop_result := jax.lax.while_loop(
                    cond_fn, body_fn, (x[0], x[1], x[2], x[3], 1)
                ),
                (loop_result[0], loop_result[2], loop_result[3]),
            )[1],
            lambda _: (key, state, match_results),
            (key, init_matches, swapped_state, match_results, 0),
        )

        key, _, _, match_results, _ = jax.lax.while_loop(
            cond_fn, body_fn, (key, init_matches, swapped_state, match_results, 0)
        )

        return key, state, match_results

    @staticmethod
    def __generate_random_grid(
        key: chex.PRNGKey, params: MatchThreeGameGridParams
    ) -> Tuple[chex.PRNGKey, chex.Array]:
        key, subkey = jax.random.split(key)
        grid = jax.random.randint(
            subkey, (GRID_SIZE, GRID_SIZE), 1, params.num_symbols + 1
        )
        grid = jnp.where(params.mask == 0, grid, -1)
        return key, grid

    @staticmethod
    def __translate_grid_vertical(grid, direction) -> chex.Array:

        translated_grid = jnp.roll(grid, direction, axis=0)

        p = jax.lax.cond(direction == 1, lambda: 0, lambda: -1)
        grid = translated_grid.at[p, :].set(-1)  # mask_vertical_fn(translated_grid, p)

        return grid

    @staticmethod
    def __translate_grid_horizontal(grid, direction) -> chex.Array:

        translated_grid = jnp.roll(grid, direction, axis=1)

        p = jax.lax.cond(direction == 1, lambda: 0, lambda: -1)
        grid = translated_grid.at[:, p].set(-1)  # mask_vertical_fn(translated_grid, p)

        return grid

    @staticmethod
    def __find_matches(grid: chex.Array) -> chex.Array:
        grid_roll_up = MatchThreeGameGridFunctions.__translate_grid_vertical(grid, -1)
        grid_roll_down = MatchThreeGameGridFunctions.__translate_grid_vertical(grid, 1)
        grid_roll_left = MatchThreeGameGridFunctions.__translate_grid_horizontal(
            grid, -1
        )
        grid_roll_right = MatchThreeGameGridFunctions.__translate_grid_horizontal(
            grid, 1
        )

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

    @staticmethod
    def __remove_matches(grid: chex.Array, matches: chex.Array) -> chex.Array:
        return jnp.where(matches, 0, grid)

    @staticmethod
    def __collapse_grid(grid: chex.Array) -> chex.Array:
        def process_column(col_v: chex.Array) -> chex.Array:
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

    @staticmethod
    def __refill_grid(
        key: chex.PRNGKey,
        grid: chex.Array,
        params: MatchThreeGameGridParams,
    ) -> Tuple[chex.PRNGKey, chex.Array]:
        key, subkey = jax.random.split(key)
        key, random_grid = MatchThreeGameGridFunctions.__generate_random_grid(
            subkey, params
        )
        print(grid)
        return key, jnp.where(grid == 0, random_grid, grid)

    @staticmethod
    def __check_no_valid_moves(
        grid: chex.Array, params: MatchThreeGameGridParams
    ) -> bool:
        def __translate_row(grid, row_idx, direction) -> chex.Array:
            row_v = grid.at[row_idx, :].get()
            row_v = jnp.roll(row_v, direction)
            row_v = jax.lax.cond(
                direction > 0, lambda: row_v.at[0].set(-1), lambda: row_v.at[-1].set(-1)
            )
            return grid.at[row_idx, :].set(row_v)

        def __translate_column(
            grid: chex.Array, col_idx: int, direction: int
        ) -> chex.Array:
            col_v = grid.at[:, col_idx].get()
            col_v = jnp.roll(col_v, direction)
            col_v = jax.lax.cond(
                direction > 0, lambda: col_v.at[0].set(-1), lambda: col_v.at[-1].set(-1)
            )
            return grid.at[:, col_idx].set(col_v)

        def __reapply_mask(grid: chex.Array, mask: chex.Array) -> chex.Array:
            return jnp.where(mask == 0, grid, -1)

        def __get_translated_grids_tuple(
            grid: chex.Array,
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

        grids = jnp.concatenate(__get_translated_grids_tuple(grid), axis=0)
        grids = jax.vmap(__reapply_mask, in_axes=(0, None))(grids, params.mask)
        matches = jax.vmap(MatchThreeGameGridFunctions.__find_matches)(grids)
        return jnp.all(jnp.sum(matches, axis=(1, 2)) == 0)

    @staticmethod
    def __generate_new_grid_if_no_moves(
        key: chex.PRNGKey,
        grid: chex.Array,
        params: MatchThreeGameGridParams,
    ) -> Tuple[chex.PRNGKey, chex.Array]:
        key, subkey = jax.random.split(key)
        grid = jax.lax.cond(
            MatchThreeGameGridFunctions.__check_no_valid_moves(grid, params),
            lambda: MatchThreeGameGridFunctions.generate_game_grid(subkey, params)[
                1
            ].grid,
            lambda: grid,
        )
        return key, grid

from typing import Tuple

import chex
import jax
import jax.numpy as jnp


def conv_action_to_swap(
    grid_size: Tuple[int, int], action: int
) -> Tuple[chex.Array, int]:
    rswap_num = grid_size[0] * (grid_size[1] - 1)
    bswap_num = (grid_size[0] - 1) * grid_size[1]
    assert 0 <= action < rswap_num + bswap_num

    if action < rswap_num:
        return jnp.array([action // (grid_size[1] - 1), action % (grid_size[1] - 1)]), 3

    action -= rswap_num
    return jnp.array([action % (grid_size[0] - 1), action // (grid_size[0] - 1)]), 2


def conv_action_to_swap_jit(
    grid_size: Tuple[int, int], action: int
) -> Tuple[chex.Array, int]:
    height, width = grid_size
    rswap_num = height * (width - 1)

    grid_cell, direction = jax.lax.cond(
        action < rswap_num,
        lambda: (jnp.array([action // (width - 1), action % (width - 1)]), 3),
        lambda: (
            (_action := action - rswap_num),
            (jnp.array([_action % (height - 1), _action // (height - 1)]), 2),
        )[1],
    )
    return grid_cell, direction

def conv_action_to_swap_continuous_jit(
    grid_size: Tuple[int, int], action: chex.Array
) -> Tuple[chex.Array, int]:
    # action = [height, width, direction]
    # height and width are in [0, 1]
    # direction is in [0, 1]. 0 is swap with top (horizontal), 1 is swap with left (vertical).
    
    height, width = grid_size
    action = jnp.clip(action, 0, 1)

    grid_cell = jnp.array([1 + action[0] * (height - 1), 1 + action[1] * (width - 1)], dtype=jnp.int32)
    direction = jax.lax.cond(
        action[2] < 0.5,
        lambda: 0,
        lambda: 1,
    )
    return grid_cell, jnp.int32(direction)
    
    


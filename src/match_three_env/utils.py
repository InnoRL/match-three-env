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
        return jnp.array([action // (grid_size[1] - 1), action % (grid_size[1] - 1)]), 2

    action -= rswap_num
    return jnp.array([action % (grid_size[0] - 1), action // (grid_size[0] - 1)]), 3


# @partial(jax.jit, static_argnums=(0,))
def conv_action_to_swap_jit(
    grid_size: Tuple[int, int], action: int
) -> Tuple[chex.Array, int]:
    height, width = grid_size
    rswap_num = height * (width - 1)
    # bswap_num = (height - 1) * width

    grid_cell, direction = jax.lax.cond(
        action < rswap_num,
        lambda: (jnp.array([action // (width - 1), action % (width - 1)]), 2),
        lambda: (
            (_action := action - rswap_num),
            (jnp.array([_action % (height - 1), _action // (height - 1)]), 3),
        )[1],
    )
    return grid_cell, direction

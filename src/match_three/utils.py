import jax.numpy as jnp
import chex
from typing import Tuple


def conv_action_to_swap(grid_size: Tuple[int, int], action: int) -> Tuple[chex.Array, int]:
    rswap_num = grid_size[0] * (grid_size[1] - 1)
    bswap_num = (grid_size[0] - 1) * grid_size[1]
    assert 0 <= action < rswap_num + bswap_num

    if action < rswap_num:
        return jnp.array([action // (grid_size[1] - 1), action % (grid_size[1] - 1)]), 2
    
    action -= rswap_num
    return jnp.array([action % (grid_size[0] - 1), action // (grid_size[0] - 1)]), 3

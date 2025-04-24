from typing import Callable
from flax import linen as nn
import jax.numpy as jnp


class Critic(nn.Module):
    precision_dtype: jnp.dtype
    rl_init_fn: Callable
    small_init_fn: Callable

    @nn.compact
    def __call__(self, x):
        # x is the output from CNN
        x = nn.Dense(
            256,
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            128,
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x) # Introduced for stability
        value = nn.Dense(
            1,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=self.small_init_fn(),
        )(x)
        return value

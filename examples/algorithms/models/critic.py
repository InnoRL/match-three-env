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
            bias_init=nn.initializers.zeros_init()
        )(x)
        # x = nn.relu(x)
        # x = nn.tanh(x)
        # x = nn.LayerNorm()(x)  # Before activation
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(
            256,
            dtype=self.precision_dtype,
            # dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
            bias_init=nn.initializers.zeros_init()
        )(x)
        x = x.astype(jnp.float32)
        # x = nn.relu(x)
        # x = nn.tanh(x)
        # x = nn.LayerNorm()(x)  # Before activation!
        x = nn.leaky_relu(x, negative_slope=0.01)
        value = nn.Dense(
            1,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=self.small_init_fn(),
        )(x)
        return value

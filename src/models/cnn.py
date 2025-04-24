from typing import Callable
from flax import linen as nn
import jax.numpy as jnp


class CNN(nn.Module):
    """CNN for processing 9x9 grids"""

    precision_dtype: jnp.dtype
    rl_init_fn: Callable

    @nn.compact
    def __call__(self, x):
        # Input shape: (9, 9, K_MAX+1)
        # print("CNN input shape: ", x.shape)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.relu(x)
        x = x.reshape((-1))  # Flatten
        # print("CNN output shape: ", x.shape)
        return x

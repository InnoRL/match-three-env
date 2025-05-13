from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn


class ResidualBlock(nn.Module):
    """Pre-activation residual block with bottleneck (similar to ResNet v2)"""

    features: int
    precision_dtype: jnp.dtype
    rl_init_fn: Callable

    @nn.compact
    def __call__(self, x):
        residual = x
        # Pre-activation design (ReLU before layers)
        x = nn.relu(x)
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            # kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            # kernel_init=self.rl_init_fn(),
        )(x)
        # print("before residual shape: ", residual.shape)
        # print("before x shape: ", x.shape)
        # # Project residual if channel dimensions change
        # residual = jax.lax.select(
        # residual.shape[-1] != self.features,
        residual = nn.Conv(
            features=self.features,
            kernel_size=(1, 1),
            strides=(1, 1),
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            # kernel_init=self.rl_init_fn(),
        )(residual)
        #     residual,
        # )
        # print("after residual shape: ", residual.shape)
        # print("after x shape: ", x.shape)
        return x + residual


class CNN(nn.Module):
    """CNN for processing 9x9 grids"""

    precision_dtype: jnp.dtype
    rl_init_fn: Callable
    latent_dim: int  # Dimension of the latent space (bottleneck)

    @nn.compact
    def __call__(self, x):
        # Input shape: (9, 9, K_MAX+1)
        # print("CNN input shape: ", x.shape)
        x = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.relu(x)
        x = x.reshape((-1))  # Flatten
        # Residual blocks
        # x = ResidualBlock(
        #     features=128,
        #     precision_dtype=self.precision_dtype,
        #     rl_init_fn=self.rl_init_fn,
        # )(x)
        # x = ResidualBlock(
        #     features=256,
        #     precision_dtype=self.precision_dtype,
        #     rl_init_fn=self.rl_init_fn,
        # )(x)
        # x = ResidualBlock(
        #     features=256,
        #     precision_dtype=self.precision_dtype,
        #     rl_init_fn=self.rl_init_fn,
        # )(x)
        # # Global average pooling (replaces flatten)
        # x = jnp.mean(x, axis=(0, 1))  # (9,9,C) -> (C,)

        x = nn.Dense(
            name="latent_space",
            features=self.latent_dim,
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        print("CNN output shape: ", x.shape)
        return x

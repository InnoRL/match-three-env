from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class ResidualDecoderBlock(nn.Module):
    """Residual block for decoders with transposed convs"""

    features: int
    precision_dtype: jnp.dtype
    rl_init_fn: Callable

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.relu(x)
        x = nn.ConvTranspose(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),  # Maintain spatial dims
            padding="SAME",
            dtype=self.precision_dtype,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            dtype=self.precision_dtype,
            kernel_init=self.rl_init_fn(),
        )(x)
        return x + residual  # Skip connection


class CNNDecoder(nn.Module):
    """Decoder for CNN autoencoder that reconstructs 9x9 grids."""

    precision_dtype: jnp.dtype
    rl_init_fn: Callable
    k_symbols: int

    @nn.compact
    def __call__(self, z):
        # Input z: (256,), reshape to (1, 1, 256) for ConvTranspose
        z = z.reshape((1, 1, -1))  # Now shape: (1, 1, 256)

        # Upsample to 3x3
        z = nn.ConvTranspose(
            features=256,
            kernel_size=(3, 3),
            strides=(3, 3),  # Upsample by stride=3
            padding="VALID",
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(
            z
        )  # Output shape: (3, 3, 256)
        z = nn.relu(z)

        # Upsample to 9x9
        # z = nn.ConvTranspose(
        #     features=256,
        #     kernel_size=(3, 3),
        #     strides=(3, 3),
        #     padding="VALID",
        #     dtype=self.precision_dtype,
        #     param_dtype=jnp.float32,
        #     kernel_init=self.rl_init_fn(),
        # )(
        #     z
        # )  # Output shape: (9, 9, 256)
        # z = nn.relu(z)
        # z = ResidualDecoderBlock(
        #     features=256,
        #     precision_dtype=self.precision_dtype,
        #     rl_init_fn=self.rl_init_fn,
        # )(z)
        # Second upsampling: 3x3 → 9x9
        z = nn.ConvTranspose(
            features=256,
            kernel_size=(3, 3),
            strides=(3, 3),
            padding="VALID",
            dtype=self.precision_dtype,
            kernel_init=self.rl_init_fn(),
        )(z)  # (9, 9, 256)
        z = nn.relu(z)
        
        # Final layer: Output (9, 9, K)
        z = nn.ConvTranspose(
            features=self.k_symbols,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(z)
        # Output shape: (9, 9, K)
        # z = nn.softmax(z, axis=-1)  # Softmax over K classes
        print("DECODER OUTPUT SHAPE: ", z.shape)

        return z  # Shape: (9, 9, K)

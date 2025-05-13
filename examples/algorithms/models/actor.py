from typing import Callable
import jax.numpy as jnp
from flax import linen as nn


class Actor(nn.Module):
    action_dim: int
    precision_dtype: jnp.dtype
    rl_init_fn: Callable
    small_init_fn: Callable

    @nn.compact
    def __call__(self, x):
        # x is the output from CNN
        x = nn.Dense(
            256,
            dtype=self.precision_dtype,
            # dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            128,
            dtype=self.precision_dtype,
            # dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.tanh(x)
        # x = nn.LayerNorm()(x) # Introduced for stability
        # logits should not be bfloat16
        logits = nn.Dense(
            self.action_dim,
            # dtype=jnp.float32,
            dtype=self.precision_dtype,  # NOTE: we are forced to use bfloat16 here. float32 will cause an error (I do not know why)
            # dtype=jnp.float64,
            param_dtype=jnp.float32,
            kernel_init=self.small_init_fn(),
        )(x)
        logits = logits.astype(jnp.float32)
        return logits

    # def get_action(self, params, x, rng):
    #     """Sample an action from the policy"""
    #     logits = self.apply(params, x)
    #     action = jax.random.categorical(rng, logits)
    #     return action

    # def get_log_prob(self, params, x, action):
    #     """Get log probability of a specific action"""
    #     logits = self.apply(params, x)
    #     log_prob = jax.nn.log_softmax(logits)[action]
    #     return log_prob

from typing import Callable
from flax import linen as nn
import jax
import jax.numpy as jnp


class Actor(nn.Module):
    action_dim: int
    precision_dtype: jnp.dtype
    rl_init_fn: Callable
    small_init_fn: Callable

    @nn.compact
    def __call__(self, x):
        # x is the output from CNN
        x = nn.Dense(256, dtype=self.precision_dtype, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        x = nn.Dense(128, dtype=self.precision_dtype, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        # logits should not be bfloat16
        logits = nn.Dense(
            self.action_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            kernel_init=self.small_init_fn,
        )(x)
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

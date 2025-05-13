from typing import Callable
import jax.numpy as jnp
from flax import linen as nn

class ActorContinuous(nn.Module):
    action_dim: int  # Set to 3 for your case
    precision_dtype: jnp.dtype
    rl_init_fn: Callable
    small_init_fn: Callable
    log_std_min: float = -5.0  # Clipping for stability (~0.007)
    log_std_max: float = 2.0   # Clipping for stability (~7.389)

    @nn.compact
    def __call__(self, x):
        # Shared feature extractor
        x = nn.Dense(
            256,
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            128,
            dtype=self.precision_dtype,
            param_dtype=jnp.float32,
            kernel_init=self.rl_init_fn(),
        )(x)
        x = nn.tanh(x)
        
        # Mean output (μ) for each action dimension
        means_position = nn.Dense(
            self.action_dim - 1,
            dtype=self.precision_dtype,  # Force float32 for action outputs
            param_dtype=jnp.float32,
            kernel_init=self.small_init_fn(),
            name="means_position"
        )(x)
        
        logit_direction = nn.Dense(
            1,
            dtype=self.precision_dtype,  # Force float32 for action outputs
            param_dtype=jnp.float32,
            kernel_init=self.small_init_fn(),
            name="logit_direction"
        )(x)
                
        # Learned log_std (independent of state)
        log_stds_position = self.param(
            "log_stds",
            nn.initializers.constant(0.0),  # Start with small std
            (self.action_dim - 1,),
        )
        
        # Clip log_std for numerical stability
        # We use log stds to ensure that the std is always positive
        # log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        
        return means_position, logit_direction, log_stds_position


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

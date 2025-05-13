import jax
import jax.numpy as jnp
import optax
from jax.nn.initializers import variance_scaling

from match3_env.game_grid import K_MAX


@jax.jit
def encode_grid(grid):
    """One-hot encode the grid"""
    # TODO: check if this is correct
    vals = jnp.arange(0, K_MAX + 1)
    vals = vals[:, jnp.newaxis, jnp.newaxis]
    return jnp.array(grid == vals, dtype=jnp.float32).transpose(1, 2, 0)


# ----------------------
# Custom Initializers
# ----------------------
def rl_init(scale: float = 2.0, mode: str = "fan_in"):
    """Kaiming/He initialization optimized for ReLU networks in RL"""
    return variance_scaling(scale, mode, "truncated_normal", dtype=jnp.float32)


def small_init(scale: float = 0.01):
    """Small initialization for value/policy heads"""
    return variance_scaling(scale, "fan_in", "truncated_normal", dtype=jnp.float32)


# ----------------------
def cosine_annealing_with_warmup(warmup_steps, total_steps, base_lr=0.1):
    """
    Implements cosine annealing with warmup in Optax.

    Args:
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total training steps.
        base_lr (float): Initial learning rate.

    Returns:
        An Optax learning rate scheduler.
    """
    return optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=0.0, end_value=base_lr, transition_steps=warmup_steps
            ),  # Warmup
            optax.cosine_decay_schedule(
                init_value=base_lr, decay_steps=total_steps - warmup_steps
            ),  # Cosine Annealing
        ],
        boundaries=[warmup_steps],  # Switch from warmup to cosine decay
    )

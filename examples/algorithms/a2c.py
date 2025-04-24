import os
from functools import partial

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from models import CNN, Actor, Critic
from tqdm import tqdm
from utils import (
    cosine_annealing_with_warmup,
    encode_grid,
    rl_init,
    small_init,
)

from match_three_env.env import EnvParams, MatchThree
from match_three_env.game_grid import GRID_SIZE

NUM_ENVS = 2
GAMMA = 0.99
LAMBDA = 0.98
LEARNING_RATE = 0.0001
NUM_UPDATES = 1000
WARMUP_RATIO = 0.01
PRECISION = "bfloat16"

CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_INTERVAL = 100
CHECKPOINT_MAX_TO_KEEP = 5
CHECKPOINT_RESTORE_DIR = None  # "./checkpoints/latest"

jax.config.update("jax_default_matmul_precision", PRECISION)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Wandb
wandb.init(
    project="match-three",
    group="InnoRL",
    name="match-three-a2c-test",
    config={
        "num_updates": NUM_UPDATES,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        # TODO: add other hyperparameters here
    },
)


def restore_model(ckpt_dir):
    mngr = ocp.CheckpointManager(ckpt_dir, ocp.PyTreeCheckpointer())
    return mngr.restore(mngr.latest_step())


# Combined network
class ActorCritic(nn.Module):
    action_dim: int
    precision_dtype: jnp.dtype

    def setup(self):
        self.cnn = CNN(self.precision_dtype, rl_init_fn=rl_init)
        self.actor = Actor(
            self.action_dim,
            self.precision_dtype,
            rl_init_fn=rl_init,
            small_init_fn=partial(small_init, scale=0.01),
        )
        self.critic = Critic(
            self.precision_dtype,
            rl_init_fn=rl_init,
            small_init_fn=partial(small_init, scale=1.0),
        )

    def __call__(self, x):
        encoded = encode_grid(x)
        features = self.cnn(encoded)
        return self.actor(features), self.critic(features)

    def get_action(self, x, key):
        encoded = encode_grid(x)
        # print("encoded shape: ", encoded.shape)
        features = self.cnn(encoded)
        logits = self.actor(features)
        # key = self.make_rng("key")
        action = jax.random.categorical(key, logits)
        return action

    def get_value(self, x):
        encoded = encode_grid(x)
        features = self.cnn(encoded)
        return self.critic(features)


# Training state
@chex.dataclass
class TrainState:
    params: dict
    opt_state: optax.OptState
    key: jax.random.PRNGKey


def train():
    # Initialize environment
    # TODO: get this using gymnax
    env_params = EnvParams()
    env = MatchThree(env_params)

    if PRECISION == "bfloat16":
        precision_dtype = jnp.bfloat16
    else:
        precision_dtype = jnp.float32

    max_steps_in_episode = env_params.max_steps_in_episode
    num_actions = env.num_actions

    # # test encoder
    # key = jax.random.PRNGKey(0)
    # key, subkey = jax.random.split(key)
    # obs, state = env.reset(subkey, env_params)
    # enc = encode_grid(obs)
    # print(enc, enc.shape)

    # Initialize networks
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    dummy_input = jnp.zeros(
        (GRID_SIZE, GRID_SIZE)
    )  # TODO: verify the actual input shape
    ac = ActorCritic(action_dim=num_actions, precision_dtype=precision_dtype)
    params = ac.init(subkey, dummy_input)
    # NOTE: env_params are jnp.float32, but some of the operations are jnp.bfloat16.
    # This is manual mixed precision, done for performance reasons.

    # Cosine annealing with warmup
    warmup_steps = int(WARMUP_RATIO * NUM_UPDATES)
    total_steps = NUM_UPDATES
    learning_rate_scheduler = cosine_annealing_with_warmup(
        warmup_steps, total_steps, base_lr=LEARNING_RATE
    )

    # Initialize optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(
            learning_rate=learning_rate_scheduler,
            # NOTE: commented out since they are default values
            # b1=0.9,
            # b2=0.999,
            # weight_decay=0.01,
            # eps=1e-8,
        ),
    )
    opt_state = optimizer.init(params)
    # Initialize training state
    train_state = TrainState(params=params, opt_state=opt_state, key=key)

    # Training loop
    def update_step(train_state, _) -> tuple[TrainState, dict]:
        # Storage for rollouts for each environment (NUM_ENVS)
        rollouts = {
            "obs_batch": jnp.zeros(
                (NUM_ENVS, max_steps_in_episode, GRID_SIZE, GRID_SIZE)
            ),
            # Actions are categorical
            "actions_batch": jnp.zeros(
                (NUM_ENVS, max_steps_in_episode), dtype=jnp.int32
            ),
            "rewards_batch": jnp.zeros(
                (NUM_ENVS, max_steps_in_episode), dtype=precision_dtype
            ),
            "dones_batch": jnp.zeros((NUM_ENVS, max_steps_in_episode)),
            "values_batch": jnp.zeros(
                (NUM_ENVS, max_steps_in_episode), dtype=precision_dtype
            ),
            "log_probs_batch": jnp.zeros((NUM_ENVS, max_steps_in_episode)),
        }

        # Vectorized functions
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_apply = jax.vmap(ac.apply, in_axes=(None, 0))
        get_action_fn = lambda params, obs, key: ac.apply(
            params, obs, key, method=ac.get_action
        )
        vmap_get_action = jax.vmap(get_action_fn, in_axes=(None, 0, 0))
        vmap_split = jax.vmap(jax.random.split, in_axes=(0, None))

        # Collect rollout using vmap over NUM_ENVS
        def _env_step(carry, _):
            train_state, keys, obses, states = carry
            keys = vmap_split(
                keys, 2
            )  # IN shape (NUM_ENVS, 2); OUT shape (NUM_ENVS, 2, 2)
            # jax.debug.print("keys shape: {x}", x=keys.shape)
            # print("keys shape: ", keys.shape)
            keys, subkeys = keys[:, 0], keys[:, 1]

            # Get actions and values
            logits, values = vmap_apply(train_state.params, obses)
            # TODO: check if this is correct. Wont this squeeze batches?
            # NOTE: squeeze removes one or more length-1 axes from array.
            # This means that it should not be a problem
            values = values.squeeze()
            actions = vmap_get_action(train_state.params, obses, subkeys)
            # TODO: check if this will work. do we need to do a vmap here?
            # NOTE: solved by using axis=1
            # print("obses shape: ", obses.shape)
            # print("values shape: ", values.shape)
            # print("logits shape: ", logits.shape)
            # print("actions shape: ", actions.shape)
            log_probs = jax.nn.log_softmax(logits, axis=1)[
                jnp.arange(NUM_ENVS), actions
            ]

            # Step environment
            keys = vmap_split(
                keys, 2
            )  # IN shape (NUM_ENVS, 2); OUT shape (NUM_ENVS, 2, 2)
            keys, subkeys = keys[:, 0], keys[:, 1]
            next_obses, next_states, rewards, dones, _ = vmap_step(
                subkeys, states, actions, env_params
            )

            # Update carry and store data
            carry = (train_state, keys, next_obses, next_states)
            data = {
                "obs_batch": obses,
                "actions_batch": actions,
                "rewards_batch": rewards,
                "dones_batch": dones,
                "values_batch": values,
                "log_probs_batch": log_probs,
            }
            return carry, data

        # Reset environments
        key = train_state.key
        keys = jax.random.split(
            key, num=(NUM_ENVS + 1)
        )  # add one for key, NUM_ENVS for subkeys
        key = keys[0]
        subkeys = keys[1:]
        obses, states = vmap_reset(subkeys, env_params)
        # we dont need to split the keys here because they will be split in _env_step
        carry = (train_state, subkeys, obses, states)

        # Collect episode rollout for each environment (NUM_ENVS).
        # We assume that the episode terminates only when truncated (i.e., reached max_steps_in_episode)
        carry, rollouts = jax.lax.scan(
            _env_step, carry, jnp.arange(max_steps_in_episode)
        )
        # train_state is not changed in _env_step, so we need to update only the key
        train_state = train_state.replace(key=key)
        print("Done collecting rollouts")

        # Compute advantages and returns
        # Get (NUM_ENVS) last values
        _, last_values = vmap_apply(train_state.params, rollouts["obs_batch"][-1])
        last_values = (
            last_values.squeeze()
        )  # TODO: check if this is correct. NOTE: should be correct.

        # NOTE: here rewards, dones, values are (max_steps_in_episode,) and last_value is scalar.
        def _compute_gae_single_env(rewards, dones, values, last_value):
            """Compute GAE for a single environment trajectory."""
            last_advantage = 0.0

            # Reverse computation through time
            def _body(carry, xs):
                last_advantage, last_value = carry
                reward, done, value = xs

                mask = 1.0 - done
                delta = reward + GAMMA * last_value * mask - value
                last_advantage = delta + GAMMA * LAMBDA * last_advantage * mask
                last_value = value
                return (last_advantage, last_value), last_advantage

            # Scan through time steps in reverse
            _, advantages = jax.lax.scan(
                _body,
                (last_advantage, last_value),
                (rewards, dones, values),
                reverse=True,
            )

            returns = advantages + values
            return advantages, returns

        # Compute GAE for each environment
        # NOTE: Shape of rewards, dones, values is (max_steps_in_episode, NUM_ENVS,). This is probably due to scan()
        advantages, returns = jax.vmap(_compute_gae_single_env, in_axes=(1, 1, 1, 0))(
            rollouts["rewards_batch"],  # (NUM_ENVS, max_steps_in_episode)
            rollouts["dones_batch"],  # (NUM_ENVS, max_steps_in_episode)
            rollouts["values_batch"],  # (NUM_ENVS, max_steps_in_episode)
            last_values,  # (NUM_ENVS,)
        )
        print("Done computing GAE")

        # NOTE: Onwards we are no longer using vmap.
        # Flatten batch.
        b_obs = rollouts["obs_batch"].reshape((-1, GRID_SIZE, GRID_SIZE))
        b_actions = rollouts["actions_batch"].flatten()
        b_log_probs = rollouts["log_probs_batch"].flatten()
        b_advantages = advantages.flatten()
        b_returns = returns.flatten()

        # NOTE: after flattening the shapes are:
        # b_obs:        (NUM_ENVS * max_steps_in_episode, GRID_SIZE, GRID_SIZE) float32
        # b_actions:    (NUM_ENVS * max_steps_in_episode,) int32
        # b_log_probs:  (NUM_ENVS * max_steps_in_episode,) float32
        # b_advantages: (NUM_ENVS * max_steps_in_episode,) float32
        # b_returns:    (NUM_ENVS * max_steps_in_episode,) float32

        # Calculate loss and update.
        # NOTE: No need to wrap this in a jit function since it will be wrapped in value_and_grad
        def _loss_fn(params, obs, actions, old_log_probs, advantages, returns):
            # NOTE: We need to ensure that the gradients are computed using float32.
            # Calculate policy loss
            logits, values = jax.vmap(ac.apply, in_axes=(None, 0))(params, obs)
            logits = logits.astype(jnp.float32)  # ensure logits are float32
            # logits are of shape (NUM_ENVS * max_steps_in_episode, num_actions)
            values = values.astype(jnp.float32)  # ensure values are float32
            values = values.squeeze()
            # print("logits shape: ", logits.shape)
            # print("actions shape: ", actions.shape)
            log_probs = jax.nn.log_softmax(logits, axis=1)[
                jnp.arange(NUM_ENVS * max_steps_in_episode), actions
            ]

            # Policy gradient loss
            advantages = advantages.astype(jnp.float32)  # ensure advantages are float32
            returns = returns.astype(jnp.float32)  # ensure returns are float32
            ratio = jnp.exp(log_probs - old_log_probs)
            policy_loss1 = ratio * advantages
            policy_loss2 = jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
            policy_loss = -jnp.minimum(policy_loss1, policy_loss2).mean()

            # Value loss
            value_loss = 0.5 * jnp.square(values - returns).mean()

            # Entropy bonus
            entropy = -jnp.sum(
                jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1
            ).mean()

            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            return total_loss, (policy_loss, value_loss, entropy)

        # TODO: will the grads be computed correctly since the return of _loss_fn is a tuple?
        # NOTE: solved by using has_aux=True. This means that the return of _loss_fn is a tuple of (loss, aux).
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        (total_loss, (policy_loss, value_loss, entropy)), grads = grad_fn(
            train_state.params, b_obs, b_actions, b_log_probs, b_advantages, b_returns
        )
        print("Done computing gradients")
        # Update parameters
        updates, new_opt_state = optimizer.update(
            grads, train_state.opt_state, params=train_state.params
        )
        new_params = optax.apply_updates(train_state.params, updates)

        # Update training state
        new_train_state = train_state.replace(
            params=new_params,
            opt_state=new_opt_state,
        )
        print("Done updating parameters")
        # Logging
        metrics = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "returns": b_returns.mean(),
        }

        return new_train_state, metrics

    # # Run training
    # key = jax.random.PRNGKey(0)
    # train_state = TrainState(env_params, opt_state, key)
    # NOTE: moved to top (see above update_step function)

    # JIT the update step
    jit_update_step_fn = jax.jit(update_step)

    # Setup checkpointing
    ckpt_dir = CHECKPOINT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    # checkpointer = ocp.PyTreeCheckpointer()
    # options = ocp.CheckpointManagerOptions(
    #     max_to_keep=CHECKPOINT_MAX_TO_KEEP, save_interval_steps=CHECKPOINT_INTERVAL
    # )
    # mngr = ocp.CheckpointManager(ckpt_dir, checkpointer, options)
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    mngr = ocp.CheckpointManager(
        ckpt_dir,
        checkpointer,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=CHECKPOINT_MAX_TO_KEEP,
            save_interval_steps=CHECKPOINT_INTERVAL,
            step_prefix="checkpoint_",
        ),
    )

    # Training loop
    with tqdm(range(NUM_UPDATES), desc="Training") as pbar:
        for i in pbar:
            train_state, metrics = update_step(train_state, None)
            # train_state, metrics = jit_update_step_fn(train_state, None)

            # Logging
            if i % 10 == 0:
                pbar.set_postfix(
                    {
                        "policy_loss": f"{metrics[0]:.3f}",
                        "value_loss": f"{metrics[1]:.3f}",
                        "entropy": f"{metrics[2]:.3f}",
                    }
                )
                wandb.log(metrics)

            # NOTE: we dont need the condition here since orbax will deal with intervals
            # mngr.save(
            #     i,
            #     args=ocp.args.Composite(
            #         params=ocp.args.StandardSave(train_state.params),
            #         step=ocp.args.StandardSave(i),
            #     ),
            # )
            mngr.save(
                step=i,
                items={"params": train_state.params, "step": i},
                save_kwargs={"save_args": ocp.SaveArgs(aggregate=True)},
            )
    wandb.finish()

    return train_state


# TODO: add loading from checkpoint

# Start training
if __name__ == "__main__":
    final_state = train()
    print("Training completed and model saved!")

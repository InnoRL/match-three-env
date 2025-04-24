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

NUM_ENVS = 100
GAMMA = 0.99
LAMBDA = 0.98
LEARNING_RATE = 0.00005 # Default is 0.0001
GRADIENT_CLIP = 0.3 # default is 0.5
NUM_UPDATES = 1000
WARMUP_RATIO = 0.01
PRECISION = "bfloat16"
# PRECISION = "float32"

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

if PRECISION == "bfloat16":
    precision_dtype = jnp.bfloat16
else:
    precision_dtype = jnp.float32

class ActorCritic(nn.Module):
    action_dim: int
    precision_dtype: jnp.dtype = precision_dtype

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
        (GRID_SIZE, GRID_SIZE), dtype=jnp.int32
    )  # TODO: verify the actual input shape
    ac = ActorCritic(action_dim=num_actions, precision_dtype=precision_dtype)
    params = ac.init(subkey, dummy_input)
    ac_apply = jax.jit(ac.apply)
    
    print("Dummy input shape: ", dummy_input.shape)
    print("params keys: ", params.keys())
    print("params['params'] keys: ", params['params'].keys())
    print("max steps in episode: ", max_steps_in_episode)
    print("num actions: ", num_actions)
    print("num envs: ", NUM_ENVS)

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
        optax.clip_by_global_norm(GRADIENT_CLIP),
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
        vmap_apply = jax.vmap(ac_apply, in_axes=(None, 0))
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
        # Transpose the rollouts to get from  (max_steps_in_episode, NUM_ENVS, ...) to (NUM_ENVS, max_steps_in_episode, ...)
        rollouts = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), rollouts)
        # print("rollouts shape: ", rollouts["obs_batch"].shape)
        # train_state is not changed in _env_step, so we need to update only the key
        train_state = train_state.replace(key=key)
        # print("Done collecting rollouts")

        # Compute advantages and returns
        # Get (NUM_ENVS) last values
        _, last_values = vmap_apply(train_state.params, rollouts["obs_batch"][:, -1])
        last_values = last_values.squeeze() 

        # NOTE: here rewards, dones, values are (max_steps_in_episode,) and last_value is scalar.
        def _compute_gae_single_env(rewards, dones, values, last_value):
            """Compute GAE for a single environment trajectory."""
            last_advantage = 0.0
            # print("rewards shape: ", rewards.shape)
            # print("dones shape: ", dones.shape)
            # print("values shape: ", values.shape)
            # print("last_value shape: ", last_value.shape)

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
        advantages, returns = jax.vmap(_compute_gae_single_env, in_axes=(0, 0, 0, 0))(
            rollouts["rewards_batch"],  # (NUM_ENVS, max_steps_in_episode)
            rollouts["dones_batch"],  # (NUM_ENVS, max_steps_in_episode)
            rollouts["values_batch"],  # (NUM_ENVS, max_steps_in_episode)
            last_values,  # (NUM_ENVS,)
        )
        # print("Done computing GAE")

        def _single_loss_fn(params, obs, action, old_log_prob, advantage, return_):
            # print("obs shape: ", obs.shape)
            # print("obs dtype: ", obs.dtype)
            # print("action shape: ", action.shape)
            # print("old_log_prob shape: ", old_log_prob.shape)
            # print("advantage shape: ", advantage.shape)
            # print("return_ shape: ", return_.shape)
            
            # print("params keys: ", params.keys())
            # print("params['params'] keys: ", params['params'].keys())
            
            logits, value = ac.apply(params, obs)
            logits = logits.astype(jnp.float32)
            value = value.astype(jnp.float32)
            log_prob = jax.nn.log_softmax(logits)[action]

            advantage = advantage.astype(jnp.float32)
            return_ = return_.astype(jnp.float32)
            ratio = jnp.exp(log_prob - old_log_prob)
            policy_loss1 = ratio * advantage
            policy_loss2 = jnp.clip(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantage
            policy_loss = -jnp.minimum(policy_loss1, policy_loss2).mean()
            value_loss = 0.5 * jnp.square(value - return_).mean()
            entropy = -jnp.sum(
                jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1
            ).mean()
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            return total_loss, (policy_loss, value_loss, entropy)

        def _rollout_loss_fn(params, obs, actions, old_log_probs, advantages, returns):
            # print("obs shape: ", obs.shape)
            # print("actions shape: ", actions.shape)
            # print("old_log_probs shape: ", old_log_probs.shape)
            # print("advantages shape: ", advantages.shape)
            # print("returns shape: ", returns.shape)
            losses, aux = jax.vmap(_single_loss_fn, in_axes=(None, 0, 0, 0, 0, 0))(
                params, obs, actions, old_log_probs, advantages, returns
            )
            total_loss = losses[0].mean()
            policy_loss = aux[0].mean()
            value_loss = aux[1].mean()
            entropy = aux[2].mean()
            return total_loss, (policy_loss, value_loss, entropy)

        def _batch_loss_fn(
            params,
            obs_batch,
            actions_batch,
            old_log_probs_batch,
            advantages_batch,
            returns_batch,
        ):
            # print("obs_batch shape: ", obs_batch.shape)
            # print("actions_batch shape: ", actions_batch.shape)
            # print("old_log_probs_batch shape: ", old_log_probs_batch.shape)
            # print("advantages_batch shape: ", advantages_batch.shape)
            # print("returns_batch shape: ", returns_batch.shape)
            losses, aux = jax.vmap(_rollout_loss_fn, in_axes=(None, 0, 0, 0, 0, 0))(
                params,
                obs_batch,
                actions_batch,
                old_log_probs_batch,
                advantages_batch,
                returns_batch,
            )
            total_loss = losses[0].mean()
            policy_loss = aux[0].mean()
            value_loss = aux[1].mean()
            entropy = aux[2].mean()
            return total_loss, (policy_loss, value_loss, entropy)

        # TODO: will the grads be computed correctly since the return of _loss_fn is a tuple?
        # NOTE: solved by using has_aux=True. This means that the return of _loss_fn is a tuple of (loss, aux).
        grad_fn = jax.value_and_grad(_batch_loss_fn, has_aux=True)
        (total_loss, (policy_loss, value_loss, entropy)), grads = grad_fn(
            train_state.params,
            rollouts["obs_batch"],
            rollouts["actions_batch"],
            rollouts["log_probs_batch"],
            advantages,
            returns,
        )
        # print("Done computing gradients")
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
        # print("Done updating parameters")
        # Logging
        metrics = {
            "learning_rate": learning_rate_scheduler(new_opt_state[1][0].count), # Access count of the adam state
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "returns": returns.mean(),
        }

        return new_train_state, metrics

    # # Run training
    # key = jax.random.PRNGKey(0)
    # train_state = TrainState(env_params, opt_state, key)
    # NOTE: moved to top (see above update_step function)

    # JIT the update step
    jit_update_step_fn = jax.jit(update_step)

    # Setup checkpointing
    ckpt_dir = os.path.abspath(CHECKPOINT_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    # checkpointer = ocp.PyTreeCheckpointer()
    # options = ocp.CheckpointManagerOptions(
    #     max_to_keep=CHECKPOINT_MAX_TO_KEEP, save_interval_steps=CHECKPOINT_INTERVAL
    # )
    # mngr = ocp.CheckpointManager(ckpt_dir, checkpointer, options)
    
    # Initialize AsyncCheckpointer and CheckpointManager
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    options = ocp.CheckpointManagerOptions(
        max_to_keep=CHECKPOINT_MAX_TO_KEEP,
        save_interval_steps=CHECKPOINT_INTERVAL,
        step_prefix="checkpoint_",
    )

    # Initialize CheckpointManager with options
    mngr = ocp.CheckpointManager(ckpt_dir, checkpointer, options=options)
    
    # Training loop
    with tqdm(range(NUM_UPDATES), desc="Training") as pbar:
        for i in pbar:
            # train_state, metrics = update_step(train_state, None)
            train_state, metrics = jit_update_step_fn(train_state, None)

            # Logging
            if i % 10 == 0:
                pbar.set_postfix(
                    {
                        "policy_loss": f"{metrics['policy_loss']:.3f}",
                        "value_loss": f"{metrics['value_loss']:.3f}",
                        "entropy": f"{metrics['entropy']:.3f}",
                    }
                )
                wandb.log(metrics)

            # NOTE: we dont need the condition here since orbax will deal with intervals
            # save_args = ocp.SaveArgs(aggregate=True)
            mngr.save(
                step=i,
                items={"params": train_state.params, "step": i},  # Ensure this is a dictionary
                # save_kwargs={"save_args": save_args},  # Pass SaveArgs correctly
            )
    wandb.finish()

    return train_state


# TODO: add loading from checkpoint

# Start training
if __name__ == "__main__":
    final_state = train()
    print("Training completed and model saved!")

import os
from functools import partial
import shutil

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
# from ae import AutoEncoder
from models import CNN, Actor, Critic
from tqdm import tqdm
from utils import (
    cosine_annealing_with_warmup,
    encode_grid,
    cnn_init,
    small_init,
)
from flax.core import freeze, unfreeze


from match3_env.env import EnvParams, MatchThree
from match3_env.game_grid import GRID_SIZE

NUM_ENVS = 20
GAMMA = 0.95  # 0.99
LAMBDA = 0.95  # 0.98
LEARNING_RATE_ACTOR = 1e-3  # Default is 0.0001
LEARNING_RATE_CRITIC = 1e-3  # Default is 0.0001
LEARNING_RATE_CNN = 3e-4  # Default is 0.0001

COEFFICIENT_ENTROPY = 0.005
COEFFICIENT_VALUE = 0.5
COEFFICIENT_POLICY = 1.0
CLIP_EPSILON = 0.2  # PPO hyperparameter

GRADIENT_CLIP = 1  # default is 0.5
NUM_UPDATES = 5000
WARMUP_RATIO = 0.01
PRECISION = "bfloat16"
# PRECISION = "float32"

CHECKPOINT_DIR = "./checkpoints_final"
AE_CHECKPOINT_DIR = "./checkpoints_ae"
CHECKPOINT_INTERVAL = 100
CHECKPOINT_MAX_TO_KEEP = 5
CHECKPOINT_RESTORE_DIR = None  # "./checkpoints/latest"

# jax.config.update("jax_default_matmul_precision", PRECISION)
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# jax.config.update("jax_default_matmul_precision", 'bfloat16')
jax.config.update('jax_default_matmul_precision', 'tensorfloat32')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"  # Limits memory usage
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"  # Prevents OOM
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Reduces memory fragmentation
jax.config.update('jax_enable_x64', False)  # Disable float64 computations

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


def restore_cnn_params():
    # 1. Point to checkpoint directory
    ckpt_dir = os.path.abspath("checkpoints_ae")

    # 2. Initialize CheckpointManager with new API
    checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    mngr = ocp.CheckpointManager(
        ckpt_dir,
        checkpointer,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=5,
            step_prefix="checkpoint_",
        ),
    )

    # 3. Get latest step
    latest_step = mngr.latest_step()
    if latest_step is None:
        raise ValueError(f"No valid checkpoints found in {ckpt_dir}")

    # 4. Restore with proper args
    # First option: If you saved a single PyTree
    try:
        restored = mngr.restore(latest_step)
        # print(restored["params"]["params"].keys())
        return restored["params"]["params"]["cnn_encoder"]

    # Second option: If you saved multiple items
    except ValueError:
        restore_args = ocp.args.Composite(
            params=ocp.args.PyTreeRestore(), step=ocp.args.PyTreeRestore()
        )
        restored = mngr.restore(latest_step, args=restore_args)
        return restored.params["cnn_encoder"]


# def restore_cnn_params():
#     # 1. Point to the checkpoint directory (parent directory)
#     ckpt_dir = os.path.abspath("checkpoints_ae")

#     # 2. Initialize the checkpointer SAME WAY AS SAVING
#     checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())

#     # 3. Initialize CheckpointManager with SAME OPTIONS
#     options = ocp.CheckpointManagerOptions(
#         max_to_keep=5,  # Match your saving config
#         step_prefix="checkpoint_",  # Must match!
#     )
#     mngr = ocp.CheckpointManager(ckpt_dir, checkpointer, options=options)

#     # 4. Get latest step
#     latest_step = mngr.latest_step()
#     if latest_step is None:
#         raise ValueError(f"No valid checkpoints found in {ckpt_dir}")

#     # 5. Restore JUST THE PARAMS (matches your save structure)
#     restored = mngr.restore(latest_step, items={"params": None, "step": None})
#     return restored["params"]["cnn_encoder"]  # Adjust based on your actual param structure

# Combined network

if PRECISION == "bfloat16":
    precision_dtype = jnp.bfloat16
else:
    precision_dtype = jnp.float32


class ActorCritic(nn.Module):
    action_dim: int
    precision_dtype: jnp.dtype = precision_dtype

    def setup(self):
        self.cnn = CNN(
            self.precision_dtype,
            rl_init_fn=cnn_init,
            latent_dim=1024,
        )
        self.step_embedding = nn.Embed(
            num_embeddings=100,
            features=64,
            dtype=self.precision_dtype
            # embedding_init=small_init,
        )
        self.actor = Actor(
            self.action_dim,
            self.precision_dtype,
            rl_init_fn=partial(cnn_init, scale=0.1),
            small_init_fn=partial(small_init, scale=0.1),
        )
        self.critic = Critic(
            self.precision_dtype,
            rl_init_fn=partial(cnn_init, scale=50.0),
            small_init_fn=partial(small_init, scale=5.0),
        )

    def __call__(self, x, timestep):
        encoded = encode_grid(x)
        features = self.cnn(encoded)
        timestep_embedding = self.step_embedding(timestep).flatten()
        # add the normalized timestep to the features
        features = jnp.concatenate([features, timestep_embedding], axis=-1)
        # print("features shape: ", features.shape)
        print("features type", features.dtype)
        return self.actor(features), self.critic(features)

    @staticmethod
    def get_action(logits, key):
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
    params_behavior: dict
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
    dummy_input = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.int32) # TODO: verify the actual input shape
    ac = ActorCritic(action_dim=num_actions, precision_dtype=precision_dtype)
    params = ac.init(subkey, dummy_input, jnp.zeros(1, dtype=jnp.int32))
    ac_apply = jax.jit(ac.apply)

    # Replace CNN params with pre-trained weights
    params["params"]["cnn"] = restore_cnn_params()

    # print("Dummy input shape: ", dummy_input.shape)
    # print("params keys: ", params.keys())
    # print("params['params'] keys: ", params["params"].keys())
    print("max steps in episode: ", max_steps_in_episode)
    print("num actions: ", num_actions)
    print("num envs: ", NUM_ENVS)
    # print(jax.tree_util.tree_structure(params))

    # NOTE: env_params are jnp.float32, but some of the operations are jnp.bfloat16.
    # This is manual mixed precision, done for performance reasons.

    # Cosine annealing with warmup
    warmup_steps = int(WARMUP_RATIO * NUM_UPDATES)
    total_steps = NUM_UPDATES
    # learning_rate_scheduler = cosine_annealing_with_warmup(
    #     warmup_steps, total_steps, base_lr=LEARNING_RATE
    # )

    # Separate schedulers for actor/critic
    actor_lr_scheduler = cosine_annealing_with_warmup(
        warmup_steps, total_steps, base_lr=LEARNING_RATE_ACTOR
    )
    critic_lr_scheduler = cosine_annealing_with_warmup(
        warmup_steps, total_steps, base_lr=LEARNING_RATE_CRITIC
    )
    cnn_lr_scheduler = cosine_annealing_with_warmup(
        warmup_steps, total_steps, base_lr=LEARNING_RATE_CNN
    )

    # # Initialize optimizer
    # optimizer = optax.chain(
    #     optax.clip_by_global_norm(GRADIENT_CLIP),
    #     optax.adamw(
    #         learning_rate=learning_rate_scheduler,
    #         # NOTE: commented out since they are default values
    #         # b1=0.9,
    #         # b2=0.999,
    #         # weight_decay=0.01,
    #         # eps=1e-8,
    #     ),
    # )

    def label_fn(params):
        def _get_label(path, _):
            path_str = "/".join([p.key for p in path])
            if "cnn" in path_str:
                return "cnn"
                # # Only allow updates to the last layer
                # if "latent_space" in path_str:
                #     return "cnn"
                # else:
                #     return "frozen"  # Mark other CNN layers as frozen
            elif "actor" in path_str:
                return "actor"
            elif "critic" in path_str:
                return "critic"
            return "frozen"  # Default for any other parameters

        return jax.tree_util.tree_map_with_path(
            _get_label,
            params,
            is_leaf=lambda x: False,
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP),
        optax.multi_transform(
            transforms={
                "actor": optax.adamw(learning_rate=actor_lr_scheduler),
                "critic": optax.adamw(learning_rate=critic_lr_scheduler),
                "cnn": optax.adamw(learning_rate=cnn_lr_scheduler),
                "frozen": optax.set_to_zero(),
            },
            param_labels=label_fn,
        ),
    )
    opt_state = optimizer.init(params)
    # print(jax.tree_util.tree_structure(opt_state))

    # Initialize training state
    train_state = TrainState(
        params=params, params_behavior=params, opt_state=opt_state, key=key
    )

    # def update_params(outer_params, opt_state, grads):
    #     inner_params = outer_params["params"]
    #     updates, new_opt_state = optimizer.update(grads, opt_state, inner_params)
    #     new_inner_params = optax.apply_updates(inner_params, updates)
    #     return {"params": new_inner_params}, new_opt_state

    # Training loop
    def update_step(train_state, _) -> tuple[TrainState, dict]:
        # Storage for rollouts for each environment (NUM_ENVS)
        # rollouts = {
        #     "obs_batch": jnp.zeros(
        #         (NUM_ENVS, max_steps_in_episode, GRID_SIZE, GRID_SIZE)
        #     ),
        #     # Actions are categorical
        #     "actions_batch": jnp.zeros(
        #         (NUM_ENVS, max_steps_in_episode), dtype=jnp.int32
        #     ),
        #     "rewards_batch": jnp.zeros(
        #         (NUM_ENVS, max_steps_in_episode), dtype=precision_dtype
        #     ),
        #     "dones_batch": jnp.zeros((NUM_ENVS, max_steps_in_episode)),
        #     "values_batch": jnp.zeros(
        #         (NUM_ENVS, max_steps_in_episode), dtype=precision_dtype
        #     ),
        #     "log_probs_batch": jnp.zeros((NUM_ENVS, max_steps_in_episode)),
        # }

        # Vectorized functions
        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_apply = jax.vmap(ac_apply, in_axes=(None, 0, 0))
        # get_action_fn = lambda params, obs, key: ac.apply(
        #     params, obs, key, method=ac.get_action
        # )
        vmap_get_action = jax.vmap(ac.get_action, in_axes=(0, 0))  # (logits, keys)
        vmap_split = jax.vmap(jax.random.split, in_axes=(0, None))
        # vmap_get_normalized_timestep = jax.vmap(lambda x: jnp.array([x.time/env_params.max_steps_in_episode], dtype=jnp.float32), in_axes=(0))
        vmap_get_normalized_timestep = jax.vmap(lambda x: jnp.array([x.time], dtype=jnp.int32), in_axes=(0))

        # Collect rollout using vmap over NUM_ENVS
        def _env_step(carry, _):
            train_state, keys, obses, states = carry
            keys = vmap_split(
                keys, 2
            )  # IN shape (NUM_ENVS, 2); OUT shape (NUM_ENVS, 2, 2)
            # jax.debug.print("keys shape: {x}", x=keys.shape)
            # print("keys shape: ", keys.shape)
            keys, subkeys = keys[:, 0], keys[:, 1]
            normalized_timesteps = vmap_get_normalized_timestep(states)
            # Get actions and values
            logits, values = vmap_apply(train_state.params_behavior, obses, normalized_timesteps)
            # TODO: check if this is correct. Wont this squeeze batches?
            # NOTE: squeeze removes one or more length-1 axes from array.
            # This means that it should not be a problem
            # jax.debug.print("values shape: {x}", x=values.shape)
            # values = values.squeeze()
            actions = vmap_get_action(logits, subkeys)
            # TODO: check if this will work. do we need to do a vmap here?
            # NOTE: solved by using axis=1
            # print("obses shape: ", obses.shape)
            # print("values shape: ", values.shape)
            # print("logits shape: ", logits.shape)
            # print("actions shape: ", actions.shape)
            log_probs = jax.nn.log_softmax(logits, axis=1)[
                jnp.arange(NUM_ENVS), actions
            ]
            # log_probs = jax.nn.log_softmax(logits)
            # log_probs = jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]

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
                "normalized_timestep": normalized_timesteps,
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
        
        # print("normalized_timestep shape: ", rollouts["normalized_timestep"].shape)
        # print("normalized_timestep: ", rollouts["normalized_timestep"][:, -1])
        # jax.debug.print("normalized_timestep: {x}", x=rollouts["normalized_timestep"][:, -1])
        
        _, last_values = vmap_apply(
            train_state.params_behavior, rollouts["obs_batch"][:, -1], rollouts["normalized_timestep"][:, -1]
        )
        # last_values = last_values.squeeze()

        # NOTE: here rewards, dones, values are (max_steps_in_episode,) and last_value is scalar.
        def _compute_gae_single_env(rewards, dones, values, last_value):
            """Compute GAE for a single environment trajectory."""
            last_advantage = jnp.asarray([0.0])
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
        # normalization_factor = 0.001
        advantages, returns = jax.vmap(_compute_gae_single_env, in_axes=(0, 0, 0, 0))(
            rollouts["rewards_batch"],  # (NUM_ENVS, max_steps_in_episode)
            rollouts["dones_batch"],  # (NUM_ENVS, max_steps_in_episode)
            rollouts["values_batch"],  # (NUM_ENVS, max_steps_in_episode)
            last_values,  # (NUM_ENVS,)
        )
        jax.debug.print("MAX VALUE: {x}", x=jnp.max(rollouts["values_batch"]))
        jax.debug.print("MAX ADVANTAGE: {x}", x=jnp.max(advantages))
        jax.debug.print("MAX RETURN: {x}", x=jnp.max(returns))
        jax.debug.print("MAX REWARD: {x}", x=jnp.max(rollouts["rewards_batch"]))

        def _rollout_loss_fn(
            params,
            obs,
            normalized_timestep,
            actions,
            old_log_probs,
            values,
            advantages,
            returns,
            returns_stats,
        ):
            # ALL SHAPES ARE (max_steps_in_episode, ...)
            # Vectorized computation across all dimensions
            logits, values_target = vmap_apply(params, obs, normalized_timestep)
            logits = logits.astype(jnp.float32)
            values = values.astype(jnp.float32)

            # # Policy loss (vectorized)
            # log_probs = jax.nn.log_softmax(logits)
            # action_log_probs = jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]
            # policy_loss = -jnp.mean(action_log_probs * advantages)
            # --- Policy Loss (PPO Clipping) ---
            log_probs = jax.nn.log_softmax(logits, axis=1)
            # action_log_probs = jnp.take_along_axis(
            #     log_probs, actions[..., None], axis=-1
            # )[..., 0]
            action_log_probs = log_probs[jnp.arange(max_steps_in_episode), actions]
            # print("actions shape: ", actions.shape)
            # print("logits shape: ", logits.shape)
            # action_log_probs = jax.nn.log_softmax(logits, axis=1)[
            #     jnp.arange(max_steps_in_episode), actions
            # ]

            # Ratio of new to old policy probabilities
            ratios = jnp.exp(action_log_probs - old_log_probs)
            # jax.debug.print("ratios min: {x}", x=ratios.min())
            # jax.debug.print("ratios mean: {x}", x=ratios.mean())
            # jax.debug.print("ratios max: {x}", x=ratios.max())

            # Clipped surrogate objective
            clipped_ratios = jnp.clip(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON)
            policy_loss = -jnp.mean(
                jnp.minimum(ratios * advantages, clipped_ratios * advantages)
            )

            # Value loss (vectorized)
            # value_loss = 0.5 * jnp.mean(jnp.square(values - returns))
            # Standardize returns and values
            return_mean, return_std = returns_stats

            # returns = (returns - return_mean) / (return_std + 1e-8)
            # values = (values - return_mean) / (return_std + 1e-8)
            # values_target = (values_target - return_mean) / (return_std + 1e-8)
            returns = returns / 100
            values = values / 100
            values_target = values_target / 100
            # value_loss = optax.huber_loss(values, returns, delta=1.0)
            value_clipped = values + (values_target - values).clip(
                -CLIP_EPSILON, CLIP_EPSILON
            )
            value_loss_1 = optax.huber_loss(values_target, returns, delta=1.0)
            # OR jnp.square(values_target - returns)
            value_loss_2 = optax.huber_loss(value_clipped, returns, delta=1.0)
            # OR jnp.square(value_clipped - returns)
            value_loss = jnp.maximum(value_loss_1, value_loss_2).mean()
            # scale = 1.0 / (return_std + 1e-8)  # Adaptive scaling
            # value_loss = value_loss / (scale ** 2)
            # td_error = values - returns
            # td_error = jnp.clip(td_error, -10.0, 10.0)
            # value_loss = 0.5 * jnp.mean(td_error**2)

            # Entropy (vectorized)
            probs = jax.nn.softmax(
                logits, axis=1
            )  # softmax per each step, hence axis=1
            entropy = -jnp.mean(jnp.sum(probs * jnp.log(probs), axis=1))

            # Total loss
            # total_loss = (
            #     COEFFICIENT_POLICY * policy_loss
            #     + COEFFICIENT_VALUE * value_loss
            #     - COEFFICIENT_ENTROPY * entropy
            # )

            return policy_loss, value_loss, entropy

        def _batch_loss_fn(
            params,
            obs_batch,
            normalized_timestep_batch,
            actions_batch,
            old_log_probs_batch,
            values_batch,
            advantages_batch,
            returns_batch,
        ):
            returns_stats = (returns_batch.mean(), returns_batch.std())
            losses = jax.vmap(_rollout_loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, None))(
                params,
                obs_batch,
                normalized_timestep_batch,
                actions_batch,
                old_log_probs_batch,
                values_batch,
                advantages_batch,
                returns_batch,
                returns_stats,
            )
            policy_loss = losses[0]
            value_loss = losses[1]
            entropy = losses[2]

            total_loss = (
                COEFFICIENT_POLICY * policy_loss
                + COEFFICIENT_VALUE * value_loss
                - COEFFICIENT_ENTROPY * entropy
            )
            total_loss = total_loss.mean()
            policy_loss = losses[0].mean()
            value_loss = losses[1].mean()
            entropy = losses[2].mean()
            return total_loss, (policy_loss, value_loss, entropy)

        # TODO: will the grads be computed correctly since the return of _loss_fn is a tuple?
        # NOTE: solved by using has_aux=True. This means that the return of _loss_fn is a tuple of (loss, aux).
        # returns_for_grads = (returns - returns.mean()) / (returns.std() + 1e-8)  # Standardize returns. done for stability.
        # advantages_for_grads = jnp.clip(advantages, -10.0, 10.0)
        advantages_for_grads = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )  # Standardize advantages. done for stability.
        grad_fn = jax.value_and_grad(_batch_loss_fn, has_aux=True)
        # jax.debug.print("advantage mean: {x}", x=advantages_for_grads.mean())
        # jax.debug.print("advantage std: {x}", x=advantages_for_grads.std())
        # jax.debug.print("advantage min: {x}", x=advantages_for_grads.min())
        # jax.debug.print("advantage max: {x}", x=advantages_for_grads.max())

        (total_loss, (policy_loss, value_loss, entropy)), grads = grad_fn(
            train_state.params,
            rollouts["obs_batch"],
            rollouts["normalized_timestep"],
            rollouts["actions_batch"],
            rollouts["log_probs_batch"],
            rollouts["values_batch"],
            advantages_for_grads,
            returns,
        )
        # print("Done computing gradients")
        # Update parameters
        updates, new_opt_state = optimizer.update(
            grads, train_state.opt_state, params=train_state.params
        )
        new_params = optax.apply_updates(train_state.params, updates)

        # new_params, new_opt_state = update_params(
        #     train_state.params, train_state.opt_state, grads
        # )

        # Update training state
        new_train_state = train_state.replace(
            params=new_params,
            opt_state=new_opt_state,
        )
        # Logging
        true_returns = jnp.mean(jnp.sum(rollouts["rewards_batch"], axis=1))

        # Optimizer states
        critic_state = new_opt_state[1].inner_states["critic"].inner_state[0]

        actor_state = new_opt_state[1].inner_states["actor"].inner_state[0]
        cnn_state = new_opt_state[1].inner_states["cnn"].inner_state[0]

        metrics = {
            "lr/critic": critic_lr_scheduler(critic_state.count),
            "lr/actor": actor_lr_scheduler(actor_state.count),
            "lr/cnn": cnn_lr_scheduler(cnn_state.count),
            "loss/total": total_loss,
            "loss/policy": policy_loss,
            "loss/value": value_loss,
            "stats/entropy": entropy,
            "stats/returns": true_returns,
        }
        return new_train_state, metrics

    # # Run training
    # key = jax.random.PRNGKey(0)
    # train_state = TrainState(env_params, opt_state, key)
    # NOTE: moved to top (see above update_step function)

    # JIT the update step
    jit_update_step_fn = jax.jit(update_step).lower(train_state, None).compile()
    # jit_update_step_fn = update_step

    # Setup checkpointing
    ckpt_dir = os.path.abspath(CHECKPOINT_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    if os.path.exists(ckpt_dir):
        try:
            shutil.rmtree(ckpt_dir)
            print(f"Cleared existing checkpoint directory: {ckpt_dir}")
        except Exception as e:
            print(f"Error clearing directory {ckpt_dir}: {e}")
            raise
    # checkpointer = ocp.PyTreeCheckpointer()
    # options = ocp.CheckpointManagerOptions(
    #     max_to_keep=CHECKPOINT_MAX_TO_KEEP, save_interval_steps=CHECKPOINT_INTERVAL
    # )
    # mngr = ocp.CheckpointManager(ckpt_dir, checkpointer, options)

    # Initialize AsyncCheckpointer and CheckpointManager
    # checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    options = ocp.CheckpointManagerOptions(
        max_to_keep=CHECKPOINT_MAX_TO_KEEP,
        save_interval_steps=CHECKPOINT_INTERVAL,
        step_prefix="checkpoint_",
    )
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    # orbax_checkpointer.save('/tmp/flax_ckpt/orbax/single_save', ckpt, save_args=save_args)

    # # Initialize CheckpointManager with options
    mngr = ocp.CheckpointManager(ckpt_dir, orbax_checkpointer, options=options)

    # Training loop
    with tqdm(range(NUM_UPDATES), desc="Training") as pbar:
        for i in pbar:
            # train_state, metrics = update_step(train_state, None)
            train_state, metrics = jit_update_step_fn(train_state, None)

            # Logging
            if i % 10 == 0:
                pbar.set_postfix(
                    {
                        "policy_loss": f"{metrics['loss/policy']:.3f}",
                        "value_loss": f"{metrics['loss/value']:.3f}",
                        "entropy": f"{metrics['stats/entropy']:.3f}",
                    }
                )
            if i % 5 == 0:
                train_state = train_state.replace(params_behavior=train_state.params)
            wandb.log(metrics)

            # NOTE: we dont need the condition here since orbax will deal with intervals
            # save_args = ocp.SaveArgs(aggregate=True)
            # Initialize CheckpointManager with options

            mngr.save(
                step=i,
                items={
                    "params": train_state.params["params"],
                    # "step": i,
                },  # Ensure this is a dictionary
                # save_kwargs={"save_args": save_args},  # Pass SaveArgs correctly
            )
    wandb.finish()

    return train_state


# TODO: add loading from checkpoint

# Start training
if __name__ == "__main__":
    print("JAX version:", jax.__version__)
    print("Backend:", jax.lib.xla_bridge.get_backend().platform)
    print("CUDA visible:", jax.lib.xla_bridge.get_backend().platform_version)
    final_state = train()
    print("Training completed and model saved!")

import os
from functools import partial

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from models import CNN, CNNDecoder
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

NUM_ENVS = 10000
LEARNING_RATE_ENCODER = 1e-4  # Default is 0.0001
LEARNING_RATE_DECODER = 1e-3  # Default is 0.0001

GRADIENT_CLIP = 1  # default is 0.5
NUM_UPDATES = 1000
WARMUP_RATIO = 0.05
PRECISION = "bfloat16"
# PRECISION = "float32"

CHECKPOINT_DIR = "./checkpoints_ae"
CHECKPOINT_INTERVAL = 100
CHECKPOINT_MAX_TO_KEEP = 5
CHECKPOINT_RESTORE_DIR = None  # "./checkpoints/latest"

K_SYMBOLS_MAX = 8

jax.config.update("jax_default_matmul_precision", PRECISION)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Wandb
wandb.init(
    project="match-three",
    group="InnoRL",
    name="match-three-ae",
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


class AutoEncoder(nn.Module):
    precision_dtype: jnp.dtype = precision_dtype
    latent_dim: int = 2048
    k_symbols: int = 4

    def setup(self):
        self.cnn_encoder = CNN(
            self.precision_dtype, rl_init_fn=cnn_init, latent_dim=self.latent_dim
        )
        self.cnn_decoder = CNNDecoder(
            self.precision_dtype, rl_init_fn=cnn_init, k_symbols=self.k_symbols
        )

    def __call__(self, x):
        encoded = encode_grid(x)
        cnn_encoder = self.cnn_encoder(encoded)
        cnn_decoder = self.cnn_decoder(cnn_encoder)
        return cnn_decoder
    
    @staticmethod
    def apply_static_encoder(x):
        encoded = encode_grid(x)
        return encoded


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

    print("env_params: ", env_params)
    # k_symbols = env_params.grid_params.num_symbols + 1
    k_symbols = K_SYMBOLS_MAX

    # Initialize networks
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    dummy_input = jnp.zeros(
        (GRID_SIZE, GRID_SIZE), dtype=jnp.int32
    )  # TODO: verify the actual input shape
    ae = AutoEncoder(precision_dtype=precision_dtype, k_symbols=k_symbols)
    params = ae.init(subkey, dummy_input)
    ae_apply = jax.jit(ae.apply)

    print("Dummy input shape: ", dummy_input.shape)
    print("params keys: ", params.keys())
    print("params['params'] keys: ", params["params"].keys())
    print("num envs: ", NUM_ENVS)
    print(jax.tree_util.tree_structure(params))

    # Cosine annealing with warmup
    warmup_steps = int(WARMUP_RATIO * NUM_UPDATES)
    total_steps = NUM_UPDATES

    # Separate schedulers for actor/critic
    encoder_lr_scheduler = cosine_annealing_with_warmup(
        warmup_steps, total_steps, base_lr=LEARNING_RATE_ENCODER
    )
    decoder_lr_scheduler = cosine_annealing_with_warmup(
        warmup_steps, total_steps, base_lr=LEARNING_RATE_DECODER
    )

    def label_fn(params):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: ("encoder" if "cnn_encoder" in path else "decoder"),
            params,
            is_leaf=lambda x: False,
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(GRADIENT_CLIP),
        optax.multi_transform(
            transforms={
                "encoder": optax.adamw(learning_rate=encoder_lr_scheduler),
                "decoder": optax.adamw(learning_rate=decoder_lr_scheduler),
            },
            param_labels=label_fn,
        ),
    )
    opt_state = optimizer.init(params)
    print(jax.tree_util.tree_structure(opt_state))

    # Initialize training state
    train_state = TrainState(
        params=params, opt_state=opt_state, key=key
    )

    # Vectorized functions
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    # vmap_apply = jax.vmap(ae_apply, in_axes=(None, 0))
    vmap_apply_static_encoder = jax.vmap(AutoEncoder.apply_static_encoder, in_axes=(0,))

    # Training loop
    def update_step(train_state, _) -> tuple[TrainState, dict]:

        # Collect data using vmap over NUM_ENVS
        def _generate_data(train_state, subkeys):
            obses, _ = vmap_reset(subkeys, env_params)
            
            data = {
                "obs_batch": obses,
                # "input_batch": inputs,
            }
            return data

        # Generate keys
        key = train_state.key
        # add one for key, NUM_ENVS for subkeys
        keys = jax.random.split(key, num=(NUM_ENVS + 1))
        key = keys[0]
        subkeys = keys[1:]
        # print("Subkeys shape: ", subkeys.shape)
        # Generate data
        data = _generate_data(train_state, subkeys)
        train_state = train_state.replace(key=key)
        
        print("data['obs_batch'] shape: ", data["obs_batch"].shape)

        # Compute losses
        def _reconstruction_loss_fn(
            params,
            obs_batch,
        ):
            outputs_logits = ae_apply(params, obs_batch)
            print("obs_batch: ", obs_batch.shape)
            inputs = ae.apply_static_encoder(obs_batch)            
            
            inputs = inputs.reshape(-1, k_symbols).astype(jnp.float32)
            outputs_logits = outputs_logits.reshape(-1, k_symbols).astype(jnp.float32)
            outputs = nn.softmax(outputs_logits, axis=-1)
            
            print("outputs_logits: ", outputs_logits.shape)
            print("inputs: ", inputs.shape)
            print("outputs: ", outputs.shape)
            # CCE, MSE, MAE
            cce_loss = optax.softmax_cross_entropy(outputs_logits, inputs).mean()#-jnp.mean(inputs * jnp.log(outputs + 1e-8))
            mse_loss = optax.l2_loss(outputs, inputs).mean() # jnp.mean((inputs - outputs) ** 2)
            mae_loss = jnp.mean(jnp.abs(inputs - outputs))
            return cce_loss, mse_loss, mae_loss

        def _batch_loss_fn(
            params,
            obs_batch,
        ):
            # outputs = ae_apply(params, obs_batch)
            
            # inputs = vmap_apply_static_encoder(obs_batch)            
            # inputs = inputs.astype(jnp.float32)
            # outputs = outputs.astype(jnp.float32)
            
            losses = jax.vmap(
                _reconstruction_loss_fn, in_axes=(None, 0)
            )(
                params,
                obs_batch,
            )
            cce_loss = losses[0].mean()
            mse_loss = losses[1].mean()
            mae_loss = losses[2].mean()

            return cce_loss, (mse_loss, mae_loss)

        grad_fn = jax.value_and_grad(_batch_loss_fn, has_aux=True)

        (cce_loss, (mse_loss, mae_loss)), grads = grad_fn(
            train_state.params,
            data["obs_batch"],
        )
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
        # Optimizer states
        encoder_state = new_opt_state[1].inner_states["encoder"].inner_state[0]
        decoder_state = new_opt_state[1].inner_states["decoder"].inner_state[0]

        metrics = {
            "lr/decoder": decoder_lr_scheduler(decoder_state.count),
            "lr/encoder": encoder_lr_scheduler(encoder_state.count),
            "loss/cce": cce_loss,
            "loss/mse": mse_loss,
            "loss/mae": mae_loss,
        }
        return new_train_state, metrics

    # Run training
    # JIT the update step
    jit_update_step_fn = jax.jit(update_step).lower(train_state, None).compile()

    # Setup checkpointing
    ckpt_dir = os.path.abspath(CHECKPOINT_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    
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
                        "MSE": f"{metrics['loss/mse']:.3f}",
                        "MAE": f"{metrics['loss/mae']:.3f}",
                        "CCE": f"{metrics['loss/cce']:.3f}",
                    }
                )
            wandb.log(metrics)

            # NOTE: we dont need the condition here since orbax will deal with intervals
            # save_args = ocp.SaveArgs(aggregate=True)
            mngr.save(
                step=i,
                items={
                    "params": train_state.params,
                    "step": i,
                },  # Ensure this is a dictionary
                # save_kwargs={"save_args": save_args},  # Pass SaveArgs correctly
            )
    wandb.finish()

    return train_state


# TODO: add loading from checkpoint

# Start training
if __name__ == "__main__":
    final_state = train()
    print("Training completed and model saved!")

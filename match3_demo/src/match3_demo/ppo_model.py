import os
import sys
from typing import Callable
import jax
import numpy as np
import orbax.checkpoint as ocp


from jax import jit, devices
from jax import numpy as jnp
from jax.sharding import SingleDeviceSharding

from orbax.checkpoint import CheckpointManager, args


from pathlib import Path

# 1. Get the absolute path to the parent directory (match-three-env)
parent_dir = str(Path(__file__).parent.parent.parent.parent.absolute())
print("PARENT DIR", parent_dir)

# 2. Add to Python path if not already present
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    print(f"Added to path: {parent_dir}")

from examples.algorithms.a2c_ppo import ActorCritic

CHECKPOINT_DIR = "../examples/checkpoints_final"


class PPOModel:
    def __init__(self):
        self.params = self.__restore_ppo_params()
        self.apply = self.__get_apply()

    def get_action(self, observation, timestep):
        return self.apply(observation, timestep)

    # @staticmethod
    # def __restore_ppo_params():
    #     # 1. Point to checkpoint directory
    #     ckpt_dir = os.path.abspath(CHECKPOINT_DIR)

    #     # 2. Initialize CheckpointManager with new API
    #     checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    #     mngr = ocp.CheckpointManager(
    #         ckpt_dir,
    #         checkpointer,
    #         options=ocp.CheckpointManagerOptions(
    #             max_to_keep=5,
    #             step_prefix="checkpoint_",
    #         ),
    #     )

    #     # 3. Get latest step
    #     latest_step = mngr.latest_step()
    #     if latest_step is None:
    #         raise ValueError(f"No valid checkpoints found in {ckpt_dir}")

    #     # 4. Restore with proper args
    #     # First option: If you saved a single PyTree
    #     try:
    #         print("HERE! LATEST STEP", latest_step)
    #         restored = mngr.restore(latest_step)
    #         # print(restored["params"]["params"].keys())
    #         return restored["params"]["params"]

    #     # Second option: If you saved multiple items
    #     except ValueError:
    #         restore_args = ocp.args.Composite(
    #             params=ocp.args.PyTreeRestore(), step=ocp.args.PyTreeRestore()
    #         )
    #         restored = mngr.restore(latest_step, args=restore_args)
    #         return restored.params

    @staticmethod
    def __restore_ppo_params():
        """
        Load model checkpoint using Orbax's CheckpointManager.

        Args:
            ckpt_dir (str): The directory where checkpoints are stored.
            checkpoint_step (int): The specific checkpoint step to load.
            state_type (PyTreeType): The type of the model state (e.g., the train_state class).

        Returns:
            train_state: The loaded model state.
        """
        ckpt_dir = os.path.abspath(CHECKPOINT_DIR)

        # Initialize the CheckpointManager with the same options used in saving
        options = ocp.CheckpointManagerOptions(
            step_prefix="checkpoint_",
        )

        checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
        mngr = ocp.CheckpointManager(ckpt_dir, checkpointer, options=options)
        latest_step = mngr.latest_step()
        structure = mngr.metadata(latest_step)
        print("Checkpoint Dir", ckpt_dir)
        print("Checkpoint Step", latest_step)
        print("Checkpoint Structure", structure)
        # meta = structure.item_metadata["params"].tree
        cpu = jax.devices("cpu")[0]
        cpu_sharding = SingleDeviceSharding(cpu)
        
        item_metadata = structure.item_metadata
        print("Item Metadata", item_metadata)

        # Assuming 'params' are the object we want to restore
        params_meta = item_metadata.get('params', None)
        if params_meta is None:
            raise ValueError("Params not found in item_metadata.")
        
        # Restoring using StandardRestore with the proper arguments
        # restore_args = args.StandardRestore(params_meta)  # Use the appropriate restore method for params
        restore_args = ocp.PyTreeRestoreArgs(item=params_meta)

        
        # Perform the restore
        restored_params = mngr.restore(step=latest_step, args=restore_args)["params"]
        
        print("Restored Params:", restored_params)



        # target = jax.tree_util.tree_map(lambda x: np.asarray(x), structure.item_metadata.tree)
        # Load the checkpoint for the specified step
        checkpoint_data = mngr.restore(
            latest_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(structure.item_metadata.tree),
            )
            # args=ocp.args.PyTreeRestore(
            #     restore_kwargs={
            #         "restore_args": jax.tree.map(
            #             lambda _: ocp.RestoreArgs(restore_type=np.ndarray), structure
            #         )
            #     },
            # ),
        )

        # Ensure the checkpoint contains the right structure (params)
        if "params" not in checkpoint_data:
            raise ValueError(f"Checkpoint does not contain 'params' key")

        # Load the state (train_state) from the checkpoint
        # loaded_state = state_type(params=checkpoint_data['params'])
        loaded_state = checkpoint_data["params"]

        return loaded_state

    @staticmethod
    def __get_apply():
        model = ActorCritic(action_dim=144, precision_dtype=jnp.bfloat16)
        return jit(model.apply)

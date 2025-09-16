import os 
import socket
import numpy as np

import matplotlib.pyplot as plt

import jax 
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh

from flax.training import train_state

from utils.datautils import trim_to_batches, batch_iterator, prefetch_batches

class ModelTrainState(train_state.TrainState):
    pass

def train_step(userloss):
    @jax.jit
    def func(state, batch, rng, *args, **kwargs):
        rng, dropout_key = jax.random.split(rng)

        def loss_fn(training_params):
            preds = state.apply_fn(
                training_params,
                batch['enc'],
                batch['dec_in'],
                deterministic=False,
                rngs={'dropout': dropout_key}
            )
            loss = userloss(preds, batch['dec_target'], *args, **kwargs)
            return loss

        # Loss + grads
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        # Update state
        state = state.apply_gradients(grads=grads)
        return state, loss
    return func

def eval_step(userloss):
    @jax.jit
    def func(state, batch, *args, **kwargs):
        """
        state: TrainState
        batch: dict with "x" and "y"
            x: (batch, time, input_features)
            y: (batch, features)
        """
        preds = state.apply_fn(
            state.params,
            batch['enc'],
            batch['dec_in'],
            deterministic=True,
        )
        preds.astype(jnp.float32)
        loss = userloss(preds, batch['dec_target'], *args, **kwargs)

        return loss, preds
    return func

class TrainModel():
    def __init__(self):
        # initialize distributed environment
        visible_devices = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]          

        jax.distributed.initialize(
            local_device_ids=visible_devices
        )

        print(f"[JAX] ProcID: {jax.process_index()}")
        print(f"[JAX] Local devices: {jax.local_devices()}")
        print(f"[JAX] Global devices: {jax.devices()}")

        self.id = jax.process_index()
        self.n_local_devices = jax.local_device_count()
        self.n_total_devices = jax.device_count()

        print(f"[JAX] Host {self.id} Name: {socket.gethostname()} sees {self.n_local_devices} local devices")
    
    def set_training_params(self, training_params):
        self.model = training_params["model"]
        self.train_set = training_params["train_set"]
        self.val_set = training_params["val_set"]
        self.n_epochs = training_params["n_epochs"]
        self.optimizer = training_params["optimizer"]
        self.loss_fn = training_params["loss_fn"]
        self.device_batches = training_params["device_batches"]
        self.batch_size = self.device_batches * self.n_total_devices
        self.seed = training_params["seed"]
        
        for split in [self.train_set, self.val_set]:
            split["enc"] = trim_to_batches(split["enc"], self.device_batches)
            split["dec_in"] = trim_to_batches(split["dec_in"], self.device_batches)
            split["dec_target"] = trim_to_batches(split["dec_target"], self.device_batches)

        self.key = jax.random.PRNGKey(self.seed)
        enc_in = jnp.zeros((self.batch_size, *self.train_set["enc"].shape[1:]))
        dec_in = jnp.zeros((self.batch_size, *self.train_set["dec_in"].shape[1:]))

        self.model_params = self.model.init(self.key, enc_in, dec_in)
        self.state = ModelTrainState.create(
            apply_fn=self.model.apply, params=self.model_params, tx=self.optimizer
        )

        print("[TrainModel] Initialized model parameters")
        print("[TrainModel] Encoder input shape:", enc_in.shape)
        print("[TrainModel] Decoder input shape:", dec_in.shape)
        print("[TrainModel] Model output shape:", self.model.apply(self.model_params, enc_in, dec_in).shape)

    def _set_sharded_functions(self):
        mesh = Mesh(jax.devices(), ('batch',))

        # Sharding specs
        param_spec = P()# fully replicated parameters
        in_spec = P('batch', None, None)  # shard batch dimension
        out_spec = P('batch', None, None, None)  # shard batch, rest replicated
        rng_spec = P()                     # RNG fully replicated

        in_sharding = NamedSharding(mesh, in_spec)
        out_sharding = NamedSharding(mesh, out_spec)
        param_sharding = NamedSharding(mesh, param_spec)
        rng_sharding = NamedSharding(mesh, rng_spec)

        # Train step sharded
        self.p_train_step = jax.jit(
            train_step(self.loss_fn),
            in_shardings=(
                param_sharding, in_sharding, rng_sharding
            ),
            out_shardings=(
                param_sharding,  # updated params (replicated)
                param_sharding   # loss (replicated)
            )
        )

        # Eval step sharded
        self.p_eval_step = jax.jit(
            eval_step(self.loss_fn),
            in_shardings=(
                param_sharding, in_sharding   # batch tuple
            ),
            out_shardings=(
                param_sharding,    # loss (replicated)
                out_sharding  # predictions sharded along batch
            )
        )
    
    def train(self, in_stations, out_stations):
        self._set_sharded_functions()
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.n_epochs):
            # Regular batch iterator
            train_gen = batch_iterator(self.train_set, batch_size=self.batch_size, shuffle=True)
            train_prefetch = prefetch_batches(train_gen, prefetch_size=2)

            # Regular batch iterator
            val_gen = batch_iterator(self.val_set, batch_size=self.batch_size, shuffle=True)
            val_prefetch = prefetch_batches(val_gen, prefetch_size=2)

            # Training phase
            train_loss = []
            val_loss = []

            for batch in train_prefetch:
                batch = {k: v for k, v in batch.items() if k != "time"}
                state, loss = self.p_train_step(self.state, batch, self.key)
                train_loss.append(loss)

            # Validation phase - No teacher forcing
            for batch in val_prefetch:
                batch = {k: v for k, v in batch.items() if k != "time"}
                batch_size, dec_len, n_signals = batch['dec_target'].shape

                # Initialize decoder input with zeros
                dec_in = jnp.zeros((batch_size, dec_len, n_signals))

                # Set first timestep to last encoder step
                dec_in = dec_in.at[:, 0, :].set(batch['enc'][:, -1, out_stations])

                # Now you can pass it to your eval function
                batch = {
                    'enc': batch['enc'],
                    'dec_in': dec_in,
                    'dec_target': batch['dec_target']
                }

                loss, _ = self.p_eval_step(self.state, batch)
                val_loss.append(loss)

            # Compute epoch averages

            loss_train = np.mean(train_loss)
            loss_val = np.mean(val_loss)

            self.train_losses.append(loss_train)
            self.val_losses.append(loss_val)
            
            if self.id == 0:
                print(f"_"*65)
                print(f"[Training loop] Host Name {socket.gethostname()}")
                print(f"[Training loop] Epoch {epoch+1}")
                print(f"[Training loop] Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}")

    def plot_losses(self):
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(self.train_losses, label="Train Loss")
        ax.plot(self.val_losses, label="Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25, which="both")
        ax.set_yscale("log")
        fig.legend()
        fig.savefig(f"images/{self.model.__name__}_loss.png")
        plt.close()

    def save_checkpoint(self):
        a = 0
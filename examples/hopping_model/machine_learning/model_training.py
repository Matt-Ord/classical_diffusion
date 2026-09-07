import os

import jax.numpy as jnp
import optax
from scipy.constants import Boltzmann

from classical_diffusion.jax.langevin import (
    KramersParameters,
)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"

import equinox as eqx
import jax
import numpy as np

jax.config.update("jax_enable_x64", val=False)


CHECK_EVERY = 100
EARLY_STOP = 0.0001  # Improvement to loss over CHECK_EVERY epochs deemed small enough to have reached training plateau
NUM_EPOCHS = 2000
BATCH_SIZE = (
    500  # Number of trajectories to test on at a time (does this need to be limited?)
)


class ResidualBlock(eqx.Module):
    """Residual Block in ResNet architecture. H(x) = F(x) + x."""

    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, dim: int, *, key: jax.Array) -> None:
        key1, key2 = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(dim, dim, key=key1)
        self.linear2 = eqx.nn.Linear(dim, dim, key=key2)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run Residual Block layers."""
        residual = x
        x = jax.nn.relu(self.linear1(x))
        x = self.linear2(x)
        # Residual Net: activated linear F(x) is added to shortcut connection, residual
        return jax.nn.relu(x + residual)


class ResNet(eqx.Module):
    """ResNet model."""

    input_layer: eqx.nn.Linear
    residual_block: ResidualBlock
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        key: jax.Array,
    ) -> None:
        input_key, residual_block_key, lin1_key, lin2_key, output_key = (
            jax.random.split(key, 5)
        )

        # Project input channel up to hidden layer channels
        self.input_layer = eqx.nn.Linear(
            in_features=1, out_features=hidden_dim, key=input_key
        )

        # Run residual block
        self.residual_block = ResidualBlock(dim=hidden_dim, key=residual_block_key)

        self.linear1 = eqx.nn.Linear(hidden_dim, hidden_dim, key=lin1_key)

        self.linear2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=lin2_key)

        # Project hidden channels down to output
        self.output_layer = eqx.nn.Linear(hidden_dim, 1, key=output_key)

    def __call__(self, x: jnp.ndarray) -> jax.Array:
        """Propagate input through model layers."""
        # Normalise inputs
        # x = x.at[4].set(
        #     x[4] * Boltzmann
        # )  # Turn temperature into kbt for simplified calculation

        # Run model
        x = jax.nn.relu(self.input_layer(x))  # shape = (hidden_dim,)
        x = jax.nn.relu(self.residual_block(x))  # shape = (hidden_dim,)
        x = jax.nn.relu(self.linear1(x))
        x = jax.nn.relu(self.linear2(x))
        return self.output_layer(x)  # shape = (1,)


default_params = KramersParameters(
    omega_well=1.0,
    omega_barrier=1.0,
    barrier_energy=3.0,
    m=1.0,
    temperature=0.5 / Boltzmann,
    gamma=0.1,
)


@eqx.filter_jit
def loss_fn(
    model: eqx.Module,
    *,
    test_data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jax.Array:
    """Loss function for an ISF hopping rate prediction model."""
    print("\nCompile Loss Function\n")

    test_params, test_hop_times, test_num_hops = test_data
    # Test params: Array of parameters sets in the test data
    # Test hop times: Array of the corresponding hop times derived from mean dwell time in filtered trajectory
    # Test num hops: Array of the corresponding number of hops (i.e. weighting of error) in filtered trajectory

    # Pass batched params through the model to predict hopping rates and isf offsets
    model_hop_times = jax.vmap(model, (0))(test_params[:, 0:1])  # ty: ignore[invalid-argument-type]

    reshaped_model_hop_times = model_hop_times.squeeze(-1)

    diff = test_hop_times - reshaped_model_hop_times

    loss = (diff**2) * test_num_hops

    # Return the average error
    return jnp.mean(loss)


def train_model(
    training_data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    *,
    resume: bool = False,
) -> eqx.Module:
    """Train a ResNet model."""
    # Set up constants
    key = jax.random.PRNGKey(1)

    fresh_model = ResNet(hidden_dim=16, key=key)
    if resume:
        model = eqx.tree_deserialise_leaves("model_checkpoint.eqx", fresh_model)
    else:
        model = fresh_model

    optimizer = optax.adam(learning_rate=1e-3)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    training_params, training_hop_times, training_num_hops = training_data

    # Define number batches per epoch
    num_trajectories = len(training_params)
    num_batches = num_trajectories // BATCH_SIZE
    usable_len = num_batches * BATCH_SIZE

    loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)

    def batch_step(
        carry: tuple[eqx.Module, optax.OptState],
        batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[eqx.Module, optax.OptState], jnp.ndarray]:
        model, opt_state = carry
        batch_params, batch_hop_times, batch_num_hops = batch

        batch_loss, gradients = loss_and_grad_fn(
            model, test_data=(batch_params, batch_hop_times, batch_num_hops)
        )

        updates, opt_state = optimizer.update(gradients, opt_state)

        model = eqx.apply_updates(model, updates)

        return (model, opt_state), batch_loss

    @eqx.filter_jit
    def run_epoch(
        model: eqx.Module,
        opt_state: optax.OptState,
        params: jnp.ndarray,
        hop_times: jnp.ndarray,
        num_hops: jnp.ndarray,
    ) -> tuple[eqx.Module, optax.OptState, jnp.ndarray]:
        """Run one epoch of ML algorithm."""
        batched_params = params[:usable_len].reshape(
            num_batches, BATCH_SIZE, *params.shape[1:]
        )
        batched_hop_times = hop_times[:usable_len].reshape(
            num_batches, BATCH_SIZE, *hop_times.shape[1:]
        )
        batched_num_hops = num_hops[:usable_len].reshape(
            num_batches, BATCH_SIZE, *num_hops.shape[1:]
        )

        (model, opt_state), batch_losses = jax.lax.scan(
            batch_step,
            (model, opt_state),
            (batched_params, batched_hop_times, batched_num_hops),
        )

        return model, opt_state, jnp.sum(batch_losses)

    def control_epoch(
        carry: tuple[jax.Array, ResNet, optax.OptState, np.ndarray, int],
    ) -> tuple[tuple[jax.Array, ResNet, optax.OptState, np.ndarray, int], bool]:
        key, model, optimizer_state, losses, epoch = carry

        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, num_trajectories)

        shuffled_params = training_params[permutation]
        shuffled_hop_times = training_hop_times[permutation]
        shuffled_num_hops = training_num_hops[permutation]

        model, optimizer_state, epoch_loss = run_epoch(
            model,
            optimizer_state,
            shuffled_params,
            shuffled_hop_times,
            shuffled_num_hops,
        )

        losses = np.append(losses, float(epoch_loss))
        print(f"Epoch {epoch + 1:03d} | Loss: {epoch_loss:.5f}")

        flag = False
        if (epoch + 1) % CHECK_EVERY == 0:
            eqx.tree_serialise_leaves("model_checkpoint.eqx", model)
            loss_difference = losses[-CHECK_EVERY] - epoch_loss
            if len(losses) >= CHECK_EVERY and abs(loss_difference) < EARLY_STOP:
                print("Early stopping triggered: plateau reached.")
                flag = True

        return ((key, model, optimizer_state, losses, epoch + 1), flag)  # ty: ignore[invalid-return-type]

    # Control loop outside jit
    carry = (key, model, optimizer_state, np.array([]), 0)
    for _epoch in range(NUM_EPOCHS):
        carry, flag = control_epoch(carry)
        if flag:
            break

    # Return trained model
    return model

import os

import jax.numpy as jnp
import numpy as np
import optax
from matplotlib import ticker
from scipy.constants import Boltzmann

from classical_diffusion.hopping import (
    CanonicalLattice,
    Lattice1D,
    get_deterministic_isf,
    get_deterministic_probabilities,
    get_kramers_rate,
)
from classical_diffusion.jax import get_measured_data, get_pairwise_isf
from classical_diffusion.jax.hopping import (
    get_deterministic_isf as get_deterministic_isf_jax,
)
from classical_diffusion.jax.hopping import (
    get_deterministic_probabilities as get_deterministic_probabilities_jax,
)
from classical_diffusion.jax.langevin import (
    KramersParameters,
    KramersSystem1D,
    get_isf_offset,
)
from classical_diffusion.jax.langevin import (
    solve_many_overdamped as solve_many_overdamped_jax,
)
from classical_diffusion.plot import get_fancy_figure, get_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import timed

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"

import operator
import pickle  # ruff: ignore[suspicious-pickle-import]
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", False)  # ruff: ignore[boolean-positional-value-in-call]


if TYPE_CHECKING:
    from collections.abc import Callable

    from classical_diffusion.langevin import CanonicalSystem

CHECK_EVERY = 1
EARLY_STOP = 1  # Improvement to loss over CHECK_EVERY epochs deemed small enough to have reached training plateau
NUM_EPOCHS = 100
BATCH_SIZE = 16  # Number of isfs to test on at a time (does this need to be limited?)
LOSS_BATCH_SIZE = 8  # Number of isfs to calculate loss of at a time


class JaxEnsembleResults(TypedDict):
    """Jax compatible output from an ensemble run."""

    parameters: KramersParameters
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating[Any]]],
        np.ndarray[Any, np.dtype[np.floating[Any]]],
    ]
    result: tuple[jnp.ndarray, jnp.ndarray]


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
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        key: jax.Array,
    ) -> None:
        input_key, residual_block_key, output_key = jax.random.split(key, 3)

        # Project input channel up to hidden layer channels
        self.input_layer = eqx.nn.Linear(
            in_features=6, out_features=hidden_dim, key=input_key
        )

        # Run residual block
        self.residual_block = ResidualBlock(dim=hidden_dim, key=residual_block_key)

        # Project hidden channels down to output
        self.output_layer = eqx.nn.Linear(hidden_dim, 2, key=output_key)
        self.output_layer = eqx.tree_at(
            lambda l: l.bias,  # ruff: ignore[ambiguous-variable-name]
            self.output_layer,
            jnp.array([1.0, 0.0]),  # Sets initial hop_time log-scale bias
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Propagate input through model layers."""
        # Normalise inputs
        x = x.at[4].set(
            x[4] * Boltzmann
        )  # Turn temperature into kbt for simplified calculation
        x_mean = jnp.mean(x, axis=0)
        x_std = jnp.std(x, axis=0) + 1e-6
        x = (x - x_mean) / x_std

        # Run model
        x = jax.nn.relu(self.input_layer(x))  # shape = (hidden_dim,)
        x = self.residual_block(x)  # shape = (hidden_dim,)
        x = self.output_layer(x)  # shape = (2,)

        hop_time = jnp.exp(x[0])
        offset = 1.0
        return jnp.array([hop_time, offset])


# Model functions


@jax.checkpoint
@jax.jit
def get_deterministic_isf_directly(
    hop_time: float, time_span: TimeSpan, delta_k: float
) -> jnp.ndarray:
    """Get the isf from hop time directly, without vmapping over get probabilities and get isf."""
    lattice = CanonicalLattice(1.0, hop_time)

    probabilities, _ = get_deterministic_probabilities_jax(
        lattice, time_span, shape=(1000,)
    )

    return get_deterministic_isf_jax(lattice, probabilities, (delta_k,))


@eqx.filter_jit
def loss_fn(  # ruff: ignore[too-many-arguments]
    model: eqx.Module,
    *,
    time_span: TimeSpan,
    delta_k: float,
    test_isfs: jnp.ndarray,
    test_offsets: jnp.ndarray,
    test_params: jnp.ndarray,
) -> jax.Array:
    """Loss function for an ISF hopping rate prediction model."""
    print("\nCompile Loss Function\n")

    # Index of 1s in times
    index_1s = time_span.n_steps // time_span.t_end
    index_half_s = int(0.5 * index_1s)
    index_5s = int(5 * index_1s)

    # Pass batched isfs through the model to predict hopping rates and isf offsets
    predictions = jax.vmap(model, (0))(test_params)  # ty: ignore[invalid-argument-type]
    hopping_times = predictions[:, 0]

    total_samples = hopping_times.shape[0]
    num_chunks = total_samples // LOSS_BATCH_SIZE
    usable_samples = num_chunks * LOSS_BATCH_SIZE

    hop_time_chunks = hopping_times[:usable_samples].reshape(
        num_chunks, LOSS_BATCH_SIZE
    )

    # Parallel evaluation inside each batch via vmap
    def _process_loss_batch(hop_time_chunk: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(get_deterministic_isf_directly, (0, None, None))(
            hop_time_chunk, time_span, delta_k
        )

    # Sequential execution over micro-batches via lax.map
    isfs = jax.lax.map(_process_loss_batch, hop_time_chunks)
    isfs = isfs.reshape(usable_samples, -1)

    corrected_isfs = test_offsets[:usable_samples, None] * isfs
    targets = test_isfs[:usable_samples]

    # Calculate the error by
    errors = jnp.mean(
        (corrected_isfs[:, index_half_s:index_5s] - targets[:, index_half_s:index_5s])
        ** 2,
        axis=-1,
    )

    # Return the average error
    return jnp.mean(errors)


def get_hopping_time_and_offset(
    model: ResNet, params: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the hopping time and initial offset from a pre-trained model."""
    outputs = model(params)
    return outputs[0], outputs[1]


# Training functions


def train_model(  # ruff: ignore[too-many-locals]
    training_data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    *,
    time_span: TimeSpan,
    delta_k: float,
    resume: bool = False,
) -> ResNet:
    """Train a ResNet model."""
    # Set up constants
    key = jax.random.key(1)

    fresh_model = ResNet(hidden_dim=16, key=key)
    if resume:
        model = eqx.tree_deserialise_leaves("model_checkpoint.eqx", fresh_model)
    else:
        model = fresh_model

    optimizer = optax.adam(learning_rate=1e-3)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    training_isfs, training_params, training_offsets = training_data

    # Define number batches per epoch
    num_isfs = len(training_isfs)
    num_batches = num_isfs // BATCH_SIZE
    usable_len = num_batches * BATCH_SIZE

    loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)

    def batch_step(
        carry: tuple[eqx.Module, optax.OptState],
        batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[eqx.Module, optax.OptState], jnp.ndarray]:
        model, opt_state = carry
        batch_isfs, batch_params, batch_offsets = batch

        batch_loss, gradients = loss_and_grad_fn(
            model,
            time_span=time_span,
            delta_k=delta_k,
            test_isfs=batch_isfs,
            test_params=batch_params,
            test_offsets=batch_offsets,
        )

        updates, opt_state = optimizer.update(gradients, opt_state)

        model = eqx.apply_updates(model, updates)

        return (model, opt_state), batch_loss

    @eqx.filter_jit
    def run_epoch(
        model: eqx.Module,
        opt_state: optax.OptState,
        isfs: jnp.ndarray,
        params: jnp.ndarray,
        offsets: jnp.ndarray,
    ) -> tuple[eqx.Module, optax.OptState, jnp.ndarray]:
        """Run one epoch of ML algorithm."""
        batched_isfs = isfs[:usable_len].reshape(
            num_batches, BATCH_SIZE, *isfs.shape[1:]
        )
        batched_params = params[:usable_len].reshape(
            num_batches, BATCH_SIZE, *params.shape[1:]
        )
        batched_offsets = offsets[:usable_len].reshape(
            num_batches, BATCH_SIZE, *offsets.shape[1:]
        )

        (model, opt_state), batch_losses = jax.lax.scan(
            batch_step,
            (model, opt_state),
            (batched_isfs, batched_params, batched_offsets),
        )

        return model, opt_state, jnp.sum(batch_losses)

    # Control loop outside jit
    losses = []
    for epoch in range(NUM_EPOCHS):
        key, perm_key = jax.random.split(key)
        permutation = jax.random.permutation(perm_key, num_isfs)

        shuffled_isfs = training_isfs[permutation]
        shuffled_params = training_params[permutation]
        shuffled_offsets = training_offsets[permutation]

        model, optimizer_state, epoch_loss = run_epoch(
            model, optimizer_state, shuffled_isfs, shuffled_params, shuffled_offsets
        )
        losses.append(float(epoch_loss))
        print(f"Epoch {epoch + 1:03d} | Loss: {epoch_loss:.5f}")

        if (epoch + 1) % CHECK_EVERY == 0:
            eqx.tree_serialise_leaves("model_checkpoint.eqx", model)
            loss_difference = losses[-CHECK_EVERY] - epoch_loss
            if len(losses) >= CHECK_EVERY and abs(loss_difference) < EARLY_STOP:
                print("Early stopping triggered: plateau reached.")
                break

    # Return trained model
    return model  # ty: ignore[invalid-return-type]


# Simulation input generator functions
default_params = KramersParameters(
    omega_well=1.0,
    omega_barrier=5.0,
    barrier_energy=1.0,
    m=1.0,
    temperature=0.5 / Boltzmann,
    gamma=0.1,
)


@jax.jit
def _vary_barrier_energy(
    _key: jax.Array,
) -> tuple[tuple[float, float, float, float, float, float], "CanonicalSystem"]:  # ruff: ignore[quoted-annotation]

    rand = jax.random.uniform(_key, shape=(), minval=0.2, maxval=2.0)

    params = (
        default_params.omega_well,
        default_params.omega_barrier,
        rand.astype(jnp.float32),
        default_params.m,
        default_params.temperature,
        default_params.gamma,
    )
    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=params[0],
            omega_barrier=params[1],
            barrier_energy=params[2],  # ty: ignore[invalid-argument-type]
            m=params[3],
            temperature=params[4],
            gamma=params[5],
        )
    ).as_canonical()

    return params, system  # ty: ignore[invalid-return-type]


@jax.jit
def _inits_constant(
    _key: jax.Array,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
]:
    const_initial_cond = jnp.full((1, 1), 0.0)
    return (const_initial_cond, const_initial_cond)


# jax simulation run
@eqx.filter_jit
def run_langevin_trajectories(
    *,
    time_span: TimeSpan,
    param_keys: jax.Array,  # shape: (num_isfs)
    sim_keys: jax.Array,  # shape: (num_isfs, traj_per_isf)
    generate_params: "Callable[        [jax.Array],        tuple[tuple[float, float, float, float, float, float], CanonicalSystem],    ]",  # ruff: ignore[quoted-annotation]
) -> JaxEnsembleResults:
    """Run langevin trajectories."""

    def run_group_trajectories(
        p_key: jax.Array, group_sim_keys: jax.Array
    ) -> JaxEnsembleResults:
        # Generate a system for the group
        params, system = generate_params(p_key)

        def run_one_trajectory(sim_key: jax.Array) -> JaxEnsembleResults:

            cond_key, solve_key = jax.random.split(sim_key, 2)

            initial_conditions = _inits_constant(cond_key)

            times, positions, _ = solve_many_overdamped_jax(
                system,
                time_span,
                initial_conditions,
                _key=solve_key,
            )

            return {
                "parameters": params,  # ty: ignore[invalid-argument-type]
                "initial_conditions": initial_conditions,
                "result": (times, positions),
            }  # ty: ignore[invalid-return-type]

        return jax.vmap(run_one_trajectory, (0))(group_sim_keys)

    grouped_results = jax.vmap(run_group_trajectories, (0, 0))(param_keys, sim_keys)

    # Flatten outer dimension (num_isfs, traj_per_isf) into (n_traj)
    return jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), grouped_results)


# Training data generation functions


@timed
def generate_single_clean_isf(
    folderpath: str,
    *,
    time_span: TimeSpan,
    delta_k: float,
    params: KramersParameters = default_params,
) -> None:
    """Run a langevin ensemble and save the resulting clean isf to file."""
    system = KramersSystem1D(params=params).as_canonical()

    times, x_points, _ = solve_many_overdamped_jax(
        system,
        time_span,
        initial_conditions=(jnp.full((500, 1), 0.0), jnp.full((500, 1), 0.0)),
        _key=jax.random.key(33),
    )

    isf = get_pairwise_isf(x_points, jnp.asarray((delta_k,)))
    avg_isf = jnp.mean(isf, axis=0)
    sem_isf = jnp.std(isf, axis=0) / np.sqrt(isf.shape[0])

    avg_data = get_measured_data(avg_isf, "real")
    sem_data = get_measured_data(sem_isf, "real")

    clean_isf_path = folderpath + "/langevin_clean_isf.pkl"
    with Path(clean_isf_path).open("wb") as file:
        isf_record = {"time": times, "isf": avg_data, "error": sem_data}
        pickle.dump(isf_record, file)

    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line,) = ax.plot(times, avg_data)
    line.set_label("ISF")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    ax.set_xlim(0, right=20)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Clean Langevin ISF")

    fig.savefig("./examples/hopping_model/test.clean_isf.pdf")


@timed
def generate_many_langevin_trajectories(
    traj_filepath: str,
    num_isfs: int,
    traj_per_isf: int,
    *,
    time_span: TimeSpan,
    generate_params: "Callable[        [jax.Array],        tuple[tuple[float, float, float, float, float, float], CanonicalSystem],    ]" = _vary_barrier_energy,  # ruff: ignore[quoted-annotation]
) -> None:
    """Run the langevin simulations and save to file."""
    param_key, sim_key = jax.random.split(jax.random.key(100))
    param_keys = jax.random.split(param_key, num_isfs)
    sim_keys = jax.random.split(sim_key, (num_isfs, traj_per_isf))

    # Transfer batched array directly from GPU to CPU as contiguous blocks
    batched_trajectories = jax.tree.map(
        jnp.asarray,
        run_langevin_trajectories(
            time_span=time_span,
            param_keys=param_keys,
            sim_keys=sim_keys,
            generate_params=generate_params,
        ),
    )

    # Save directly without slicing into 10,000 Python objects
    with Path(traj_filepath).open("wb") as file:
        pickle.dump(batched_trajectories, file, protocol=pickle.HIGHEST_PROTOCOL)


@timed
def generate_isfs(
    traj_filepath: str,
    isfs_filepath: str,
    traj_per_isf: int,
    *,
    delta_k: float,
    generate_offsets: "Callable[ [jnp.ndarray, float, float], jnp.ndarray    ]" = get_isf_offset,  # ruff: ignore[quoted-annotation]
) -> None:
    """Generate isfs from trajectories saved in ML pipeline and save to file."""
    # Open the trajectories file and load in the trajectories
    with Path(traj_filepath).open("rb") as file:
        batched = pickle.load(file)  # ruff: ignore[suspicious-pickle-usage]

    times_all, x_points_all = batched["result"]
    parameters_all = batched["parameters"]
    n_traj = x_points_all.shape[0]

    # Open the isfs file and calculate, then save, the isfs
    with Path(isfs_filepath).open("wb") as file:
        for i in range(0, n_traj, traj_per_isf):
            x_chunk = x_points_all[i : i + traj_per_isf]
            times = times_all[i]
            params = jax.tree.map(operator.itemgetter(i), parameters_all)

            isf = get_pairwise_isf(jnp.array(x_chunk), jnp.asarray((delta_k,)))
            avg_isf = jnp.mean(isf, axis=0)
            sem_isf = jnp.std(isf, axis=0) / np.sqrt(isf.shape[0])

            avg_data = get_measured_data(avg_isf, "real")[0]
            sem_data = get_measured_data(sem_isf, "real")[0]

            isf_offset = generate_offsets(params, delta_k, times[-1])

            isf_record = {
                "parameters": params,
                "isf": {"time": times, "isf": avg_data, "error": sem_data},
                "offset": isf_offset,
            }
            pickle.dump(isf_record, file)


# Train & Test functions


def untrained_test(folderpath: str) -> None:
    """Test an untrained model with a clean isf input."""
    print("\n\nRunning untrained test\n")

    key = jax.random.key(40)
    model = ResNet(key=key)

    with Path(folderpath + "/langevin_clean_isf.pkl").open("rb") as file:
        isf_record = pickle.load(file)  # ruff: ignore[suspicious-pickle-usage]
    x_input = jnp.array([isf_record.get("isf")])
    output = model(x_input)

    print("Input shape:", x_input.shape)
    print("Output shape:", output.shape)
    print("Output value:", output)


# Broken
def single_clean_test(folderpath: str) -> None:  # ruff: ignore[too-many-locals]
    """Train and test a model on a single, clean ISF."""
    print("\n\nRunning test with a single, clean isf\n")

    # Experimental parameters
    time_span = TimeSpan(t_start=0, t_end=400, n_steps=200)
    delta_k = 0.5

    # Define filepath
    clean_isf_path = folderpath + "/langevin_clean_isf.pkl"

    # Ensure isf exists
    if not Path(clean_isf_path).exists():
        print("No data, generating a new clean isf")
        generate_single_clean_isf(folderpath, time_span=time_span, delta_k=delta_k)

    # Load isf
    with Path(clean_isf_path).open("rb") as file:
        clean_isf = pickle.load(file)  # ruff: ignore[suspicious-pickle-usage]

    # Train model on clean isf
    training_isf = jnp.array(clean_isf.get("isf"))
    training_error = jnp.array(clean_isf.get("error"))

    print("Training model")
    trained_model = train_model(
        (
            training_isf[None, None, :],
            training_error[None, None, :],
            jnp.full((1, 1, training_isf.shape[0]), 1.0),
        ),
        time_span=time_span,
        delta_k=delta_k,
    )

    print("\nModel trained! Getting model's isf")

    # Test model on clean isf
    hopping_time, offset = get_hopping_time_and_offset(
        trained_model, training_isf[None, :]
    )
    hopping_time = float(hopping_time)
    offset = float(offset)

    predicted_lattice = Lattice1D(1.0, hopping_time)
    model_isf = get_deterministic_isf(
        get_deterministic_probabilities(predicted_lattice, (1000,), time_span),
        (delta_k,),
    )

    corrected_model_isf = offset * model_isf

    # Plot source and model isfs for comparison
    print("Plotting ISFs")
    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line1,) = ax.plot(clean_isf.get("time"), clean_isf.get("isf"))
    line1.set_label("Langevin ISF")

    (line2,) = ax.plot(clean_isf.get("time"), corrected_model_isf)
    line2.set_label("Model ISF")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    ax.set_xlim(0, right=20)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Single clean ISF comparison")

    fig.savefig("./examples/hopping_model/model_test_single_clean.isf.pdf")


def many_test(folderpath: str, n_isfs: int, *, resume: bool = False) -> None:  # ruff: ignore[too-many-locals, too-many-statements]
    """Train and test a model on n_isfs equivalent but noisy ISFs."""
    print(f"\n\nRunning test with {n_isfs} equivalent isfs\n")

    # Experimental parameters
    time_span = TimeSpan(t_start=0, t_end=400, n_steps=200)
    delta_k = 0.5
    traj_per_isf = 10

    # Define filepaths
    traj_filepath = folderpath + f"/langevin_{n_isfs * traj_per_isf}_equivalent.pkl"
    isfs_filepath = (
        folderpath
        + f"/langevin_{n_isfs}_equivalent_isf_averaged_over_{traj_per_isf}.pkl"
    )

    # Ensure isfs exist
    if not Path(traj_filepath).exists():
        print("No data, generating new equivalent trajectories")
        generate_many_langevin_trajectories(
            traj_filepath, n_isfs, traj_per_isf, time_span=time_span
        )
    if not Path(isfs_filepath).exists():
        print("No isfs, generating new equivalent isfs")
        generate_isfs(traj_filepath, isfs_filepath, traj_per_isf, delta_k=delta_k)

    # Load isfs
    training_isf_data = []
    with Path(isfs_filepath).open("rb") as file:
        while True:
            try:
                training_isf_data.append(pickle.load(file))  # ruff: ignore[suspicious-pickle-usage]
            except EOFError:
                break

    # Train model: select training data
    training_isfs = jnp.array(
        [data.get("isf").get("isf") for data in training_isf_data]
    )
    training_errors = jnp.array(
        [data.get("isf").get("error") for data in training_isf_data]
    )
    training_offsets = jnp.array(
        [data.get("isf").get("offset") for data in training_isf_data]
    )

    # Train model: train model
    print("Training model")
    trained_model = train_model(
        (
            training_isfs[:, None, :],
            training_errors[:, None, :],
            training_offsets[:, None, :],
        ),
        time_span=time_span,
        delta_k=delta_k,
        resume=resume,
    )

    print("\nModel trained! Getting model's isf")

    # Test model: get test data
    clean_isf_path = folderpath + "/langevin_clean_isf.pkl"

    # Ensure isf exists
    if not Path(clean_isf_path).exists():
        print("No data, generating a new clean isf")
        generate_single_clean_isf(folderpath, time_span=time_span, delta_k=delta_k)

    # Load isf
    with Path(clean_isf_path).open("rb") as file:
        test_isf = pickle.load(file)  # ruff: ignore[suspicious-pickle-usage]

    # Test model: get model output
    hopping_time, offset = get_hopping_time_and_offset(
        trained_model, jnp.array([test_isf.get("isf")])
    )
    hopping_time = float(hopping_time)
    offset = float(offset)

    predicted_lattice = Lattice1D(1.0, hopping_time)
    model_isf = get_deterministic_isf(
        get_deterministic_probabilities(predicted_lattice, (1000,), time_span),
        (delta_k,),
    )

    corrected_model_isf = offset * model_isf

    print("Plotting ISFs")
    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line1,) = ax.plot(test_isf.get("time"), test_isf.get("isf"))
    line1.set_label("Langevin ISF")

    (line2,) = ax.plot(test_isf.get("time"), corrected_model_isf)
    line2.set_label("Model ISF")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    ax.set_xlim(0, right=400)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Model trained on many, tested on clean")

    fig.savefig("./examples/hopping_model/model_test_many_equiv.isf.pdf")

    for i in range(len(training_isf_data)):
        data = training_isf_data[i]
        isf = data.get("isf")
        hopping_time, offset = get_hopping_time_and_offset(
            trained_model, jnp.array([isf.get("isf")])
        )
        hopping_time = float(hopping_time)
        offset = float(offset)

        predicted_lattice = Lattice1D(1.0, hopping_time)
        model_isf = get_deterministic_isf(
            get_deterministic_probabilities(predicted_lattice, (1000,), time_span),
            (delta_k,),
        )

        corrected_model_isf = offset * model_isf

        fig, ax = get_fancy_figure()
        fig, ax = get_figure(ax)
        (line1,) = ax.plot(isf.get("time"), isf.get("isf"))
        line1.set_label("Langevin ISF")

        (line2,) = ax.plot(isf.get("time"), corrected_model_isf)
        line2.set_label("Model ISF")

        fill = ax.fill_between(
            isf.get("time"),
            isf.get("isf") - isf.get("error"),
            isf.get("isf") + isf.get("error"),
        )
        fill.set_alpha(0.3)
        fill.set_color(line1.get_color())

        ax.set_xlabel("Time / s")
        ax.set_ylabel("ISF")

        ax.set_xlim(0, right=200)
        ax.set_ylim(-1, 1)
        ax.legend()
        ax.set_title("Model trained on many, tested on random")

        fig.savefig(f"./examples/hopping_model/training_isfs/test_{i}.isf.pdf")

        if i == 15:  # ruff: ignore[magic-value-comparison]
            break


def _kramers_rate(params: KramersParameters) -> jnp.ndarray:
    return (
        (params.omega_well * params.omega_barrier) / (2 * jnp.pi * params.gamma)
    ) * jnp.exp(-params.barrier_energy / params.kbt)


def kramers_rate_plot_test(folderpath: str, *, resume: bool = False) -> None:  # ruff: ignore[too-many-locals, too-many-statements]
    """Train a model on a range of barrier_energy values."""
    print("\n\nRunning kramers rate theory\n")

    # Experimental parameters
    time_span = TimeSpan(t_start=0, t_end=25, n_steps=500)
    delta_k = jnp.pi / 2.9

    # Test parameters
    traj_per_isf = 100
    n_parameter_data_points = 40
    # 19 training isfs per test isf
    num_isfs = 20 * n_parameter_data_points

    # Define filepaths
    traj_filepath = (
        folderpath
        + f"/vary_barrier_energy/langevin_{num_isfs * traj_per_isf}_trajectories.pkl"
    )
    Path(traj_filepath).parent.mkdir(parents=True, exist_ok=True)
    isfs_filepath = (
        folderpath
        + f"/vary_barrier_energy/langevin_{num_isfs}_isf_averaged_over_{traj_per_isf}.pkl"
    )

    # Ensure isfs exist
    if not Path(traj_filepath).exists():
        print(f"No data, generating {num_isfs * traj_per_isf} equivalent trajectories")
        generate_many_langevin_trajectories(
            traj_filepath,
            num_isfs,
            traj_per_isf,
            time_span=time_span,
            generate_params=_vary_barrier_energy,
        )
    if not Path(isfs_filepath).exists():
        print("No isfs, generating new equivalent isfs")
        generate_isfs(traj_filepath, isfs_filepath, traj_per_isf, delta_k=delta_k)

    _check_training_data(
        path
        + f"/vary_barrier_energy/langevin_{num_isfs * traj_per_isf}_trajectories.pkl",
        path
        + f"/vary_barrier_energy/langevin_{num_isfs}_isf_averaged_over_{traj_per_isf}.pkl",
        time_span=time_span,
    )

    # Load isfs
    generated_isf_data = []
    with Path(isfs_filepath).open("rb") as file:
        while True:
            try:
                generated_isf_data.append(pickle.load(file))  # ruff: ignore[suspicious-pickle-usage]
            except EOFError:
                break

    @timed
    def split_isfs_by_barrier_energy(
        generated_isf_data: list[dict], n: int
    ) -> tuple[list[dict], list[dict]]:
        total_len = len(generated_isf_data)

        # Sort isfs by parameter of interest: barrier_energy
        barrier_energy_values = jnp.array(
            [data["parameters"][2] for data in generated_isf_data]
        )

        sorted_indices = jnp.argsort(barrier_energy_values)
        sorted_list = barrier_energy_values[sorted_indices]

        targets = jnp.linspace(sorted_list[0], sorted_list[-1], n)
        target_positions = jnp.searchsorted(sorted_list, targets)
        target_positions = jnp.clip(target_positions, 0, total_len - 1)

        selected_indices = np.asarray(sorted_indices[target_positions])

        is_test = np.zeros(total_len, dtype=bool)
        is_test[selected_indices] = True

        test_list = [
            i for i, in_test in zip(generated_isf_data, is_test, strict=True) if in_test
        ]
        training_list = [
            i
            for i, in_test in zip(generated_isf_data, is_test, strict=True)
            if not in_test
        ]

        return test_list, training_list

    print("\nSplitting isfs into training and evenly spaced test set")
    test_list, training_list = split_isfs_by_barrier_energy(
        generated_isf_data, n_parameter_data_points
    )

    # Train model: select training data
    training_isfs = jnp.array([data["isf"]["isf"] for data in training_list])
    training_params = jnp.array([data["parameters"] for data in training_list])
    training_offsets = jnp.array([data["offset"] for data in training_list])

    print("\nTraining model")
    trained_model = train_model(
        (training_isfs, training_params, training_offsets),
        time_span=time_span,
        delta_k=delta_k,
        resume=resume,
    )

    print("\nModel trained! Testing on test data")
    _check_model_fits(trained_model, test_list, time_span=time_span, delta_k=delta_k)

    test_params = jnp.array([data["parameters"] for data in test_list])

    # Test model: get model output
    hopping_times, _ = jax.vmap(get_hopping_time_and_offset, (None, 0))(
        trained_model, test_params
    )
    hopping_rate = 1.0 / hopping_times
    barrier_energies = jnp.array([data["parameters"][2] for data in test_list])

    # Plot hopping rate against barrier_energy
    print("Plot Kramers stuff")
    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    line1 = ax.scatter(barrier_energies, hopping_rate)
    line1.set_label("Model")

    def dephasing_rate(params: KramersParameters, delta_k: float) -> jnp.ndarray:
        lattice_parameter = params.delta_x
        rate = _kramers_rate(params)
        return rate * (2 - 2 * jnp.cos(delta_k * lattice_parameter))

    (line2,) = ax.plot(
        sorted(barrier_energies),
        [
            _kramers_rate(
                KramersParameters(
                    omega_well=default_params.omega_well,
                    omega_barrier=default_params.omega_barrier,
                    barrier_energy=barrier_energy,
                    m=default_params.m,
                    temperature=default_params.temperature,
                    gamma=default_params.gamma,
                )
            )
            for barrier_energy in sorted(barrier_energies)
        ],
    )
    line2.set_label("Kramers")

    ax.set_xlabel("Barrier energy")
    ax.set_ylabel("Hopping rate")

    ax.set_xlim(0, right=max(sorted(barrier_energies)) + 0.5)
    ax.set_ylim(
        0,
        float(
            _kramers_rate(
                KramersParameters(
                    omega_well=default_params.omega_well,
                    omega_barrier=default_params.omega_barrier,
                    barrier_energy=min(barrier_energies),
                    m=default_params.m,
                    temperature=default_params.temperature,
                    gamma=default_params.gamma,
                )
            )
            + 0.5
        ),
    )
    ax.legend()
    ax.set_title("Kramers")

    fig.savefig("./examples/hopping_model/kramers.compare.pdf")

    print(hopping_rate)


def pre_generate_on_gpu(folderpath: str) -> None:
    """Pre-generate training isfs."""
    # Experimental parameters
    time_span = TimeSpan(t_start=0, t_end=400, n_steps=200)

    num_traj = [10000, 100000, 1000000]

    for num in num_traj:
        traj_filepath = (
            folderpath + f"/vary_barrier_energy/langevin_{num}_equivalent.pkl"
        )
        Path(traj_filepath).parent.mkdir(parents=True, exist_ok=True)

        if not Path(traj_filepath).exists():
            print(f"Generating {num} trajectories")
            generate_many_langevin_trajectories(
                traj_filepath, num // 10, 10, time_span=time_span
            )


def _check_training_data(
    traj_filepath: str, isf_filepath: str, time_span: TimeSpan
) -> None:
    params = KramersParameters(
        omega_well=default_params.omega_well,
        omega_barrier=default_params.omega_barrier,
        barrier_energy=1.0,
        m=default_params.m,
        temperature=default_params.temperature,
        gamma=default_params.gamma,
    )
    print("delta_x = ", params.delta_x)
    print("kramer's hop time = ", 1 / get_kramers_rate(params))
    print("\nChecking trajectory plot")
    # Open the trajectories file and load in the trajectories
    with Path(traj_filepath).open("rb") as file:
        batched = pickle.load(file)  # ruff: ignore[suspicious-pickle-usage]

    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)

    lines = []
    for i in range(min(len(batched["result"][0]), 10)):
        times = batched["result"][0][i]
        x_points = batched["result"][1][i]
        barrier_energy = batched["parameters"][2][i]
        (line,) = ax.plot(times, x_points[0][0])
        line.set_color("C0")
        lines.append(line)

    lines[-1].set_color("C1")

    # Apply tick intervals
    ax.yaxis.set_major_locator(ticker.MultipleLocator(params.delta_x))

    # Display grid lines aligned with ticks
    # Use axis='y' for horizontal lines, axis='x' for vertical lines, or axis='both'
    ax.grid(True, axis="y", color="gray", linestyle="--", linewidth=0.7)  # ruff: ignore[boolean-positional-value-in-call]

    ax.set_xlabel("$time$")
    ax.set_ylabel("$x$")
    ax.set_xlim(times[0], times[-1])
    ax.set_title(f"Blue barrier energy: {barrier_energy:.2f}")

    fig.savefig("./examples/hopping_model/training_isfs/training.traj.pdf")

    print("\nChecking ISFs")
    # Load isfs
    generated_isf_data = []
    with Path(isf_filepath).open("rb") as file:
        while True:
            try:
                generated_isf_data.append(pickle.load(file))  # ruff: ignore[suspicious-pickle-usage]
            except EOFError:
                break

    times = generated_isf_data[0]["isf"]["time"]
    for i in range(min(len(generated_isf_data), 15)):
        isf = generated_isf_data[i]["isf"]["isf"]
        params = generated_isf_data[i]["parameters"]
        offset = generated_isf_data[i]["offset"]

        fig, ax = get_fancy_figure()
        fig, ax = get_figure(ax)
        (line1,) = ax.plot(times, isf)
        line1.set_label("Langevin ISF")

        (line2,) = ax.plot(times, np.full((times.shape[0],), offset))
        line2.set_label("Offset")

        ax.set_xlabel("Time / s")
        ax.set_ylabel("ISF")

        ax.set_xlim(0, right=time_span.t_end)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f"Barrier_energy = {params[2]:.2f}")

        fig.savefig(f"./examples/hopping_model/training_isfs/training_{i}.isf.pdf")


def _check_model_fits(  # ruff: ignore[too-many-locals]
    trained_model: ResNet, test_list: list[dict], *, time_span: TimeSpan, delta_k: float
) -> None:

    # Test model: get model output
    test_params = jnp.array([data["parameters"] for data in test_list])
    hopping_times, _ = jax.vmap(get_hopping_time_and_offset, (None, 0))(
        trained_model, test_params
    )
    times = jnp.linspace(time_span.t_start, time_span.t_end, time_span.n_steps + 1)

    for i in range(min(len(test_list), 15)):
        isf = test_list[i]["isf"]["isf"]
        offset = test_list[i]["offset"]
        params = test_list[i]["parameters"]

        hopping_time = hopping_times[i]
        predicted_lattice = Lattice1D(1.0, float(hopping_time))
        model_isf = get_deterministic_isf(
            get_deterministic_probabilities(predicted_lattice, (1000,), time_span),
            (delta_k,),
        )

        corrected_model_isf = offset * model_isf

        kramers_rate = _kramers_rate(
            KramersParameters(
                omega_well=params[0],
                omega_barrier=params[1],
                barrier_energy=params[2],
                m=params[3],
                temperature=params[4],
                gamma=params[5],
            )
        )
        kramers_time = 1.0 / kramers_rate

        kramers_lattice = Lattice1D(np.pi / delta_k, float(kramers_time))
        kramers_isf = get_deterministic_isf(
            get_deterministic_probabilities(kramers_lattice, (1000,), time_span),
            (delta_k,),
        )

        corrected_kramers_isf = offset * kramers_isf

        fig, ax = get_fancy_figure()
        fig, ax = get_figure(ax)
        (line1,) = ax.plot(times, isf)
        line1.set_label("Langevin ISF")

        (line2,) = ax.plot(times, corrected_model_isf)
        line2.set_label("Corrected Model ISF")

        (line3,) = ax.plot(times, corrected_kramers_isf)
        line3.set_label("Corrected Kramers ISF")

        ax.set_xlabel("Time / s")
        ax.set_ylabel("ISF")

        ax.set_xlim(0, right=5)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f"Barrier_energy = {params[2]:.2f}")

        fig.savefig(f"./examples/hopping_model/training_isfs/kramers_test_{i}.isf.pdf")


if __name__ == "__main__":
    path = "./examples/data"
    kramers_rate_plot_test(path, resume=True)

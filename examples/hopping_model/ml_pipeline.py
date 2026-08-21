import operator
import pickle
from pathlib import Path
from typing import Any, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from classical_diffusion.analysis import get_isf
from classical_diffusion.hopping import (
    KramersParameters,
    Lattice1D,
    deterministic_probabilities_jax,
    get_deterministic_isf,
    get_deterministic_isf_jax,
    get_deterministic_probabilities,
)
from classical_diffusion.langevin import (
    CanonicalSystem,
    KramersSystem1D,
    solve_overdamped_ensemble,
    solve_overdamped_ensemble_jax,
)
from classical_diffusion.plot import get_fancy_figure, get_figure, get_measured_data
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import timed


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

    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d

    def __init__(self, channels: int, kernel_size: int = 8, *, key: jax.Array) -> None:
        key1, key2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv1d(
            channels, channels, kernel_size, padding="same", key=key1
        )
        self.conv2 = eqx.nn.Conv1d(
            channels, channels, kernel_size, padding="same", key=key2
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run Residual Block layers."""
        residual = x
        x = jax.nn.relu(self.conv1(x))
        x = self.conv2(x)
        # Residual Net: activated linear F(x) is added to shortcut connection, residual
        return jax.nn.relu(x + residual)


class ResNet(eqx.Module):
    """ResNet model."""

    input_layer: eqx.nn.Conv1d
    residual_block: ResidualBlock
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        *,
        hidden_channels: int = 16,
        key: jax.Array,
    ) -> None:
        input_key, residual_block_key, output_key = jax.random.split(key, 3)

        # Project input channel up to hidden layer channels
        self.input_layer = eqx.nn.Conv1d(
            in_channels=1, out_channels=hidden_channels, kernel_size=8, key=input_key
        )

        # Run residual block
        self.residual_block = ResidualBlock(
            channels=hidden_channels, key=residual_block_key
        )

        # Project hidden channels down to output
        self.output_layer = eqx.nn.Linear(hidden_channels, 2, key=output_key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Propagate input through model layers."""
        x = jax.nn.relu(self.input_layer(x))  # shape = (hidden_channels, n_time_steps)
        x = self.residual_block(x)  # shape = (hidden_channels, n_time_steps)

        x = jnp.mean(x, axis=-1)  # shape = (hidden_channels,)
        x = self.output_layer(x)  # shape = (2,)
        return x.at[1].set(0.5 + 0.5 * jax.nn.tanh(x[1]))


# Model functions


@jax.jit
def get_deterministic_isf_directly(
    hop_time: float, times: jnp.ndarray, delta_k: float
) -> jnp.ndarray:

    probabilities = deterministic_probabilities_jax(hop_time, times)

    return get_deterministic_isf_jax(probabilities, delta_k)


@jax.jit
def loss_fn(
    model: eqx.Module,
    times: jnp.ndarray,
    delta_k: float,
    test_isfs: jnp.ndarray,
) -> jax.Array:
    """Loss function for an ISF hopping rate prediction model."""
    print("\nCompile Loss Function")
    # Pass batched isfs through the model to predict hopping rates and isf offsets
    predictions = jax.vmap(model)(test_isfs)  # ty: ignore[invalid-argument-type]

    # For each prediction, generate an isf
    hopping_times = predictions[:, 0]
    offsets = predictions[:, 1]

    isfs = jax.vmap(get_deterministic_isf_directly, (0, None, None))(
        hopping_times, times, delta_k
    )  # These are already real

    corrected_isfs = offsets[:, None] * isfs

    # For each isf, compare to test isf
    errors = jnp.sum((corrected_isfs - test_isfs.squeeze(axis=1)) ** 2, axis=-1)

    # Return the average error
    return jnp.mean(errors)


@eqx.filter_jit
def training_step(  # ruff: ignore[too-many-arguments]
    model: ResNet,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformationExtraArgs,
    *,
    times: jnp.ndarray,
    delta_k: float,
    test_isfs: jnp.ndarray,
) -> tuple[Any, Any, Any]:
    """Progress the training of the model by one epoch by computing loss, gradients and updates."""
    print("\nCompile Training Step\n")

    # Compute loss and gradients for trainable parameters only
    loss, gradients = eqx.filter_value_and_grad(loss_fn)(
        model, times, delta_k, test_isfs
    )

    # Calculate parameter updates using Optax
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, model)  # ty: ignore[invalid-argument-type]

    # Apply updates to the model
    model = eqx.apply_updates(model, updates)
    return model, optimizer_state, loss


def get_hopping_time_and_offset(model: ResNet, isf: jnp.ndarray) -> tuple[float, float]:
    """Get the hopping time and initial offset from a pre-trained model."""
    outputs = model(isf)
    return float(outputs[0]), float(outputs[1])


# Training functions


def train_model(training_isfs: jnp.ndarray) -> ResNet:
    """Train a ResNet model."""
    # Set up constants
    key = jax.random.key(1)
    time_span = TimeSpan(t_start=0, t_end=40, n_steps=200)
    times = jnp.linspace(time_span.t_start, time_span.t_end, time_span.n_steps + 1)

    delta_k = 0.5

    # Initialise the model and optimizer
    model = ResNet(hidden_channels=16, key=key)

    optimizer = optax.adam(learning_rate=1e-3)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Define number of epochs to train for
    num_epochs = 100

    # Training loop
    for epoch in range(num_epochs):
        if epoch > 0:
            print(f"starting epoch {epoch + 1}")
        model, optimizer_state, loss = training_step(
            model,
            optimizer_state,
            optimizer,
            times=times,
            delta_k=delta_k,
            test_isfs=training_isfs,
        )

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d} | Loss = {loss:.5f}")

    # Return trained model
    return model


# Simulation input generator functions
@jax.jit
def _generate_canonical_kramers_system(
    _key: jax.Array,
) -> tuple[tuple[float, float, float, float, float, float], CanonicalSystem]:

    rand = jax.random.uniform(_key, shape=(), minval=0.5, maxval=3.0)
    omega_well = 1.0
    omega_barrier = 1.0
    barrier_energy = rand
    m = 1.0
    kbt = 0.5
    gamma = 0.1

    params = (omega_well, omega_barrier, barrier_energy, m, kbt, gamma)
    system = KramersSystem1D(
        omega_well=omega_well,
        omega_barrier=omega_barrier,
        barrier_energy=barrier_energy,
        m=m,
        kbt=kbt,
        gamma=gamma,
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
@jax.jit
def run_langevin_trajectories(
    keys: jax.Array,
) -> JaxEnsembleResults:
    """Run langevin trajectories."""
    # Times
    times = TimeSpan(t_start=0, t_end=40, n_steps=200)

    def body(time_span: TimeSpan, _key: jax.Array) -> JaxEnsembleResults:

        param_key, cond_key, sim_key = jax.random.split(_key, 3)

        params, system = _generate_canonical_kramers_system(param_key)

        initial_conditions = _inits_constant(cond_key)

        result = solve_overdamped_ensemble_jax(
            system,
            time_span,
            initial_conditions,
            sim_key,
        )

        return {
            "parameters": params,
            "initial_conditions": initial_conditions,
            "result": result,
        }

    return jax.vmap(body, (None, 0))(times, keys)


# Training data generation functions


@timed
def generate_single_clean_isf(folderpath: str) -> None:
    """Run a langevin ensemble and save the resulting clean isf to file."""
    time_span = TimeSpan(t_start=0, t_end=40, n_steps=200)

    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=1.0,
            omega_barrier=1.0,
            barrier_energy=1.0,
            m=1.0,
            kbt=0.5,
            gamma=0.1,
        )
    )

    delta_k = 0.5

    result = solve_overdamped_ensemble(
        system,
        time_span,
        initial_conditions=(np.full((500, 1), 0.0), np.full((500, 1), 0.0)),
        _key=jax.random.key(3),
    )

    isf = get_isf(result.x_points, (delta_k,))
    avg_isf = np.mean(isf, axis=0)

    print(result.times)
    print(avg_isf)

    clean_isf_path = folderpath + "/langevin_clean_isf.pkl"
    with Path(clean_isf_path).open("wb") as file:
        real_isf = get_measured_data(avg_isf, "real")
        isf_record = {"time": result.times, "isf": real_isf}
        pickle.dump(isf_record, file)

    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line,) = ax.plot(result.times, real_isf)
    line.set_label("ISF")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    ax.set_xlim(0, right=20)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Clean Langevin ISF")

    fig.savefig("./examples/hopping_model/test.clean_isf.pdf")


@timed
def generate_many_equiv_trajectories(traj_filepath: str, n_isfs: int) -> None:
    """Run the langevin simulations and save to file."""
    keys = jnp.full(n_isfs, jax.random.key(100))

    batched_trajectories = jax.tree.map(np.asarray, run_langevin_trajectories(keys))
    trajectories = [
        jax.tree.map(operator.itemgetter(i), batched_trajectories)
        for i in range(n_isfs)
    ]

    with Path(traj_filepath).open("wb") as file:
        pickle.dump(trajectories, file)


@timed
def generate_random_langevin_trajectories(
    traj_filepath: str,
    n_isfs: int,
) -> None:
    """Run the langevin simulations and save to file."""
    keys = jax.random.split(jax.random.key(100), n_isfs)

    batched_trajectories = jax.tree.map(np.asarray, run_langevin_trajectories(keys))
    trajectories = [
        jax.tree.map(operator.itemgetter(i), batched_trajectories)
        for i in range(n_isfs)
    ]

    with Path(traj_filepath).open("wb") as file:
        pickle.dump(trajectories, file)


@timed
def generate_isfs(traj_filepath: str, isfs_filepath: str) -> None:
    """Generate isfs from trajectories saved in ML pipeline and save to file."""
    delta_k = 0.5

    # Open the trajectories file and load in the trajectories
    with Path(traj_filepath).open("rb") as file:
        trajectory_records = pickle.load(file)

    # Open the isfs file and calculate, then save, the isfs
    with Path(isfs_filepath).open("wb") as file:
        for trajectory in trajectory_records:
            times, x_points = trajectory.get("result")
            isf = get_isf(x_points, (delta_k,))[0]
            isf = get_measured_data(isf, "real")

            isf_record = {
                "parameters": trajectory.get("parameters"),
                "isf": {"time": times, "isf": isf},
            }
            pickle.dump(isf_record, file)


# Train & Test functions


def untrained_test(folderpath: str) -> None:
    """Test an untrained model with a clean isf input."""
    print("\n\nRunning untrained test\n")

    key = jax.random.key(40)
    model = ResNet(key=key)

    with Path(folderpath + "/langevin_clean_isf.pkl").open("rb") as file:
        isf_record = pickle.load(file)
    x_input = jnp.array([isf_record.get("isf")])
    output = model(x_input)

    print("Input shape:", x_input.shape)
    print("Output shape:", output.shape)
    print("Output value:", output)


def single_clean_test(folderpath: str) -> None:
    """Train and test a model on a single, clean ISF."""
    print("\n\nRunning test with a single, clean isf\n")
    clean_isf_path = folderpath + "/langevin_clean_isf.pkl"

    if not Path(clean_isf_path).exists():
        print("No data, generating a new clean isf")
        generate_single_clean_isf(folderpath)

    print("Loading clean isf")
    with Path(clean_isf_path).open("rb") as file:
        clean_isf = pickle.load(file)

    training_isf = jnp.array(clean_isf.get("isf"))

    print("Training model")
    trained_model = train_model(training_isf[None, None, :])

    print("\nModel trained! Getting model's isf")
    # Check output
    hopping_time, offset = get_hopping_time_and_offset(
        trained_model, training_isf[None, :]
    )

    system = Lattice1D(1.0, hopping_time)
    time_span = TimeSpan(t_start=0, t_end=40, n_steps=200)
    model_isf = get_deterministic_isf(
        system,
        get_deterministic_probabilities(system, (1000,), time_span, 500).probabilities,
        0.5,
    )

    corrected_model_isf = offset * model_isf

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
    ax.set_title("Comparison please work")

    fig.savefig("./examples/hopping_model/model_test_single_clean.isf.pdf")


def many_equiv_test(folderpath: str, n_isfs: int = 10) -> None:  # ruff: ignore[too-many-locals]
    """Train and test a model on num equivalent but noisy ISFs."""
    print(f"\n\nRunning test with {n_isfs} equivalent isfs\n")
    traj_filepath = folderpath + f"/langevin_{n_isfs}_equivalent.pkl"
    isfs_filepath = folderpath + f"/langevin_{n_isfs}_equivalent_isf.pkl"

    if not Path(isfs_filepath).exists():
        print("No data, generating new equivalent isfs")
        generate_many_equiv_trajectories(traj_filepath, n_isfs)
        generate_isfs(traj_filepath, isfs_filepath)

    print("Loading equivalent isfs")
    equivalent_isf_data = []
    with Path(isfs_filepath).open("rb") as file:
        while True:
            try:
                equivalent_isf_data.append(pickle.load(file))
            except EOFError:
                break

    training_isf_data = equivalent_isf_data[:-1]

    training_isfs = jnp.array(
        [data.get("isf").get("isf") for data in training_isf_data]
    )

    print("Training model")
    trained_model = train_model(training_isfs[:, None, :])

    print("\nModel trained! Getting model's isf")
    test_isf = equivalent_isf_data[-1].get("isf")
    # Check output
    hopping_time, offset = get_hopping_time_and_offset(
        trained_model, jnp.array([test_isf.get("isf")])
    )

    system = Lattice1D(1.0, hopping_time)
    time_span = TimeSpan(t_start=0, t_end=40, n_steps=200)
    model_isf = get_deterministic_isf(
        system,
        get_deterministic_probabilities(system, (1000,), time_span, 500).probabilities,
        0.5,
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

    ax.set_xlim(0, right=20)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(
        "Model trained on many equivalent isfs, tested with this new one as input"
    )

    fig.savefig("./examples/hopping_model/model_test_many_equiv.isf.pdf")


if __name__ == "__main__":
    path = "./examples/data"
    single_clean_test(path)

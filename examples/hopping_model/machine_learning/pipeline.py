import operator
import os

import jax.numpy as jnp
import numpy as np
from model_training import ResNet, train_model

from classical_diffusion.hopping import (
    Lattice1D,
)
from classical_diffusion.jax.hopping import (
    get_deterministic_isf,
    get_deterministic_probabilities,
)
from classical_diffusion.jax.langevin import (
    KramersParameters,
)
from classical_diffusion.plot import get_fancy_figure, get_figure

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"


import jax
from generate_training_data import generate_training_data

from classical_diffusion.simulation import TimeSpan

jax.config.update("jax_enable_x64", val=False)


def _kramers_rate(params: KramersParameters) -> jnp.ndarray:
    return (
        (params.omega_well * params.omega_barrier) / (2 * jnp.pi * params.gamma)
    ) * jnp.exp(-params.barrier_energy / params.kbt)


def _generalised_kramers_rate(params: KramersParameters) -> jnp.ndarray:
    msg = "This function is yet to be implemented."
    raise NotImplementedError(msg)


def _mask_nan_values(
    training_data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], time_span: TimeSpan
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    hop_times = training_data[1]
    masked_hop_times = jnp.where(jnp.isnan(hop_times), time_span.t_end, hop_times)
    return (training_data[0], masked_hop_times, training_data[2])


def _test_model(
    model: ResNet,
    time_span: TimeSpan,
    test_data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> None:

    test_omega_wells = test_data[0][:, 0]
    filtered_hop_times = test_data[1]
    filtered_hop_rates = 1.0 / filtered_hop_times

    _fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)

    # Quick kramers
    kramers_hop_rates = ((test_omega_wells * 5.0) / (2 * np.pi * 0.1)) * np.exp(
        -1.0 / 0.5
    )

    # Quick model's
    model_hop_times = jax.vmap(model, (0))(test_data[0])
    model_hop_rates = 1.0 / model_hop_times

    (line1,) = ax.plot(test_omega_wells, kramers_hop_rates)
    line1.set_label("kramers hop rates")

    line2 = ax.scatter(test_omega_wells, filtered_hop_rates)
    line2.set_label("filtered hop rates")

    line3 = ax.scatter(test_omega_wells, model_hop_rates)
    line3.set_label("model hop rates")

    ax.set_xlabel("Omega well")
    ax.set_ylabel("Rate")

    ax.legend()

    fig.savefig("./examples/hopping_model/machine_learning/kramers_test.pdf")

    times = np.linspace(time_span.t_start, time_span.t_end, time_span.n_steps + 1)

    for i in range(min(len(test_data[0]), 10)):
        _fig, ax = get_fancy_figure()
        fig, ax = get_figure(ax)

        # Kramers prediction
        params = test_data[0][i]
        kramers_params = KramersParameters(
            omega_well=params[0],  # ty: ignore[invalid-argument-type]
            omega_barrier=params[1],  # ty: ignore[invalid-argument-type]
            barrier_energy=params[2],  # ty: ignore[invalid-argument-type]
            m=params[3],  # ty: ignore[invalid-argument-type]
            temperature=params[4],  # ty: ignore[invalid-argument-type]
            gamma=params[5],  # ty: ignore[invalid-argument-type]
        )

        print(kramers_params.omega_well)

        kramers_rate = _kramers_rate(kramers_params)
        kramers_time = 1.0 / kramers_rate
        print(f"Kramers hop time: {kramers_time}")
        kramers_lattice = Lattice1D(
            kramers_params.delta_x, float(kramers_time)
        ).as_canonical()
        kramers_isf = get_deterministic_isf(
            kramers_lattice,
            get_deterministic_probabilities(kramers_lattice, time_span, (1000,))[0],
            (jnp.pi / kramers_params.delta_x,),
        )

        # Langevin extracted rate
        langevin_time = test_data[1][i]
        print(f"Filtered trajectory hop time: {langevin_time}")
        filtered_time_lattice = Lattice1D(
            kramers_params.delta_x, float(langevin_time)
        ).as_canonical()
        filtered_time_isf = get_deterministic_isf(
            filtered_time_lattice,
            get_deterministic_probabilities(filtered_time_lattice, time_span, (1000,))[
                0
            ],
            (jnp.pi / kramers_params.delta_x,),
        )

        # Model prediction
        model_time = model(jnp.array(jnp.array(params)))[0]
        print(f"Model hop time: {model_time}")
        model_lattice = Lattice1D(
            kramers_params.delta_x, float(model_time)
        ).as_canonical()
        model_isf = get_deterministic_isf(
            model_lattice,
            get_deterministic_probabilities(model_lattice, time_span, (1000,))[0],
            (jnp.pi / kramers_params.delta_x,),
        )

        (line1,) = ax.plot(times, kramers_isf)
        line1.set_label("Kramers ISF")

        (line2,) = ax.plot(times, filtered_time_isf)
        line2.set_label("Langevin ISF")

        (line3,) = ax.plot(times, model_isf)
        line3.set_label("Model ISF")

        ax.set_xlabel("Time / s")
        ax.set_ylabel("ISF")

        ax.set_xlim(0, right=20)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f"Omega_well = {kramers_params.omega_well:.2f}")

        fig.savefig(
            f"./examples/hopping_model/machine_learning/training_isfs/kramers_test_{kramers_params.omega_well:.2f}.isf.pdf"
        )


def learn_varying_omega_well() -> None:
    """Generate training data, train a model and test it."""
    # Simulation parameters - adjust these and the system parameters to get quick, sensible data
    time_span = TimeSpan(t_end=100.0, n_steps=100)
    num_training_trajectories = 9
    num_validation_trajectories = 1

    # Generate trajectory data
    print("\nGenerate trajectory data")
    data = generate_training_data(
        time_span=time_span,
        n_trajectories=num_training_trajectories + num_validation_trajectories,
    )

    # Split data into training and validation sets
    print("\nSplit trajectory data into training and test")
    training_data = jax.tree.map(
        operator.itemgetter(slice(None, num_training_trajectories)), data
    )
    masked_training_data = _mask_nan_values(training_data, time_span)
    test_data = jax.tree.map(
        operator.itemgetter(slice(num_training_trajectories, None)), data
    )
    masked_test_data = _mask_nan_values(test_data, time_span)

    # Train model on training data
    print("\nTrain model")
    model = train_model(masked_training_data, resume=False)

    # Test model
    print("\nTest model")
    _test_model(model, time_span, masked_test_data)  # ty: ignore[invalid-argument-type]


if __name__ == "__main__":
    print("running...\n")
    learn_varying_omega_well()

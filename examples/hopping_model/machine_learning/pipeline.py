import operator
import os

import jax.numpy as jnp
import numpy as np
from model_training import ResNet, train_model

from classical_diffusion.hopping import (
    Lattice1D,
    get_lifson_jackson_rate,
)
from classical_diffusion.jax.hopping import (
    get_deterministic_isf,
    get_deterministic_probabilities,
)
from classical_diffusion.jax.langevin import (
    KramersParameters,
)
from classical_diffusion.langevin import KramersParameters as KramersParametersNotJax
from classical_diffusion.langevin import KramersSystem1D
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
    test_params = test_data[0]
    discretised_hop_times = test_data[1]
    discretised_hop_rates = 1.0 / discretised_hop_times

    sort_idx = jnp.argsort(test_omega_wells)
    sorted_params = test_params[sort_idx]
    sorted_omega_wells = test_omega_wells[sort_idx]
    sorted_disc_rates = discretised_hop_rates[sort_idx]
    test_data[2][sort_idx]

    bin_size = 30
    num_bins = max(1, int(jnp.round(len(sorted_disc_rates) / bin_size)))
    binned_rates = jnp.array_split(sorted_disc_rates, num_bins)
    binned_omegas = jnp.array_split(sorted_omega_wells, num_bins)
    # binned_num_hops = jnp.array_split(sorted_num_hops, num_bins)

    binned_omegas_means = jnp.array([jnp.mean(b) for b in binned_omegas])
    binned_rates_means = jnp.array([jnp.mean(b) for b in binned_rates])
    # binned_num_hops_sums = jnp.array([jnp.sum(b) for b in binned_num_hops])
    # jnp.array(
    #     [
    #         (1.0 / (jnp.sqrt(binned_rates_means[idx]) ** 3 * binned_num_hops_sums[idx]))
    #         for idx in range(len(binned_rates_means))
    #     ]
    # )

    _fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)

    lifson_jackson_rates = []
    for params in sorted_params:
        kramers_params = KramersParametersNotJax(
            omega_well=params[0],
            omega_barrier=params[1],
            barrier_energy=params[2],
            m=params[3],
            temperature=params[4],
            gamma=params[5],
        )
        system = KramersSystem1D(params=kramers_params)

        lifson_jackson_rates.append(
            get_lifson_jackson_rate(system, delta_x=kramers_params.delta_x)
        )

    lifson_jackson_hop_rates = np.array(lifson_jackson_rates)

    model_hop_times = jax.vmap(model, (0))(sorted_params)
    model_hop_rates = 1.0 / model_hop_times

    line2 = ax.scatter(binned_omegas_means, binned_rates_means, marker="x")
    line2.set_label("discretised hop rates")

    (line3,) = ax.plot(sorted_omega_wells, model_hop_rates, color="C1")
    line3.set_label("model hop rates")

    (line4,) = ax.plot(sorted_omega_wells, lifson_jackson_hop_rates, color="C2")
    line4.set_label("lifson jackson hop rates")

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
        get_deterministic_isf(
            kramers_lattice,
            get_deterministic_probabilities(kramers_lattice, time_span, (1000,))[0],
            (jnp.pi / kramers_params.delta_x,),
        )

        lifson_jackson_rate = get_lifson_jackson_rate(
            KramersSystem1D(params=kramers_params), delta_x=kramers_params.delta_x
        )
        lifson_jackson_time = 1.0 / lifson_jackson_rate
        print(f"Lifson Jackson hop time: {lifson_jackson_time}")
        lifson_lattice = Lattice1D(
            kramers_params.delta_x, float(lifson_jackson_time)
        ).as_canonical()
        lifson_isf = get_deterministic_isf(
            lifson_lattice,
            get_deterministic_probabilities(lifson_lattice, time_span, (1000,))[0],
            (jnp.pi / kramers_params.delta_x,),
        )

        # Langevin extracted rate
        langevin_time = test_data[1][i]
        print(f"Discretised trajectory hop time: {langevin_time}")
        discretised_time_lattice = Lattice1D(
            kramers_params.delta_x, float(langevin_time)
        ).as_canonical()
        discretised_time_isf = get_deterministic_isf(
            discretised_time_lattice,
            get_deterministic_probabilities(
                discretised_time_lattice, time_span, (1000,)
            )[0],
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

        (line2,) = ax.plot(times, discretised_time_isf, color="C0")
        line2.set_label("Langevin ISF")

        (line1,) = ax.plot(times, model_isf, color="C1")
        line1.set_label("Model ISF")

        (line1,) = ax.plot(times, lifson_isf, color="C2")
        line1.set_label("Lifson Jackson ISF")

        ax.set_xlabel("Time")
        ax.set_ylabel("ISF")

        ax.set_xlim(0, right=15)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f"Omega_well = {kramers_params.omega_well:.2f}")

        fig.savefig(
            f"./examples/hopping_model/machine_learning/training_isfs/kramers_test_{kramers_params.omega_well:.2f}.isf.pdf"
        )


def learn_varying_omega_well() -> None:
    """Generate training data, train a model and test it."""
    # Simulation parameters - adjust these and the system parameters
    time_span = TimeSpan(t_end=100.0, n_steps=1000)
    num_training_trajectories = 1000
    num_validation_trajectories = 600

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
    model = train_model(masked_training_data, resume=True)

    # Test model
    print("\nTest model")
    _test_model(model, time_span, masked_test_data)  # ty: ignore[invalid-argument-type]


if __name__ == "__main__":
    print("running...\n")
    learn_varying_omega_well()

from collections.abc import (
    Callable,  # ruff: ignore[typing-only-standard-library-import]
)
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import ticker
from scipy.constants import Boltzmann

from classical_diffusion.jax.langevin import KramersParameters, KramersSystem1D
from classical_diffusion.jax.langevin import filter_trajectory as filter_trajectory_jax
from classical_diffusion.jax.langevin import (
    get_trajectory_breakpoints as get_trajectory_breakpoints_jax,
)
from classical_diffusion.jax.langevin import (
    solve_many_overdamped as solve_many_overdamped_jax,
)
from classical_diffusion.langevin._system import (
    CanonicalSystem,  # ruff: ignore[typing-only-first-party-import]
)
from classical_diffusion.plot import get_fancy_figure, get_figure
from classical_diffusion.simulation import (
    TimeSpan,  # ruff: ignore[typing-only-first-party-import]
)
from classical_diffusion.util import cached, timed

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
def _vary_omega_well(
    _key: jax.Array,
) -> tuple[jnp.ndarray, "CanonicalSystem"]:  # ruff: ignore[quoted-annotation]

    min_value = 0.2
    max_value = 5.0

    params = [
        jax.random.uniform(_key, shape=(), minval=min_value, maxval=max_value).astype(
            jnp.float32
        ),
        default_params.omega_barrier,
        default_params.barrier_energy,
        default_params.m,
        default_params.temperature,
        default_params.gamma,
    ]
    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=params[0],  # ty: ignore[invalid-argument-type]
            omega_barrier=params[1],  # ty: ignore[invalid-argument-type]
            barrier_energy=params[2],  # ty: ignore[invalid-argument-type]
            m=params[3],  # ty: ignore[invalid-argument-type]
            temperature=params[4],  # ty: ignore[invalid-argument-type]
            gamma=params[5],  # ty: ignore[invalid-argument-type]
        )
    ).as_canonical()

    return params, system  # ty: ignore[invalid-return-type]


@jax.jit
def _inits_constant() -> tuple[
    jnp.ndarray,
    jnp.ndarray,
]:
    const_initial_cond = jnp.full((1, 1), 0.0)
    return (const_initial_cond, const_initial_cond)


@eqx.filter_jit
def _run_langevin_trajectories(
    *,
    time_span: TimeSpan,
    keys: jax.Array,  # shape: (num_trajectories)
    generate_params: Callable[[jax.Array], tuple[jnp.ndarray, CanonicalSystem]],
) -> tuple[jnp.ndarray, jnp.ndarray]:

    def run_trajectory(_key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        p_key, s_key = jax.random.split(_key)

        # Generate a system for the group
        params, system = generate_params(p_key)

        initial_conditions = _inits_constant()

        _times, positions, _ = solve_many_overdamped_jax(
            system,
            time_span,
            initial_conditions,
            _key=s_key,
        )

        return (jnp.array(params), positions[0][0])

    return jax.vmap(run_trajectory, (0))(keys)


@timed
def run_langevin_trajectories_timed(
    *,
    time_span: TimeSpan,
    keys: jax.Array,  # shape: (num_trajectories)
    generate_params: Callable[[jax.Array], tuple[jnp.ndarray, CanonicalSystem]],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run langevin trajectories."""
    return _run_langevin_trajectories(
        time_span=time_span, keys=keys, generate_params=generate_params
    )


@timed
def get_sequence_lengths(x: jnp.ndarray, delta_x: float) -> np.ndarray:
    """Get the lengths of sequences between breakpoints in a trajectory."""
    breakpoints = get_trajectory_breakpoints_jax(x, delta_x=delta_x)

    true_indices = np.flatnonzero(breakpoints)
    # jnp.diff computes sequence lengths; [:-1] excludes the final sequence
    return np.diff(true_indices)[:-1]


def _plot_data_checks(
    params: KramersParameters,
    trajectory: jnp.ndarray,
    filtered_trajectory: jnp.ndarray,
    *,
    time_span: TimeSpan,
) -> None:
    times = jnp.linspace(time_span.t_start, time_span.t_end, time_span.n_steps + 1)

    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line1,) = ax.plot(times, trajectory)
    line1.set_label("Langevin Trajectory")

    (line1,) = ax.plot(times, filtered_trajectory)
    line1.set_label("Filtered Trajectory")

    # Apply tick intervals
    ax.yaxis.set_major_locator(ticker.MultipleLocator(params.delta_x))

    # Display grid lines aligned with ticks
    # Use axis='y' for horizontal lines, axis='x' for vertical lines, or axis='both'
    ax.grid(True, axis="y", color="gray", linestyle="--", linewidth=0.7)  # ruff: ignore[boolean-positional-value-in-call]

    ax.set_xlabel("Time")
    ax.set_ylabel("Position")

    fig.savefig(
        "./examples/hopping_model/machine_learning/training_data.trajectory.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _generate_trajectories_path(*, time_span: TimeSpan, n_trajectories: int) -> Path:
    filename = f"training_data_{hash(time_span)}_{hash(n_trajectories)}.npz"
    return Path("examples/data") / filename


@cached(_generate_trajectories_path)
def generate_training_data(
    *,
    time_span: TimeSpan,
    n_trajectories: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate machine learning data set with parameter generator defined below."""
    generate_params = _vary_omega_well

    key = jax.random.PRNGKey(234)
    keys = jax.random.split(key, n_trajectories)

    print("Run trajectories")
    all_params, all_trajectories = run_langevin_trajectories_timed(
        time_span=time_span, keys=keys, generate_params=generate_params
    )

    steps_to_seconds = (time_span.t_end - time_span.t_start) / time_span.n_steps

    out_params = []
    out_hop_times = []
    out_num_hops = []

    print("Extract hop times and number of hops")
    for index in range(len(all_trajectories)):
        params_array = all_params[index]
        params = KramersParameters(
            omega_well=params_array[0],  # ty: ignore[invalid-argument-type]
            omega_barrier=params_array[1],  # ty: ignore[invalid-argument-type]
            barrier_energy=params_array[2],  # ty: ignore[invalid-argument-type]
            m=params_array[3],  # ty: ignore[invalid-argument-type]
            temperature=params_array[4],  # ty: ignore[invalid-argument-type]
            gamma=params_array[5],  # ty: ignore[invalid-argument-type])
        )
        trajectory = all_trajectories[index]

        sequence_lengths = get_sequence_lengths(trajectory, params.delta_x)

        derived_hop_time = 2 * (np.mean(sequence_lengths) * steps_to_seconds)
        num_hops = len(sequence_lengths) + 1

        out_params.append(params_array)
        out_hop_times.append(derived_hop_time)
        out_num_hops.append(num_hops)

    _plot_data_checks(
        params,
        trajectory,
        filter_trajectory_jax(trajectory, delta_x=params.delta_x),
        time_span=time_span,
    )

    return np.array(out_params), np.array(out_hop_times), np.array(out_num_hops)

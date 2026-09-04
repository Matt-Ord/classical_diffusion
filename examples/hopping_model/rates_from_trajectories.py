import itertools
import os
from pathlib import Path
from typing import Any, TypedDict

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import ticker
from scipy.constants import Boltzmann

from classical_diffusion.analysis import plot_isf
from classical_diffusion.hopping import (
    Lattice1D,
    get_deterministic_isf,
    get_deterministic_probabilities,
)
from classical_diffusion.jax.langevin import (
    KramersParameters as KramersParametersJax,
)
from classical_diffusion.jax.langevin import (
    filter_trajectory as filter_trajectory_jax,
)
from classical_diffusion.jax.langevin import (
    get_trajectory_breakpoints as get_trajectory_breakpoints_jax,
)
from classical_diffusion.jax.langevin import (
    solve_many_overdamped as solve_many_overdamped_jax,
)
from classical_diffusion.langevin import KramersParameters, KramersSystem1D, System
from classical_diffusion.plot import get_fancy_figure, get_figure
from classical_diffusion.simulation import SimulationResult, TimeSpan
from classical_diffusion.util import cached, timed


class JaxEnsembleResults(TypedDict):
    """Jax compatible output from an ensemble run."""

    parameters: KramersParametersJax
    initial_conditions: tuple[
        np.ndarray[Any, np.dtype[np.floating[Any]]],
        np.ndarray[Any, np.dtype[np.floating[Any]]],
    ]
    result: tuple[jnp.ndarray, jnp.ndarray]


def _generate_trajectories_path(
    system: System, time_span: TimeSpan, n_trajectories: int
) -> Path:
    filename = f"overdamped_{hash(system)}_{hash(time_span)}_{hash(n_trajectories)}.npz"
    return Path("examples/data") / filename


@cached(_generate_trajectories_path)
@timed
def generate_trajectories(
    system: System, time_span: TimeSpan, n_trajectories: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate trajectories for a given system and time span."""
    solve_key = jax.random.PRNGKey(0)
    initial_conditions = (
        jnp.zeros((n_trajectories, system.n_dim)),
        jnp.zeros((n_trajectories, system.n_dim)),
    )

    times, positions, _ = solve_many_overdamped_jax(
        system.as_canonical(),
        time_span,
        initial_conditions,
        _key=solve_key,
    )

    return times, positions


def get_sequence_lengths(x: jnp.ndarray, delta_x: float) -> np.ndarray:
    """Get the lengths of sequences between breakpoints in a trajectory."""
    breakpoints = get_trajectory_breakpoints_jax(x, delta_x=delta_x)

    true_indices = np.flatnonzero(breakpoints)
    # jnp.diff computes sequence lengths; [:-1] excludes the final sequence
    return np.diff(true_indices)[:-1]


@timed
def _plot_filtered_trajectory() -> None:

    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=1.0,
            omega_barrier=1.0,
            barrier_energy=1.0,
            m=1.0,
            temperature=0.5 / Boltzmann,
            gamma=0.1,
        )
    )

    time_span = TimeSpan(t_start=0.0, t_end=100.0, n_steps=1000)

    print("Generate 1 trajectory")
    times, positions = generate_trajectories(system, time_span, n_trajectories=1)

    filtered_trajectory = filter_trajectory_jax(positions[0][0], delta_x=system.delta_x)

    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    (line1,) = ax.plot(times, positions[0][0])
    line1.set_label("Langevin Trajectory")

    (line1,) = ax.plot(times, filtered_trajectory)
    line1.set_label("Filtered Trajectory")

    # Apply tick intervals
    ax.yaxis.set_major_locator(ticker.MultipleLocator(system.delta_x))

    # Display grid lines aligned with ticks
    # Use axis='y' for horizontal lines, axis='x' for vertical lines, or axis='both'
    ax.grid(True, axis="y", color="gray", linestyle="--", linewidth=0.7)  # ruff: ignore[boolean-positional-value-in-call]

    ax.set_xlabel("Time")
    ax.set_ylabel("Position")

    fig.savefig(
        "./examples/hopping_model/1d_filtered.trajectory.pdf",
        dpi=300,
        bbox_inches="tight",
    )


@timed
def _plot_hop_intervals_histogram() -> None:

    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=1.0,
            omega_barrier=1.0,
            barrier_energy=1.0,
            m=1.0,
            temperature=0.5 / Boltzmann,
            gamma=0.1,
        )
    )

    time_span = TimeSpan(t_start=0.0, t_end=100.0, n_steps=10000)

    n_trajectories = 10

    _times, trajectories = generate_trajectories(system, time_span, n_trajectories)

    all_sequence_lengths = []
    for trajectory in trajectories:
        filtered_trajectory = filter_trajectory_jax(
            trajectory[0], delta_x=system.delta_x
        )
        all_sequence_lengths.append(
            get_sequence_lengths(filtered_trajectory, system.delta_x)
        )

    all_lengths = np.fromiter(
        itertools.chain.from_iterable(all_sequence_lengths), dtype=int
    )

    bin_width = 10

    max_length = np.max(all_lengths)
    bins = np.arange(0, max_length + bin_width, bin_width)

    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    ax.hist(
        all_lengths,
        bins=bins,
        density=True,
        edgecolor="black",
        alpha=0.6,
        color="skyblue",
        label="Langevin data",
    )

    ax.set_xlabel("Sequence Length (steps)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Sequence Lengths (Bin Size = {bin_width})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)  # ruff: ignore[boolean-positional-value-in-call]

    fig.savefig("./examples/hopping_model/1d_filtered.histogram.pdf")


def _kramers_rate(params: KramersParameters) -> jnp.ndarray:
    return (
        (params.omega_well * params.omega_barrier) / (2 * jnp.pi * params.gamma)
    ) * jnp.exp(-params.barrier_energy / params.kbt)


@timed
def _get_rate_comparison() -> None:  # ruff: ignore[too-many-locals]
    params = KramersParameters(
        omega_well=1.0,
        omega_barrier=1.0,
        barrier_energy=1.0,
        m=1.0,
        temperature=0.5 / Boltzmann,
        gamma=0.1,
    )

    # Define the System for Langevin simulation
    system = KramersSystem1D(params=params)

    # Get the hop time from Kramers' Rate Theory and define the Lattice for hopping simulation
    hop_rate = _kramers_rate(params)
    hop_time = 1.0 / hop_rate
    print(f"Kramers hop time: {hop_time}")
    kramers_lattice = Lattice1D(params.delta_x, float(hop_time))

    # Generate Langevin simulation data
    time_span = TimeSpan(t_start=0.0, t_end=100.0, n_steps=10000)

    n_trajectories = 10

    times, trajectories = generate_trajectories(system, time_span, n_trajectories)

    # Filter trajectories
    all_sequence_lengths = []
    filtered_trajectories = []
    for trajectory in trajectories:
        print("new traj")
        filtered_trajectory = filter_trajectory_jax(
            trajectory[0], delta_x=system.delta_x
        )
        sequence_lengths = get_sequence_lengths(filtered_trajectory, system.delta_x)

        all_sequence_lengths.append(sequence_lengths)

        filtered_trajectories.append(filtered_trajectory)

    all_lengths = np.fromiter(
        itertools.chain.from_iterable(all_sequence_lengths), dtype=int
    )

    bin_width = 10

    max_length = np.max(all_lengths)
    bins = np.arange(0, max_length + bin_width, bin_width)

    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    ax.hist(
        all_lengths,
        bins=bins,
        density=True,
        edgecolor="black",
        alpha=0.6,
        color="skyblue",
        label="Langevin data",
    )

    ax.set_xlabel("Sequence Length (steps)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Sequence Lengths (Bin Size = {bin_width})")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)  # ruff: ignore[boolean-positional-value-in-call]

    fig.savefig("./examples/hopping_model/1d_filtered.histogram.pdf")

    result = SimulationResult(
        times=np.array(times),
        x_points=np.array(filtered_trajectories)[:, None, :],
        system=system.as_canonical(),
    )

    derived_hop_time_both_directions = np.mean(all_lengths) * (
        (time_span.t_end - time_span.t_start) / time_span.n_steps
    )
    derived_hop_time = 2 * derived_hop_time_both_directions
    print(f"Derived hop time from trajectories: {derived_hop_time}")

    fig, ax = get_fancy_figure()
    fig, ax = get_figure(ax)
    _, ax, line, _ = plot_isf(result, ax=ax, delta_k=(np.pi / params.delta_x,))
    line.set_label("Langevin")

    kramers_isf = get_deterministic_isf(
        get_deterministic_probabilities(kramers_lattice, (1000,), time_span),
        (np.pi / params.delta_x,),
    )
    (line2,) = ax.plot(times, kramers_isf)
    line2.set_label("Kramers ISF")

    mean_rate_lattice = Lattice1D(params.delta_x, derived_hop_time)
    mean_rate_isf = get_deterministic_isf(
        get_deterministic_probabilities(mean_rate_lattice, (1000,), time_span),
        (np.pi / params.delta_x,),
    )
    (line3,) = ax.plot(times, mean_rate_isf)
    line3.set_label("Mean Rate ISF")

    ax.legend()
    fig.savefig(
        "./examples/hopping_model/1d_filtered.isfs.pdf", dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    _plot_filtered_trajectory()
    _plot_hop_intervals_histogram()
    _get_rate_comparison()

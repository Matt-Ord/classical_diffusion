from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from classical_diffusion.analysis import (
    _calculate_total_offsset_multiplications_complex,
    _time_average,
    plot_isf,
)
from classical_diffusion.hopping import Lattice1D, solve_ensemble
from classical_diffusion.hopping._system import (
    get_kramers_lattice_harmonic,
)
from classical_diffusion.langevin import (
    PeriodicSystem1D,
    plot_potential_1d,
    solve_overdamped_ensemble,
)
from classical_diffusion.langevin._system import DoubleHarmonicSystem
from classical_diffusion.plot import (
    Measure,
    get_fancy_figure,
    get_figure,
    get_measured_data,
)
from classical_diffusion.simulation import SimulationResult, TimeSpan

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


def _plot_hop_vs_periodic_langevin_isf(
    lattice: Lattice1D,
    system: PeriodicSystem1D,
    time_span: TimeSpan = TimeSpan(t_end=400, n_steps=4000),
    initial_condition: np.ndarray = np.full((4000, 1), 0.0),
) -> None:

    hopping_results = solve_ensemble(
        system=lattice,
        time_span=time_span,
        initial_condition=initial_condition,
        key=jax.random.PRNGKey(seed=100),
    )

    # Now Langevin

    langevin_result = solve_overdamped_ensemble(
        system,
        time_span,
        (initial_condition, np.full(initial_condition.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )

    fig, ax = get_fancy_figure()
    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    _, ax, line_0, _ = plot_isf(result=hopping_results, delta_k=delta_k, ax=ax)
    line_0.set_label("Hopping model")

    _, _, line, _ = plot_isf(
        result=langevin_result, ax=ax, delta_k=delta_k, pairwise=True, measure="real"
    )
    line.set_label("Overdamped Langevin")

    ax.set_xlim(0, right=40)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"Hopping vs Langevin for barrier energy = {system.barrier_energy}")
    fig.savefig(f"./examples/1d_comparison_{system.barrier_energy}.isf.pdf")
    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig(f"./examples/1d_comparison_{system.barrier_energy}.isf.log.pdf")


def _plot_hop_vs_harmonic_langevin_isf(
    lattice: Lattice1D,
    system: DoubleHarmonicSystem,
    time_span: TimeSpan = TimeSpan(t_end=400, n_steps=4000),
    initial_condition: np.ndarray = np.full((4000, 1), 0.0),
) -> None:

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    # First, Hopping model with Kramer's rates

    hopping_results = solve_ensemble(
        system=lattice,
        time_span=time_span,
        initial_condition=initial_condition,
        key=jax.random.PRNGKey(seed=100),
    )

    # Calculate intra-well relaxation correction factor

    relaxation_correction_factor = np.exp(
        -1 * delta_k[0] ** 2 * system.temperature / (system.m * system.omegas[0])
    )

    # Now Langevin

    langevin_result = solve_overdamped_ensemble(
        system,
        time_span,
        (initial_condition, np.full(initial_condition.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )

    fig, ax = get_fancy_figure()

    _, ax, line_0, _ = plot_relaxation_corrected_hopping_isf(
        result=hopping_results,
        delta_k=delta_k,
        ax=ax,
        correction_factor=relaxation_correction_factor,
    )
    line_0.set_label("Hopping model")

    _, _, line, _ = plot_isf(
        result=langevin_result, ax=ax, delta_k=delta_k, pairwise=True, measure="real"
    )
    line.set_label("Overdamped Langevin")

    ax.set_xlim(0, right=100)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"Hopping vs Langevin for barrier energy = {system.barrier_energy}")
    fig.savefig(f"./examples/1d_comparison_{system.barrier_energy}.isf.pdf")
    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig(f"./examples/1d_comparison_{system.barrier_energy}.isf.log.pdf")


def get_relaxation_corrected_hopping_isf(
    positions: np.ndarray[Any, np.dtype[np.floating]],
    delta_k: tuple[float, ...],
    correction_factor: float,
    *,
    pairwise: bool = True,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Get the restored displacement of a wavepacket."""
    if not pairwise:
        phase = np.einsum(
            "i,...ij->...j",
            delta_k,
            positions - positions[..., 0].reshape((*positions.shape[:-1], 1)),
        )
        return np.exp(1j * phase)

    delta_k = (float(delta_k[0]),)
    positions = np.asarray(positions, dtype=np.float64)

    scatter = np.exp(-1j * np.einsum("i,...ij->...j", delta_k, positions))

    # convolution_j = \sum_i^N-j e^(ik.x_i+j) e^(-ik.x_i)
    convolution = np.apply_along_axis(
        lambda m: _calculate_total_offsset_multiplications_complex(m, m),
        axis=-1,
        arr=scatter,
    )
    raw_isf = _time_average(convolution)

    return raw_isf * correction_factor


def plot_relaxation_corrected_hopping_isf(
    result: SimulationResult,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
    delta_k: tuple[int | float, ...],
    correction_factor: float,
) -> tuple[Figure, Axes, Line2D, PolyCollection]:
    """Plot the ensemble-averaged ISF over time, with a shaded ±1 SEM band."""
    fig, ax = get_figure(ax)

    isf = get_relaxation_corrected_hopping_isf(
        result.x_points, delta_k, correction_factor
    )
    n_trajectories = isf.shape[0]
    avg_isf = np.mean(isf, axis=0)
    sem_isf = np.std(isf, axis=0) / np.sqrt(n_trajectories)

    avg_data = get_measured_data(avg_isf, measure)
    sem_data = get_measured_data(sem_isf, measure)

    (line,) = ax.plot(result.times, avg_data)
    line.set_label("ISF")

    fill = ax.fill_between(result.times, avg_data - sem_data, avg_data + sem_data)
    fill.set_alpha(0.3)
    fill.set_label("SEM")
    fill.set_color(line.get_color())

    line.set_label("SEM")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    return fig, ax, line, fill


def _kramers_test() -> None:

    initial_position: np.ndarray = np.full((100, 1), 0.0)
    time_span = TimeSpan(t_end=200, n_steps=2000)

    system = DoubleHarmonicSystem(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        barrier_energy=3,
        omega_a=2,
        omega_b=1,
    )

    fig, ax = get_fancy_figure()
    fig, _ax, _ = plot_potential_1d(system, 0, 10, ax=ax)
    fig.savefig("./examples/1d_harmonic_potential.pdf")

    lattice = get_kramers_lattice_harmonic(system)
    print(f"hop time = {lattice.hop_time}")
    print(f"Lattice spacing = {lattice.lattice_spacing}")

    _plot_hop_vs_harmonic_langevin_isf(
        lattice=lattice,
        system=system,
        time_span=time_span,
        initial_condition=initial_position,
    )


if __name__ == "__main__":
    print("running")
    _kramers_test()

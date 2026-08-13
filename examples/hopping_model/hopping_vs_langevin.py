from typing import TYPE_CHECKING

import jax
import numpy as np

from classical_diffusion.analysis import (
    get_isf,
    plot_isf,
)
from classical_diffusion.hopping import (
    KramersParameters,
    Lattice1D,
    get_kramers_lattice,
    get_kramers_parameters_cosine,
    get_kramers_rate,
    solve_ensemble,
)
from classical_diffusion.langevin import (
    KramersSystem1D,
    PeriodicSystem1D,
    System,
    plot_potential_1d,
    solve_overdamped_ensemble,
)
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

default_time_span = TimeSpan(t_end=400, n_steps=4000)
default_init_condition = np.full((4000, 1), 0.0)


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

    isf = get_isf(result.x_points, delta_k) * correction_factor
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


def _plot_hop_vs_langevin_isf(
    systems: tuple[Lattice1D, System],
    omega_well: float,
    delta_k: tuple[float,],
    time_span: TimeSpan = default_time_span,
    initial_condition: np.ndarray = default_init_condition,
) -> tuple[Figure, Axes]:
    lattice, system = systems

    # First, Hopping model with Kramer's rates

    hopping_results = solve_ensemble(
        system=lattice,
        time_span=time_span,
        initial_condition=initial_condition,
        key=jax.random.PRNGKey(seed=100),
    )

    # Calculate intra-well relaxation correction factor

    relaxation_correction_factor = np.exp(
        -1 * delta_k[0] ** 2 * system.temperature / (system.m * omega_well)
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

    return fig, ax


def _kramers_harmonic_comparison() -> None:

    initial_position: np.ndarray = np.full((100, 1), 0.0)
    time_span = TimeSpan(t_end=200, n_steps=2000)

    system = KramersSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        kramers_params=KramersParameters(2.0, 1.0, 3.0),
    )

    fig, ax = get_fancy_figure()
    fig, _ax, _ = plot_potential_1d(system, 0, 10, ax=ax)
    fig.savefig("./examples/1d_harmonic_potential.pdf")

    lattice = get_kramers_lattice(
        system.delta_x,
        get_kramers_rate(system.gamma, system.kbt, system.kramers_params),
    )

    fig, ax = _plot_hop_vs_langevin_isf(
        systems=(lattice, system),
        omega_well=system.omega_well,
        delta_k=(0.5 * 2 * np.pi / system.delta_x,),
        time_span=time_span,
        initial_condition=initial_position,
    )

    ax.set_xlim(0, right=100)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Hopping vs Langevin for a harmonic potential")
    fig.savefig("./examples/1d_comparison_harmonic.isf.pdf")
    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig("./examples/1d_comparison_harmonic.isf.log.pdf")


def _kramers_sinusoid_comparison() -> None:

    initial_position: np.ndarray = np.full((100, 1), 0.0)
    time_span = TimeSpan(t_end=200, n_steps=2000)

    system = PeriodicSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=3,
    )

    fig, ax = get_fancy_figure()
    fig, _ax, _ = plot_potential_1d(system, 0, 10, ax=ax)
    fig.savefig("./examples/1d_sinusoid_potential.pdf")

    kramers_parameters = get_kramers_parameters_cosine(system)
    lattice = get_kramers_lattice(
        system.delta_x, get_kramers_rate(system.gamma, system.kbt, kramers_parameters)
    )

    fig, ax = _plot_hop_vs_langevin_isf(
        systems=(lattice, system),
        omega_well=kramers_parameters.omega_well,
        delta_k=(0.5 * 2 * np.pi / system.delta_x,),
        time_span=time_span,
        initial_condition=initial_position,
    )

    ax.set_xlim(0, right=100)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Hopping vs Langevin for a sinusoidal potential")
    fig.savefig("./examples/1d_comparison_sinusoidal.isf.pdf")
    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig("./examples/1d_comparison_sinusoidal.isf.log.pdf")


if __name__ == "__main__":
    _kramers_harmonic_comparison()
    _kramers_sinusoid_comparison()

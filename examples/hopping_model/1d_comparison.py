from typing import TYPE_CHECKING

import jax
import numpy as np

from classical_diffusion.analysis import (
    get_isf,
    plot_isf,
)
from classical_diffusion.hopping import (
    KramersParameters,
    get_kramers_parameters_cosine,
    lattice_1d_from_kramers_parameters,
    solve_ensemble,
)
from classical_diffusion.langevin import (
    HarmonicSystem,
    KramersSystem1D,
    PeriodicSystem1D,
    get_exact_harmonic_isf,
    plot_force_1d,
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
from classical_diffusion.util import timed

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D


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
    fill.set_color(line.get_color())

    ax.set_xlabel("Time / s")
    ax.set_ylabel("ISF")

    return fig, ax, line, fill


@timed
def _plot_kramers_system() -> None:
    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=2.0, omega_barrier=1.0, barrier_energy=3.0, m=1.0, gamma=0.1, kbt=0.5
        ),
    )

    fig, ax = get_fancy_figure()
    _, _, line0 = plot_potential_1d(system, 0, 10, ax=ax)
    line0.set_label("Kramers potential")

    _, _, line1 = plot_force_1d(system, 0, 10, ax=ax.twinx())  # cspell: disable-line
    line1.set_label("Kramers force")
    line1.set_color("C1")

    ax.legend()
    fig.savefig("./examples/hopping_model/1d_comparison.kramers.pdf")


def _kramers_harmonic_comparison() -> None:

    system = KramersSystem1D(
        params=KramersParameters(
            omega_well=2.0, omega_barrier=1.0, barrier_energy=3.0, m=1.0, gamma=0.1, kbt=0.5
        ),
    )

    fig, ax = get_fancy_figure()
    initial_position = np.full((100, 1), 0.0)
    time_span = TimeSpan(t_end=200, n_steps=2000)

    langevin_result = solve_overdamped_ensemble(
        system,
        time_span,
        (initial_position, np.full(initial_position.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    lattice = lattice_1d_from_kramers_parameters(system.kramers_params)

    lattice_result = solve_ensemble(
        system=lattice,
        time_span=time_span,
        initial_condition=initial_position,
        key=jax.random.PRNGKey(seed=100),
    )

    relaxation_correction_factor = get_exact_harmonic_isf(
        HarmonicSystem(
            omega=system.omega_well,
            temperature=system.temperature,
            m=system.m,
            gamma=system.gamma,
        ),
        delta_k=delta_k,
        times=np.array([1000 / system.gamma]),
    ).item()

    _, _, line_0, _ = plot_relaxation_corrected_hopping_isf(
        result=lattice_result,
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
    ax.set_title("Hopping vs Langevin for a harmonic potential")

    fig.savefig("./examples/hopping_model/1d_comparison.harmonic.isf.pdf")

    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig("./examples/hopping_model/1d_comparison.harmonic.isf.log.pdf")
    print("harmonic comparison figures saved")


def _kramers_sinusoid_comparison() -> None:

    system = PeriodicSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=3,
    )

    fig, ax = get_fancy_figure()
    initial_position = np.full((100, 1), 0.0)
    time_span = TimeSpan(t_end=200, n_steps=2000)

    langevin_result = solve_overdamped_ensemble(
        system,
        time_span,
        (initial_position, np.full(initial_position.shape, 0.0)),
        _key=jax.random.PRNGKey(seed=100),
    )

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)
    _, _, line, _ = plot_isf(
        result=langevin_result, ax=ax, delta_k=delta_k, pairwise=True, measure="real"
    )
    line.set_label("Overdamped Langevin")

    kramers_params = get_kramers_parameters_cosine(system)
    lattice = lattice_1d_from_kramers_parameters(kramers_params)

    lattice_result = solve_ensemble(
        system=lattice,
        time_span=time_span,
        initial_condition=initial_position,
        key=jax.random.PRNGKey(seed=100),
    )

    relaxation_correction_factor = get_exact_harmonic_isf(
        HarmonicSystem(
            omega=kramers_params.omega_well,
            temperature=system.temperature,
            m=system.m,
            gamma=system.gamma,
        ),
        delta_k=delta_k,
        times=np.array([1000 / system.gamma]),
    ).item()

    _, _, line_0, _ = plot_relaxation_corrected_hopping_isf(
        result=lattice_result,
        delta_k=delta_k,
        ax=ax,
        correction_factor=relaxation_correction_factor,
    )
    line_0.set_label("Hopping model")

    ax.set_xlim(0, right=100)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Hopping vs Langevin for a cosine potential")

    fig.savefig("./examples/hopping_model/1d_comparison.cosine.isf.pdf")

    ax.set_yscale("symlog", linthresh=1e-3)
    fig.savefig("./examples/hopping_model/1d_comparison.cosine.isf.log.pdf")


if __name__ == "__main__":
    _plot_kramers_system()
    _kramers_harmonic_comparison()
    _kramers_sinusoid_comparison()

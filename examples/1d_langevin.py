from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann

from classical_diffusion.analysis import (
    plot_isf,
    plot_x_evolution_1d,
)
from classical_diffusion.langevin import (
    PeriodicSystem1D,
    plot_force_1d,
    plot_periodic_potential_1d,
    solve_ensemble,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cache_base_path


def _plot_periodic_system() -> None:
    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5 / Boltzmann, m=1.0, delta_x=5, barrier_energy=2
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    _, _, _ = plot_force_1d(system, 0, system.delta_x, ax=ax)
    fig.savefig("examples/1d_langevin.potential.pdf")


def _plot_periodic_trajectory() -> None:
    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5 / Boltzmann, m=1.0, delta_x=5, barrier_energy=2
    )

    result = solve_ensemble(
        system,
        TimeSpan(t_end=40 / system.gamma, n_steps=4000),
        n_samples=5,
    )

    fig, ax = get_fancy_figure()

    _, _, _ = plot_x_evolution_1d(result=result, ax=ax)

    fig.savefig("./examples/1d_langevin.trajectory.pdf")


def _plot_periodic_isf() -> None:
    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5 / Boltzmann, m=1.0, delta_x=5, barrier_energy=2
    )

    result = solve_ensemble(
        system,
        TimeSpan(t_end=40 / system.gamma, n_steps=4000),
        n_samples=20,
    )

    fig, ax = get_fancy_figure()

    delta_k = (0.7 * 2 * np.pi / system.delta_x,)
    _, ax, line_0, _fill_0 = plot_isf(result=result, ax=ax, delta_k=delta_k)
    line_0.set_label("full simulation")

    result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_end=10 / system.gamma, n_steps=1000),
        n_samples=10000,
    )

    _, ax, line_1, _ = plot_isf(result=result, ax=ax, delta_k=delta_k, pairwise=False)
    line_1.set_label("ballistic simulation")

    ax.set_xlim(0, 4 / system.gamma)
    ax.set_ylim(0, 1)

    ax.legend(handles=[line_0, line_1])
    fig.savefig("examples/1d_langevin.isf.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _plot_periodic_system()
        _plot_periodic_trajectory()
        _plot_periodic_isf()

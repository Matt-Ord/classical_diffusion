from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann

from classical_diffusion.analysis import plot_isf, plot_x_evolution_1d
from classical_diffusion.langevin import (
    HarmonicSystem,
    plot_exact_flat_isf,
    plot_exact_harmonic_isf,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cache_base_path


def _plot_harmonic_isf() -> None:

    system = HarmonicSystem(
        gamma=0.1,
        temperature=0.5 / Boltzmann,
        m=1.0,
        omega=3,
    )

    result = solve_ensemble(system, TimeSpan(t_end=100, n_steps=5000), n_samples=200)

    fig, ax = get_fancy_figure()

    delta_k = (2 * np.pi / 5,)
    _, ax, line_simulated, _ = plot_isf(
        result=result,
        ax=ax,
        delta_k=delta_k,
    )
    line_simulated.set_label("simulation")

    ax.axhline(
        y=np.exp(
            -(delta_k[0] ** 2) * system.temperature / (system.m * system.omega**2)
        ),
        linestyle=":",
        color="black",
    )

    _, ax, line_exact = plot_exact_harmonic_isf(system, delta_k, result.times, ax=ax)
    line_exact.set_label("exact")
    ax.legend(
        loc="upper right",
        handles=[line_simulated, line_exact],
        labels=["Simulation", "Exact"],
    )
    ax.set_xlim(0, 2 / system.gamma)
    ax.set_ylim(0, 1)
    fig.savefig("./examples/analytical_isf.harmonic.pdf")


def _plot_flat_isf() -> None:

    system = HarmonicSystem(gamma=0.1, temperature=0.5 / Boltzmann, m=1.0, omega=0)

    result = solve_ensemble(
        system, TimeSpan(t_end=50 / system.gamma, n_steps=5000), n_samples=200
    )

    fig, ax = get_fancy_figure()

    delta_k = (2 * np.pi / 40,)
    _, ax, line_simulated, _ = plot_isf(result=result, ax=ax, delta_k=delta_k)
    line_simulated.set_label("simulation")

    _, ax, line_exact = plot_exact_flat_isf(system, delta_k, result.times, ax=ax)
    line_exact.set_label("exact")
    ax.legend(
        loc="upper right",
        handles=[line_simulated, line_exact],
        labels=["Simulation", "Exact"],
    )
    ax.set_xlim(0, 2 / system.gamma)
    ax.set_ylim(0, 1)
    fig.savefig("./examples/analytical_isf.flat.pdf")

    fig, ax = get_fancy_figure()
    plot_x_evolution_1d(result, ax=ax)
    fig.savefig("./examples/analytical_isf.flat_x_evolution.pdf")


def _plot_flat_isf_2d() -> None:

    system = HarmonicSystem(
        gamma=0.1, temperature=0.5 / Boltzmann, m=1.0, omega=0, n_dim=2
    )

    result = solve_ensemble(
        system, TimeSpan(t_end=50 / system.gamma, n_steps=5000), n_samples=200
    )

    fig, ax = get_fancy_figure()

    delta_k = (2 * np.pi / 40, 0)
    _, ax, line_simulated, _ = plot_isf(result=result, ax=ax, delta_k=delta_k)
    line_simulated.set_label("simulation")

    _, ax, line_exact = plot_exact_flat_isf(system, delta_k, result.times, ax=ax)
    line_exact.set_label("exact")
    ax.legend(
        loc="upper right",
        handles=[line_simulated, line_exact],
        labels=["Simulation", "Exact"],
    )
    ax.set_xlim(0, 2 / system.gamma)
    ax.set_ylim(0, 1)
    fig.savefig("./examples/analytical_isf.flat_2d.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _plot_harmonic_isf()
        _plot_flat_isf()
        _plot_flat_isf_2d()

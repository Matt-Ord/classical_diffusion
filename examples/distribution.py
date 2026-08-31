from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.animation import ArtistAnimation
from scipy.constants import Boltzmann

from classical_diffusion.langevin import (
    HarmonicSystem,
    LangevinSimulationResult,
    PeriodicSystem1D,
    get_random_initial_conditions,
    plot_kinetic_probability,
    plot_p_histogram,
    plot_phase_space_density,
    plot_x_distribution_histogram,
    plot_x_distribution_kde,
    shift_origin_to_unit_cell_1d,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cache_base_path


def fold_results(
    result: LangevinSimulationResult[Any],
    delta: float,
) -> LangevinSimulationResult[Any]:
    """Fold x into first BZ zone."""
    return LangevinSimulationResult(
        system=result.system,
        times=result.times,
        x_points=result.x_points % delta,
        p_points=result.p_points,
    )


def _plot_xp_distributions_periodic() -> None:

    system = PeriodicSystem1D(
        gamma=5e11,
        temperature=110,
        m=8e-27,
        delta_x=3e-10,
        barrier_energy=1.6e-21,
    )

    fig, ax = get_fancy_figure()

    result = solve_ensemble(
        system,
        TimeSpan(t_end=system.units.time_into(1000 / system.gamma), n_steps=1000),
        n_samples=10,
    )

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_phase_space_density(result, ax=ax)
    ax.set_ylim(-1e-23, 1e-23)
    mesh.set_rasterized(True)
    fig.savefig("examples/distribution.1d_periodic.phase_space.pdf", dpi=1000)

    fig, ax = get_fancy_figure()
    _, ax, _bars = plot_p_histogram(result=result, ax=ax)
    ax.set_ylim(-1e23, 1e-23)
    fig.savefig("examples/distribution.1d_periodic.p.pdf")

    result_folded = fold_results(result, delta=system.delta_x)

    fig, ax = get_fancy_figure()
    _, ax, _bars = plot_x_distribution_histogram(result=result_folded, ax=ax)
    ax.set_xlim(0, system.delta_x)
    fig.savefig("examples/distribution.1d_periodic.x.pdf")

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_phase_space_density(result=result_folded, ax=ax)
    ax.set_xlim(0, system.delta_x)
    mesh.set_rasterized(True)
    fig.savefig("examples/distribution.1d_periodic.phase_space.folded.pdf", dpi=1000)


def sample_results(
    result: LangevinSimulationResult[Any], step: int
) -> LangevinSimulationResult[Any]:
    """Sample the results of a Langevin simulation."""
    return LangevinSimulationResult(
        system=result.system,
        times=result.times[::step],
        x_points=result.x_points[..., ::step],
        p_points=result.p_points[..., ::step],
    )


def _plot_x_distribution_spread() -> None:

    system = PeriodicSystem1D(
        gamma=0.5, temperature=0.5 / Boltzmann, m=1.0, delta_x=5, barrier_energy=2
    )

    result = solve_ensemble(
        system, TimeSpan(t_end=100 / system.gamma, n_steps=100), n_samples=4000
    )
    result = shift_origin_to_unit_cell_1d(result)

    fig, ax = get_fancy_figure()

    x_points = np.linspace(-2 * system.delta_x, 2 * system.delta_x, 1000)
    _, ax, lines = plot_x_distribution_kde(result=result, x_points=x_points, ax=ax)
    ax.set_ylim(0, 0.5)

    anim = ArtistAnimation(fig, [[c] for c in lines])
    anim.save(
        "examples/distribution.1d_periodic.x_spread.gif",
        writer="pillow",
        dpi=200,
        fps=30,
    )

    fig, ax = get_fancy_figure()
    result = sample_results(result, step=10)
    _, ax, lines = plot_x_distribution_kde(result=result, x_points=x_points, ax=ax)
    ax.set_ylim(0, 0.5)

    fig.savefig("examples/distribution.1d_periodic.x_spread.pdf")


def _plot_xp_distributions_harmonic() -> None:
    system = HarmonicSystem(gamma=5e11, temperature=110, m=8e-27, omega=10e12)

    fig, ax = get_fancy_figure()

    result = solve_ensemble(
        system,
        TimeSpan(t_end=system.units.time_into(2000 / system.gamma), n_steps=2000),
        n_samples=10,
    )

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_phase_space_density(result=result, ax=ax)
    mesh.set_rasterized(True)
    fig.savefig("examples/distribution.1d_harmonic.phase_space.pdf", dpi=1000)

    fig, ax = get_fancy_figure()
    _, ax, _bars = plot_x_distribution_histogram(result=result, ax=ax)
    fig.savefig("examples/distribution.1d_harmonic.x.pdf")

    fig, ax = get_fancy_figure()
    _, ax, _bars = plot_p_histogram(result=result, ax=ax)
    fig.savefig("examples/distribution.1d_harmonic.p.pdf")

    fig, ax = get_fancy_figure()
    _, _, (line0, bars) = plot_kinetic_probability(result=result, max_energy=6, ax=ax)
    for patch in bars.patches:
        patch.set_alpha(1)
    ax.legend(
        loc="upper right",
        handles=[line0],
        labels=["Theoretical"],
    )
    fig.savefig("examples/distribution.1d_harmonic.kinetic.pdf")


def _plot_x_distribution_spread_ballistic_sample() -> None:

    system = PeriodicSystem1D(
        gamma=0.5, temperature=0.5 / Boltzmann, m=1.0, delta_x=5, barrier_energy=2
    )
    x_points, p_points = get_random_initial_conditions(
        system, n_samples=4000, minimum_energy=0.0
    )
    result = LangevinSimulationResult(
        system=system,
        times=np.array([0.0]),
        x_points=x_points[..., np.newaxis],
        p_points=p_points[..., np.newaxis],
    )

    fig, ax = get_fancy_figure()
    result_folded = fold_results(result, delta=system.delta_x)
    _, ax, _lines = plot_x_distribution_histogram(result=result_folded, ax=ax)
    ax.set_xlim(0, system.delta_x)

    fig.savefig("examples/distribution.1d_periodic.x_histogram.pdf")


if __name__ == "__main__":
    # These examples plot the distribution of the
    # classical coordinates of (x,p) for a periodic and harmonic system.
    # This is a good test of convergence - there is a known analytical distribution for both systems,
    # and a lack of high momentum states (or the wrong distribution of x) indicates that the simulation is not converged
    # and therefore smaller timesteps are needed.
    #
    # To generate statistically independent samples, the simulation is
    # sampled once every 1 / gamma, and the first 1 / gamma time is discarded as burn-in.
    #
    # In the periodic system, the x distribution is folded into the first Brillouin zone.
    with cache_base_path(Path("examples/data")):
        _plot_xp_distributions_periodic()
        _plot_x_distribution_spread()
        _plot_xp_distributions_harmonic()
        _plot_x_distribution_spread_ballistic_sample()

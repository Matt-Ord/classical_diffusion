import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    SimulationResult,
    TimeSpan,
    plot_p_histogram,
    plot_phase_space_density,
    plot_x_histogram,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import HarmonicSystem, PeriodicSystem1D, System


def fold_results[S: System](
    result: SimulationResult[S],
    delta: float,
) -> SimulationResult[S]:
    """Fold x into first BZ zone."""
    return SimulationResult(
        times=result.times,
        x_points=result.x_points % delta,
        p_points=result.p_points,
        system=result.system,
    )


def _plot_xp_distributions_periodic() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5, m=1.0, delta_x=5, barrier_energy=1.5
    )

    result = solve_ensemble(
        system,
        TimeSpan(
            t0=1 / system.gamma,
            t1=10 / system.gamma,
            dt=1 / system.gamma,
            dt_step=0.05 / system.gamma,
        ),
        (np.full((2000, 1), 0.0), np.full((2000, 1), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()
    # TODO: x range...
    _, ax, mesh = plot_phase_space_density(result=result, ax=ax)
    mesh.set_rasterized(True)
    fig.savefig("examples/distribution.1d_periodic_phase_space.pdf", dpi=1000)

    fig, ax = get_fancy_figure()
    _, ax, _bars = plot_x_histogram(result=result, ax=ax)
    ax.set_xlim(-4 * system.delta_x, 4 * system.delta_x)
    fig.savefig("examples/distribution.1d_periodic_x.pdf")

    result_folded = fold_results(result, delta=system.delta_x)

    fig, ax = get_fancy_figure()
    _, ax, _bars = plot_x_histogram(result=result_folded, ax=ax)
    ax.set_xlim(0, system.delta_x)
    fig.savefig("examples/distribution.1d_periodic_x.folded.pdf")

    fig, ax = get_fancy_figure()
    _, ax, _bars = plot_p_histogram(result=result_folded, ax=ax)
    fig.savefig("examples/distribution.1d_periodic_p.pdf")

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_phase_space_density(result=result_folded, ax=ax)
    ax.set_xlim(0, system.delta_x)
    mesh.set_rasterized(True)
    fig.savefig(
        "examples/distribution.1d_periodic_phase_space.density.folded.pdf", dpi=1000
    )


def _plot_xp_distributions_harmonic() -> None:
    key = jrandom.PRNGKey(100)

    system = HarmonicSystem(gamma=0.1, temperature=0.5, m=1.0, omega=1.0)

    result = solve_ensemble(
        system,
        TimeSpan(
            t0=1 / system.gamma,
            t1=500 / system.gamma,
            dt=1 / system.gamma,
            dt_step=0.05 / system.gamma,
        ),
        (np.full((200, 1), 0.0), np.full((200, 1), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_phase_space_density(result=result, ax=ax)
    mesh.set_rasterized(True)
    fig.savefig("examples/distribution.1d_harmonic_phase_space.pdf", dpi=1000)

    fig, ax = get_fancy_figure()
    _, ax, _bars = plot_x_histogram(result=result, ax=ax)
    fig.savefig("examples/distribution.1d_harmonic_x.pdf")

    fig, ax = get_fancy_figure()
    _, ax, _bars = plot_p_histogram(result=result, ax=ax)
    fig.savefig("examples/distribution.1d_harmonic_p.pdf")


if __name__ == "__main__":
    _plot_xp_distributions_periodic()
    _plot_xp_distributions_harmonic()

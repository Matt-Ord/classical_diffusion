import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    TimeSpan,
    plot_2d_trajectory,
    plot_isf,
    solve_ballistic_ensemble,
    solve_ensemble,
    solve_single,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import (
    PeriodicSystemFCC,
    plot_periodic_potential_fcc,
)


def _plot_periodic_system() -> None:
    system = PeriodicSystemFCC(
        gamma=0.1, temperature=1.0, m=1.0, delta_x=5.0, barrier_energy=1.5
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_fcc(system, ax=ax)
    fig.savefig("examples/2d_system.potential.pdf")


def _plot_2d_periodic_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystemFCC(
        gamma=0.1, temperature=0.5, m=1.0, delta_x=5, barrier_energy=1.5
    )

    result = solve_ensemble(
        system,
        TimeSpan(
            t0=0,
            t1=50 / system.gamma,
            dt=0.01 / system.gamma,
        ),
        (np.full((2000, 2), 0.0), np.full((2000, 2), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)
    _, ax, line_0, _ = plot_isf(
        result=result,
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("simulation")

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t0=0,
            t1=4 / system.gamma,
            dt=0.01 / system.gamma,
        ),
        n_samples=2000,
        _key=key,
    )
    _, ax, line_1, _ = plot_isf(result=result, ax=ax, delta_k=delta_k, pairwise=False)
    line_1.set_label("ballistic simulation")

    ax.set_xlim(0, 4 / system.gamma)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1])
    fig.savefig("./examples/2d_system.isf.pdf", dpi=300, bbox_inches="tight")


def _plot_2d_trajectory() -> None:
    key = jrandom.PRNGKey(100)
    system = PeriodicSystemFCC(
        gamma=0.1, temperature=0.5, m=1.0, delta_x=5, barrier_energy=1.5
    )

    result = solve_single(
        system,
        TimeSpan(
            t0=0,
            t1=100 / system.gamma,
            dt=0.01 / system.gamma,
        ),
        (np.full((2,), 0.0), np.full((2,), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()
    _, ax, _line = plot_2d_trajectory(result=result, ax=ax)

    fig.savefig("examples/2d.periodic.trajectory.pdf")


if __name__ == "__main__":
    _plot_periodic_system()
    _plot_2d_periodic_isf()
    _plot_2d_trajectory()

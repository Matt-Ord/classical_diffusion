from typing import TYPE_CHECKING

import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np

from classical_diffusion.analysis import plot_isf
from classical_diffusion.langevin import (
    PeriodicSystemFCC,
    breakdown_ballistic_trajectory,
    plot_2d_trajectory,
    plot_periodic_potential_fcc,
    solve_ballistic_ensemble,
    solve_ensemble,
    solve_single,
)
from classical_diffusion.plot import (
    get_fancy_figure,
    setup_fancy_figure,
    setup_rc_params,
)
from classical_diffusion.simulation import TimeSpan

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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
            t_end=50 / system.gamma,
            n_steps=5000,
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
            t_end=4 / system.gamma,
            n_steps=400,
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
    # TODO: add elastic and inelastic trajectories to the plot
    key = jrandom.PRNGKey(100)
    system = PeriodicSystemFCC(
        gamma=0.1, temperature=0.5, m=1.0, delta_x=5, barrier_energy=1.5
    )

    result = solve_single(
        system,
        TimeSpan(
            t_end=100 / system.gamma,
            n_steps=10000,
        ),
        (np.full((2,), 0.0), np.full((2,), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()
    _, ax, _line = plot_2d_trajectory(result=result, ax=ax)

    fig.savefig("examples/2d_system.trajectory.pdf")


def _get_two_panel_figure() -> tuple[Figure, list[Axes]]:
    setup_rc_params()
    fig, ax = plt.subplots(layout="constrained", ncols=2, figsize=(6, 2.5))
    setup_fancy_figure(fig, ax)
    return fig, ax


def _plot_2d_ballistic_trajectory() -> None:

    key = jrandom.PRNGKey(100)
    system = PeriodicSystemFCC(
        gamma=0, temperature=0.5, m=1.0, delta_x=5, barrier_energy=1.5
    )

    result = solve_single(
        system,
        TimeSpan(t_end=1000, n_steps=1000),
        # TODO: replace with random "free" initial condition?
        (np.full((2,), 0.0), np.full((2,), 1.0)),
        _key=key,
    )
    elastic, inelastic = breakdown_ballistic_trajectory(result)

    fig, ax = _get_two_panel_figure()

    _, _ax_0, line = plot_2d_trajectory(result=result, ax=ax[0])
    _, _ax_0, line_e = plot_2d_trajectory(result=elastic, ax=ax[0])
    _, _ax_0, line_i = plot_2d_trajectory(result=inelastic, ax=ax[0])

    _, _ax_1, line_1 = plot_2d_trajectory(result=inelastic, ax=ax[1])
    line_1.set_color(line_i.get_color())

    ax[0].legend(
        handles=[line, line_e, line_i],
        labels=["full", "elastic", "inelastic"],
    )

    fig.savefig("examples/2d_system.ballistic_trajectory.pdf")


if __name__ == "__main__":
    _plot_periodic_system()
    _plot_2d_periodic_isf()
    _plot_2d_trajectory()
    _plot_2d_ballistic_trajectory()

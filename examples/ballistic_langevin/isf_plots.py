import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    PeriodicSystem1D,
    PeriodicSystemFCC,
    breakdown_ballistic_trajectory,
    solve_ensemble,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan


def _plot_periodic_isf_1d() -> None:
    system = PeriodicSystem1D(
        gamma=5e11,
        temperature=155,
        m=8e-27,
        delta_x=5e-10,
        barrier_energy=4e-21,
    )

    full_result = solve_ensemble(
        system,
        TimeSpan(t_end=20e-12, n_steps=1000),
        n_samples=500,
    )

    fig, ax = get_fancy_figure()
    delta_k = (2 * np.pi / system.delta_x * 0.3,)

    _, ax, line_0, _fill_0 = plot_isf(
        result=full_result,
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("full simulation")

    ballistic_result = solve_ensemble_ballistic(
        system,
        TimeSpan(
            t_end=system.units.time_into(20e-12),
            n_steps=2000,
        ),
        n_samples=5000,
    )

    _, ax, line_1, _ = plot_isf(
        result=ballistic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_1.set_label("ballistic simulation")

    elastic_result, inelastic_result = breakdown_ballistic_trajectory(
        ballistic_result,
        filter_timescale=1 / system.gamma,
    )

    _, ax, line_2, _ = plot_isf(
        result=inelastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_2.set_label("inelastic")

    _, ax, line_3, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_3.set_label("elastic")

    ax.set_xlim(0, 6e-12)

    ax.set_ylim(0, 1.0)

    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig(
        "examples/ballistic_langevin/1d_periodic.isf.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _plot_periodic_isf_2d() -> None:

    system = PeriodicSystemFCC(
        gamma=5e11,
        temperature=114,
        m=8e-27,
        delta_x=3e-10,
        barrier_energy=4e-21,
    )

    full_result = solve_ensemble(
        system,
        TimeSpan(t_end=10e-12, n_steps=1000),
        n_samples=500,
    )

    direction = np.array([0, 1])
    delta_k = tuple(
        2 * np.pi / system.delta_x * 0.5 * direction / np.linalg.norm(direction)
    )

    fig, ax = get_fancy_figure()
    _, ax, line_0, _fill_0 = plot_isf(
        result=full_result,
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("full simulation")

    ballistic_result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_end=10e-12, n_steps=1000),
        n_samples=2000,
    )

    _, ax, line_1, _ = plot_isf(
        result=ballistic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_1.set_label("ballistic simulation")

    elastic_result, inelastic_result = breakdown_ballistic_trajectory(
        ballistic_result,
        filter_timescale=1 / system.gamma,
    )

    _, ax, line_2, _ = plot_isf(
        result=inelastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_2.set_label("inelastic")

    _, ax, line_3, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_3.set_label("elastic")

    ax.set_xlim(0, 2e-12)

    ax.set_ylim(0.0, 1.0)

    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig(
        "examples/ballistic_langevin/2d_fcc.isf.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_periodic_isf_1d()
    _plot_periodic_isf_2d()

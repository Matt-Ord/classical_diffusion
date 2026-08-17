import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    PeriodicSystem1D,
    breakdown_ballistic_trajectory,
    get_diffusion_time,
    solve_ballistic_ensemble,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan


def _plot_periodic_isf() -> None:
    system = PeriodicSystem1D(
        gamma=5e11,
        temperature=155,
        m=8e-27,
        delta_x=5e-10,
        barrier_energy=4e-21,
    ).with_normalized_units()

    key = jrandom.PRNGKey(100)
    full_result = solve_ensemble(
        system,
        TimeSpan(
            t_end=system.units.time_into(10e-12),
            n_steps=1000,
        ),
        (np.full((500, 1), 0.0), np.full((500, 1), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    delta_k = (7e9,)
    _, ax, line_0, _fill_0 = plot_isf(
        result=full_result.with_si_units(),
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("full simulation")

    ballistic_result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t_end=system.units.time_into(3e-12),
            n_steps=1000,
        ),
        n_samples=5000,
        _key=key,
    )

    _, ax, line_1, _ = plot_isf(
        result=ballistic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_1.set_label("ballistic simulation")

    elastic_result, inelastic_result = breakdown_ballistic_trajectory(
        ballistic_result,
        minimum_timescale=get_diffusion_time(
            system, characteristic_length=system.delta_x
        ),
    )

    _, ax, line_2, _ = plot_isf(
        result=elastic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_2.set_label("elastic")
    line_2.set_linestyle(":")

    _, ax, line_3, _ = plot_isf(
        result=inelastic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_3.set_label("inelastic")
    line_3.set_linestyle(":")

    ax.set_xlim(
        0,
        2e-12,
    )

    ax.set_ylim(
        0.6,
        1.0,
    )

    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig(
        "examples/ballistic_langevin/full_isf_plots/1d_periodic.isf.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_periodic_isf()

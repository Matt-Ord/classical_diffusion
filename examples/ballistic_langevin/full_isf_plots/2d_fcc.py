import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    breakdown_filtered_ballistic_trajectory_butterworth,
    get_initial_conditions,
    solve_ballistic_ensemble,
    solve_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import (
    PeriodicSystemFCC,
    UnitSystem,
    get_diffusion_time,
)

system = PeriodicSystemFCC(
    gamma=1e11,
    temperature=114,
    m=8e-27,
    delta_x=3e-10,
    barrier_energy=4e-21,
    units=UnitSystem(),
)

normalized_system = system.with_normalized_units()

key = jrandom.PRNGKey(100)


def _plot_periodic_isf() -> None:

    full_result = solve_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(10e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        (np.full((500, 2), 1.0), np.full((500, 2), 0.0)),
        _key=key,
    )

    direction = np.array([0, 1])
    delta_k = tuple(7e9 * direction / np.linalg.norm(direction))

    fig, ax = get_fancy_figure()
    _, ax, line_0, _fill_0 = plot_isf(
        result=full_result.with_si_units(),
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("full simulation")

    initial_conditions = get_initial_conditions(normalized_system, n_samples=100000)

    ballistic_result = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(10e-12, units=UnitSystem()),
            n_steps=5000,
        ),
        initial_conditions=initial_conditions,
        _key=key,
    )

    _, ax, line_1, _ = plot_isf(
        result=ballistic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_1.set_label("ballistic simulation")

    elastic_result, inelastic_result = (
        breakdown_filtered_ballistic_trajectory_butterworth(
            ballistic_result,
            minimum_timescale=get_diffusion_time(
                normalized_system, characteristic_length=normalized_system.delta_x
            ),
        )
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
        0.0,
        1.0,
    )

    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig(
        "examples/ballistic_langevin/full_isf_plots/2d_fcc.isf.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_periodic_isf()

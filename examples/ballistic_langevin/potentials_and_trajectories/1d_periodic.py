import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_single_x_evolution,
)
from classical_diffusion.langevin import (
    breakdown_filtered_ballistic_trajectory_butterworth,
    plot_periodic_potential_1d,
    solve_single,
)
from classical_diffusion.plot import _get_two_panel_figure, get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import (
    PeriodicSystem1D,
    UnitSystem,
    get_diffusion_time,
)

system = PeriodicSystem1D(
    gamma=4e11,
    temperature=100,
    m=6e-27,
    delta_x=1.48e-10,
    barrier_energy=10e-21,
    units=UnitSystem(),
)

normalized_system = system.with_normalized_units()

key = jrandom.PRNGKey(100)


def _plot_periodic_system() -> None:

    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig(
        "examples/ballistic_langevin/Trajectories_and_potentials/1d_periodic.potential.pdf"
    )


def _plot_ballistic_trajectory() -> None:

    key = jrandom.PRNGKey(100)

    result = solve_single(
        normalized_system.with_gamma(0.0),
        TimeSpan(
            t_end=normalized_system.units.time_into(10e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        (np.full((1,), 0.0), np.full((1,), 2.43)),
        _key=key,
    )

    elastic, inelastic = breakdown_filtered_ballistic_trajectory_butterworth(
        result,
        minimum_timescale=get_diffusion_time(
            normalized_system, characteristic_length=normalized_system.delta_x / 0.1
        ),
    )

    fig, ax = _get_two_panel_figure()

    _, _ax_0, line = plot_single_x_evolution(result=result.with_si_units(), ax=ax[0])
    _, _ax_0, line_e = plot_single_x_evolution(result=elastic.with_si_units(), ax=ax[0])

    _, _ax_1, line_i = plot_single_x_evolution(
        result=inelastic.with_si_units(), ax=ax[1]
    )

    line_i.set_color("C2")

    ax[0].legend(
        handles=[line, line_e],
        labels=["full", "elastic"],
    )
    ax[1].legend(
        handles=[line_i],
        labels=["inelastic"],
    )

    ax[0].set_xlim(1e-12, 2e-12)
    ax[0].set_ylim(5.25e-10, 1.15e-9)
    ax[1].set_xlim(1e-12, 2e-12)
    fig.savefig(
        "examples/ballistic_langevin/Trajectories_and_potentials/1d_periodic.trajectory.pdf"
    )


if __name__ == "__main__":
    _plot_periodic_system()
    _plot_ballistic_trajectory()

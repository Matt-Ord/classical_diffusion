import numpy as np

from classical_diffusion.analysis import (
    plot_x_evolution_1d,
)
from classical_diffusion.langevin import (
    PeriodicSystem1D,
    breakdown_ballistic_trajectory,
    get_diffusion_time,
    plot_periodic_potential_1d,
    solve_single_ballistic,
)
from classical_diffusion.plot import get_fancy_figure, get_two_panel_figure
from classical_diffusion.simulation import TimeSpan

system = PeriodicSystem1D(
    gamma=4e11,
    temperature=100,
    m=6e-27,
    delta_x=1.48e-10,
    barrier_energy=10e-21,
)


def _plot_periodic_system() -> None:

    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig(
        "examples/ballistic_langevin/Trajectories_and_potentials/1d_periodic.potential.pdf"
    )


def _plot_ballistic_trajectory() -> None:

    result = solve_single_ballistic(
        system,
        TimeSpan(
            t_end=system.units.time_into(10e-12),
            n_steps=1000,
        ),
        (np.full((1,), 0.0), np.full((1,), 2.43)),
    )

    elastic, inelastic = breakdown_ballistic_trajectory(
        result,
        minimum_timescale=get_diffusion_time(
            system, characteristic_length=system.delta_x / 0.1
        ),
    )

    fig, ax = get_two_panel_figure()

    _, _ax_0, [line] = plot_x_evolution_1d(result=result.with_si_units(), ax=ax[0])
    _, _ax_0, [line_e] = plot_x_evolution_1d(result=elastic.with_si_units(), ax=ax[0])

    _, _ax_1, [line_i] = plot_x_evolution_1d(result=inelastic.with_si_units(), ax=ax[1])

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

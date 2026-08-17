import jax.random as jrandom
import numpy as np

from classical_diffusion.langevin import (
    PeriodicSystemFCC,
    breakdown_ballistic_trajectory,
    get_diffusion_time,
    plot_2d_trajectory_single,
    plot_periodic_potential_fcc,
    solve_single,
)
from classical_diffusion.plot import get_fancy_figure, get_two_panel_figure
from classical_diffusion.simulation import TimeSpan

system = PeriodicSystemFCC(
    gamma=4e11,
    temperature=102,
    m=6e-26,
    delta_x=20e-10,
    barrier_energy=1.6e-21,
)

normalized_system = system.with_normalized_units()

key = jrandom.PRNGKey(100)


def _plot_periodic_system() -> None:

    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_fcc(system, ax=ax)
    fig.savefig(
        "examples/ballistic_langevin/potentials_and_trajectories2d_fcc.potential.pdf"
    )


def _plot_ballistic_trajectory() -> None:

    key = jrandom.PRNGKey(100)

    result = solve_single(
        normalized_system.with_gamma(0.0),
        TimeSpan(
            t_end=normalized_system.units.time_into(50e-12),
            n_steps=1000,
        ),
        (np.full((2,), 1), np.full((2,), 0.01)),
        _key=key,
    )

    elastic, inelastic = breakdown_ballistic_trajectory(
        result,
        minimum_timescale=get_diffusion_time(
            normalized_system, 1 / normalized_system.gamma
        ),
    )

    fig, ax = get_two_panel_figure()

    _, _ax_0, line = plot_2d_trajectory_single(
        result=result.with_si_units(), ax=ax[0], start_step=30, end_step=30
    )
    _, _ax_0, line_e = plot_2d_trajectory_single(
        result=elastic.with_si_units(), ax=ax[0], start_step=30, end_step=30
    )

    _, _ax_1, line_i = plot_2d_trajectory_single(
        result=inelastic.with_si_units(), ax=ax[1], start_step=30, end_step=30
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

    fig.savefig(
        "examples/ballistic_langevin/potentials_and_trajectories/2d_fcc.trajectory.pdf"
    )


if __name__ == "__main__":
    _plot_periodic_system()
    _plot_ballistic_trajectory()

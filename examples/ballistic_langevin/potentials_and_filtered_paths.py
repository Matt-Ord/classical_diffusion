from classical_diffusion.analysis import (
    plot_x_evolution_1d,
    plot_x_evolution_2d,
)
from classical_diffusion.langevin import (
    LangevinSimulationResult,
    PeriodicSystem1D,
    PeriodicSystemFCC,
    System,
    breakdown_ballistic_trajectory,
    plot_periodic_potential_1d,
    plot_periodic_potential_fcc,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import get_fancy_figure, get_two_panel_figure
from classical_diffusion.simulation import TimeSpan


def _plot_periodic_system_1d() -> None:

    system = PeriodicSystem1D(
        gamma=4e11,
        temperature=150,
        m=6e-27,
        delta_x=1.48e-10,
        barrier_energy=10e-21,
    )

    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig("examples/ballistic_langevin/1d_periodic.potential.pdf")


def _truncate_results[S: System](
    result: LangevinSimulationResult[S],
    times: tuple[float, float],
) -> LangevinSimulationResult[S]:
    mask = (result.times >= times[0]) & (result.times <= times[1])
    return LangevinSimulationResult(
        system=result.system,
        times=result.times[mask],
        x_points=result.x_points[:, :, mask],
        p_points=result.p_points[:, :, mask],
    )


def _plot_filtered_ballistic_trajectory_1d() -> None:

    system = PeriodicSystem1D(
        gamma=4e11,
        temperature=150,
        m=6e-27,
        delta_x=1.48e-10,
        barrier_energy=10e-22,
    )

    result = solve_ensemble_ballistic.call_uncached(
        system,
        TimeSpan(t_start=-20e-12, t_end=20e-12, n_steps=5000),
        n_samples=1,
    )

    elastic, inelastic = breakdown_ballistic_trajectory(
        result, filter_timescale=1 / system.gamma
    )
    elastic = _truncate_results(elastic, times=(0, 0.5e-12))
    inelastic = _truncate_results(inelastic, times=(0, 0.5e-12))
    result = _truncate_results(result, times=(0, 0.5e-12))

    fig, ax = get_two_panel_figure()

    _, _ax_0, lines = plot_x_evolution_1d(result=result, ax=ax[0])
    _, _ax_0, lines_e = plot_x_evolution_1d(result=elastic, ax=ax[0])

    _, _ax_1, lines_i = plot_x_evolution_1d(result=inelastic, ax=ax[1])

    lines[0].set_color("C1")
    lines_e[0].set_color("C3")
    lines_i[0].set_color("C2")

    ax[0].legend(handles=[lines[0], lines_e[0]], labels=["ballistic", "elastic"])
    ax[1].legend(handles=[lines_i[0]], labels=["inelastic"])

    fig.savefig("examples/ballistic_langevin/1d_periodic.trajectory.pdf")


def _plot_periodic_system_fcc() -> None:

    system = PeriodicSystemFCC(
        gamma=4e11,
        temperature=102,
        m=6e-26,
        delta_x=20e-10,
        barrier_energy=1.6e-21,
    )

    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_fcc(system, ax=ax, height=10, width=8)
    fig.savefig("examples/ballistic_langevin/2d_fcc.potential.pdf")


def _plot_filtered_ballistic_trajectory_2d() -> None:

    system = PeriodicSystemFCC(
        gamma=4e11,
        temperature=105,
        m=6e-27,
        delta_x=1.48e-10,
        barrier_energy=1.6e-21,
    )

    result = solve_ensemble_ballistic.call_uncached(
        system,
        TimeSpan(t_start=-10e-11, t_end=10e-11, n_steps=1000),
        n_samples=1,
    )

    elastic, inelastic = breakdown_ballistic_trajectory(
        result,
        filter_timescale=1 / system.gamma,
    )

    fig, ax = get_two_panel_figure()

    _, _ax_0, line = plot_x_evolution_2d(result=result, ax=ax[0])
    _, _ax_0, line_e = plot_x_evolution_2d(result=elastic, ax=ax[0])

    _, _ax_1, line_i = plot_x_evolution_2d(result=inelastic, ax=ax[1])

    line[0].set_color("C1")
    line_e[0].set_color("C3")
    line_i[0].set_color("C2")

    ax[0].legend(
        handles=[line[0], line_e[0]],
        labels=["ballistic", "elastic"],
    )
    ax[1].legend(
        handles=[line_i[0]],
        labels=["inelastic"],
    )

    fig.savefig("examples/ballistic_langevin/2d_fcc.trajectory.pdf")


if __name__ == "__main__":
    _plot_periodic_system_1d()
    _plot_filtered_ballistic_trajectory_1d()
    _plot_periodic_system_fcc()
    _plot_filtered_ballistic_trajectory_2d()

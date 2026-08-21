from classical_diffusion.analysis import (
    plot_x_evolution_1d,
)
from classical_diffusion.langevin import (
    LangevinSimulationResult,
    PeriodicSystem1D,
    System,
    breakdown_ballistic_trajectory,
    plot_periodic_potential_1d,
    solve_ballistic_ensemble,
)
from classical_diffusion.plot import get_fancy_figure, get_two_panel_figure
from classical_diffusion.simulation import TimeSpan

system = PeriodicSystem1D(
    gamma=4e11,
    temperature=150,
    m=6e-27,
    delta_x=1.48e-10,
    barrier_energy=10e-21,
)


def _plot_periodic_system() -> None:

    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig(
        "examples/ballistic_langevin/potentials_and_trajectories/1d_periodic.potential.pdf"
    )


def _filter_results[S: System](
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


def _plot_ballistic_trajectory() -> None:

    result = solve_ballistic_ensemble.call_uncached(
        system,
        TimeSpan(t_start=-10e-12, t_end=10e-12, n_steps=5000),
        n_samples=1,
    )

    elastic, inelastic = breakdown_ballistic_trajectory(
        result, filter_timescale=1 / system.gamma
    )
    elastic = _filter_results(elastic, times=(0, 0.5e-12))
    inelastic = _filter_results(inelastic, times=(0, 0.5e-12))
    result = _filter_results(result, times=(0, 0.5e-12))

    fig, ax = get_two_panel_figure()

    _, _ax_0, lines = plot_x_evolution_1d(result=result, ax=ax[0])
    _, _ax_0, lines_e = plot_x_evolution_1d(result=elastic, ax=ax[0])

    _, _ax_1, lines_i = plot_x_evolution_1d(result=inelastic, ax=ax[1])

    lines_i[0].set_color("C2")

    ax[0].legend(handles=[lines[0], lines_e[0]], labels=["full", "elastic"])
    ax[1].legend(handles=[lines_i[0]], labels=["inelastic"])

    fig.savefig(
        "examples/ballistic_langevin/potentials_and_trajectories/1d_periodic.trajectory.pdf"
    )


if __name__ == "__main__":
    _plot_periodic_system()
    _plot_ballistic_trajectory()

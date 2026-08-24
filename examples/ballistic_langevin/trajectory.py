import numpy as np

from classical_diffusion.analysis import (
    plot_x_evolution_1d,
    plot_x_evolution_2d,
)
from classical_diffusion.langevin import (
    LangevinSimulationResult,
    PeriodicSystem1D,
    System,
    breakdown_ballistic_trajectory,
    physical_system,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan


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


def _wrap_1d_results[S: PeriodicSystem1D](
    result: LangevinSimulationResult[S],
) -> LangevinSimulationResult[S]:
    time = np.argmin(np.abs(result.times))
    x_0 = result.x_points[:, 0, time] % result.system.delta_x
    x_points = (
        result.x_points - (result.x_points[:, 0, time] - x_0)[:, np.newaxis, np.newaxis]
    )
    return LangevinSimulationResult(
        system=result.system,
        times=result.times,
        x_points=x_points,
        p_points=result.p_points,
    )


def _plot_ballistic_trajectory_1d() -> None:
    system = physical_system.SODIUM_COPPER_1D
    plot_time = 0.5 / (system.gamma)
    simulation_time = TimeSpan(
        t_start=-(2 / system.gamma), t_end=plot_time + (2 / system.gamma), n_steps=5000
    )

    result = solve_ensemble_ballistic(
        system,
        simulation_time,
        n_samples=1,
        energy_range=(-np.inf, system.barrier_energy),
    )
    result = _wrap_1d_results(result)

    elastic, _inelastic = breakdown_ballistic_trajectory(
        result, filter_timescale=1 / system.gamma
    )
    elastic = _filter_results(elastic, times=(0, plot_time))
    result = _filter_results(result, times=(0, plot_time))

    fig, ax = get_fancy_figure()

    _, _ax_0, lines = plot_x_evolution_1d(result=result, ax=ax)
    _, _ax_0, lines_e_trapped = plot_x_evolution_1d(result=elastic, ax=ax)
    lines[0].set_color("C0")
    lines_e_trapped[0].set_color("C1")

    result = solve_ensemble_ballistic(
        system,
        simulation_time,
        n_samples=1,
        energy_range=(system.barrier_energy, np.inf),
    )
    result = _wrap_1d_results(result)

    elastic, _inelastic = breakdown_ballistic_trajectory(
        result, filter_timescale=1 / system.gamma
    )
    elastic = _filter_results(elastic, times=(0, plot_time))
    result = _filter_results(result, times=(0, plot_time))

    _, _ax_0, lines = plot_x_evolution_1d(result=result, ax=ax)
    _, _ax_0, lines_e = plot_x_evolution_1d(result=elastic, ax=ax)
    lines[0].set_color("C0")
    lines_e[0].set_color("C1")
    lines_e[0].set_linestyle("--")

    ax.legend(handles=[lines[0], lines_e_trapped[0]], labels=["full", "filtered"])

    fig.savefig("examples/ballistic_langevin/trajectory.1d.pdf")


def _plot_ballistic_trajectory_2d() -> None:

    system = physical_system.SODIUM_COPPER_2D
    plot_time = 20 / (system.gamma)

    result = solve_ensemble_ballistic(
        system,
        TimeSpan(
            t_start=-(2 / system.gamma),
            t_end=plot_time + (2 / system.gamma),
            n_steps=5000,
        ),
        n_samples=1,
        energy_range=(system.barrier_energy, np.inf),
    )

    elastic, _inelastic = breakdown_ballistic_trajectory(
        result, filter_timescale=1 / system.gamma
    )
    elastic = _filter_results(elastic, times=(0, plot_time))
    result = _filter_results(result, times=(0, plot_time))

    fig, ax = get_fancy_figure()

    _, _ax_0, line = plot_x_evolution_2d(result=result, ax=ax)
    _, _ax_0, line_e = plot_x_evolution_2d(result=elastic, ax=ax)

    ax.legend(handles=[line[0], line_e[0]], labels=["full", "filtered"])

    fig.savefig("examples/ballistic_langevin/trajectory.2d.pdf")


if __name__ == "__main__":
    _plot_ballistic_trajectory_1d()
    _plot_ballistic_trajectory_2d()

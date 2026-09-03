from pathlib import Path

import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    LangevinSimulationResult,
    System,
    breakdown_ballistic_trajectory,
    solve_ensemble,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import unit_with_trajectory_scale
from classical_diffusion.util import cache_base_path


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


def _plot_periodic_isf_1d() -> None:
    system = SODIUM_COPPER_SYSTEM_1D
    system = system.with_units(
        unit_with_trajectory_scale(
            system.units, length=system.delta_x, time=1 / system.gamma
        )
    )
    full_result = solve_ensemble(
        system,
        TimeSpan(t_end=1 / system.gamma, n_steps=500),
        n_samples=2000,
    )

    fig, ax = get_fancy_figure()
    delta_k = (2 * np.pi / system.delta_x * 0.3,)

    _, ax, line_0, _fill_0 = plot_isf(result=full_result, ax=ax, delta_k=delta_k)
    line_0.set_label("full simulation")

    ballistic_result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_start=-2 / system.gamma, t_end=3 / system.gamma, n_steps=2000),
        n_samples=2000,
    )

    _, ax, line_1, _ = plot_isf(
        result=ballistic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_1.set_label("ballistic simulation")

    elastic_result, inelastic_result = breakdown_ballistic_trajectory(
        ballistic_result, filter_timescale=1 / system.gamma
    )
    elastic_result = _truncate_results(elastic_result, times=(0, 1 / system.gamma))
    inelastic_result = _truncate_results(inelastic_result, times=(0, 1 / system.gamma))

    _, ax, line_2, _ = plot_isf(
        result=inelastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_2.set_label("inelastic")

    _, ax, line_3, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_3.set_label("elastic")

    ax.set_xlim(0, 1 / system.gamma)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$\gamma t$")

    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig("examples/ballistic_langevin/isf_plots.1d.pdf")


def _plot_periodic_isf_2d() -> None:

    system = SODIUM_COPPER_SYSTEM_2D
    system = system.with_units(
        unit_with_trajectory_scale(
            system.units, length=system.delta_x, time=1 / system.gamma
        )
    )
    full_result = solve_ensemble(
        system,
        TimeSpan(t_end=4 / system.gamma, n_steps=1000),
        n_samples=2000,
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
        pairwise=False,
    )
    line_0.set_label("full simulation")

    ballistic_result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_end=4 / system.gamma, n_steps=1000),
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

    ax.set_xlim(0, 0.5 / system.gamma)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$\gamma t$")

    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig("examples/ballistic_langevin/isf_plots.2d.pdf")


if __name__ == "__main__":
    # These examples plot the ensemble isf of the provided system for a full langevin and ballistic trajectory.
    # It also splits ballistic trajectories into elastic and inelastic components (see filtered_paths) and plots
    # the corresponding ensemble isf on the same figure
    with cache_base_path(Path("examples/data")):
        _plot_periodic_isf_1d()
        _plot_periodic_isf_2d()

from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from classical_diffusion.analysis import (
    plot_x_evolution_1d,
    plot_x_evolution_2d,
)
from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    LangevinSimulationResult,
    PeriodicSystemFCC,
    System,
    breakdown_ballistic_trajectory,
    plot_periodic_potential_1d,
    plot_periodic_potential_fcc,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import get_fancy_figure, get_figure, get_two_panel_figure
from classical_diffusion.simulation import TimeSpan

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _annotate_bridge_site_energy(
    ax: Axes,
    system: PeriodicSystemFCC,
) -> None:
    a1, a2 = system.lattice_vectors
    origin = a1 + a2
    bridge_point = origin + a2 + a1 / 2  # midpoint of the top edge

    potential_func = sp.lambdify(
        system.lambda_symbols,
        system.potential_expr,
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )
    bridge_energy = float(
        potential_func(bridge_point[0], bridge_point[1], *system.params)
    )

    ax.plot(
        *bridge_point,
        marker="x",
        markersize=9,
        markeredgewidth=2,
        zorder=10,
    )
    ax.annotate(
        f"{bridge_energy:.3g} J".strip(),
        xy=bridge_point,
        xytext=(18.0, 12.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        zorder=10,
    )


def _annotate_top_site_energy(
    ax: Axes,
    system: PeriodicSystemFCC,
    origin_site: tuple[int, int] = (0, 0),
) -> None:
    a1, a2 = system.lattice_vectors
    top_point = origin_site[0] * a1 + origin_site[1] * a2

    potential_func = sp.lambdify(
        system.lambda_symbols,
        system.potential_expr,
        modules=[{"DerivativeSafeMod": np.mod}, "numpy"],
    )
    top_energy = float(potential_func(top_point[0], top_point[1], *system.params))

    ax.plot(
        *top_point,
        marker="x",
        markersize=9,
        markeredgewidth=2,
        zorder=10,
    )
    ax.annotate(
        f"{top_energy:.3g} J".strip(),
        xy=top_point,
        xytext=(12.0, -12.0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=9,
        zorder=10,
    )


def _annotate_hollow_site_distance(
    system: PeriodicSystemFCC,
    *,
    label_offset: tuple[float, float] = (12.0, 0.0),
    ax: Axes | None = None,
) -> None:
    _fig, ax = get_figure(ax)

    a1, a2 = system.lattice_vectors

    hollow_a = (a1 + a2) / 3
    hollow_b = 2 * (a1 + a2) / 3

    (line,) = ax.plot(*np.array([hollow_a, hollow_b]).T)
    line.set_marker("o")

    ax.annotate(
        f"{np.linalg.norm(hollow_b - hollow_a):.3g} m".strip(),
        xy=(hollow_a + hollow_b) / 2,
        xytext=label_offset,
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        zorder=10,
    )


def _plot_periodic_system_1d() -> None:

    system = SODIUM_COPPER_SYSTEM_1D

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

    system = SODIUM_COPPER_SYSTEM_1D

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

    system = SODIUM_COPPER_SYSTEM_2D

    fig, ax = get_fancy_figure()
    fig, ax, mesh, _unit_cell = plot_periodic_potential_fcc(system, ax=ax, shape=(5, 4))
    mesh.set_rasterized(True)

    _annotate_bridge_site_energy(ax, system)
    _annotate_top_site_energy(ax, system)
    _annotate_hollow_site_distance(system, ax=ax)

    fig.savefig("examples/ballistic_langevin/2d_fcc.potential.pdf", dpi=600)


def _plot_filtered_ballistic_trajectory_2d() -> None:

    system = SODIUM_COPPER_SYSTEM_2D

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
    # _plot_periodic_system_1d()
    # _plot_filtered_ballistic_trajectory_1d()
    _plot_periodic_system_fcc()
    # _plot_filtered_ballistic_trajectory_2d()

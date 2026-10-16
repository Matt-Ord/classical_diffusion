from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from classical_diffusion.analysis import (
    plot_p_evolution_1d,
    plot_x_evolution_1d,
    plot_x_evolution_2d,
)
from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    LangevinSimulationResult,
    PeriodicSystem1D,
    PeriodicSystemFCC,
    System,
    breakdown_ballistic_trajectory,
    plot_periodic_potential_1d,
    plot_periodic_potential_fcc,
    solve_ensemble_ballistic,
    solve_many_ballistic,
)
from classical_diffusion.plot import (
    CAM_BLUE,
    CAM_CHERRY,
    get_fancy_figure,
    get_figure,
    get_two_panel_figure,
)
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import unit_with_trajectory_scale
from classical_diffusion.util import cache_base_path

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
    system = system.with_units(
        unit_with_trajectory_scale(
            system.units,
            length=system.delta_x,
            time=1 / system.gamma,
            momentum=np.sqrt(system.m * system.kbt),
        )
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig("examples/ballistic_langevin/filtered_paths.1d.potential.pdf")

    result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_start=-2 / system.gamma, t_end=6 / system.gamma, n_steps=4000),
        n_samples=1,
        energy_range=(-np.inf, system.barrier_energy),
    )

    elastic, inelastic = breakdown_ballistic_trajectory(
        result, filter_timescale=1 / system.gamma
    )
    elastic = _truncate_results(elastic, times=(0, 4 / system.gamma))
    inelastic = _truncate_results(inelastic, times=(0, 4 / system.gamma))
    result = _truncate_results(result, times=(0, 4 / system.gamma))

    fig, ax = get_two_panel_figure()

    _, _ax_0, lines = plot_x_evolution_1d(result=result, ax=ax[0])
    _, _ax_0, lines_e = plot_x_evolution_1d(result=elastic, ax=ax[0])
    _, _ax_1, lines_i = plot_x_evolution_1d(result=inelastic, ax=ax[1])

    lines[0].set_color("C1")
    lines_e[0].set_color("C3")
    lines_i[0].set_color("C2")

    ax[0].legend(handles=[lines[0], lines_e[0]], labels=["ballistic", "elastic"])
    ax[1].legend(handles=[lines_i[0]], labels=["inelastic"])

    fig.savefig("examples/ballistic_langevin/filtered_paths.1d_x.pdf")

    fig, ax = get_two_panel_figure()

    _, _ax_0, lines = plot_p_evolution_1d(result=result, ax=ax[0])
    _, _ax_0, lines_e = plot_p_evolution_1d(result=elastic, ax=ax[0])
    _, _ax_1, lines_i = plot_p_evolution_1d(result=inelastic, ax=ax[1])

    lines[0].set_color("C1")
    lines_e[0].set_color("C3")
    lines_i[0].set_color("C2")

    ax[0].legend(handles=[lines[0], lines_e[0]], labels=["ballistic", "elastic"])
    ax[1].legend(handles=[lines_i[0]], labels=["inelastic"])

    fig.savefig("examples/ballistic_langevin/filtered_paths.1d_p.pdf")


def _add_direction_arrows(  # ruff:ignore[too-many-arguments]
    ax: Axes,
    path: tuple[np.ndarray, np.ndarray],
    x_targets: list[float],
    *,
    color: str,
    n_crossings: int = 1,
    reverse: bool = False,
) -> None:
    """Add arrows along a phase space path, pointing in the direction of travel."""
    x_data, p_data = path
    if reverse:
        x_data, p_data = x_data[::-1], p_data[::-1]
    step = max(len(x_data) // 400, 1)
    for x_target in x_targets:
        side = np.sign(x_data - x_target)
        # Ignore the breaks in the line, where x_data is nan
        crossings = np.flatnonzero((side[1:] * side[:-1]) < 0)
        # Skip the initial point, where every trajectory starts
        crossings = crossings[(crossings > step) & (crossings < len(x_data) - step)]
        for i in crossings[:n_crossings]:
            ax.annotate(
                "",
                xy=(x_data[i + step], p_data[i + step]),
                xytext=(x_data[i], p_data[i]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 0,
                    "mutation_scale": 12,
                },
            )


def _wrap_into_cell(
    x_points: np.ndarray,
    p_points: np.ndarray,
    delta_x: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap x into [-delta_x / 2, delta_x / 2), breaking the line at each wrap."""
    x_wrapped = np.mod(x_points + delta_x / 2, delta_x) - delta_x / 2
    breaks = np.flatnonzero(np.abs(np.diff(x_wrapped)) > delta_x / 2) + 1
    return (
        np.insert(x_wrapped, breaks, np.nan),
        np.insert(p_points, breaks, np.nan),
    )


def _plot_separatrix_1d(system: PeriodicSystem1D, ax: Axes) -> None:
    """Plot the separatrix E = E_b over a single period of the potential."""
    x = np.linspace(-system.delta_x / 2, system.delta_x / 2, 1000)
    potential = (
        0.5 * system.barrier_energy * (1 - np.cos(2 * np.pi * x / system.delta_x))
    )
    separatrix = np.sqrt(2 * system.m * (system.barrier_energy - potential))
    for sign in (1, -1):
        ax.plot(x, sign * separatrix, color=CAM_BLUE.dark, linestyle="--", lw=1)


def _plot_phase_space_trajectories_1d() -> None:
    """Plot a left escaped, right escaped and oscillating trajectory in (x, p)."""
    system = SODIUM_COPPER_SYSTEM_1D
    system = system.with_units(
        unit_with_trajectory_scale(
            system.units,
            length=system.delta_x,
            time=1 / system.gamma,
            momentum=np.sqrt(system.m * system.kbt),
        )
    )
    # All trajectories start in the hollow site at x = 0, where V(x) = 0.
    # Escaped trajectories have energy above the barrier, the oscillating
    # trajectory is trapped below it.
    escaped_p = np.sqrt(2 * system.m * system.barrier_energy * 1.4)
    trapped_p = np.sqrt(2 * system.m * system.barrier_energy * 0.6)
    result = solve_many_ballistic(
        system,
        TimeSpan(t_end=1 / system.gamma, n_steps=4000),
        (np.zeros((3, 1)), np.array([[-escaped_p], [escaped_p], [trapped_p]])),
    )

    fig, ax = get_fancy_figure()
    _plot_separatrix_1d(system, ax)

    escaped_targets = [-0.25 * system.delta_x, 0.25 * system.delta_x]
    left, right, trapped = result
    for res in (left, right):
        path = _wrap_into_cell(res.x_points[0], res.p_points[0], system.delta_x)
        ax.plot(*path, color=CAM_BLUE.warm)
        _add_direction_arrows(ax, path, escaped_targets, color=CAM_BLUE.warm)

    # The trapped trajectory is drawn with arrows in both directions (cw and ccw)
    path = (trapped.x_points[0], trapped.p_points[0])
    ax.plot(*path, color=CAM_CHERRY.base)
    for x_target, reverse in ((0.05, False), (-0.05, True)):
        _add_direction_arrows(
            ax,
            path,
            [x_target * system.delta_x],
            color=CAM_CHERRY.base,
            n_crossings=2,
            reverse=reverse,
        )

    ax.set_xlim(-system.delta_x / 2, system.delta_x / 2)
    ax.set_xticks(np.linspace(-0.5, 0.5, 5) * system.delta_x)
    ax.set_xlabel(r"$x / \Delta x$")
    ax.set_ylabel(r"$p / \sqrt{m k_B T}$")

    fig.savefig("examples/ballistic_langevin/filtered_paths.1d_phase_space.pdf")


def _plot_filtered_ballistic_trajectory_2d() -> None:

    system = SODIUM_COPPER_SYSTEM_2D

    fig, ax = get_fancy_figure()
    _, _, mesh, _unit_cell = plot_periodic_potential_fcc(system, ax=ax, shape=(5, 4))
    mesh.set_rasterized(True)

    _annotate_bridge_site_energy(ax, system)
    _annotate_top_site_energy(ax, system)
    _annotate_hollow_site_distance(system, ax=ax)

    fig.savefig(
        "examples/ballistic_langevin/filtered_paths.2d_fcc.potential.pdf", dpi=600
    )
    result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_start=-20 / system.gamma, t_end=20 / system.gamma, n_steps=1000),
        n_samples=1,
        energy_range=(system.barrier_energy, np.inf),
    )

    elastic, inelastic = breakdown_ballistic_trajectory(
        result,
        filter_timescale=0.5 / system.gamma,
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

    fig.savefig("examples/ballistic_langevin/filtered_paths.2d_fcc.pdf")


if __name__ == "__main__":
    # These examples plot the potential of the provided system and to comparison of the
    # trajectory before and after a butterworth filter has been applied to split
    # into elastic and inelastic components of the motion.
    #
    # A filter timescale of 1 / gamma removes all frequencies of motion that would
    # be washed away once friction is reintroduced
    #
    # Time spans positive and negative values s=to remove edge effects, from applying the filter
    # at t = 0.
    with cache_base_path(Path("examples/data")):
        _plot_filtered_ballistic_trajectory_1d()
        _plot_phase_space_trajectories_1d()
        _plot_filtered_ballistic_trajectory_2d()

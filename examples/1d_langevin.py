from pathlib import Path

import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
    plot_root_mean_square_x,
    plot_x_evolution_1d,
)
from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    plot_force_1d,
    plot_frictional_force_distribution,
    plot_periodic_potential_1d,
    shift_origin_to_unit_cell_1d,
    solve_ensemble,
    solve_ensemble_ballistic,
    solve_single,
    solve_single_ballistic,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import unit_with_trajectory_scale
from classical_diffusion.util import cache_base_path


def _plot_periodic_system() -> None:
    system = SODIUM_COPPER_SYSTEM_1D
    fig, ax = get_fancy_figure()
    _, _, line0 = plot_periodic_potential_1d(system, ax=ax)
    line0.set_label("potential")

    _, _, line1 = plot_force_1d(system, 0, system.delta_x, ax=ax.twinx())
    line1.set_color("C1")
    line1.set_label("force")

    ax.legend(handles=[line0, line1])
    fig.savefig("examples/1d_langevin.potential.pdf")


def _plot_periodic_trajectory() -> None:
    system = SODIUM_COPPER_SYSTEM_1D
    # Convert to a nice unit system where gamma = 1 and delta_x = 1
    system = system.with_units(
        unit_with_trajectory_scale(
            system.units, length=system.delta_x, time=1 / system.gamma
        )
    )

    result = solve_ensemble(
        system,
        TimeSpan(t_end=5 / system.gamma, n_steps=4000),
        n_samples=20,
    )
    result = shift_origin_to_unit_cell_1d(result)

    fig, ax = get_fancy_figure()
    _, _, _ = plot_x_evolution_1d(result=result, ax=ax)
    ax.set_ylim(-8, 8)
    ax.set_xlim(0, 3)
    ax.set_ylabel(r"$x \, / \, \Delta x$")
    ax.set_xlabel(r"$t \, / \, \gamma^{-1}$")

    fig.savefig("./examples/1d_langevin.trajectory.pdf")


def _plot_ballistic_trajectory() -> None:
    system = SODIUM_COPPER_SYSTEM_1D
    # Convert to a nice unit system where gamma = 1 and delta_x = 1
    system = system.with_units(
        unit_with_trajectory_scale(
            system.units, length=system.delta_x, time=1 / system.gamma
        )
    )

    time_span = TimeSpan(t_end=1 / system.gamma, n_steps=4000)
    initial_condition = (np.asarray([0]), np.asarray([9e-24]))
    langevin_result = solve_single(
        system, time_span, initial_condition=initial_condition
    )
    ballistic_result = solve_single_ballistic(
        system, time_span, initial_condition=initial_condition
    )

    fig, ax = get_fancy_figure()
    _, _, langevin_line = plot_x_evolution_1d(result=langevin_result, ax=ax)
    langevin_line[0].set_label("langevin")
    _, _, ballistic_line = plot_x_evolution_1d(result=ballistic_result, ax=ax)
    ballistic_line[0].set_label("ballistic")

    ax.set_ylim(-0.2, 0.2)
    ax.set_xlim(0, 1)
    ax.set_ylabel(r"$x \, / \, \Delta x$")
    ax.set_xlabel(r"$t \, / \, \gamma^{-1}$")

    fig.savefig("./examples/1d_langevin.trajectory.ballistic_trajectory.pdf")


def _plot_periodic_isf() -> None:
    system = SODIUM_COPPER_SYSTEM_1D
    result = solve_ensemble(
        system, TimeSpan(t_end=2 / system.gamma, n_steps=4000), n_samples=1000
    )

    fig, ax = get_fancy_figure()

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)
    _, ax, line_0, _fill_0 = plot_isf(result=result, ax=ax, delta_k=delta_k)
    line_0.set_label("full simulation")

    result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_end=2 / system.gamma, n_steps=1000),
        n_samples=10000,
    )

    _, ax, line_1, _ = plot_isf(result=result, ax=ax, delta_k=delta_k, pairwise=False)
    line_1.set_label("ballistic simulation")

    ax.set_xlim(0, 2 / system.gamma)
    ax.set_ylim(0, 1)

    ax.legend(handles=[line_0, line_1])
    fig.savefig("examples/1d_langevin.isf.pdf")


def _plot_force() -> None:
    system = SODIUM_COPPER_SYSTEM_1D
    result = solve_ensemble(
        system, TimeSpan(t_end=2 / system.gamma, n_steps=4000), n_samples=1000
    )

    fig, ax = get_fancy_figure()

    _, _, (_line, _bars) = plot_frictional_force_distribution(
        result=result, ax=ax, n_bins=100
    )
    fig.savefig("examples/1d_langevin.frictional_force.pdf", dpi=1000)


def _plot_root_mean_square_x() -> None:
    system = SODIUM_COPPER_SYSTEM_1D
    result = solve_ensemble(
        system, TimeSpan(t_end=2 / system.gamma, n_steps=4000), n_samples=1000
    )

    fig, ax = get_fancy_figure()

    _, ax, _line_0 = plot_root_mean_square_x(result=result, ax=ax)
    fig.savefig("examples/1d_langevin.rms.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _plot_periodic_system()
        _plot_periodic_trajectory()
        _plot_ballistic_trajectory()
        # The particle experiences a frictional force,
        # which scales as sqrt(2 m gamma k_B T / dt).
        _plot_force()
        _plot_periodic_isf()
        _plot_root_mean_square_x()

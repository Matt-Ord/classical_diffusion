import dataclasses
from pathlib import Path

import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    breakdown_ballistic_trajectory,
    get_effective_mass,
    get_under_barrier_occupation,
    plot_exact_flat_ballistic_isf,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import CAM_BLUE, CAM_CHERRY, get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.util import cache_base_path


def _plot_effective_mass_fitted_isf_1d() -> None:

    system = SODIUM_COPPER_SYSTEM_1D
    fig, ax = get_fancy_figure()
    delta_k = (2 * np.pi / system.delta_x * 0.2,)

    result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_end=8 / system.gamma, n_steps=5000),
        n_samples=5000,
    )
    elastic_result, _ = breakdown_ballistic_trajectory(
        result,
        filter_timescale=1 / system.gamma,
    )

    _, ax, line_0, fill_0 = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")
    line_0.set_color(CAM_CHERRY.warm)
    fill_0.set_color(CAM_CHERRY.warm)

    under_barrier_probability = get_under_barrier_occupation(
        system,
        x_points=result.x_points[:, :, 0],
        p_points=result.p_points[:, :, 0],
        barrier_energy=system.barrier_energy,
    )

    _, ax, line_1 = plot_exact_flat_ballistic_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        offset=under_barrier_probability,
        times=result.times,
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")
    line_1.set_color(CAM_BLUE.dark)

    effective_mass = get_effective_mass(
        result,
        filter_timescale=1 / system.gamma,
        under_barrier_probability=under_barrier_probability,
    )

    _, ax, line_2 = plot_exact_flat_ballistic_isf(
        system=dataclasses.replace(system.as_canonical(), m=effective_mass),
        ax=ax,
        delta_k=delta_k,
        offset=under_barrier_probability,
        times=result.times,
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")
    line_2.set_color(CAM_CHERRY.dark)

    ax.set_xlim(0, 0.5 / system.gamma)
    ax.set_ylim(0.5, 1)
    ax.legend(handles=[line_0, line_1, line_2])
    fig.savefig("examples/ballistic_langevin/effective_mass_fit.1d.pdf")


def _plot_effective_mass_fitted_isf_2d() -> None:

    system = SODIUM_COPPER_SYSTEM_2D
    fig, ax = get_fancy_figure()
    direction = np.array([0, 1])
    delta_k = tuple(
        2 * np.pi / system.delta_x * 0.2 * direction / np.linalg.norm(direction)
    )

    result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_end=8 / system.gamma, n_steps=1000),
        n_samples=100,
    )
    elastic_result, _ = breakdown_ballistic_trajectory(
        result, filter_timescale=1 / system.gamma
    )

    _, ax, line_0, fill_0 = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")
    line_0.set_color(CAM_CHERRY.warm)
    fill_0.set_color(CAM_CHERRY.warm)

    under_barrier_probability = get_under_barrier_occupation(
        system,
        x_points=result.x_points[:, :, 0],
        p_points=result.p_points[:, :, 0],
        barrier_energy=system.barrier_energy,
    )

    _, ax, line_1 = plot_exact_flat_ballistic_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        offset=under_barrier_probability,
        times=result.times,
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")
    line_1.set_color(CAM_BLUE.dark)

    effective_mass = get_effective_mass(
        result,
        filter_timescale=1 / system.gamma,
        under_barrier_probability=under_barrier_probability,
    )

    _, ax, line_2 = plot_exact_flat_ballistic_isf(
        system=dataclasses.replace(system.as_canonical(), m=effective_mass),
        ax=ax,
        delta_k=delta_k,
        offset=under_barrier_probability,
        times=result.times,
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")
    line_2.set_color(CAM_CHERRY.dark)

    ax.set_xlim(0, 3e-12)
    ax.set_ylim(0.5, 1)
    ax.legend(handles=[line_0, line_1, line_2])
    fig.savefig("examples/ballistic_langevin/effective_mass_fit.2d.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _plot_effective_mass_fitted_isf_1d()
        _plot_effective_mass_fitted_isf_2d()

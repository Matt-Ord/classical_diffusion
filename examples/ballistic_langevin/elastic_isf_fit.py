import dataclasses
from pathlib import Path

import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    SODIUM_COPPER_SYSTEM_1D,
    LangevinSimulationResult,
    System,
    breakdown_ballistic_trajectory,
    get_effective_mass,
    get_exact_elastic_isf,
    get_gamma_elastic_isf,
    get_under_barrier_occupation,
    plot_exact_flat_ballistic_isf,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import CAM_BLUE, CAM_CHERRY, get_fancy_figure
from classical_diffusion.simulation import TimeSpan
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


def _plot_elastic_isf_fit_1d() -> None:

    system = SODIUM_COPPER_SYSTEM_1D

    fig, ax = get_fancy_figure()
    delta_k = (2 * np.pi / system.delta_x * 0.2,)

    result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_end=8 / system.gamma, n_steps=4000),
        n_samples=10000,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(
        result,
        filter_timescale=1 / system.gamma,
    )
    elastic_result = _truncate_results(
        elastic_result, times=(2 / system.gamma, 4 / system.gamma)
    )

    _, ax, line_0, fill_0 = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")
    line_0.set_color(CAM_CHERRY.warm)
    fill_0.set_color(CAM_CHERRY.warm)

    times = elastic_result.times - elastic_result.times[0]

    (line_1,) = ax.plot(times, get_exact_elastic_isf(system, delta_k, times))
    line_1.set_label("exact")
    line_1.set_linestyle("--")
    line_1.set_color(CAM_BLUE.dark)

    trapped_probability = get_under_barrier_occupation(
        result,
        system.barrier_energy,
    )
    effective_mass = get_effective_mass(
        result,
        filter_timescale=1 / system.gamma,
    )

    _, ax, line_2 = plot_exact_flat_ballistic_isf(
        system=dataclasses.replace(system.as_canonical(), m=effective_mass),
        ax=ax,
        delta_k=delta_k,
        offset=trapped_probability,
        times=times,
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")
    line_2.set_color(CAM_CHERRY.dark)

    (line_3,) = ax.plot(times, get_gamma_elastic_isf(system, delta_k, times))
    line_3.set_label("gamma")
    line_3.set_linestyle("-.")
    line_3.set_color(CAM_BLUE.warm)

    ax.set_title("")
    ax.set_xlim(0, 0.6 / system.gamma)
    ax.set_ylim(0.7, 1)
    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig("examples/ballistic_langevin/elastic_isf_fit.1d.pdf")


if __name__ == "__main__":
    with cache_base_path(Path("examples/data")):
        _plot_elastic_isf_fit_1d()

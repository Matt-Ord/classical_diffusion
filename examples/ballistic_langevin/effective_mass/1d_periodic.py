import dataclasses

import numpy as np
from scipy.constants import Avogadro
from scipy.integrate import quad
from scipy.special import ellipk

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    PeriodicSystem1D,
    breakdown_ballistic_trajectory,
    get_effective_mass,
    get_under_barrier_occupation,
    plot_exact_flat_ballistic_isf,
    solve_ensemble_ballistic,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import UnitSystem


def _get_free_effective_mass_exact_1d_periodic_directly(
    system: PeriodicSystem1D,
) -> float:
    u0 = system.barrier_energy / (2 * system.kbt)

    def integrand_denominator(epsilon: float) -> float:
        return np.sqrt(epsilon) / ellipk(1 / epsilon) * np.exp(-2 * u0 * epsilon)

    def integrand_running(epsilon: float) -> float:
        return (
            2 * (1 / np.sqrt(epsilon)) * ellipk(1 / epsilon) * np.exp(-2 * u0 * epsilon)
        )

    denominator_integral, _ = quad(integrand_denominator, 1, np.inf)
    running_integral, _ = quad(integrand_running, 1, np.inf)

    partition = running_integral

    return system.m * partition / (denominator_integral * 2 * u0 * np.pi**2)


def _plot_effective_mass_offset_isf() -> None:

    system = PeriodicSystem1D(
        gamma=5e11,
        temperature=155,
        m=3.8175458e-26,
        delta_x=(1 / np.sqrt(3)) * 2.558e-10,
        barrier_energy=(416.78 - 414.24) * 1e3 / Avogadro,
    )
    fig, ax = get_fancy_figure()
    delta_k = (2 * np.pi / system.delta_x * 0.2,)

    result = solve_ensemble_ballistic(
        system,
        TimeSpan(t_start=-20e-12, t_end=20e-12, n_steps=5000),
        n_samples=5_000,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(
        result, filter_timescale=1 / system.gamma
    )

    _, ax, line_0, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    result_free = solve_ensemble_ballistic(
        system,
        TimeSpan(t_start=-20e-12, t_end=20e-12, n_steps=5000),
        n_samples=500,
        minimum_energy=system.barrier_energy,
    )

    prob_under_barrier = get_under_barrier_occupation(
        system,
        x_points=result.x_points[:, :, 0],
        p_points=result.p_points[:, :, 0],
        barrier_energy=system.barrier_energy,
    )

    _, ax, line_1 = plot_exact_flat_ballistic_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        offset=prob_under_barrier,
        times=result.times,
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")

    effective_mass = UnitSystem().mass_into(
        get_effective_mass(result_free, filter_timescale=1 / system.gamma),
        units=system.units,
    )

    _, ax, line_2 = plot_exact_flat_ballistic_isf(
        system=dataclasses.replace(system.as_canonical(), m=effective_mass),
        ax=ax,
        delta_k=delta_k,
        offset=prob_under_barrier,
        times=result.times,
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")

    ax.set_xlim(0, 2e-12)
    ax.set_ylim(0.5, 1)
    ax.legend(handles=[line_0, line_1, line_2])
    fig.savefig(
        "examples/ballistic_langevin/offset_effective_mass/1d_periodic.offset.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_effective_mass_offset_isf()

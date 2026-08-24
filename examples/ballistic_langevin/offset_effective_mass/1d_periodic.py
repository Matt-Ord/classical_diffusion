import dataclasses

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    PeriodicSystem1D,
    breakdown_ballistic_trajectory,
    get_diffusion_time,
    get_effective_mass,
    get_under_barrier_probability,
    plot_exact_flat_ballistic_isf,
    solve_ballistic_ensemble,
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


system = PeriodicSystem1D(
    gamma=4e11,
    temperature=100,
    m=6e-27,
    delta_x=1.48e-10,
    barrier_energy=1.6e-21,
)


def _plot_effective_mass_offset_isf() -> None:
    fig, ax = get_fancy_figure()

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t_end=system.units.time_into(5e-12),
            n_steps=1000,
        ),
        n_samples=2000,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(
        result,
        filter_timescale=get_diffusion_time(
            system, characteristic_length=system.delta_x / 0.5
        ),
    )

    delta_k = (2 * np.pi / system.delta_x * 0.2,)

    _, ax, line_0, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    result_free = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t_end=system.units.time_into(50e-12),
            n_steps=1000,
        ),
        n_samples=5000,
    )

    prob_under_barrier = get_under_barrier_probability(
        system, barrier_energy=system.barrier_energy
    )

    _, ax, line_1 = plot_exact_flat_ballistic_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        offset=prob_under_barrier,
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")

    effective_mass = UnitSystem().mass_into(
        get_effective_mass(result_free),
        units=system.units,
    )

    _, ax, line_2 = plot_exact_flat_ballistic_isf(
        system=dataclasses.replace(system.as_canonical(), m=effective_mass),
        ax=ax,
        delta_k=delta_k,
        offset=prob_under_barrier,
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")

    effective_mass_exact = UnitSystem().mass_into(
        _get_free_effective_mass_exact_1d_periodic_directly(system),
        units=system.units,
    )

    _, ax, line_3 = plot_exact_flat_ballistic_isf(
        system=dataclasses.replace(system.as_canonical(), m=effective_mass_exact),
        ax=ax,
        delta_k=delta_k,
        offset=prob_under_barrier,
    )
    line_3.set_label("effective mass exact")
    line_3.set_linestyle(":")

    ax.set_xlim(0, 1.0e-12)
    ax.set_ylim(0.8, 1)
    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig(
        "examples/ballistic_langevin/offset_effective_mass/1d_periodic.offset.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_effective_mass_offset_isf()

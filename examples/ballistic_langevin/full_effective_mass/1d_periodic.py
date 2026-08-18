import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    PeriodicSystem1D,
    UnitSystem,
    breakdown_ballistic_trajectory,
    calculate_probability_under_barrier_1d,
    get_diffusion_time,
    get_full_effective_mass_exact_1d_periodic_directly,
    get_full_effective_mass_from_free,
    plot_exact_gaussian_isf,
    solve_ballistic_ensemble,
    solve_over_barrier_ballistic_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan

system = PeriodicSystem1D(
    gamma=4e11,
    temperature=105,
    m=6e-27,
    delta_x=1.48e-10,
    barrier_energy=4e-21,
)

normalized_system = system.with_normalized_units()


def _plot_effective_mass_isf() -> None:

    fig, ax = get_fancy_figure()
    delta_k = (7e9,)

    result_full = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(5e-12),
            n_steps=1000,
        ),
        n_samples=5000,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(
        result_full,
        minimum_timescale=get_diffusion_time(
            normalized_system, characteristic_length=normalized_system.delta_x / 0.5
        ),
    )

    _, ax, line_0, _ = plot_isf(
        result=elastic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    _, ax, line_1 = plot_exact_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=np.array([[system.m]]),
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")

    result_free = solve_over_barrier_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(10e-12),
            n_steps=1000,
        ),
        barrier_energy=normalized_system.barrier_energy,
        n_samples=2000,
    )

    elastic_result_free, _ = breakdown_ballistic_trajectory(
        result_free,
        minimum_timescale=get_diffusion_time(
            normalized_system, characteristic_length=normalized_system.delta_x / 0.5
        ),
    )

    prob_under_barrier = calculate_probability_under_barrier_1d(
        normalized_system, barrier_energy=normalized_system.barrier_energy
    )
    effective_mass = UnitSystem().mass_into(
        get_full_effective_mass_from_free(
            elastic_result_free,
            prob_under_barrier=prob_under_barrier,
        ),
        units=normalized_system.units,
    )

    _, ax, line_2 = plot_exact_gaussian_isf(
        system=system.with_si_units(),
        ax=ax,
        delta_k=delta_k,
        effective_mass=effective_mass,
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")

    effective_mass_exact = UnitSystem().mass_into(
        get_full_effective_mass_exact_1d_periodic_directly(system=normalized_system),
        units=normalized_system.units,
    )
    _, ax, line_3 = plot_exact_gaussian_isf(
        system=system.with_si_units(),
        ax=ax,
        delta_k=delta_k,
        effective_mass=np.array([[effective_mass_exact]]),
    )
    line_3.set_label("effective mass exact")
    line_3.set_linestyle(":")

    ax.set_xlim(0, 1e-12)
    ax.set_ylim(0.8, 1.0)
    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig(
        "examples/ballistic_langevin/full_effective_mass/1d_periodic.full_effective_mass.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_effective_mass_isf()

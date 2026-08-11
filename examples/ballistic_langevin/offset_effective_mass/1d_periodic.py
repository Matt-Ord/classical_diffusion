import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    breakdown_filtered_ballistic_trajectory_butterworth,
    calculate_probability_under_barrier_1d,
    get_effective_mass,
    get_free_effective_mass_exact_1d_periodic_directly,
    get_initial_conditions,
    get_over_barrier_initial_conditions,
    plot_exact_offset_gaussian_isf,
    solve_ballistic_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import (
    PeriodicSystem1D,
    UnitSystem,
    get_diffusion_time,
)

system = PeriodicSystem1D(
    gamma=4e11,
    temperature=100,
    m=6e-27,
    delta_x=1.48e-10,
    barrier_energy=1.6e-21,
    units=UnitSystem(),
)

normalized_system = system.with_normalized_units()

key = jrandom.PRNGKey(100)


def _plot_effective_mass_offset_isf() -> None:
    fig, ax = get_fancy_figure()

    initial_conditions = get_initial_conditions(normalized_system, n_samples=2000)
    result = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(5e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions,
        _key=key,
    )

    elastic_result, _ = breakdown_filtered_ballistic_trajectory_butterworth(
        result,
        minimum_timescale=get_diffusion_time(
            normalized_system, characteristic_length=normalized_system.delta_x / 0.5
        ),
    )

    delta_k = (2 * np.pi / system.delta_x * 0.2,)

    _, ax, line_0, _ = plot_isf(
        result=elastic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    initial_conditions = get_over_barrier_initial_conditions(
        system=normalized_system,
        barrier_energy=normalized_system.barrier_energy,
        n_samples=5000,
    )
    result_free = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(50e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions=initial_conditions,
        _key=key,
    )

    prob_under_barrier = calculate_probability_under_barrier_1d(
        normalized_system, barrier_energy=normalized_system.barrier_energy
    )

    _, ax, line_1 = plot_exact_offset_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=np.array([[system.m]]),
        offset=prob_under_barrier,
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")

    effective_mass = UnitSystem().mass_into(
        get_effective_mass(result_free),
        units=normalized_system.units,
    )

    _, ax, line_2 = plot_exact_offset_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=effective_mass,
        offset=prob_under_barrier,
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")

    effective_mass_exact = UnitSystem().mass_into(
        get_free_effective_mass_exact_1d_periodic_directly(normalized_system),
        units=normalized_system.units,
    )

    _, ax, line_3 = plot_exact_offset_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=np.array([[effective_mass_exact]]),
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

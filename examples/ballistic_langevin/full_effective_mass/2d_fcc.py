import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    breakdown_filtered_ballistic_trajectory_butterworth,
    get_full_effective_mass_from_free,
    get_initial_conditions,
    get_over_barrier_initial_conditions,
    get_under_barrier_probability_ballistic,
    plot_exact_gaussian_isf,
    solve_ballistic_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import (
    PeriodicSystemFCC,
    UnitSystem,
    get_diffusion_time,
)

system = PeriodicSystemFCC(
    gamma=4e11,
    temperature=105,
    m=6e-27,
    delta_x=1.48e-10,
    barrier_energy=1.6e-21,
    units=UnitSystem(),
)

normalized_system = system.with_normalized_units()

key = jrandom.PRNGKey(100)


def _plot_effective_mass_isf() -> None:

    fig, ax = get_fancy_figure()
    direction = np.array([1, 0])
    delta_k = tuple(7e9 * direction / np.linalg.norm(direction))

    initial_conditions = get_initial_conditions(normalized_system, n_samples=2000)
    result_full = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(5e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions,
        _key=key,
    )

    prob_under_barrier = get_under_barrier_probability_ballistic(
        normalized_system,
        result_full.x_points,
        result_full.p_points,
        normalized_system.barrier_energy,
    )

    elastic_result, _ = breakdown_filtered_ballistic_trajectory_butterworth(
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

    initial_conditions = get_over_barrier_initial_conditions(
        system=normalized_system,
        barrier_energy=normalized_system.barrier_energy,
        n_samples=500,
    )
    result_free = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(5e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions=initial_conditions,
        _key=key,
    )

    elastic_result_free, _ = breakdown_filtered_ballistic_trajectory_butterworth(
        result_free,
        minimum_timescale=get_diffusion_time(
            normalized_system, characteristic_length=normalized_system.delta_x / 0.5
        ),
    )

    effective_mass = UnitSystem().mass_into(
        get_full_effective_mass_from_free(
            elastic_result_free,
            prob_under_barrier=prob_under_barrier,
        ),
        units=normalized_system.units,
    )
    print(effective_mass)

    _, ax, line_2 = plot_exact_gaussian_isf(
        system=system.with_si_units(),
        ax=ax,
        delta_k=delta_k,
        effective_mass=effective_mass,
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")

    ax.set_xlim(0, 1e-12)
    ax.legend(handles=[line_0, line_1, line_2])
    fig.savefig(
        "examples/ballistic_langevin/full_effective_mass/2d_fcc.full_effective_mass.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_effective_mass_isf()

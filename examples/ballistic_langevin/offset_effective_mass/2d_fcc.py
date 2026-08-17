import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    PeriodicSystemFCC,
    breakdown_ballistic_trajectory,
    get_diffusion_time,
    get_effective_mass,
    get_under_barrier_probability_ballistic,
    plot_exact_offset_gaussian_isf,
    solve_ballistic_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import UnitSystem

system = PeriodicSystemFCC(
    gamma=4e11,
    temperature=105,
    m=6e-27,
    delta_x=5e-10,
    barrier_energy=5e-21,
)

normalized_system = system.with_normalized_units()

key = jrandom.PRNGKey(100)


def _plot_effective_mass_offset_isf() -> None:

    fig, ax = get_fancy_figure()
    direction = np.array([0, 1])
    delta_k = tuple(
        2 * np.pi / system.delta_x * 0.2 * direction / np.linalg.norm(direction)
    )

    result_full = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(5e-12),
            n_steps=1000,
        ),
        n_samples=2000,
        _key=key,
    )

    under_barrier_prob = get_under_barrier_probability_ballistic(
        normalized_system,
        result_full.x_points,
        result_full.p_points,
        normalized_system.barrier_energy,
    )

    print(under_barrier_prob)

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

    _, ax, line_1 = plot_exact_offset_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=system.m * np.eye(system.n_dim),
        offset=under_barrier_prob,
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")

    result_free = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(50e-12),
            n_steps=1000,
        ),
        n_samples=1000,
        _key=key,
    )

    elastic_result_free, _ = breakdown_ballistic_trajectory(
        result_free,
        minimum_timescale=get_diffusion_time(
            normalized_system, characteristic_length=normalized_system.delta_x / 0.5
        ),
    )

    effective_mass = UnitSystem().mass_into(
        get_effective_mass(elastic_result_free),
        units=normalized_system.units,
    )

    print(effective_mass)

    _, ax, line_2 = plot_exact_offset_gaussian_isf(
        system=system.with_si_units(),
        ax=ax,
        delta_k=delta_k,
        effective_mass=effective_mass,
        offset=under_barrier_prob,
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")

    ax.set_xlim(0, 1.5e-12)
    ax.legend(handles=[line_0, line_1, line_2])
    fig.savefig(
        "examples/ballistic_langevin/offset_effective_mass/2d_fcc.full_effective_mass.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    _plot_effective_mass_offset_isf()

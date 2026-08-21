import dataclasses

import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
)
from classical_diffusion.langevin import (
    PeriodicSystemFCC,
    breakdown_ballistic_trajectory,
    get_diffusion_time,
    get_effective_mass,
    get_under_barrier_occupation,
    plot_exact_flat_ballistic_isf,
    solve_ballistic_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import UnitSystem

system = PeriodicSystemFCC(
    gamma=4e11,
    temperature=105,
    m=6e-27,
    delta_x=1.48e-10,
    barrier_energy=1.6e-21,
)


def _plot_effective_mass_isf() -> None:

    fig, ax = get_fancy_figure()
    direction = np.array([1, 0])
    delta_k = tuple(7e9 * direction / np.linalg.norm(direction))

    result_full = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t_end=system.units.time_into(5e-12),
            n_steps=1000,
        ),
        n_samples=2000,
    )

    prob_under_barrier = get_under_barrier_occupation(
        system,
        result_full.x_points,
        result_full.p_points,
        system.barrier_energy,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(
        result_full,
        filter_timescale=get_diffusion_time(
            system, characteristic_length=system.delta_x / 0.5
        ),
    )

    _, ax, line_0, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    _, ax, line_1 = plot_exact_flat_ballistic_isf(system=system, ax=ax, delta_k=delta_k)
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")

    result_free = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t_end=system.units.time_into(5e-12),
            n_steps=1000,
        ),
        n_samples=500,
        minimum_energy=system.barrier_energy,
    )

    effective_mass = UnitSystem().mass_into(
        get_effective_mass(
            result_free,
            under_barrier_probability=prob_under_barrier,
            filter_timescale=get_diffusion_time(
                system, characteristic_length=system.delta_x / 0.5
            ),
        ),
        units=system.units,
    )
    print(effective_mass)

    _, ax, line_2 = plot_exact_flat_ballistic_isf(
        system=dataclasses.replace(system, m=effective_mass),
        ax=ax,
        delta_k=delta_k,
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

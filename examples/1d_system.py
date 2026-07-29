import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
    plot_isf_with_delta_k,
    plot_x_evolution,
)
from classical_diffusion.langevin import (
    PeriodicSystem1D,
    breakdown_ballistic_trajectory,
    get_effective_mass,
    get_full_effective_mass_from_free,
    plot_isf,
    plot_isf_with_delta_k,
    solve_ballistic_ensemble,
    solve_ensemble,
    solve_free_ballistic_ensemble,
)
from classical_diffusion.langevin._langevin import get_over_barrier_initial_conditions
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import (
    PeriodicSystem1D,
    UnitSystem,
    calculate_probability_under_barrier,
    get_diffusion_time,
    get_free_effective_mass_exact_1d_periodic,
    get_full_effective_mass_exact_1d_periodic,
    plot_exact_gaussian_isf,
    plot_exact_offset_gaussian_isf,
    plot_periodic_potential_1d,
    solve_ballistic_ensemble,
    solve_ensemble,
)


def _plot_periodic_system() -> None:
    system = PeriodicSystem1D(
        gamma=4e10,
        temperature=106,
        m=6.64e-27,
        delta_x=1.48e-10,
        barrier_energy=4.22e-21,
        units=UnitSystem(),
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig("examples/1d_system.potential.pdf")


def _plot_1d_periodic_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=4e11,
        temperature=110,
        m=8e-27,
        delta_x=3e-10,
        barrier_energy=1.6e-21,
        units=UnitSystem(),
    )

    normalized_system = system.with_normalized_units()

    result = solve_ensemble(
        normalized_system,
        TimeSpan(
            t_end=10e-12
            / get_diffusion_time(system=system, characteristic_length=system.delta_x),
            n_steps=1000,
        ),
        (np.full((200, 1), 0.0), np.full((200, 1), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    delta_k = (7.1e9,)
    _, ax, line_0, _fill_0 = plot_isf(
        result=result.with_si_units(),
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("full simulation")

    result = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=2e-12
            / get_diffusion_time(system=system, characteristic_length=system.delta_x),
            n_steps=1000,
        ),
        n_samples=2000,
        _key=key,
    )

    _, ax, line_1, _ = plot_isf(
        result=result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_1.set_label("ballistic simulation")

    elastic_result, inelastic_result = breakdown_ballistic_trajectory(result)

    _, ax, line_2, _ = plot_isf(
        result=elastic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_2.set_label("elastic")
    line_2.set_linestyle(":")

    _, ax, line_3, _ = plot_isf(
        result=inelastic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_3.set_label("inelastic")
    line_3.set_linestyle(":")

    ax.set_xlim(
        0,
        2e-12,
    )
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig("./examples/1d_system.isf.pdf", dpi=300, bbox_inches="tight")


def _plot_1d_inelastic_trends() -> None:

    system = PeriodicSystem1D(
        gamma=4e11,
        temperature=110,
        m=8e-27,
        delta_x=3e-10,
        barrier_energy=1.6e-21,
        units=UnitSystem(),
    )

    normalized_system = system.with_normalized_units()

    key = jrandom.PRNGKey(100)

    result = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=2e-12
            / get_diffusion_time(system=system, characteristic_length=system.delta_x),
            n_steps=1000,
        ),
        n_samples=2000,
        _key=key,
    )

    _elastic_result, inelastic_result = breakdown_ballistic_trajectory(result)
    delta_k_values = np.linspace(0.1, 2.0, 9) * (7.1e9)

    fig, ax = get_fancy_figure()
    _, ax = plot_isf_with_delta_k(
        result=inelastic_result.with_si_units(),
        ax=ax,
        delta_k_values=delta_k_values,
        pairwise=False,
    )
    ax.set_xlim(
        0,
        1e-12,
    )
    fig.savefig(
        "./examples/1d_system.inelastic_trends.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _plot_effective_mass_isf() -> None:  # ruff:ignore[too-many-locals]
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=4e11,
        temperature=110,
        m=8e-27,
        delta_x=3e-10,
        barrier_energy=1.6e-21,
        units=UnitSystem(),
    )

    normalized_system = system.with_normalized_units()

    fig, ax = get_fancy_figure()

    delta_k = (7.1e9,)

    result_full = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(5e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        n_samples=2000,
        _key=key,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(result_full)

    _, ax, line_0, _ = plot_isf(
        result=elastic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    _, ax, line_1 = plot_exact_gaussian_isf(
        system=system, ax=ax, delta_k=delta_k, effective_mass=system.m
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")

    initial_conditions = get_over_barrier_initial_conditions(
        system=normalized_system,
        barrier_energy=normalized_system.barrier_energy,
        n_samples=500,
    )
    result_free = solve_free_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(5e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions=initial_conditions,
        _key=key,
    )

    prob_under_barrier = calculate_probability_under_barrier(
        normalized_system, barrier_energy=normalized_system.barrier_energy
    )
    effective_mass = UnitSystem().mass_into(
        get_full_effective_mass_from_free(
            result_free, prob_under_barrier=prob_under_barrier
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
        get_full_effective_mass_exact_1d_periodic(
            system=normalized_system, initial_conditions=initial_conditions
        ),
        units=normalized_system.units,
    )
    _, ax, line_3 = plot_exact_gaussian_isf(
        system=system.with_si_units(),
        ax=ax,
        delta_k=delta_k,
        effective_mass=effective_mass_exact,
    )
    line_3.set_label("effective mass exact")
    line_3.set_linestyle(":")

    ax.set_xlim(0, 1e-12)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig("./examples/1d_system.effective_mass.pdf", dpi=300, bbox_inches="tight")


def _plot_effective_mass_offset_isf() -> None:  # ruff:ignore[too-many-locals]
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=4e11,
        temperature=110,
        m=8e-27,
        delta_x=3e-10,
        barrier_energy=1.6e-21,
        units=UnitSystem(),
    )

    normalized_system = system.with_normalized_units()

    fig, ax = get_fancy_figure()

    result = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=2e-12
            / get_diffusion_time(system=system, characteristic_length=system.delta_x),
            n_steps=1000,
        ),
        n_samples=2000,
        _key=key,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(result)

    delta_k = (7.1e9,)

    _, ax, line_0, _ = plot_isf(
        result=elastic_result.with_si_units(), ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    initial_conditions = get_over_barrier_initial_conditions(
        system=normalized_system,
        barrier_energy=normalized_system.barrier_energy,
        n_samples=500,
    )
    result_free = solve_free_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t0=0,
            t1=normalized_system.units.time_into(5e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions=initial_conditions,
        _key=key,
    )

    prob_under_barrier = calculate_probability_under_barrier(
        normalized_system, barrier_energy=normalized_system.barrier_energy
    )

    _, ax, line_1 = plot_exact_offset_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=system.m,
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
        get_free_effective_mass_exact_1d_periodic(
            normalized_system, initial_conditions
        ),
        units=normalized_system.units,
    )
    _, ax, line_3 = plot_exact_offset_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=effective_mass_exact,
        offset=prob_under_barrier,
    )
    line_3.set_label("effective mass exact")
    line_3.set_linestyle(":")

    ax.set_xlim(0, 2e-12)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig(
        "./examples/1d_system.effective_mass_offset.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _plot_1d_trajectory() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5, m=1.0, delta_x=5, barrier_energy=0.5
    )

    result = solve_ensemble(
        system,
        TimeSpan(
            t_end=40 / system.gamma,
            n_steps=4000,
        ),
        (np.full((20, 1), 0.0), np.full((20, 1), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    _, _, _ = plot_x_evolution(result=result, ax=ax)

    fig.savefig("./examples/1d_system.trajectory.pdf")


if __name__ == "__main__":
    _plot_1d_trajectory()
    _plot_periodic_system()
    _plot_1d_periodic_isf()
    _plot_1d_inelastic_trends()
    _plot_effective_mass_isf()
    _plot_effective_mass_offset_isf()

import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import plot_isf
from classical_diffusion.langevin import (
    breakdown_ballistic_trajectory,
    get_effective_mass,
    get_initial_conditions,
    plot_2d_trajectory,
    plot_exact_gaussian_isf,
    plot_periodic_potential_fcc,
    solve_ballistic_ensemble,
    solve_ensemble,
    solve_single,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import (
    PeriodicSystemFCC,
    UnitSystem,
)


def _plot_periodic_system() -> None:
    system = PeriodicSystemFCC(
        gamma=4e11,
        temperature=110,
        m=3e-27,
        delta_x=3e-10,
        barrier_energy=5e-21,
        units=UnitSystem(),
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_fcc(system, ax=ax)
    fig.savefig("examples/2d_system.potential.pdf")


def _plot_2d_periodic_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystemFCC(
        gamma=4e11,
        temperature=110,
        m=3e-27,
        delta_x=3e-10,
        barrier_energy=5e-21,
        units=UnitSystem(),
    )

    normalized_system = system.with_normalized_units()

    result = solve_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(10e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        (np.full((50, 2), 0.0), np.full((50, 2), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    delta_k = (7.1e9, 7.1e9)
    _, ax, line_0, _fill_0 = plot_isf(
        result=result.with_si_units(),
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("full simulation")

    initial_conditions = get_initial_conditions(normalized_system, n_samples=1000)

    result = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(2e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions=initial_conditions,
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
    fig.savefig("./examples/2d_system.isf.pdf", dpi=300, bbox_inches="tight")


def _plot_effective_mass_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystemFCC(
        gamma=4e11,
        temperature=110,
        m=8e-27,
        delta_x=3e-10,
        barrier_energy=1.6e-21,
        units=UnitSystem(),
    )

    normalized_system = system.with_normalized_units()

    fig, ax = get_fancy_figure()

    delta_k = (7.1e9, 7.1e9)

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

    _, ax, line_1 = plot_exact_gaussian_isf(
        system=system, ax=ax, delta_k=delta_k, effective_mass=system.m
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")

    effective_mass = UnitSystem().mass_into(
        get_effective_mass(result_full),
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

    ax.set_xlim(0, 1e-12)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1, line_2])
    fig.savefig("./examples/2d_system.effective_mass.pdf", dpi=300, bbox_inches="tight")


def _plot_2d_trajectory() -> None:
    # TODO: add elastic and inelastic trajectories to the plot
    key = jrandom.PRNGKey(100)
    system = PeriodicSystemFCC(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=1.5,
        units=UnitSystem(Boltzmann=1.0, angstrom=1.0, atomic_mass=1.0),
    )

    result = solve_single(
        system,
        TimeSpan(
            t_end=100 / system.gamma,
            n_steps=10000,
        ),
        (np.full((2,), 0.0), np.full((2,), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()
    _, ax, _line = plot_2d_trajectory(result=result, ax=ax)

    fig.savefig("examples/2d.periodic.trajectory.pdf")


if __name__ == "__main__":
    _plot_periodic_system()
    _plot_2d_periodic_isf()
    _plot_effective_mass_isf()

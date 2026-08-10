from dataclasses import dataclass

import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import (
    plot_isf,
    plot_isf_with_delta_k,
    plot_x_evolution,
)
from classical_diffusion.langevin import (
    breakdown_ballistic_trajectory,
    calculate_probability_under_barrier,
    get_effective_mass,
    get_free_effective_mass_exact_1d_periodic_directly,
    get_full_effective_mass_exact_1d_periodic_directly,
    get_full_effective_mass_from_free,
    get_initial_conditions,
    get_over_barrier_initial_conditions,
    plot_exact_gaussian_isf,
    plot_exact_offset_gaussian_isf,
    plot_periodic_potential_1d,
    solve_ballistic_ensemble,
    solve_ensemble,
)
from classical_diffusion.langevin.analysis import (
    breakdown_ballistic_trajectory,
    get_effective_mass,
    plot_isf_with_delta_k,
    split_escaped_and_trapped,
    under_barrier_probability_ballistic,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import (
    PeriodicSystem1D,
    plot_periodic_potential_1d,
)
from classical_diffusion.system._system import UnitSystem
from classical_diffusion.system.analysis import (
    plot_exact_gaussian_isf,
    plot_exact_offset_gaussian_isf,
)


@dataclass(frozen=True, kw_only=True)
class CamColor:
    """A class to hold CAM color palettes."""

    light: str
    warm: str
    base: str
    dark: str


CAM_BLUE = CamColor(
    light="#D1F9F1",
    warm="#00BDB6",
    base="#8EE8D8",
    dark="#133844",
)
CAM_CHERRY = CamColor(
    light="#F2CAD8",
    warm="#E18AAC",
    base="#CD3572",
    dark="#911449",
)


def _plot_periodic_system() -> None:
    system = PeriodicSystem1D(
        gamma=4e11,
        temperature=103,
        m=8e-27,
        delta_x=3e-10,
        barrier_energy=1.6e-21,
        units=UnitSystem(),
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig("examples/1d_system.potential.pdf")


def _plot_1d_periodic_isf() -> None:  # ruff:ignore[too-many-locals]
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=4e11,
        temperature=103,
        m=8e-27,
        delta_x=3e-10,
        barrier_energy=1.6e-21,
        units=UnitSystem(),
    )

    normalized_system = system.with_normalized_units()

    result = solve_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(10e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        (np.full((200, 1), 0.0), np.full((200, 1), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    delta_k = (5e9,)
    _, ax, line_0, _fill_0 = plot_isf(
        result=result.with_si_units(),
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("full simulation")
    line_0.set_color(CAM_BLUE.dark)
    fill_0.set_color(CAM_BLUE.dark)

    initial_conditions = get_initial_conditions(normalized_system, n_samples=2000)

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
    line_1.set_color(CAM_BLUE.warm)
    fill_1.set_color(CAM_BLUE.warm)

    elastic_result, inelastic_result = breakdown_ballistic_trajectory(
        result, stride_time=1 / system.gamma
    )

    _, ax, line_2, fill_2 = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_2.set_label("elastic")
    line_2.set_linestyle(":")
    line_2.set_color(CAM_CHERRY.light)
    fill_2.set_color(CAM_CHERRY.light)

    _, ax, line_3, fill_3 = plot_isf(
        result=inelastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_3.set_label("inelastic")
    line_3.set_linestyle(":")
    line_3.set_color(CAM_CHERRY.dark)
    fill_3.set_color(CAM_CHERRY.dark)

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

    delta_k_values = np.linspace(0.1, 2.0, 9) * (0.5 * 2 * np.pi / system.delta_x)

    fig, ax = get_fancy_figure()
    _, ax = plot_isf_with_delta_k(
        result=inelastic_result, ax=ax, delta_k_values=delta_k_values, pairwise=False
    )
    ax.set_xlim(0, 4 / system.gamma)
    fig.savefig(
        "./examples/1d_system.isf.ballsitic.inelastic.delta_k.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _plot_effective_mass_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=0.5,
        units=UnitSystem(),
    )

    fig, ax = get_fancy_figure()

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t0=0,
            t1=100 / system.gamma,
            dt=0.01 / system.gamma,
            dt_step=0.01 / system.gamma,
        ),
        n_trajectories=1,
        _key=key,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(
        result, stride_time=1 / system.gamma
    )

    _, ax, line_0, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    _, ax, line_1 = plot_exact_gaussian_isf(
        system=system, ax=ax, delta_k=delta_k, effective_mass=system.m
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")
    line_1.set_color(CAM_BLUE.dark)

    effective_mass = get_effective_mass(result)

    _, ax, line_2 = plot_exact_gaussian_isf(
        system=system, ax=ax, delta_k=delta_k, effective_mass=effective_mass
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")
    line_2.set_color(CAM_CHERRY.dark)

    ax.set_xlim(0, 0.3 / system.gamma)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1, line_2])
    fig.savefig(
        "./examples/1d_system.isf.effective.mass.pdf", dpi=300, bbox_inches="tight"
    )


def _plot_effective_mass_offset_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=0.5,
        units=UnitSystem(),
    )

    fig, ax = get_fancy_figure()

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t0=0,
            t1=100 / system.gamma,
            dt=0.01 / system.gamma,
            dt_step=0.01 / system.gamma,
        ),
        n_trajectories=1,
        _key=key,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(
        result, stride_time=1 / system.gamma
    )

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    _, ax, line_0, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    _, ax, line_1 = plot_exact_offset_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=system.m,
        offset=np.average(under_barrier_probability_ballistic(result)),
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")
    line_1.set_color(CAM_BLUE.dark)

    free_result, _ = split_escaped_and_trapped(result)

    effective_mass = get_effective_mass(free_result)

    _, ax, line_2 = plot_exact_offset_gaussian_isf(
        system=system,
        ax=ax,
        delta_k=delta_k,
        effective_mass=effective_mass,
        offset=np.average(under_barrier_probability_ballistic(result)),
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")
    line_2.set_color(CAM_CHERRY.dark)

    ax.set_xlim(0, 1.0 / system.gamma)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1, line_2])
    fig.savefig(
        "./examples/1d_system.isf.effective.mass.offset.pdf",
        dpi=300,
        bbox_inches="tight",
    )


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
    initial_conditions = get_initial_conditions(normalized_system, n_samples=2000)

    key = jrandom.PRNGKey(100)

    result = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(2e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions,
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

    initial_conditions = get_over_barrier_initial_conditions(
        system=normalized_system,
        barrier_energy=normalized_system.barrier_energy,
        n_samples=2000,
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
        get_full_effective_mass_exact_1d_periodic_directly(system=normalized_system),
        units=normalized_system.units,
    )
    print(effective_mass)
    print(effective_mass_exact)
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
    result_free = solve_ballistic_ensemble(
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
        get_free_effective_mass_exact_1d_periodic_directly(normalized_system),
        units=normalized_system.units,
    )
    print(effective_mass)
    print(effective_mass_exact)
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
        gamma=4e11,
        temperature=103,
        m=8e-27,
        delta_x=3e-10,
        barrier_energy=1.6e-21,
        units=UnitSystem(),
    )

    normalized_system = system.with_normalized_units()
    result = solve_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(5e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        (np.full((20, 1), 0.0), np.full((20, 1), 0.0)),
        _key=key,
    )

    fig, ax = get_fancy_figure()

    _, _, _ = plot_x_evolution(result=result, ax=ax, n_trajectories=20)

    fig.savefig("./examples/1d_system.trajectory.pdf")


if __name__ == "__main__":
    _plot_periodic_system()
    _plot_1d_trajectory()
    _plot_1d_periodic_isf()
    _plot_effective_mass_isf()
    _plot_effective_mass_offset_isf()

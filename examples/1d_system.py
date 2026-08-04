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
    get_under_barrier_probability_ballistic,
    plot_exact_gaussian_isf,
    plot_exact_offset_gaussian_isf,
    plot_periodic_potential_1d,
    solve_ballistic_ensemble,
    solve_ensemble,
    solve_overdamped_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.simulation import TimeSpan


def _plot_periodic_system() -> None:
    system = PeriodicSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=0.5,
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_1d(system, ax=ax)
    fig.savefig("examples/1d_system.potential.pdf")


def _plot_1d_periodic_isf() -> None:
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

    delta_k = (0.7 * 2 * np.pi / system.delta_x,)
    _, ax, line_0, _fill_0 = plot_isf(
        result=result,
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("full simulation")

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t_end=10 / system.gamma,
            n_steps=1000,
        ),
        n_samples=10000,
        _key=key,
    )

    _, ax, line_1, _ = plot_isf(result=result, ax=ax, delta_k=delta_k, pairwise=False)
    line_1.set_label("ballistic simulation")

    elastic_result, inelastic_result = breakdown_ballistic_trajectory(result)

    _, ax, line_2, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_2.set_label("elastic")
    line_2.set_linestyle(":")

    _, ax, line_3, _ = plot_isf(
        result=inelastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_3.set_label("inelastic")
    line_3.set_linestyle(":")

    ax.set_xlim(0, 4 / system.gamma)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1, line_2, line_3])
    fig.savefig("./examples/1d_system.isf.pdf", dpi=300, bbox_inches="tight")


def _plot_1d_periodic_isf_overdamped() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5, m=1.0, delta_x=5, barrier_energy=0.5
    )

    fig, ax = get_fancy_figure()
    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    result = solve_overdamped_ensemble(
        system,
        TimeSpan(t_end=40 / system.gamma, n_steps=4000),
        (np.full((80, 1), 0.0), np.full((80, 1), 0.0)),
        _key=key,
    )
    _, _, line, _ = plot_isf(
        result=result, ax=ax, delta_k=delta_k, pairwise=True, measure="real"
    )
    line.set_label("overdamped")

    times = np.linspace(0, 1 / system.gamma, 4000)
    expected = np.exp(-(system.kbt / system.gamma) * (delta_k[0] ** 2) * times)
    (line_1,) = ax.plot(times, expected, label="flat surface", linestyle=":")

    ax.set_xlim(0, 0.4 / system.gamma)
    ax.set_ylim(0, 1)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend(handles=[line, line_1])
    fig.savefig("./examples/1d_system.isf.overdamped.pdf", dpi=300, bbox_inches="tight")


def _plot_1d_inelastic_trends() -> None:

    system = PeriodicSystem1D(
        gamma=0.1, temperature=0.5, m=1.0, delta_x=5, barrier_energy=0.5
    )

    key = jrandom.PRNGKey(100)

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t_end=10 / system.gamma,
            n_steps=1000,
        ),
        n_samples=10000,
        _key=key,
    )

    _elastic_result, inelastic_result = breakdown_ballistic_trajectory(result)
    delta_k_values = np.linspace(0.1, 2.0, 9) * (0.5 * 2 * np.pi / system.delta_x)

    fig, ax = get_fancy_figure()
    _, ax = plot_isf_with_delta_k(
        result=inelastic_result, ax=ax, delta_k_values=delta_k_values, pairwise=False
    )
    ax.set_xlim(0, 4 / system.gamma)
    fig.savefig(
        "./examples/1d_system.inelastic_trends.pdf",
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
    )

    fig, ax = get_fancy_figure()

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t_end=100 / system.gamma,
            n_steps=10000,
        ),
        n_samples=2000,
        _key=key,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(result)

    _, ax, line_0, _ = plot_isf(
        result=elastic_result, ax=ax, delta_k=delta_k, pairwise=False
    )
    line_0.set_label("elastic")

    _, ax, line_1 = plot_exact_gaussian_isf(
        system=system, ax=ax, delta_k=delta_k, effective_mass=system.m
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")

    effective_mass = get_effective_mass(result)

    _, ax, line_2 = plot_exact_gaussian_isf(
        system=system, ax=ax, delta_k=delta_k, effective_mass=effective_mass
    )
    line_2.set_label("effective mass")
    line_2.set_linestyle(":")

    ax.set_xlim(0, 0.3 / system.gamma)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1, line_2])
    fig.savefig("./examples/1d_system.effective_mass.pdf", dpi=300, bbox_inches="tight")


def _plot_effective_mass_offset_isf() -> None:
    key = jrandom.PRNGKey(100)

    system = PeriodicSystem1D(
        gamma=0.1,
        temperature=0.5,
        m=1.0,
        delta_x=5,
        barrier_energy=0.5,
    )

    _fig, ax = get_fancy_figure()

    result = solve_ballistic_ensemble(
        system,
        TimeSpan(
            t_end=100 / system.gamma,
            n_steps=10000,
        ),
        n_samples=2000,
        _key=key,
    )

    elastic_result, _ = breakdown_ballistic_trajectory(result)

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
        offset=np.average(get_under_barrier_probability_ballistic(result)),
    )
    line_1.set_label("actual mass")
    line_1.set_linestyle(":")


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
    _plot_1d_periodic_isf_overdamped()
    _plot_1d_inelastic_trends()
    _plot_effective_mass_isf()
    _plot_effective_mass_offset_isf()

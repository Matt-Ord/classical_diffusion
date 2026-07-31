import jax.random as jrandom
import numpy as np

from classical_diffusion.analysis import plot_isf
from classical_diffusion.langevin import (
    get_initial_conditions,
    plot_2d_trajectory,
    plot_periodic_potential_fcc,
    solve_ballistic_ensemble,
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
        gamma=0.1,
        temperature=1.0,
        m=1.0,
        delta_x=5.0,
        barrier_energy=1.5,
        units=UnitSystem(),
    )
    fig, ax = get_fancy_figure()
    _, _, _ = plot_periodic_potential_fcc(system, ax=ax)
    fig.savefig("examples/2d_system.potential.pdf")


def _plot_2d_periodic_isf() -> None:
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

    delta_k = (0.5 * 2 * np.pi / system.delta_x,)
    _, ax, line_0, _ = plot_isf(
        result=result,
        ax=ax,
        delta_k=delta_k,
    )
    line_0.set_label("simulation")

    normalized_system = system.with_normalized_units()
    initial_conditions = get_initial_conditions(normalized_system, n_samples=100)
    result = solve_ballistic_ensemble(
        normalized_system,
        TimeSpan(
            t_end=normalized_system.units.time_into(10e-12, units=UnitSystem()),
            n_steps=1000,
        ),
        initial_conditions=initial_conditions,
        _key=key,
    )

    _, ax, line_1, _ = plot_isf(result=result, ax=ax, delta_k=delta_k, pairwise=False)
    line_1.set_label("ballistic simulation")

    ax.set_xlim(0, 4 / system.gamma)
    ax.set_ylim(0, 1)
    ax.legend(handles=[line_0, line_1])
    fig.savefig("./examples/2d_system.isf.pdf", dpi=300, bbox_inches="tight")


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
    _plot_2d_trajectory()

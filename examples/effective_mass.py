import jax.random as jrandom
import numpy as np
from tqdm import tqdm

from classical_diffusion.langevin import (
    TimeSpan,
    get_effective_mass,
    plot_effective_mass_periodic_1D,
    solve_ballistic_ensemble,
    split_escaped_and_trapped,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import (
    PeriodicSystem1D,
)


def _full_effective_mass_plot() -> None:

    barrier_energy = np.linspace(0.1, 0.75, 10)
    inertial_mass = np.linspace(2.5, 10, 10)

    barrier_energy, inertial_mass = np.meshgrid(barrier_energy, inertial_mass)

    keys = jrandom.split(jrandom.PRNGKey(100), barrier_energy.size)
    effective_mass = np.zeros(barrier_energy.shape)

    for idx, (i, j) in enumerate(
        tqdm(np.ndindex(barrier_energy.shape), total=barrier_energy.size)
    ):
        system = PeriodicSystem1D(
            gamma=0.1,
            temperature=0.5,
            m=inertial_mass[i, j],
            delta_x=5.0,
            barrier_energy=barrier_energy[i, j],
        )

        result = solve_ballistic_ensemble(
            system,
            TimeSpan(
                t0=0,
                t1=100 / system.gamma,
                dt=0.01 / system.gamma,
            ),
            n_samples=10,
            _key=keys[idx],
        )

        effective_mass[i, j] = get_effective_mass(result) / inertial_mass[i, j]

    print(effective_mass)
    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_effective_mass_periodic_1D(
        effective_mass=effective_mass,
        barrier_energy=barrier_energy,
        inertial_mass=inertial_mass,
        ax=ax,
    )
    mesh.set_rasterized(True)
    fig.savefig("examples/effective_mass.full.pdf", dpi=1000)


def _free_effective_mass_plot() -> None:

    barrier_energy = np.linspace(0.1, 0.75, 10)
    inertial_mass = np.linspace(2.5, 10, 10)

    barrier_energy, inertial_mass = np.meshgrid(barrier_energy, inertial_mass)

    keys = jrandom.split(jrandom.PRNGKey(100), barrier_energy.size)
    effective_mass = np.zeros(barrier_energy.shape)

    for idx, (i, j) in enumerate(
        tqdm(np.ndindex(barrier_energy.shape), total=barrier_energy.size)
    ):
        system = PeriodicSystem1D(
            gamma=0.1,
            temperature=0.5,
            m=inertial_mass[i, j],
            delta_x=5.0,
            barrier_energy=barrier_energy[i, j],
        )

        result = solve_ballistic_ensemble(
            system,
            TimeSpan(
                t0=0,
                t1=100 / system.gamma,
                dt=0.01 / system.gamma,
            ),
            n_samples=15,
            _key=keys[idx],
        )
        free_result, _ = split_escaped_and_trapped(result)
        effective_mass[i, j] = get_effective_mass(free_result) / inertial_mass[i, j]

    print(effective_mass)
    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_effective_mass_periodic_1D(
        effective_mass=effective_mass,
        barrier_energy=barrier_energy,
        inertial_mass=inertial_mass,
        ax=ax,
    )
    mesh.set_rasterized(True)
    fig.savefig("examples/effective_mass.free.pdf", dpi=1000)


if __name__ == "__main__":
    _full_effective_mass_plot()
    _free_effective_mass_plot()

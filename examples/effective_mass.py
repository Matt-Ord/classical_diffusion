from pathlib import Path

import jax
import jax.random as jrandom
import numpy as np
from scipy.constants import Boltzmann, hbar
from tqdm import tqdm

from classical_diffusion.langevin import (
    TimeSpan,
    get_effective_mass,
    get_full_effective_mass_from_free,
    plot_effective_mass_ratio_periodic_1D,
    solve_free_ballistic_ensemble,
)
from classical_diffusion.langevin._langevin import get_over_barrier_initial_conditions
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import (
    PeriodicSystem1D,
    UnitSystem,
    calculate_probability_under_barrier,
    get_free_effective_mass_exact_1d_periodic,
    get_full_effective_mass_exact_1d_periodic,
)
from classical_diffusion.util import cached, disabled_timing, hash_array

key = jrandom.PRNGKey(100)


def _solve_effective_mass_path(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    temperature: float,
    delta_x: float,
    end_time: float,
    n_samples: int,
    _key: jax.Array,
    barrier_energy: np.ndarray,
    inertial_mass: np.ndarray,
) -> Path:
    filename = f"{temperature}_{delta_x}_{end_time}_{n_samples}_{key}_{hash_array((barrier_energy,))}_{hash_array((inertial_mass,))}.npz"
    return Path("examples/data") / filename


@cached(_solve_effective_mass_path)
def _effective_mass_simulation(  # ruff:ignore[too-many-arguments, too-many-locals, too-many-positional-arguments]
    temperature: float,
    delta_x: float,
    end_time: float,
    n_samples: int,
    _key: jax.Array,
    barrier_energy: np.ndarray,
    inertial_mass: np.ndarray,
) -> tuple:
    keys = jrandom.split(jrandom.PRNGKey(100), barrier_energy.size)
    barrier_energy, inertial_mass = np.meshgrid(barrier_energy, inertial_mass)
    prob_under_barrier = np.zeros_like(barrier_energy)
    full_effective_mass_ratio = np.zeros_like(barrier_energy)
    free_effective_mass_ratio = np.zeros_like(barrier_energy)
    full_effective_mass_exact_ratio = np.zeros_like(barrier_energy)
    free_effective_mass_exact_ratio = np.zeros_like(barrier_energy)

    kbt = temperature * Boltzmann
    m0 = 4 * (np.pi) ** 2 * (hbar) ** 2 / (kbt * (delta_x) ** 2)

    m_grid = inertial_mass * m0
    barrier_energy_grid = barrier_energy * kbt

    with disabled_timing():
        for idx, (i, j) in enumerate(
            tqdm(np.ndindex(barrier_energy.shape), total=barrier_energy.size)
        ):
            system = PeriodicSystem1D(
                gamma=0,
                temperature=temperature,
                m=m_grid[i, j],
                delta_x=delta_x,
                barrier_energy=barrier_energy_grid[i, j],
                units=UnitSystem(),
            )
            normalized_system = system.with_normalized_units()
            initial_conditions = get_over_barrier_initial_conditions(
                system=normalized_system,
                barrier_energy=normalized_system.barrier_energy,
                n_samples=n_samples,
            )
            result = solve_free_ballistic_ensemble(
                normalized_system,
                TimeSpan(
                    t0=0,
                    t1=normalized_system.units.time_into(end_time, units=UnitSystem()),
                    n_steps=1000,
                ),
                initial_conditions=initial_conditions,
                _key=keys[idx],
            )

            prob_under_barrier_val = calculate_probability_under_barrier(
                system=normalized_system,
                barrier_energy=normalized_system.barrier_energy,
            )
            prob_under_barrier[i, j] = prob_under_barrier_val

            full_effective_mass_ratio[i, j] = get_full_effective_mass_from_free(
                result, prob_under_barrier=prob_under_barrier_val
            )

            free_effective_mass_ratio[i, j] = get_effective_mass(result)

            full_effective_mass_exact_ratio[i, j] = (
                get_full_effective_mass_exact_1d_periodic(
                    normalized_system, initial_conditions
                )
            )

            free_effective_mass_exact_ratio[i, j] = (
                get_free_effective_mass_exact_1d_periodic(
                    normalized_system, initial_conditions
                )
            )

        return (
            prob_under_barrier,
            full_effective_mass_ratio,
            free_effective_mass_ratio,
            full_effective_mass_exact_ratio,
            free_effective_mass_exact_ratio,
        )


if __name__ == "__main__":
    barrier_energy = np.linspace(0, 4, 50)
    inertial_mass = np.linspace(0.1, 20, 50)
    (
        prob_under_barrier,
        full_effective_mass_ratio,
        free_effective_mass_ratio,
        full_effective_mass_exact_ratio,
        free_effective_mass_exact_ratio,
    ) = _effective_mass_simulation(
        temperature=115,
        delta_x=1.48e-10,
        end_time=5e-12,
        n_samples=50,
        _key=key,
        barrier_energy=barrier_energy,
        inertial_mass=inertial_mass,
    )

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_effective_mass_ratio_periodic_1D(
        effective_mass_ratio=full_effective_mass_ratio,
        barrier_energy=barrier_energy,
        inertial_mass=inertial_mass,
        ax=ax,
    )
    mesh.set_rasterized(True)
    fig.savefig("examples/effective_mass.full.pdf", dpi=1000)

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_effective_mass_ratio_periodic_1D(
        effective_mass_ratio=free_effective_mass_ratio,
        barrier_energy=barrier_energy,
        inertial_mass=inertial_mass,
        ax=ax,
    )
    mesh.set_rasterized(True)
    fig.savefig("examples/effective_mass.free.pdf", dpi=1000)

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_effective_mass_ratio_periodic_1D(
        effective_mass_ratio=free_effective_mass_exact_ratio,
        barrier_energy=barrier_energy,
        inertial_mass=inertial_mass,
        ax=ax,
    )
    mesh.set_rasterized(True)
    fig.savefig("examples/effective_mass.exact.free.pdf", dpi=1000)

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_effective_mass_ratio_periodic_1D(
        effective_mass_ratio=full_effective_mass_exact_ratio,
        barrier_energy=barrier_energy,
        inertial_mass=inertial_mass,
        ax=ax,
    )
    mesh.set_rasterized(True)
    fig.savefig("examples/effective_mass.exact.full.pdf", dpi=1000)

    full_error = (
        abs(full_effective_mass_exact_ratio - full_effective_mass_ratio)
        / full_effective_mass_exact_ratio
    )
    free_error = (
        abs(free_effective_mass_exact_ratio - free_effective_mass_ratio)
        / free_effective_mass_exact_ratio
    )

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_effective_mass_ratio_periodic_1D(
        effective_mass_ratio=full_error,
        barrier_energy=barrier_energy,
        inertial_mass=inertial_mass,
        ax=ax,
    )
    mesh.set_rasterized(True)
    fig.savefig("examples/effective_mass.error.free.pdf", dpi=1000)

    fig, ax = get_fancy_figure()
    _, ax, mesh = plot_effective_mass_ratio_periodic_1D(
        effective_mass_ratio=free_error,
        barrier_energy=barrier_energy,
        inertial_mass=inertial_mass,
        ax=ax,
    )
    mesh.set_rasterized(True)
    fig.savefig("examples/effective_mass.error.full.pdf", dpi=1000)

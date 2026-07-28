from pathlib import Path

import jax
import jax.random as jrandom
import numpy as np
from tqdm import tqdm

from classical_diffusion.langevin import (
    TimeSpan,
    get_effective_mass,
    get_effective_mass_weighted,
    plot_effective_mass_ratio_periodic_1D,
    solve_free_ballistic_ensemble,
)
from classical_diffusion.plot import get_fancy_figure
from classical_diffusion.system import (
    PeriodicSystem1D,
    UnitSystem,
    calculate_probability_under_barrier,
)
from classical_diffusion.util import cached, disabled_timing

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
    filename = f"{temperature}_{delta_x}_{end_time}_{n_samples}_{key}_{hash(barrier_energy.tobytes())}_{hash(inertial_mass.tobytes())}.npz"
    return Path("examples/data") / filename


@cached(_solve_effective_mass_path)
def _effective_mass_simulation(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
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

    with disabled_timing():
        for idx, (i, j) in enumerate(
            tqdm(np.ndindex(barrier_energy.shape), total=barrier_energy.size)
        ):
            system = PeriodicSystem1D(
                gamma=0,
                temperature=temperature,
                m=inertial_mass[i, j],
                delta_x=delta_x,
                barrier_energy=barrier_energy[i, j],
                units=UnitSystem(Boltzmann=1.0, atomic_mass=inertial_mass[i, j]),
            )
            normalized_system = system.with_normalized_units()
            result = solve_free_ballistic_ensemble(
                normalized_system,
                TimeSpan(
                    t0=0,
                    t1=normalized_system.units.time_into(end_time, units=UnitSystem()),
                    n_steps=1000,
                ),
                n_samples=n_samples,
                _key=keys[idx],
                barrier_energy=normalized_system.barrier_energy,
            )

            prob_under_barrier_val = calculate_probability_under_barrier(
                system=normalized_system,
                barrier_energy=normalized_system.barrier_energy,
            )
            prob_under_barrier[i, j] = prob_under_barrier_val

            full_effective_mass = UnitSystem().mass_into(
                get_effective_mass_weighted(
                    result, prob_under_barrier=prob_under_barrier_val
                ),
                units=normalized_system.units,
            )
            full_effective_mass_ratio[i, j] = (
                full_effective_mass / system.with_si_units().m
            )

            free_effective_mass = UnitSystem().mass_into(
                get_effective_mass(result), units=normalized_system.units
            )
            free_effective_mass_ratio[i, j] = (
                free_effective_mass / system.with_si_units().m
            )

        return full_effective_mass_ratio, prob_under_barrier, free_effective_mass_ratio


if __name__ == "__main__":
    barrier_energy = np.linspace(1, 4, 20)
    inertial_mass = np.linspace(1, 20, 20)
    full_effective_mass_ratio, prob_under_barrier, free_effective_mass_ratio = (
        _effective_mass_simulation(
            temperature=110,
            delta_x=1.48e-10,
            end_time=2e-12,
            n_samples=50,
            _key=key,
            barrier_energy=barrier_energy,
            inertial_mass=inertial_mass,
        )
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

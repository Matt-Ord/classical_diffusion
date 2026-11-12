import jax
import numpy as np
import pytest
import sympy as sp

from classical_diffusion.jax.langevin import FixedStep
from classical_diffusion.langevin import (
    LITHIUM_COPPER_SYSTEM_2D,
    PeriodicSystem,
    build_2d_fcc,
    solve_ensemble_ballistic,
)
from classical_diffusion.langevin._experimental_systems import (  # ruff: ignore[import-private-name]
    LITHIUM_COPPER_BRIDGE_ENERGY,
    LITHIUM_COPPER_GAMMA,
    LITHIUM_COPPER_SPACING,
    LITHIUM_COPPER_TOP_ENERGY,
    LITHIUM_MASS,
)
from classical_diffusion.langevin._langevin import (  # ruff: ignore[import-private-name]
    get_random_initial_conditions,
)
from classical_diffusion.simulation import TimeSpan
from classical_diffusion.system import UnitSystem


def _reduced_energy(system: PeriodicSystem, x: np.ndarray) -> np.ndarray:
    """Return V / kT at positions x, with the coordinate on the last axis."""
    f = sp.lambdify(system.lambda_symbols, system.potential_expr, "numpy")
    v = np.asarray(f(*np.moveaxis(x, -1, 0), *system.params)) / system.kbt
    return v * np.ones(x.shape[:-1])


def _boltzmann_mean_energy(system: PeriodicSystem, n: int = 400) -> float:
    """Return <V> / kT, averaged over the unit cell with the Boltzmann weight."""
    u = np.linspace(0, 1, n, endpoint=False)
    grid = np.stack(np.meshgrid(*[u] * system.n_dim, indexing="ij"), axis=-1)
    v = _reduced_energy(system, grid @ system.lattice_vectors)
    weight = np.exp(-v)
    return float(np.sum(v * weight) / np.sum(weight))


def test_initial_positions_are_boltzmann_when_the_cell_is_large() -> None:
    # Lithium on Cu(111) at 80 K, whose top site (the origin) is about 29 kT.
    # In units where the cell is much larger than the sampling region of the
    # Gaussian proposal around the origin, that proposal started the
    # ensemble near the top site, with <V> far above its Boltzmann value.
    system = build_2d_fcc(
        gamma=LITHIUM_COPPER_GAMMA,
        temperature=80,
        barrier_energy=LITHIUM_COPPER_BRIDGE_ENERGY,
        top_energy=LITHIUM_COPPER_TOP_ENERGY,
        delta_x=LITHIUM_COPPER_SPACING,
        m=LITHIUM_MASS,
    )
    cell_length = 1000
    units = UnitSystem(
        boltzmann=1 / system.temperature,
        atomic_mass=system.units.atomic_mass / system.m,
        angstrom=system.units.angstrom * cell_length / LITHIUM_COPPER_SPACING,
    )
    system = system.with_units(units)

    x, p = get_random_initial_conditions(system, 20_000, _key=jax.random.key(0))

    assert np.mean(_reduced_energy(system, x)) == pytest.approx(
        _boltzmann_mean_energy(system), rel=0.03
    )
    assert np.mean(p**2) / (system.m * system.kbt) == pytest.approx(1, rel=0.03)


def test_ballistic_ensemble_starts_in_the_unit_cell() -> None:
    # The solvers convert the system twice on the way to the sampler, so this
    # checks that the unit cell survives to it.
    system = LITHIUM_COPPER_SYSTEM_2D
    result = solve_ensemble_ballistic.call_uncached(
        system,
        TimeSpan(t_end=1e-13, n_steps=2),
        500,
        step_config=FixedStep(step_size=0.05),
        _key=jax.random.key(0),
    )

    fractional = result.x_points[:, :, 0] @ np.linalg.inv(system.lattice_vectors)
    assert np.all((fractional >= 0) & (fractional < 1))

import numpy as np
import pytest
import sympy as sp

from classical_diffusion.langevin import (
    LITHIUM_COPPER_100_SYSTEM_1D,
    LITHIUM_COPPER_100_SYSTEM_2D,
    LITHIUM_COPPER_110_ACROSS_SYSTEM_1D,
    LITHIUM_COPPER_110_SYSTEM_1D,
    LITHIUM_COPPER_110_SYSTEM_2D,
    LITHIUM_COPPER_SYSTEM_2D,
    PeriodicSystem1D,
    PeriodicSystem2D,
    PeriodicSystemRectangular,
    PeriodicSystemSquare,
    System,
)
from classical_diffusion.langevin._langevin import (  # ruff: ignore[import-private-name]
    _get_langevin_units,
)


def _energy_at(system: System, *point: float) -> float:
    f = sp.lambdify(system.lambda_symbols, system.potential_expr, "numpy")
    return float(f(*point, *system.params))


def _potential_grid(system: PeriodicSystem2D, n: int = 200) -> np.ndarray:
    f = sp.lambdify(system.lambda_symbols, system.potential_expr, "numpy")
    u = np.linspace(0, 1, n, endpoint=False)
    grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
    a1, a2 = system.lattice_vectors
    x = grid_u * a1[0] + grid_v * a2[0]
    y = grid_u * a1[1] + grid_v * a2[1]
    return np.asarray(f(x, y, *system.params)) * np.ones_like(x)


@pytest.mark.parametrize("top_energy", [None, 1.5, 3.0])
def test_square_site_energies(top_energy: float | None) -> None:
    system = PeriodicSystemSquare(
        gamma=1.0,
        temperature=1.0,
        m=1.0,
        delta_x=2.0,
        barrier_energy=1.0,
        top_energy=top_energy,
    )
    assert _energy_at(system, 0, 0) == pytest.approx(0, abs=1e-12)
    assert _energy_at(system, 1, 0) == pytest.approx(1)
    assert _energy_at(system, 1, 1) == pytest.approx(system.top_energy)
    assert system.top_energy == (2.0 if top_energy is None else top_energy)

    grid = _potential_grid(system)
    assert grid.min() == pytest.approx(0, abs=1e-12)
    assert grid.max() == pytest.approx(system.top_energy, rel=1e-3)


def test_rectangular_site_energies() -> None:
    system = PeriodicSystemRectangular(
        gamma=1.0,
        temperature=1.0,
        m=1.0,
        delta_x=2.0,
        delta_y=3.0,
        barrier_energy_x=1.0,
        barrier_energy_y=5.0,
    )
    assert _energy_at(system, 0, 0) == pytest.approx(0, abs=1e-12)
    assert _energy_at(system, 1, 0) == pytest.approx(1)
    assert _energy_at(system, 0, 1.5) == pytest.approx(5)
    assert _energy_at(system, 1, 1.5) == pytest.approx(6)


@pytest.mark.parametrize(
    ("system", "cut"),
    [
        (LITHIUM_COPPER_100_SYSTEM_2D, LITHIUM_COPPER_100_SYSTEM_1D),
        (LITHIUM_COPPER_110_SYSTEM_2D, LITHIUM_COPPER_110_SYSTEM_1D),
    ],
)
def test_1d_cut_matches_2d_along_x(system: System, cut: PeriodicSystem1D) -> None:
    for fraction in (0.1, 0.25, 0.5):
        x = fraction * cut.delta_x
        assert _energy_at(system, x, 0) == pytest.approx(_energy_at(cut, x))


def test_110_across_cut_matches_2d_along_y() -> None:
    system = LITHIUM_COPPER_110_SYSTEM_2D
    cut = LITHIUM_COPPER_110_ACROSS_SYSTEM_1D
    for fraction in (0.1, 0.25, 0.5):
        y = fraction * cut.delta_x
        assert _energy_at(system, 0, y) == pytest.approx(_energy_at(cut, y))


@pytest.mark.parametrize(
    "system",
    [
        LITHIUM_COPPER_SYSTEM_2D,
        LITHIUM_COPPER_100_SYSTEM_2D,
        LITHIUM_COPPER_110_SYSTEM_2D,
    ],
)
def test_units_round_trip(system: System) -> None:
    units = _get_langevin_units(system)
    si_system = system.with_units(units).with_si_units()
    np.testing.assert_allclose(si_system.params, system.params, rtol=1e-6)
    np.testing.assert_allclose(si_system.m, system.m, rtol=1e-6)
    np.testing.assert_allclose(si_system.gamma, system.gamma, rtol=1e-6)

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
    CosineSystem1D,
    FourierTerm,
    LangevinSimulationResult,
    PeriodicSystem,
    System,
    build_2d_fcc,
    build_2d_rectangular,
    build_2d_square,
    build_2d_triangular,
    shift_origin_to_unit_cell,
)
from classical_diffusion.langevin._langevin import (  # ruff: ignore[import-private-name]
    _get_langevin_units,
)
from classical_diffusion.system import UnitSystem


def _energy_at(system: System, *point: float) -> float:
    f = sp.lambdify(system.lambda_symbols, system.potential_expr, "numpy")
    return float(f(*point, *system.params))


def _potential_on_cell(system: PeriodicSystem, n: int = 200) -> np.ndarray:
    f = sp.lambdify(system.lambda_symbols, system.potential_expr, "numpy")
    u = np.linspace(0, 1, n, endpoint=False)
    grid = np.stack(np.meshgrid(*[u] * system.n_dim, indexing="ij"), axis=-1)
    x = grid @ system.lattice_vectors
    values = f(*np.moveaxis(x, -1, 0), *system.params)
    return np.asarray(values) * np.ones(x.shape[:-1])


def test_potential_is_periodic_on_the_lattice() -> None:
    system = PeriodicSystem(
        gamma=1.0,
        temperature=1.0,
        m=1.0,
        lattice_vectors=[[1.0, 0.3], [-0.2, 1.4]],
        terms=(
            FourierTerm(miller=(1, 0), amplitude=0.7),
            FourierTerm(miller=(2, -1), amplitude=-0.4),
        ),
        offset=0.1,
        barrier_energy=1.0,
    )
    point = np.array([0.37, -0.81])
    for shift in system.lattice_vectors:
        assert _energy_at(system, *(point + shift)) == pytest.approx(
            _energy_at(system, *point)
        )


def test_term_is_a_cosine_of_the_fractional_coordinate() -> None:
    lattice = np.array([[2.0, 0.5], [0.0, 3.0]])
    system = PeriodicSystem(
        gamma=1.0,
        temperature=1.0,
        m=1.0,
        lattice_vectors=lattice,
        terms=(FourierTerm(miller=(1, 2), amplitude=1.0),),
        barrier_energy=1.0,
    )
    fractional = np.array([0.13, 0.29])
    expected = np.cos(2 * np.pi * (fractional[0] + 2 * fractional[1]))
    assert _energy_at(system, *(fractional @ lattice)) == pytest.approx(expected)


def test_1d_cosine_site_energies() -> None:
    system = CosineSystem1D(
        gamma=1.0, temperature=1.0, m=1.0, delta_x=2.0, barrier_energy=3.0
    )
    assert _energy_at(system, 0.0) == pytest.approx(0, abs=1e-12)
    assert _energy_at(system, 1.0) == pytest.approx(3.0)
    assert system.delta_x == pytest.approx(2.0)


@pytest.mark.parametrize("top_energy", [None, 5.0, 12.0])
def test_fcc_site_energies(top_energy: float | None) -> None:
    system = build_2d_fcc(
        gamma=1.0,
        temperature=1.0,
        m=1.0,
        delta_x=2.0,
        barrier_energy=1.0,
        top_energy=top_energy,
    )
    a1, a2 = system.lattice_vectors
    top = 9.0 if top_energy is None else top_energy
    assert _energy_at(system, *((a1 + a2) / 3)) == pytest.approx(0, abs=1e-12)
    assert _energy_at(system, *(2 * (a1 + a2) / 3)) == pytest.approx(0, abs=1e-12)
    assert _energy_at(system, *(a1 / 2)) == pytest.approx(1.0)
    assert _energy_at(system, 0.0, 0.0) == pytest.approx(top)


def test_triangular_site_energies() -> None:
    system = build_2d_triangular(
        gamma=1.0, temperature=1.0, m=1.0, delta_x=2.0, barrier_energy=1.0
    )
    a1, a2 = system.lattice_vectors
    assert _energy_at(system, 0.0, 0.0) == pytest.approx(0, abs=1e-12)
    assert _energy_at(system, *(a1 / 2)) == pytest.approx(1.0)
    assert _energy_at(system, *((a1 + a2) / 3)) == pytest.approx(9 / 8)


@pytest.mark.parametrize("top_energy", [None, 1.5, 3.0])
def test_square_site_energies(top_energy: float | None) -> None:
    system = build_2d_square(
        gamma=1.0,
        temperature=1.0,
        m=1.0,
        delta_x=2.0,
        barrier_energy=1.0,
        top_energy=top_energy,
    )
    top = 2.0 if top_energy is None else top_energy
    assert _energy_at(system, 0, 0) == pytest.approx(0, abs=1e-12)
    assert _energy_at(system, 1, 0) == pytest.approx(1)
    assert _energy_at(system, 1, 1) == pytest.approx(top)

    grid = _potential_on_cell(system)
    assert grid.min() == pytest.approx(0, abs=1e-12)
    assert grid.max() == pytest.approx(top, rel=1e-3)


def test_rectangular_site_energies() -> None:
    system = build_2d_rectangular(
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
    assert system.barrier_energy == pytest.approx(1)


@pytest.mark.parametrize(
    ("system", "cut"),
    [
        (LITHIUM_COPPER_100_SYSTEM_2D, LITHIUM_COPPER_100_SYSTEM_1D),
        (LITHIUM_COPPER_110_SYSTEM_2D, LITHIUM_COPPER_110_SYSTEM_1D),
    ],
)
def test_1d_cut_matches_2d_along_x(system: System, cut: PeriodicSystem) -> None:
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
        LITHIUM_COPPER_110_SYSTEM_1D,
    ],
)
def test_units_round_trip(system: PeriodicSystem) -> None:
    units = _get_langevin_units(system)
    si_system = system.with_units(units).with_units(UnitSystem())
    np.testing.assert_allclose(si_system.params, system.params, rtol=1e-6)
    np.testing.assert_allclose(
        si_system.lattice_vectors, system.lattice_vectors, rtol=1e-6
    )
    np.testing.assert_allclose(
        si_system.barrier_energy, system.barrier_energy, rtol=1e-6
    )
    np.testing.assert_allclose(si_system.m, system.m, rtol=1e-6)
    np.testing.assert_allclose(si_system.gamma, system.gamma, rtol=1e-6)


def _cosine_on_cell(cell: float, miller: int, barrier: float = 1.0) -> PeriodicSystem:
    return PeriodicSystem(
        gamma=1.0,
        temperature=1.0,
        m=1.0,
        lattice_vectors=[[cell]],
        terms=(FourierTerm(miller=(miller,), amplitude=-0.5),),
        offset=0.5,
        barrier_energy=barrier,
    )


def test_equality_sees_the_lattice_and_barrier() -> None:
    cosine = _cosine_on_cell(2.0, 1)
    doubled_cell = _cosine_on_cell(4.0, 2)
    assert cosine.params == doubled_cell.params
    assert cosine != doubled_cell
    assert hash(cosine) != hash(doubled_cell)

    assert cosine != _cosine_on_cell(2.0, 1, barrier=2.0)
    assert cosine == _cosine_on_cell(2.0, 1)
    assert hash(cosine) == hash(_cosine_on_cell(2.0, 1))


def test_cosine_system_is_a_periodic_system() -> None:
    cosine = CosineSystem1D(
        gamma=1.0, temperature=1.0, m=1.0, delta_x=2.0, barrier_energy=1.0
    )
    general = _cosine_on_cell(2.0, 1)
    assert cosine.params == general.params
    assert cosine.with_units(cosine.units) == cosine
    assert isinstance(cosine.with_units(cosine.units), CosineSystem1D)


def _periodic(
    *,
    lattice: list[list[float]] | None = None,
    terms: tuple[FourierTerm, ...] | None = None,
    offset: float = 0.5,
    barrier: float = 1.0,
) -> PeriodicSystem:
    return PeriodicSystem(
        gamma=1.0,
        temperature=1.0,
        m=1.0,
        lattice_vectors=lattice or [[2.0, 0.0], [0.0, 3.0]],
        terms=terms
        or (
            FourierTerm(miller=(1, 0), amplitude=-0.5),
            FourierTerm(miller=(0, 1), amplitude=-0.25),
        ),
        offset=offset,
        barrier_energy=barrier,
    )


@pytest.mark.parametrize(
    "changed",
    [
        _periodic(lattice=[[2.0, 0.0], [0.0, 4.0]]),
        _periodic(
            terms=(
                FourierTerm(miller=(1, 0), amplitude=-0.5),
                FourierTerm(miller=(0, 2), amplitude=-0.25),
            )
        ),
        _periodic(
            terms=(
                FourierTerm(miller=(1, 0), amplitude=-0.5),
                FourierTerm(miller=(0, 1), amplitude=-0.3),
            )
        ),
        _periodic(terms=(FourierTerm(miller=(1, 0), amplitude=-0.5),)),
        _periodic(offset=0.0),
        _periodic(barrier=2.0),
    ],
    ids=["lattice", "miller", "amplitude", "number of terms", "offset", "barrier"],
)
def test_any_change_to_the_potential_breaks_equality(changed: PeriodicSystem) -> None:
    reference = _periodic()
    assert reference == _periodic()
    assert hash(reference) == hash(_periodic())
    assert changed != reference
    assert hash(changed) != hash(reference)


@pytest.mark.parametrize(
    "system",
    [
        CosineSystem1D(
            gamma=1.0, temperature=1.0, m=1.0, delta_x=2.0, barrier_energy=1.0
        ),
        build_2d_fcc(
            gamma=1.0, temperature=1.0, m=1.0, delta_x=2.0, barrier_energy=1.0
        ),
    ],
    ids=["1d", "fcc"],
)
def test_shift_origin_to_unit_cell(system: PeriodicSystem) -> None:
    rng = np.random.default_rng(0)
    lattice = system.lattice_vectors
    start_in_cell = rng.uniform(-0.45, 0.45, size=(5, system.n_dim)) @ lattice
    lattice_offset = rng.integers(-3, 4, size=(5, system.n_dim)) @ lattice
    path = rng.normal(size=(5, system.n_dim, 4))
    path[:, :, 0] = 0.0
    x_points = (start_in_cell + lattice_offset)[:, :, None] + path
    result = LangevinSimulationResult(
        system=system,
        x_points=x_points,
        p_points=np.zeros_like(x_points),
        times=np.arange(4.0),
    )
    shifted = shift_origin_to_unit_cell(result)
    np.testing.assert_allclose(shifted.x_points[:, :, 0], start_in_cell, atol=1e-12)
    np.testing.assert_allclose(
        shifted.x_points - shifted.x_points[:, :, :1], x_points - x_points[:, :, :1]
    )

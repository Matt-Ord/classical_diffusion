import zlib
from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import final, override

import jax
import numpy as np
import sympy as sp

from classical_diffusion.hopping import KramersParameters
from classical_diffusion.system import CanonicalUnitSystem, UnitSystem


def _hash_sympy_expr(expr: sp.Expr) -> int:
    stable_string = sp.srepr(expr)
    return zlib.crc32(stable_string.encode("utf-8"))


@cache
def _get_force_expr(
    potential: sp.Expr, coordinates: tuple[sp.Symbol, ...]
) -> tuple[sp.Expr, ...]:
    """Return the force, -grad V, shared by every system with the same potential."""
    return tuple(-sp.simplify(sp.diff(potential, c)) for c in coordinates)


@dataclass(frozen=True, kw_only=True)
class System:
    """Parameters representing a physical system."""

    gamma: float
    temperature: float
    m: float
    potential: tuple[int, sp.Expr]
    params: tuple[float, ...]
    units: UnitSystem = field(default_factory=UnitSystem)

    @property
    def n_dim(self) -> int:
        """The number of dimensions of the system."""
        return self.potential[0]

    @cached_property
    def coordinate_symbols(self) -> tuple[sp.Symbol, ...]:
        """The symbols of each coordinate of the system."""
        return sp.symbols(f"x0:{self.n_dim}")

    @property
    def n_params(self) -> int:
        """The number of parameters of the system."""
        return len(self.params)

    @cached_property
    def parameter_symbols(self) -> tuple[sp.Symbol, ...]:
        """The symbols of each parameter of the system."""
        return sp.symbols(f"s0:{self.n_params}")

    @cached_property
    def lambda_symbols(self) -> tuple[sp.Symbol, ...]:
        """The symbols of the system, including coordinates and parameters."""
        return (*self.coordinate_symbols, *self.parameter_symbols)

    @property
    def potential_expr(self) -> sp.Expr:
        """The potential expression of the system."""
        return self.potential[1]

    @cached_property
    def force_expr(self) -> list[sp.Expr]:
        """The symbolic force of the system."""
        return list(_get_force_expr(self.potential_expr, self.coordinate_symbols))

    def as_canonical(self) -> CanonicalSystem:
        """Return the canonical form of the system."""
        return CanonicalSystem(
            gamma=self.gamma,
            temperature=self.temperature,
            m=self.m,
            potential=self.potential,
            params=self.params,
            units=self.units.as_canonical(),
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.gamma,
                self.temperature,
                self.m,
                self.params,
                (
                    self.potential[0],
                    _hash_sympy_expr(self.potential[1]),
                ),
            )
        )

    @property
    def kbt(self) -> float:
        """Kbt."""
        return self.units.boltzmann * self.temperature

    def with_units(self, units: UnitSystem) -> System:
        """Return the system converted to the given units."""
        length_factor = self.units.length_factor / units.length_factor
        energy_factor = self.units.energy_factor / units.energy_factor

        scaled_expr = (
            self.potential_expr.subs(
                {sym: sym * length_factor for sym in self.coordinate_symbols}
            )
            / energy_factor
        )
        return System(
            m=self.units.mass_into(self.m, units),
            gamma=self.units.frequency_into(self.gamma, units),
            units=units,
            temperature=self.temperature,
            potential=(self.n_dim, scaled_expr),
            params=self.params,
        )

    def with_si_units(self) -> System:
        """Return the si parameters of the system."""
        return self.with_units(UnitSystem())


@jax.tree_util.register_dataclass
@final
@dataclass(frozen=True, kw_only=True)
class CanonicalSystem(System):
    """Parameters representing a physical system."""

    potential: tuple[int, sp.Expr] = field(metadata={"static": True})
    units: CanonicalUnitSystem = field(default_factory=CanonicalUnitSystem)

    def with_units(self, units: UnitSystem) -> CanonicalSystem:
        """Return the system converted to the given units."""
        return super().with_units(units).as_canonical()


class HarmonicSystem(System):
    """Parameters representing a simple harmonic oscillator system."""

    _omega: float

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        omega: float,
        units: UnitSystem | None = None,
        n_dim: int = 1,
    ) -> None:
        s0 = sp.Symbol("s0")
        potential = 0.5 * s0**2 * sp.symbols("x0") ** 2

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
            params=(omega,),
            units=units or UnitSystem(),
        )

    @property
    def omega(self) -> float:
        """The angular frequency of the system."""
        return self.params[0]

    @override
    def with_units(self, units: UnitSystem) -> HarmonicSystem:

        return HarmonicSystem(
            gamma=self.units.frequency_into(self.gamma, units),
            temperature=self.temperature,
            m=self.units.mass_into(self.m, units),
            omega=self.units.frequency_into(self.omega, units),
            n_dim=self.n_dim,
            units=units,
        )


@dataclass(frozen=True, kw_only=True)
class FourierTerm:
    """One term, amplitude cos(2 pi miller . u), of a periodic potential.

    u are the fractional coordinates: the position measured along each lattice
    vector, in units of that vector, so the term is periodic on the lattice.
    """

    miller: tuple[int, ...]
    amplitude: float


def _get_periodic_param_layout(n_dim: int, n_terms: int) -> tuple[slice, slice, int]:
    """Return where the wavevectors, amplitudes and offset sit in the parameters."""
    n_wavevector = n_dim * n_terms
    return (
        slice(0, n_wavevector),
        slice(n_wavevector, n_wavevector + n_terms),
        n_wavevector + n_terms,
    )


@cache
def _get_periodic_potential_expr(n_dim: int, n_terms: int) -> sp.Expr:
    """Return offset + sum_k a_k cos(g_k . x).

    The parameters are the wavevectors g_k (n_dim components each), then the
    amplitudes a_k, then the offset. The cosine arguments are linear in the
    coordinates, so the force is cheap to derive, and every system with the same
    dimension and number of terms shares one expression.
    """
    coordinates = sp.symbols(f"x0:{n_dim}")
    _, amplitude_slice, offset_index = _get_periodic_param_layout(n_dim, n_terms)
    symbols = sp.symbols(f"s0:{offset_index + 1}")
    amplitudes = symbols[amplitude_slice]
    terms = (
        amplitudes[k]
        * sp.cos(
            sp.Add(*(symbols[k * n_dim + j] * coordinates[j] for j in range(n_dim)))
        )
        for k in range(n_terms)
    )
    return symbols[offset_index] + sp.Add(*terms)


def _get_fourier_series_minimum(millers: np.ndarray, amplitudes: np.ndarray) -> float:
    """Return the minimum over u of sum_k a_k cos(2 pi n_k . u).

    The minimum does not depend on the lattice. It is found on a grid of
    fractional coordinates, fine enough to resolve every term, then refined to
    machine precision by Newton steps with the exact gradient and Hessian.
    """
    n_dim = millers.shape[1]

    def value(u: np.ndarray) -> np.ndarray:
        return np.cos(2 * np.pi * u @ millers.T) @ amplitudes

    n_grid = max(16, 8 * int(np.abs(millers).max(initial=0)))
    axis = np.arange(n_grid) / n_grid
    grid = np.stack(np.meshgrid(*[axis] * n_dim, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, n_dim)
    values = value(grid)
    best = float(values.min())

    for u in grid[np.argsort(values)[:8]]:
        point = u.copy()
        for _ in range(50):
            phase = 2 * np.pi * millers @ point
            gradient = -2 * np.pi * (amplitudes * np.sin(phase)) @ millers
            hessian = -((2 * np.pi) ** 2) * np.einsum(
                "k,ki,kj->ij", amplitudes * np.cos(phase), millers, millers
            )
            if np.any(np.linalg.eigvalsh(hessian) <= 0):
                break
            step = np.linalg.solve(hessian, gradient)
            point -= step
            if np.max(np.abs(step)) < 1e-15:  # ruff: ignore[magic-value-comparison]
                break
        best = min(best, float(value(point)))
    return best


@dataclass(frozen=True, kw_only=True, init=False)
class PeriodicSystem(System):
    """A system whose potential is periodic on a lattice, with its minimum at zero.

    The potential is a Fourier series, sum_k a_k cos(2 pi n_k . u) + C, in the
    fractional coordinates u of the lattice: the position measured along each
    lattice vector. The constant C puts the minimum at zero. The series is given
    by the lattice vectors and the Miller indices n_k, and stored as the
    wavevectors g_k = 2 pi n_k . B, with B the reciprocal basis, so the
    potential in x is sum_k a_k cos(g_k . x) + C. Build the common surfaces with
    CosineSystem1D, build_2d_fcc, build_2d_triangular, build_2d_square and
    build_2d_rectangular.
    """

    _lattice: tuple[tuple[float, ...], ...]
    _millers: tuple[tuple[int, ...], ...]

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        lattice_vectors: np.ndarray,
        terms: tuple[FourierTerm, ...],
        units: UnitSystem | None = None,
    ) -> None:
        n_dim = lattice_vectors.shape[0]
        if lattice_vectors.shape != (n_dim, n_dim):
            msg = f"lattice_vectors must be square, not {lattice_vectors.shape}"
            raise ValueError(msg)
        millers = np.array([t.miller for t in terms], dtype=int).reshape(-1, n_dim)
        if any(len(t.miller) != n_dim for t in terms):
            msg = f"every Miller index must have {n_dim} components"
            raise ValueError(msg)
        amplitudes = np.array([t.amplitude for t in terms], dtype=float)

        # Rows of the reciprocal basis b_i satisfy a_i . b_j = delta_ij, so the
        # term with Miller index n has wavevector g = 2 pi n . B and phase 2 pi n . u.
        reciprocal = np.linalg.inv(lattice_vectors).T
        wavevectors = 2 * np.pi * millers @ reciprocal
        offset = -_get_fourier_series_minimum(millers, amplitudes)

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, _get_periodic_potential_expr(n_dim, len(terms))),
            params=(*wavevectors.ravel().tolist(), *amplitudes.tolist(), offset),
            units=units or UnitSystem(),
        )
        object.__setattr__(
            self, "_lattice", tuple(map(tuple, lattice_vectors.astype(float).tolist()))
        )
        object.__setattr__(self, "_millers", tuple(map(tuple, millers.tolist())))

    @property
    def lattice_vectors(self) -> np.ndarray:
        """Primitive lattice vectors, as rows."""
        return np.array(self._lattice)

    @property
    def terms(self) -> tuple[FourierTerm, ...]:
        """The Fourier terms of the potential."""
        _, amplitude_slice, _ = _get_periodic_param_layout(
            self.n_dim, len(self._millers)
        )
        return tuple(
            FourierTerm(miller=n, amplitude=a)
            for n, a in zip(self._millers, self.params[amplitude_slice], strict=True)
        )

    @override
    def __hash__(self) -> int:
        # System's hash is stable across processes; add the lattice and Miller indices
        return hash((super().__hash__(), self._lattice, self._millers))

    @override
    def with_units(self, units: UnitSystem) -> PeriodicSystem:
        """Return the system in the given units."""
        energy = self.units.energy_into(1.0, units)
        return PeriodicSystem(
            gamma=self.units.frequency_into(self.gamma, units),
            temperature=self.temperature,
            m=self.units.mass_into(self.m, units),
            lattice_vectors=self.units.length_into(self.lattice_vectors, units),
            terms=tuple(
                FourierTerm(miller=t.miller, amplitude=t.amplitude * energy)
                for t in self.terms
            ),
            units=units,
        )


class CosineSystem1D(PeriodicSystem):
    """A 1D cosine potential, V = (E_b / 2) (1 - cos(2 pi x / delta_x)).

    The minima are at x = n delta_x and the maxima at E_b halfway between. The
    type guarantees a single cosine, so the period delta_x and the barrier E_b
    are well defined, which analytic results such as the Kramers mapping rely on.
    """

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        delta_x: float,
        barrier_energy: float,
        units: UnitSystem | None = None,
    ) -> None:
        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            lattice_vectors=np.array([[delta_x]]),
            terms=(FourierTerm(miller=(1,), amplitude=-barrier_energy / 2),),
            units=units,
        )

    @property
    def delta_x(self) -> float:
        """The period of the potential."""
        return abs(self._lattice[0][0])

    @property
    def barrier_energy(self) -> float:
        """The height of the maxima above the minima."""
        return -2 * self.terms[0].amplitude

    @override
    def with_units(self, units: UnitSystem) -> CosineSystem1D:
        """Return the system in the given units."""
        return CosineSystem1D(
            gamma=self.units.frequency_into(self.gamma, units),
            temperature=self.temperature,
            m=self.units.mass_into(self.m, units),
            delta_x=self.units.length_into(self.delta_x, units),
            barrier_energy=self.units.energy_into(self.barrier_energy, units),
            units=units,
        )


def _get_hexagonal_lattice(delta_x: float) -> np.ndarray:
    """Return the lattice vectors of a hexagonal lattice with nearest-neighbor spacing delta_x."""
    return np.array([[0.0, delta_x], [np.sqrt(3.0) / 2.0 * delta_x, 0.5 * delta_x]])


_HEXAGONAL_SHELL = ((1, 0), (0, 1), (1, 1))
"""The Miller indices of the three shortest reciprocal lattice vectors at 120 degrees."""


def build_2d_fcc(  # ruff:ignore[too-many-arguments]
    *,
    gamma: float,
    temperature: float,
    m: float,
    delta_x: float,
    barrier_energy: float,
    top_energy: float | None = None,
    units: UnitSystem | None = None,
) -> PeriodicSystem:
    """Build the potential of an fcc(111) surface, with nearest-neighbor spacing delta_x.

    The fcc and hcp hollow sites are degenerate minima at zero, the bridge sites
    are saddle points at barrier_energy and the top sites are maxima at
    top_energy. This uses the first two shells of the hexagonal lattice,
    A_1 sum_j cos(G_j . r) + A_2 sum_j cos(2 G_j . r), with
    A_1 = (E_t - E_b) / 4 and A_2 = (9 E_b - E_t) / 36.
    top_energy defaults to 9 barrier_energy, for which A_2 = 0.
    """
    top_energy = 9 * barrier_energy if top_energy is None else top_energy
    first = (top_energy - barrier_energy) / 4
    second = (9 * barrier_energy - top_energy) / 36
    return PeriodicSystem(
        gamma=gamma,
        temperature=temperature,
        m=m,
        lattice_vectors=_get_hexagonal_lattice(delta_x),
        terms=(
            *(FourierTerm(miller=n, amplitude=first) for n in _HEXAGONAL_SHELL),
            *(
                FourierTerm(miller=(2 * n[0], 2 * n[1]), amplitude=second)
                for n in _HEXAGONAL_SHELL
            ),
        ),
        units=units,
    )


def build_2d_triangular(  # ruff:ignore[too-many-arguments]
    *,
    gamma: float,
    temperature: float,
    m: float,
    delta_x: float,
    barrier_energy: float,
    units: UnitSystem | None = None,
) -> PeriodicSystem:
    """Build -(E_b / 4) sum_j cos(G_j . r), with one minimum per hexagonal cell.

    The minima are at the lattice points, a spacing delta_x apart, the bridge
    sites are saddle points at E_b and the hollow sites are maxima at 9 E_b / 8.
    Along a close-packed row the potential reduces to CosineSystem1D with the
    same barrier and period.
    """
    return PeriodicSystem(
        gamma=gamma,
        temperature=temperature,
        m=m,
        lattice_vectors=_get_hexagonal_lattice(delta_x),
        terms=tuple(
            FourierTerm(miller=n, amplitude=-barrier_energy / 4)
            for n in _HEXAGONAL_SHELL
        ),
        units=units,
    )


def build_2d_square(  # ruff:ignore[too-many-arguments]
    *,
    gamma: float,
    temperature: float,
    m: float,
    delta_x: float,
    barrier_energy: float,
    top_energy: float | None = None,
    units: UnitSystem | None = None,
) -> PeriodicSystem:
    """Build the potential of an fcc(100) surface, with hollow sites a spacing delta_x apart.

    The hollow sites are minima at zero, the bridge sites are saddle points at
    barrier_energy and the top sites are maxima at top_energy. The potential is
    -A (cos k x + cos k y) - B (cos k (x + y) + cos k (x - y)), with
    k = 2 pi / delta_x, A = E_t / 4 and B = (2 E_b - E_t) / 8. top_energy
    defaults to 2 barrier_energy, for which B = 0.
    """
    top_energy = 2 * barrier_energy if top_energy is None else top_energy
    first = top_energy / 4
    second = (2 * barrier_energy - top_energy) / 8
    return PeriodicSystem(
        gamma=gamma,
        temperature=temperature,
        m=m,
        lattice_vectors=np.array([[delta_x, 0.0], [0.0, delta_x]]),
        terms=(
            FourierTerm(miller=(1, 0), amplitude=-first),
            FourierTerm(miller=(0, 1), amplitude=-first),
            FourierTerm(miller=(1, 1), amplitude=-second),
            FourierTerm(miller=(1, -1), amplitude=-second),
        ),
        units=units,
    )


def build_2d_rectangular(  # ruff:ignore[too-many-arguments]
    *,
    gamma: float,
    temperature: float,
    m: float,
    delta_x: float,
    delta_y: float,
    barrier_energy_x: float,
    barrier_energy_y: float,
    units: UnitSystem | None = None,
) -> PeriodicSystem:
    """Build the potential of an fcc(110) surface, one cosine along each axis.

    The hollow sites are minima at zero on a rectangular lattice, delta_x apart
    along x and delta_y along y. A hop along x crosses a saddle point at
    barrier_energy_x, a hop along y one at barrier_energy_y, and the maxima are
    at their sum. Along each axis the potential reduces to CosineSystem1D.
    """
    return PeriodicSystem(
        gamma=gamma,
        temperature=temperature,
        m=m,
        lattice_vectors=np.array([[delta_x, 0.0], [0.0, delta_y]]),
        terms=(
            FourierTerm(miller=(1, 0), amplitude=-barrier_energy_x / 2),
            FourierTerm(miller=(0, 1), amplitude=-barrier_energy_y / 2),
        ),
        units=units,
    )


class DerivativeSafeMod(sp.Mod):
    """A subclass of sympy.Mod that is safe for differentiation.

    This class overrides the _eval_derivative method to ensure that the derivative
    of a Mod expression is computed correctly, even when the modulus depends on the
    variable of differentiation.
    """

    def _eval_derivative(self, s: sp.Symbol) -> sp.Expr:
        p, q = self.args
        if not q.has(s):
            return sp.diff(p, s)
        return sp.Derivative(self, s)


def _build_kramers_potential() -> sp.Expr:
    x0 = sp.symbols("x0")
    s0 = sp.symbols("s0")
    s1 = sp.symbols("s1")
    s2 = sp.symbols("s2")

    # To express a double harmonic potential as a sympy expression, consider moving two harmonics towards each other
    # y = 1/2 omega_well^2 x^2                      represents an upright harmonic centred at the origin
    # y = E_b - 1/2 omega_barrier^2 (x - x_0)^2     represents an inverted harmonic centred at x_0, height E_b
    # At the point where the double harmonic potential is smooth, the two harmonics just touch. So there will only
    # be one solution to the above simultaneous system. Solving these equations gives a quadratic for x: then
    # setting the determinant equal to zero gives an expression for x_0
    # From this, the overlap point, x_meet, can be found and the expression for the periodic potential is as below.

    omegas_ss = s0**2 + s1**2  # Omegas squared sum
    x_0 = sp.sqrt((2 * omegas_ss * s2) / (s0**2 * s1**2))
    x_meet = (s1**2 / omegas_ss) * x_0

    periodic_x = DerivativeSafeMod(x0 + x_meet, 2 * x_0) - x_meet

    return sp.Piecewise(
        (0.5 * s0**2 * periodic_x**2, periodic_x <= x_meet),
        (
            s2 - 0.5 * s1**2 * (periodic_x - x_0) ** 2,
            periodic_x >= x_meet,
        ),
        (0, True),
    )


class KramersSystem1D(System):
    """Parameters representing a periodic double harmonic system."""

    def __init__(
        self,
        *,
        params: KramersParameters,
        n_dim: int = 1,
    ) -> None:

        super().__init__(
            gamma=params.gamma,
            temperature=params.temperature,
            m=params.m,
            potential=(n_dim, _build_kramers_potential()),
            params=(params.omega_well, params.omega_barrier, params.barrier_energy),
            units=params.units,
        )

    @property
    def delta_x(self) -> float:
        """The delta x of the system."""
        return self.kramers_params.delta_x

    @property
    def kramers_params(self) -> KramersParameters:
        return KramersParameters(
            omega_well=self.params[0],
            omega_barrier=self.params[1],
            barrier_energy=self.params[2],
            temperature=self.temperature,
            m=self.m,
            gamma=self.gamma,
            units=self.units,
        )

    @property
    def omega_well(self) -> float:
        """The harmonic frequency at the bottom of the well."""
        return self.kramers_params.omega_well

    @property
    def omega_barrier(self) -> float:
        """The harmonic frequency at the top of the barrier."""
        return self.kramers_params.omega_barrier

    @property
    def barrier_energy(self) -> float:
        """The barrier energy of the system."""
        return self.kramers_params.barrier_energy

    @override
    def with_units(self, units: UnitSystem) -> KramersSystem1D:
        """Return the parameters of the system in the specified units."""
        return KramersSystem1D(
            params=self.kramers_params.with_units(units),
            n_dim=self.n_dim,
        )


def get_diffusion_time(system: System, characteristic_length: float) -> float:
    """Return the average time for a particle to traverse a characteristic length."""
    return np.sqrt(system.m * characteristic_length**2 / system.kbt)


def get_characteristic_periodic_mass(system: CosineSystem1D) -> float:
    """Return the characteristic mass of a periodic system, kT delta_x^2 / gamma^2."""
    return system.kbt * system.delta_x**2 / system.gamma**2

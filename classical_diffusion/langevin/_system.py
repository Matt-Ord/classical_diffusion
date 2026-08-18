import dataclasses
import zlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import Self, final, override

import jax
import numpy as np
import sympy as sp

from classical_diffusion.hopping import KramersParameters
from classical_diffusion.system import CanonicalUnitSystem, UnitSystem


def _hash_sympy_expr(expr: sp.Expr) -> int:
    stable_string = sp.srepr(expr)
    return zlib.crc32(stable_string.encode("utf-8"))


@dataclass(frozen=True, kw_only=True)
class System:
    """Parameters representing a physical system."""

    gamma: float
    temperature: float
    m: float
    potential: tuple[int, sp.Expr]
    params: tuple[float, ...] = ()
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
        return [
            -sp.simplify(sp.diff(self.potential_expr, c))
            for c in self.coordinate_symbols
        ]

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
        """Retrurn kbt."""
        return self.units.Boltzmann * self.temperature

    @property
    def lattice_vectors(self) -> np.ndarray:
        """Primitive lattice vectors (as columns). Identity for a Cartesian/square domain."""
        return np.eye(self.n_dim)

    @property
    def sampling_domain(self) -> tuple:
        """The domain over which the equilibrium x-density should be sampled."""
        return (-np.inf, np.inf)

    def with_units(self, units: UnitSystem) -> Self:
        """Return the system with the given units."""
        length_factor = units.angstrom / self.units.angstrom
        mass_factor = units.atomic_mass / self.units.atomic_mass
        energy_factor = units.Boltzmann / self.units.Boltzmann
        time_factor = np.sqrt(length_factor**2 * mass_factor / energy_factor)
        return dataclasses.replace(
            self,
            gamma=self.gamma / time_factor,
            temperature=self.temperature,
            m=self.m * mass_factor,
            units=units,
            potential=self.potential,
        )

    @property
    def normalized_units(self) -> UnitSystem:
        """The units suited to a simulation of the system."""
        return UnitSystem(
            Boltzmann=1 / self.temperature,
            angstrom=self.units.angstrom,
            atomic_mass=self.units.atomic_mass / self.m,
        )

    def with_normalized_units(self) -> Self:
        """Return the parameters of the simulation in normalized units."""
        return self.with_units(self.normalized_units)

    def with_si_units(self) -> Self:
        """Return the si parameters of the system."""
        return self.with_units(UnitSystem())


@jax.tree_util.register_dataclass
@final
@dataclass(frozen=True, kw_only=True)
class CanonicalSystem(System):
    """Parameters representing a physical system."""

    potential: tuple[int, sp.Expr] = field(metadata={"static": True})
    units: CanonicalUnitSystem = field(default_factory=CanonicalUnitSystem)


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

        mass_factor = units.mass_factor / self.units.mass_factor
        time_factor = units.time_factor / self.units.time_factor
        return HarmonicSystem(
            gamma=self.gamma / time_factor,
            temperature=self.temperature,
            m=self.m * mass_factor,
            omega=self.omega / time_factor,
            n_dim=self.n_dim,
            units=units,
        )

    @property
    def normalized_units(self) -> UnitSystem:
        """The units suited to a simulation of the system."""
        time_factor = 1 / self.omega
        return UnitSystem(
            Boltzmann=1 / self.temperature,
            angstrom=self.units.angstrom
            / (time_factor * np.sqrt(self.units.Boltzmann * self.temperature / self.m)),
            atomic_mass=self.units.atomic_mass / self.m,
        )


class PeriodicSystem1D(System):
    """Parameters for a 1D cosine potential system."""

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        *,
        gamma: float,
        temperature: float,
        m: float,
        delta_x: float,
        barrier_energy: float,
        units: UnitSystem | None = None,
        n_dim: int = 1,
    ) -> None:
        s0 = sp.Symbol("s0")
        s1 = sp.Symbol("s1")
        potential = 0.5 * s1 * (1 - sp.cos(2 * sp.pi * sp.symbols("x0") / s0))

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(n_dim, potential),
            params=(delta_x, barrier_energy),
            units=units or UnitSystem(),
        )

    @property
    def delta_x(self) -> float:
        """The delta x of the system."""
        return self.params[0]

    @property
    def barrier_energy(self) -> float:
        """The barrier energy of the system."""
        return self.params[1]

    @override
    @property
    def sampling_domain(self) -> tuple:
        """The domain over which the equilibrium x-density should be sampled."""
        return ((-self.delta_x / 2, self.delta_x / 2),)

    @override
    @property
    def normalized_units(self) -> UnitSystem:
        """The units suited to a simulation of the system."""
        return UnitSystem(
            Boltzmann=1 / self.temperature,
            angstrom=self.units.angstrom / self.delta_x,
            atomic_mass=self.units.atomic_mass / self.m,
        )

    @override
    def with_units(self, units: UnitSystem) -> PeriodicSystem1D:
        """Return the normalized parameters of the simulation."""
        mass_factor = units.mass_factor / self.units.mass_factor
        time_factor = units.time_factor / self.units.time_factor
        length_factor = units.length_factor / self.units.length_factor
        energy_factor = units.energy_factor / self.units.energy_factor
        return PeriodicSystem1D(
            gamma=self.gamma / time_factor,
            temperature=self.temperature,
            m=self.m * mass_factor,
            delta_x=self.delta_x * length_factor,
            barrier_energy=self.barrier_energy * energy_factor,
            n_dim=self.n_dim,
            units=units,
        )


def _get_potential_expr_fcc() -> sp.Expr:
    """Return the potential energy expression for a 2D FCC lattice."""
    x0, x1 = sp.symbols("x0 x1")
    s0, s1 = sp.symbols("s0 s1")
    c = sp.Integer(2) * sp.pi / (sp.sqrt(3.0) * s0)

    kx0 = c * (sp.Integer(-1) / sp.sqrt(3.0))
    kx1 = c * sp.Integer(1)

    ky0 = c * (sp.Integer(2) / sp.sqrt(3.0))
    ky1 = 0.0

    arg1 = x0 * kx0 + x1 * kx1
    arg2 = x0 * ky0 + x1 * ky1
    arg3 = arg1 + arg2

    cos_sum = sp.cos(arg1) + sp.cos(arg2) + sp.cos(arg3)

    return sp.Integer(2) * s1 * cos_sum + sp.Integer(3) * s1


class PeriodicSystemFCC(System):
    """Parameters for the face-centered cubic periodic system."""

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
        potential = _get_potential_expr_fcc()

        super().__init__(
            gamma=gamma,
            temperature=temperature,
            m=m,
            potential=(2, potential),
            params=(delta_x, barrier_energy),
            units=units or UnitSystem(),
        )

    @property
    def delta_x(self) -> float:
        """The delta x of the system."""
        return self.params[0]

    @property
    def barrier_energy(self) -> float:
        """The barrier energy of the system."""
        return self.params[1]

    @override
    @property
    def sampling_domain(self) -> tuple:
        """Fractional coordinate domain spanning one primitive cell."""
        return (
            (0, 1),
            (0, 1),
        )

    @property
    def lattice_vectors(self) -> np.ndarray:
        """Primitive lattice vectors (as columns) for the hexagonal unit cell."""
        a1 = np.array([0.0, np.sqrt(3.0) * self.delta_x])
        a2 = np.array([1.5 * self.delta_x, np.sqrt(3.0) / 2.0 * self.delta_x])
        return np.stack([a1, a2], axis=1)

    @override
    @property
    def normalized_units(self) -> UnitSystem:
        """The units suited to a simulation of the system."""
        return UnitSystem(
            Boltzmann=1 / self.temperature,
            angstrom=self.units.angstrom / self.delta_x,
            atomic_mass=self.units.atomic_mass / self.m,
        )

    @override
    def with_units(self, units: UnitSystem) -> PeriodicSystemFCC:
        """Return the parameters of the system in the specified units."""
        length_factor = units.length_factor / self.units.length_factor
        mass_factor = units.mass_factor / self.units.mass_factor
        energy_factor = units.energy_factor / self.units.energy_factor
        time_factor = units.time_factor / self.units.time_factor
        return PeriodicSystemFCC(
            gamma=self.gamma / time_factor,
            temperature=self.temperature,
            m=self.m * mass_factor,
            delta_x=self.delta_x * length_factor,
            barrier_energy=self.barrier_energy * energy_factor,
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


class KramersSystem1D(System):
    """Parameters representing a periodic double harmonic system."""

    def __init__(
        self,
        *,
        params: KramersParameters | None = None,
        gamma: float | jax.Array | None = None,
        kbt: float | jax.Array | None = None,
        m: float | jax.Array | None = None,
        omega_well: float | jax.Array | None = None,
        omega_barrier: float | jax.Array | None = None,
        barrier_energy: float | jax.Array | None = None,
        n_dim: int = 1,
        units: UnitSystem | None = None,
    ) -> None:
        # Fallback to creating a local KramersParameters if individual arguments are passed
        if params is None:
            params = KramersParameters(
                gamma=gamma,  # ty: ignore[invalid-argument-type]
                kbt=kbt,  # ty: ignore[invalid-argument-type]
                m=m,  # ty: ignore[invalid-argument-type]
                omega_well=omega_well,  # ty: ignore[invalid-argument-type]
                omega_barrier=omega_barrier,  # ty: ignore[invalid-argument-type]
                barrier_energy=barrier_energy,  # ty: ignore[invalid-argument-type]
            )
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

        potential = sp.Piecewise(
            (0.5 * s0**2 * periodic_x**2, periodic_x <= x_meet),
            (
                s2 - 0.5 * s1**2 * (periodic_x - x_0) ** 2,
                periodic_x >= x_meet,
            ),
            (0, True),
        )

        super().__init__(
            gamma=params.gamma,
            # Note: this currently assumes Boltzmann = 1
            temperature=params.kbt,
            m=params.m,
            potential=(n_dim, potential),
            params=(
                params.omega_well,
                params.omega_barrier,
                params.barrier_energy,
            ),
            units=units or UnitSystem(),
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
            m=self.m,
            kbt=self.temperature,
            gamma=self.gamma,
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

    @property
    def normalized_units(self) -> UnitSystem:
        """The units suited to a simulation of the system."""
        return UnitSystem(
            Boltzmann=1 / self.temperature,
            angstrom=self.units.angstrom / self.delta_x,
            atomic_mass=self.units.atomic_mass / self.m,
        )

    def with_units(self, units: UnitSystem) -> KramersSystem1D:
        """Return the parameters of the system in the specified units."""
        mass_factor = units.mass_factor / self.units.mass_factor
        time_factor = units.time_factor / self.units.time_factor
        energy_factor = units.energy_factor / self.units.energy_factor
        return KramersSystem1D(
            m=self.m * mass_factor,
            params=KramersParameters(
                omega_well=self.omega_well / time_factor,
                omega_barrier=self.omega_barrier / time_factor,
                barrier_energy=self.barrier_energy * energy_factor,
                m=self.m * mass_factor,
                kbt=self.kbt * energy_factor,
                gamma=self.gamma / time_factor,
            ),
            n_dim=self.n_dim,
            units=units,
        )

    @classmethod
    def create_canonical(
        cls,
        *,
        gamma: float | jax.Array,
        kbt: float | jax.Array,
        m: float | jax.Array,
        omega_well: float | jax.Array,
        omega_barrier: float | jax.Array,
        barrier_energy: float | jax.Array,
        n_dim: int = 1,
    ) -> CanonicalSystem:
        """Helper to return the CanonicalSystem directly from JAX/float primitives."""
        return cls(
            gamma=gamma,
            kbt=kbt,
            m=m,
            omega_well=omega_well,
            omega_barrier=omega_barrier,
            barrier_energy=barrier_energy,
            n_dim=n_dim,
        ).as_canonical()


def get_diffusion_time(system: System, characteristic_length: float) -> float:
    """Return the average time for a particle to traverse a characteristic length."""
    return np.sqrt(system.m * characteristic_length**2 / system.kbt)


def get_characteristic_periodic_mass(system: PeriodicSystem1D) -> float:
    """Return the characteristic mass for a 1D periodic system."""
    return system.kbt * system.delta_x**2 / system.gamma**2
